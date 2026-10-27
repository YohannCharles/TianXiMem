"""Rerank —— **远端 cross-encoder 的接入点**。

```text
Hybrid Retrieval → RRF → memory_id 稳定去重 → 【rerank（恰好一次）】 → 扩窗 → 合并 → 打包
                                                     ↑ 本模块在这
```

* [`Reranker`][Reranker] —— 协议。**只收文本、只回分数**，不知道 `memory_id` / SQLite / Qdrant
* [`RemoteReranker`][RemoteReranker] —— 主网关（`memory.021130.xyz`）的 HTTP 客户端
* [`RerankUnavailable`][RerankUnavailable] —— "可降级"的唯一信号

本文件保留三条关键约束：

1. **`top_n` 是静默截断旋钮**：传了就只回 k 条 ⇒ 其余候选没有分数。
   本实现**不传**，并**无条件校验 `index` 集合恰好等于输入集合**。
2. **`score()` 返回按输入位置对齐的分数**，重排留给调用方，因此候选集合不会改变。
3. **`rerank.envelope` 与 `TIANXIMEM_RERANKER_BASE_URL` 是同一端点身份的两部分**；
   两者不匹配会表现为每次检索静默降级。

## 为什么 rerank 只调用**一次**

最终 rerank 只调用一次。扩窗之后不再重排；否则名次依赖扩窗结果，而扩窗又依赖名次。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from tianximem.common.config import DEFAULT_RERANK_ENVELOPE, RERANK_ENVELOPES

__all__ = ["RemoteReranker", "RerankUnavailable", "Reranker"]

logger = logging.getLogger(__name__)

#: Total attempts, including the first call. Used for local rerankers that may
#: briefly queue under concurrent Search load.
DEFAULT_RERANK_ATTEMPTS: int = 4
DEFAULT_RERANK_BACKOFF_S: tuple[float, ...] = (0.2, 0.5, 1.0)


class RerankUnavailable(RuntimeError):
    """reranker 端点不可用（超时、连接失败、非 2xx、响应形状不符、候选映射失败）。

    ⚠ **它是"可降级"，不是"可忽略"**：调用方**必须**捕获它、退回未重排的顺序、
    **并把这件事记下来**——而不是让它冒泡成 5xx。
    理由：reranker 是**唯一位于提交链路关键路径上、又不被规则保证可用**的组件。
    Full 只有 2 次，一次真机会。
    """


@runtime_checkable
class Reranker(Protocol):
    """给候选重新打分/排序的东西。

    ⚠ **协议刻意只收文本、只回分数**——它不知道 `memory_id`、不知道位置、
    不知道 Qdrant。换一个 reranker 实现（远端 HTTP / 本地 cross-encoder）不需要
    改本包任何其它文件。
    """

    @property
    def name(self) -> str:
        """模型标识。"""
        ...

    def score(self, *, query: str, documents: Sequence[str]) -> Sequence[float]:
        """给每个 `documents[i]` 一个相关性分数（**越大越相关**）。

        * 返回长度**必须**与 `documents` 一致，否则抛 `RerankUnavailable`
          （形状不符与端点挂了一样不可用——**静默按前缀对齐**会让后半段名次错位）
        * 不可用时抛 [`RerankUnavailable`][RerankUnavailable]
        """
        ...


class RemoteReranker:
    """主网关上的 Qwen3-Reranker-4B。

    ⚠ **模型 id 由 `.env` 的 `TIANXIMEM_RERANKER_MODEL` 给**，而且**必须填对**：
    网关（vllm 直服）**校验**它——**填错就拿到 404 并降级，而服务不报错**。

    ⚠ 它**只负责"文本 → 分数"**：取正文、映射回 `memory_id`、重排、重新编号
    全部在 [`../service/pipeline.py`](../service/pipeline.py)——
    这样 `rank/` 不会长出对 SQLite 的依赖，而"候选集合不可能改变"是结构性的（见模块 docstring）。

    ``score()`` retries transport and response-shape failures up to
    ``max_attempts`` total calls. This absorbs temporary local-model queueing
    under concurrent Search load. The caller still receives
    ``RerankUnavailable`` after the final attempt and can fall back to RRF.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        envelope: str = DEFAULT_RERANK_ENVELOPE,
        client: httpx.Client | None = None,
        max_attempts: int = DEFAULT_RERANK_ATTEMPTS,
        backoff_s: Sequence[float] = DEFAULT_RERANK_BACKOFF_S,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url 与 api_key 都不得为空")
        if max_attempts < 1:
            raise ValueError("rerank max_attempts 必须 ≥ 1")
        if len(backoff_s) < max_attempts - 1:
            raise ValueError("rerank backoff_s 必须覆盖所有重试间隔")
        # 构造期拒绝未知信封；否则错误配置会表现为每次检索静默降级。
        if envelope not in RERANK_ENVELOPES:
            raise ValueError(f"envelope 只能是 {sorted(RERANK_ENVELOPES)}，收到 {envelope!r}")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._envelope = envelope
        self._client = client
        self._max_attempts = max_attempts
        self._backoff_s = tuple(float(value) for value in backoff_s[: max_attempts - 1])

        #: 诊断量，不进入 Search 响应。
        self.calls = 0
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None

    # ── 标识 ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        """完整端点（诊断用；**不含 key**）。"""
        return f"{self._base_url}/rerank"

    # ── 调用 ───────────────────────────────────────────────────────────

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def score(self, *, query: str, documents: Sequence[str]) -> list[float]:
        """给每篇文档打分，**按输入位置对齐**返回。

        **整批候选一次发送**，不按候选循环调用端点。

        ⚠ **空输入在本地短路**：端点对空 `documents` 返回 400，
        而"没有候选"根本不是错误（它是合法的检索结果）。
        """
        docs = list(documents)
        if not docs:
            return []

        last_exc: RerankUnavailable | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                body = self._post(query=query, documents=docs)
                scores = self._parse(body, expected=len(docs))
                if attempt > 1:
                    logger.info(
                        "rerank succeeded after retry model=%s attempt=%d/%d",
                        self._model,
                        attempt,
                        self._max_attempts,
                    )
                return scores
            except RerankUnavailable as exc:
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 — 见下
                # reranker 是可选远程组件；端点失败或超时时退回未重排的 RRF 顺序。
                # 同时把异常类型写入 last_error，使端点不可用与未预期缺陷仍可区分。
                message = f"rerank 未预期的失败（{type(exc).__name__}）：{exc}"
                last_exc = RerankUnavailable(message)

            self.last_error = str(last_exc)
            if attempt < self._max_attempts:
                delay = self._backoff_s[attempt - 1]
                logger.warning(
                    "rerank attempt failed model=%s attempt=%d/%d retry_in_s=%.3f error=%s",
                    self._model,
                    attempt,
                    self._max_attempts,
                    delay,
                    last_exc,
                )
                time.sleep(delay)

        assert last_exc is not None
        logger.error(
            "rerank retries exhausted model=%s attempts=%d error=%s",
            self._model,
            self._max_attempts,
            last_exc,
        )
        raise last_exc

    def _post(self, *, query: str, documents: Sequence[str]) -> object:
        """发一次请求并解出 JSON。**任何 HTTP 层问题都转成 `RerankUnavailable`。**"""
        payload: dict[str, object] = {
            "model": self._model,
            "documents": list(documents),
            # ⚠ **刻意不传 `top_n`**：传了会静默截断（见模块 docstring 的陷阱一节）。
            #    不传 ⇒ 端点返回全部候选 ⇒ `_parse` 的集合校验才有意义。
        }
        # ⚠ **查询字段的名字与类型都跟着端点变**（由 `rerank.envelope` 选择）：
        #    `queries`（数组）是 vLLM 原生 score 形状，`query`（单个字符串）是自研封装的。
        #    两者互斥；字段不匹配会收到 400 / 422 并转为降级，响应本身仍然合法。
        if self._envelope == "queries":
            payload["queries"] = [query]
        else:
            payload["query"] = query
        started = time.perf_counter()
        self.calls += 1
        try:
            resp = self._http().post(
                self.url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.HTTPError as exc:  # 网络层：连接失败、超时（TimeoutException 是它的子类）
            reason = f"{type(exc).__name__}: {exc}"
            raise RerankUnavailable(f"rerank 调用失败（{self.url}）：{reason}") from exc
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code != 200:
            # 403（坏 key）/ 422（body 非法）/ 5xx 都走这里 ⇒ 一律降级
            raise RerankUnavailable(
                f"rerank 返回 {resp.status_code}（{self.url}）：{resp.text[:300]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise RerankUnavailable(f"rerank 的响应不是合法 JSON（{self.url}）：{exc}") from exc

    def _parse(self, body: object, *, expected: int) -> list[float]:
        """把 `results` 折成**按输入位置对齐**的分数表。

        这里守住一条不变量：**reranker 只能改变顺序，不能改变候选集合。**
        四条拒绝条件各自对应一种"集合被改动了"的方式：

        | 拒绝 | 它其实是 | 不拒绝的后果 |
        | --- | --- | --- |
        | `index` 越界 | **未知 candidate** | 我们会拿着一个不存在的位次去取候选 |
        | 同一个 `index` 两次 | **重复 candidate** | 重复项会被排两次、占两个名额 |
        | 有 `index` 没出现 | **静默丢 candidate** | 少一条证据，**而响应看起来完全正常** |
        | 容器不是数组 / 项不是对象 / 分数不是有限数 | 格式非法 | 崩在后面的算术里 |

        ⚠ **容器与分数键各收两个**（`results` / `data`，`score` / `relevance_score`）——
        **四种组合都过**。收两个不是"猜"——**同一台网关上的
        信封并不稳定**（它随网关侧的部署变）；
        `index` 集合校验把它们的**共同不变量**守住了，所以多认一个键**不会削弱任何检查**。

        ⚠ **多余的键一律忽略**（真实响应就带 `object` / `created` / `usage` / `id` /
        `document` / `text`）——"严格"不等于"拒绝我们不需要的字段"；
        把额外字段当错误会让实现**对着真端点反而失败**。
        """
        # ⚠ **容器名收两个**：`results`（自研封装）与 `data`（vLLM 原生 score 形状——
        #    的 nginx rewrite 之后走的就是它）。
        #   与分数键收两个同理：**共同的不变量由下面的 `index` 集合校验守住**，
        #   多认一个容器名**不削弱任何检查**。
        results = body.get("results", body.get("data")) if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise RerankUnavailable(
                f"响应形状既不是 {{'results': [...]}} 也不是 {{'data': [...]}}：{str(body)[:200]}"
            )

        scores: list[float | None] = [None] * expected
        for item in results:
            if not isinstance(item, dict):
                raise RerankUnavailable(f"`results` 里出现了非对象项：{str(item)[:120]}")
            index = item.get("index")
            # ⚠ `bool` 要先挡：`True` 是 `int` 的实例，会当成 index=1 混过去
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankUnavailable(f"`index` 必须是整数，收到 {index!r}")
            if not 0 <= index < expected:
                raise RerankUnavailable(
                    f"`index={index}` 越界（输入只有 {expected} 篇）——reranker 返回了**未知候选**"
                )
            raw_score = item.get("score", item.get("relevance_score"))
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                raise RerankUnavailable(
                    f"`score` / `relevance_score` 必须是数字，收到 {raw_score!r}"
                )
            if not math.isfinite(float(raw_score)):
                raise RerankUnavailable(f"`score` 必须是有限数，收到 {raw_score!r}")  # NaN/Inf
            if scores[index] is not None:
                raise RerankUnavailable(f"`index={index}` 出现了两次——reranker 返回了**重复候选**")
            scores[index] = float(raw_score)

        missing = [i for i, value in enumerate(scores) if value is None]
        if missing:
            raise RerankUnavailable(
                f"缺了 {len(missing)} 个候选的分数（前几个 index：{missing[:5]}）——"
                "reranker **静默丢了候选**；候选集合必须与输入一致"
            )
        # `scores` 里已经是 float（上面逐个赋过），但类型收窄要显式说给类型检查器听
        return [value for value in scores if value is not None]
