"""Rerank —— **远端 cross-encoder 的接入点**（§11.2 / D12）。

```text
Hybrid Retrieval → RRF → memory_id 稳定去重 → 【rerank（恰好一次）】 → 扩窗 → 合并 → 打包
                                                     ↑ 本模块在这
```

* [`Reranker`][Reranker] —— 协议。**只收文本、只回分数**，不知道 `memory_id` / SQLite / Qdrant
* [`RemoteReranker`][RemoteReranker] —— 主网关（`memory.021130.xyz`）的 HTTP 客户端
* [`RerankUnavailable`][RerankUnavailable] —— "可降级"的唯一信号

## 线格式：**实测出来的，不是猜的**（可复现的探针：`tools/probe_reranker.py`）

```http
POST {base_url}/rerank          # base_url 形如 https://memory.021130.xyz/v1
Authorization: Bearer <key>
Content-Type: application/json

{"model": "Qwen3-Reranker-4B", "query": "...", "documents": ["...", "..."]}
```

```json
{
  "model": "/data/…/Qwen3-Reranker-4B",
  "results": [
    {"index": 1, "score": 0.8359375, "text": "the train leaves at 09:42"},
    {"index": 3, "score": 0.147,     "text": "the departure is at 09:42 …"}
  ]
}
```

| 观察 | 后果 |
| --- | --- |
| `results[].index` = **输入 `documents` 的下标** | 这是唯一能把结果映射回候选的东西 |
| `results` **按 `score` 降序** | ⚠ **不依赖它**：`score()` 按**输入位置**对齐返回，排序在调用方 |
| `score` **越大越相关** | 实测：相关句 0.836 / 无关句 1.4e-05 |
| `model` 字段**被网关忽略** | 响应里回的是服务端路径。所以**不能拿它做路由或校验** |
| 不传 `top_n` ⇒ **返回全部** | ★ 所以**不传**——见下 |
| 传 `top_n=k` ⇒ **只回 k 条** | ★ **这是一个静默丢候选的旋钮**（实测 `top_n=2` 只回 2 条） |
| 100 篇 ≈ 2.2s、200 篇 ≈ 5.3s | 与 `top_k=100` 的规模匹配（默认超时 30s 有约 10 倍余量） |
| 空 `documents` ⇒ **400** | ⇒ 空输入**在本地短路**，不打网络 |
| 坏 key ⇒ **403**；body 非法 ⇒ **422** | ⇒ "非 2xx ⇒ 降级"这一条覆盖了它们 |

> ### ⚠ `top_n` 是本文件的头号陷阱
>
> 它**默认不限**（返回全部），而一旦传了就**静默截断**。截断的后果不是"少几条"，
> 而是：**`score()` 无法为被截掉的候选给出分数** ⇒ 只能降级或错位对齐。
> 所以本实现**不传 `top_n`**，并且**无条件校验 index 集合恰好等于输入集合**——
> 即使将来有人加了 `top_n`，也会**响亮降级**而不是静默少算几条。

## 为什么 `score()` 返回的是"按输入位置对齐"的分数

**`scores[i]` 对应 `documents[i]` ⇒ 调用方只对同一个列表重排 ⇒ 候选集合不可能改变。**
§13 要求"关掉 rerank 只改变顺序、不改变候选数量"，§2.2 要求"精确计数"——
若这里返回的是"排好序的 id 列表"，那么"reranker 少回了一条"就会变成
**候选真的少了一条**，而且**看起来完全正常**（响应依旧合法，只是少一项）。
把重排留给调用方，就**没有第二条代码路径**能改变集合了。

## 为什么 rerank 只调用**一次**

rerank 是**唯一一次**模型调用（§7.2 的"每查询恰好一次"同款纪律），
`prefetch`/`RRF` 只是两次单路查询的融合。**不做第二次 rerank**——包括
"扩窗之后再排一遍"：那会让名次依赖扩窗的结果，而扩窗的输入又是名次，
成了一个没有不动点的循环。
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

__all__ = ["RemoteReranker", "RerankUnavailable", "Reranker"]


class RerankUnavailable(RuntimeError):
    """reranker 端点不可用（超时、连接失败、非 2xx、响应形状不符、候选映射失败）。

    ⚠ **它是"可降级"，不是"可忽略"**：调用方**必须**捕获它、退回未重排的顺序、
    **并把这件事记下来**（D12）——而不是让它冒泡成 5xx。
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
        """模型标识（进 run record 的配置指纹——**提交时不得更换**，D12）。"""
        ...

    def score(self, *, query: str, documents: Sequence[str]) -> Sequence[float]:
        """给每个 `documents[i]` 一个相关性分数（**越大越相关**）。

        * 返回长度**必须**与 `documents` 一致，否则抛 `RerankUnavailable`
          （形状不符与端点挂了一样不可用——**静默按前缀对齐**会让后半段名次错位）
        * 不可用时抛 [`RerankUnavailable`][RerankUnavailable]
        """
        ...


class RemoteReranker:
    """主网关上的 `Qwen3-Reranker-4B`（§11.2；选型见 D12，**提交时不得更换**）。

    ⚠ 它**只负责"文本 → 分数"**：取正文、映射回 `memory_id`、重排、重新编号
    全部在 [`../service/pipeline.py`](../service/pipeline.py)——
    这样 `rank/` 不会长出对 SQLite 的依赖，而"候选集合不可能改变"是结构性的（见模块 docstring）。

    ⚠ **不重试**：与 `embed/` 同一个理由——重试会与 AML 的 32 次重试叠加成两层，
    失败路径变得无法归因。这里失败就是失败，由调用方降级。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url 与 api_key 都不得为空")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

        #: 诊断量（§14）。**不进响应**——响应只有 §2.1 那四个字段。
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

        **整批候选一次发送**（§7.2 的"每查询恰好一次"同款纪律），不做按候选的循环。

        ⚠ **空输入在本地短路**：端点对空 `documents` 返回 400，
        而"没有候选"根本不是错误（它是合法的检索结果）。
        """
        docs = list(documents)
        if not docs:
            return []

        try:
            body = self._post(query=query, documents=docs)
            return self._parse(body, expected=len(docs))
        except RerankUnavailable as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001 — 见下
            # ★ **兜底捕获是 D12 的要求，不是偷懒**：reranker 是唯一不被规则保证可用的组件，
            #   "端点失败或超时时退回未重排的 RRF 顺序，而不是报错"。
            #   一次没排上序 vs 整个 Search 500——后者的代价是 Full 的一次真机会。
            #
            # ⚠ **但它必须留下痕迹**：`last_error` 里带着**异常类型**，
            #   所以"未预期的 bug"与"端点不可用"在诊断上分得开——
            #   这正是本项目反复强调的"静默失败比失败更坏"。
            message = f"rerank 未预期的失败（{type(exc).__name__}）：{exc}"
            self.last_error = message
            raise RerankUnavailable(message) from exc

    def _post(self, *, query: str, documents: Sequence[str]) -> object:
        """发一次请求并解出 JSON。**任何 HTTP 层问题都转成 `RerankUnavailable`。**"""
        payload: dict[str, object] = {
            "model": self._model,
            "query": query,
            "documents": list(documents),
            # ⚠ **刻意不传 `top_n`**：传了会静默截断（见模块 docstring 的陷阱一节）。
            #    不传 ⇒ 端点返回全部候选 ⇒ `_parse` 的集合校验才有意义。
        }
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

        这里守着 §11.2 的一条不变量：**reranker 只能改变顺序，不能改变候选集合。**
        四条拒绝条件各自对应一种"集合被改动了"的方式：

        | 拒绝 | 它其实是 | 不拒绝的后果 |
        | --- | --- | --- |
        | `index` 越界 | **未知 candidate** | 我们会拿着一个不存在的位次去取候选 |
        | 同一个 `index` 两次 | **重复 candidate** | 重复项会被排两次、占两个名额 |
        | 有 `index` 没出现 | **静默丢 candidate** | 少一条证据，**而响应看起来完全正常** |
        | `results` 不是数组 / 项不是对象 / `score` 不是有限数 | 格式非法 | 崩在后面的算术里 |

        ⚠ **多余的键一律忽略**（真实响应就带 `text` 与 `model`）——"严格"不等于
        "拒绝我们不需要的字段"；把额外字段当错误会让实现**对着真端点反而失败**。
        """
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise RerankUnavailable(f"响应形状不是 {{'results': [...]}}：{str(body)[:200]}")

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
            raw_score = item.get("score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                raise RerankUnavailable(f"`score` 必须是数字，收到 {raw_score!r}")
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
