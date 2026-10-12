"""OpenAI 兼容的 `/chat/completions` 客户端——**开发期与提交期共用的那一个实现**。

```http
POST {base_url}/chat/completions        # base_url 形如 https://memory2.021130.xyz/v1
Authorization: Bearer <key>
Content-Type: application/json

{"model": "...", "messages": [{"role": "user", "content": "<prompt>"}],
 "temperature": 0.0, "max_tokens": 8}
```

## 为什么只有一个实现，而不是 `gpt_4o_mini.py` + `qwen_local.py`

`llm/CLAUDE.md` 原计划按模型分两个文件。**实测后并成一个**：开发期的
`Qwen/Qwen3.5-9B`（`memory2` / `memory3`）与提交期的 `gpt-4o-mini`（DashScope）
**说的是同一套线格式**（OpenAI 兼容的 `/chat/completions`），差别只有
`base_url` / `key` / `model` 三个值——而**那三个值全部来自 `.env`**。

⇒ 按模型分文件会造出**两个内容相同的类**，而它们的唯一差别只能靠"哪一行 import 了谁"
表达；那种差别**看不出来、也 diff 不出来**。R1 的接缝由
[`base.py`](./base.py) 的协议承担，**不由文件名承担**。

⚠ 这条与 `embed/` 的做法一致：那里也是"按模型名挑实现"，但**判据是模型名本身**
（`build_embedder`），因为 `text-embedding-v4` 与 Qwen3-Embedding **真的不同**
（批大小是端点硬限）。**这里没有那类差异**，所以不需要那个判据。

## 端点身份在 `.env`，且**两个对话端点各有一把 key**（D39）

`AML_BASE_URL` / `AML_API_KEY`（主）、`AML_BASE_URL_2` / `AML_API_KEY_2`（次）。
⚠ **url 与 key 成对**：混用/对调都是 **401**，而 401 只在真发请求时才出现。
服务侧只接**一对**（主端点）——`AML_ENDPOINT_INDEX` 那套轮转是 **harness 的调度概念**
（"这一次判分用第几个端点"），不是这台机器的配置，所以**不进服务链路**。
"""

from __future__ import annotations

import time

import httpx

from tianximem.llm.base import DEFAULT_MAX_OUTPUT_TOKENS, LlmUnavailable

__all__ = ["OpenAiCompatChat"]


def _usage(body: object) -> dict[str, int] | None:
    """把响应里的 `usage` 收成 `{名字: 整数}`；**形状不符就是 `None`**（不是错误）。

    ⚠ **收它不是为了让别的代码依赖它**：token 数是**成本分析的输入**，
    而"这一档端点回不回 usage / 字段叫什么"是**端点身份**的一部分
    （`prompt_tokens` 是 OpenAI 的叫法，别家可能叫别的）⇒ 只做**保守收集**：
    非整数一律丢掉，一个字段都没有就返回 `None`。
    **不给它加校验、不让它参与任何判定**——它坏掉不该影响一次分类。
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    counts = {
        str(name): value
        for name, value in usage.items()
        # ⚠ `bool` 要先挡：`True` 是 `int` 的实例，会当成 1 混进来
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return counts or None


class OpenAiCompatChat:
    """OpenAI 兼容的对话端点（§2.3 的 `gpt-4o-mini` / 开发期替代模型）。

    ⚠ **本实现只管一件事：把提示词变成文本。** 它不解析路由、不重试、不做结构化输出——
    解析是 [`../retrieve/router.py`](../retrieve/router.py) 的契约（见 `base.py` 的说明）。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url 与 api_key 都不得为空")
        # ⚠ **`model` 允许为空、但记一条**：真实网关对它的态度是"校验或忽略"两种
        #   （`docs/config-reference.md` §9 的表）——**忽略型的网关空着也能跑**。
        #   这里不抛，是因为"服务起不来"是比"分类降级"更坏的失败（同 `build_reranker` 的口径）；
        #   而**该不该起**由装配侧决定，不由本类决定。
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

        #: 诊断量（§14）。**不进响应**——响应只有 §2.1 那四个字段。
        self.calls = 0
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None
        #: 端点回的 `usage`（token 计数）。**`None` = 这一档端点没回它**——
        #: 与 `last_latency_ms` 同一个性质：只给诊断与成本分析读，**不是契约**。
        #: ⚠ 它**不进 `LlmClient` 协议**：协议是"文本进、文本出"（见 `base.py`），
        #: 而 token 数是**某一种端点的实现细节**——把它写进协议会让"换一个后端"
        #: 变成"必须也报 token 数"，而本地模型/无 usage 的封装根本报不出来。
        self.last_usage: dict[str, int] | None = None

    # ── 标识 ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        """完整端点（诊断用；**不含 key**）。"""
        return f"{self._base_url}/chat/completions"

    # ── 调用 ───────────────────────────────────────────────────────────

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        try:
            body = self._post(prompt, max_tokens=max_tokens, temperature=temperature)
            return self._parse(body)
        except httpx.HTTPError as exc:  # 网络层：连接失败、超时（TimeoutException 是它的子类）
            reason = f"{type(exc).__name__}: {exc}"
            self.last_error = reason
            raise LlmUnavailable(f"LLM 调用失败（{self.url}）：{reason}") from exc
        except LlmUnavailable as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001 — 见下
            # ★ **兜底捕获与 reranker 同一条理由**：LLM 是**不被规则保证可用**的远程组件，
            #   "端点失败或超时时退回默认路径，而不是报错"。一次没分好类 vs 整个 Search 500
            #   ——后者要烧掉 Full 的一次真机会。
            # ⚠ **但它必须留下痕迹**：`last_error` 里带着**异常类型**，所以"未预期的 bug"
            #   与"端点不可用"在诊断上分得开。
            message = f"LLM 未预期的失败（{type(exc).__name__}）：{exc}"
            self.last_error = message
            raise LlmUnavailable(message) from exc

    def _post(self, prompt: str, *, max_tokens: int, temperature: float) -> object:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        self.calls += 1
        try:
            resp = self._http().post(
                self.url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code != 200:
            # 401（key 与 host 不配对）/ 403 / 404（校验型网关不认识的 model）/ 422 / 5xx
            # 都走这里 ⇒ 一律降级。
            raise LlmUnavailable(f"LLM 返回 {resp.status_code}（{self.url}）：{resp.text[:300]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise LlmUnavailable(f"LLM 的响应不是合法 JSON（{self.url}）：{exc}") from exc
        self.last_usage = _usage(body)
        return body

    def _parse(self, body: object) -> str:
        """从 `choices[0].message.content` 取文本。

        | 拒绝 | 它其实是 | 不拒绝的后果 |
        | --- | --- | --- |
        | 没有 `choices` / 不是数组 / 空数组 | 形状不符 | 崩在后面的下标上 |
        | 项不是对象 / 没有 `message` | 形状不符 | 同上 |
        | `content` 不是字符串 | 形状不符（多段数组） | `str()` 一个数组，得到无法解析的文本 |
        | `content` 是空白 | **模型什么都没说** | 当成一次合法输出，把"端点坏了"记成"分类成功" |

        ⚠ **多余的键一律忽略**（真实响应带 `id` / `created` / `usage` / `finish_reason`）——
        "严格"不等于"拒绝我们不需要的字段"（同 `RemoteReranker._parse`）。
        """
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LlmUnavailable(f"响应里没有非空的 `choices`：{str(body)[:200]}")
        first = choices[0]
        if not isinstance(first, dict):
            raise LlmUnavailable(f"`choices[0]` 不是对象：{str(first)[:120]}")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LlmUnavailable(f"`choices[0].message` 不是对象：{str(message)[:120]}")
        content = message.get("content")
        if not isinstance(content, str):
            raise LlmUnavailable(f"`message.content` 必须是字符串，收到 {type(content).__name__}")
        text = content.strip()
        if not text:
            # ⚠ **空白 ≠ 合法输出**：把它放进解析层会变成"分类成功、结果是空串"，
            #   于是"端点一直在回空"这个故障在 §14 里长得像"模型答得怪"。
            raise LlmUnavailable("`message.content` 是空白——端点回了空内容")
        return text
