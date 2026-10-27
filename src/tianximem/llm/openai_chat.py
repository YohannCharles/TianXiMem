"""OpenAI 兼容的 `/chat/completions` 客户端——**开发期与提交期共用的那一个实现**。

```http
POST {base_url}/chat/completions        # base_url 形如 https://memory2.021130.xyz/v1
Authorization: Bearer <key>
Content-Type: application/json

{"model": "...", "messages": [{"role": "user", "content": "<prompt>"}],
 "temperature": 0.0, "max_tokens": 8}
```

## 为什么只有一个实现

开发模型与提交模型使用同一套 OpenAI 兼容线格式，差别只在 `base_url`、密钥和模型名。
这些值都来自环境配置，因此一个客户端实现即可覆盖两种部署。Embedding 适配器则因
批次上限等模型 API 差异而分开。

## 端点身份在 `.env`，且**两个对话端点各有一把 key**

`AML_BASE_URL` / `AML_API_KEY`（主）、`AML_BASE_URL_2` / `AML_API_KEY_2`（次）。
⚠ **url 与 key 成对**：混用/对调都是 **401**，而 401 只在真发请求时才出现。
服务侧只接**一对**（主端点）——`AML_ENDPOINT_INDEX` 那套轮转是 **harness 的调度概念**
（"这一次判分用第几个端点"），不是这台机器的配置，所以**不进服务链路**。
"""

from __future__ import annotations

import time

import httpx

from tianximem.common.tokens import LLM_MAX_INPUT_TOKENS, TokenCounter
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
    """OpenAI 兼容的对话端点，用于提交模型和开发期替代模型。

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
        counter: TokenCounter | None = None,
        max_input_tokens: int = LLM_MAX_INPUT_TOKENS,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url 与 api_key 都不得为空")
        if max_input_tokens <= 0:
            raise ValueError("LLM 输入 token 上限必须为正")
        # `model` 允许为空：部分网关会忽略该字段。是否需要强制配置由装配和启动校验决定。
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client
        self._counter = counter
        self._max_input_tokens = max_input_tokens

        #: 诊断量，不进入 Search 响应。
        self.calls = 0
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None
        self.last_prompt_tokens: int | None = None
        self.last_sent_tokens: int | None = None
        self.last_dropped_tokens = 0
        self.last_truncated = False
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
        prompt = self._fit_prompt(prompt)
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

    def _fit_prompt(self, prompt: str) -> str:
        """把过长的提示词截成不超过输入上限的最长 token 前缀。"""
        self.last_prompt_tokens = None
        self.last_sent_tokens = None
        self.last_dropped_tokens = 0
        self.last_truncated = False
        if self._counter is None:
            return prompt

        token_ids = self._counter.encode_ids(prompt)
        prompt_tokens = len(token_ids)
        self.last_prompt_tokens = prompt_tokens
        if prompt_tokens <= self._max_input_tokens:
            self.last_sent_tokens = prompt_tokens
            return prompt

        fitted = self._counter.decode_ids(token_ids[: self._max_input_tokens])
        self.last_sent_tokens = self._max_input_tokens
        self.last_dropped_tokens = prompt_tokens - self._max_input_tokens
        self.last_truncated = True
        return fitted

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
            # 空白不是合法输出；在这里失败可区分端点返回空内容与模型输出无效。
            raise LlmUnavailable("`message.content` 是空白——端点回了空内容")
        return text
