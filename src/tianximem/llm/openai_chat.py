from __future__ import annotations

import time

import httpx

from tianximem.common.tokens import LLM_MAX_INPUT_TOKENS, TokenCounter
from tianximem.llm.base import DEFAULT_MAX_OUTPUT_TOKENS, LlmUnavailable

__all__ = ["OpenAiCompatChat"]


def _usage(body: object) -> dict[str, int] | None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    counts = {
        str(name): value
        for name, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return counts or None


class OpenAiCompatChat:
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
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client
        self._counter = counter
        self._max_input_tokens = max_input_tokens

        self.calls = 0
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None
        self.last_prompt_tokens: int | None = None
        self.last_sent_tokens: int | None = None
        self.last_dropped_tokens = 0
        self.last_truncated = False
        self.last_usage: dict[str, int] | None = None

    @property
    def name(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        return f"{self._base_url}/chat/completions"

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
        except httpx.HTTPError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.last_error = reason
            raise LlmUnavailable(f"LLM 调用失败（{self.url}）：{reason}") from exc
        except LlmUnavailable as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001
            message = f"LLM 未预期的失败（{type(exc).__name__}）：{exc}"
            self.last_error = message
            raise LlmUnavailable(message) from exc

    def _fit_prompt(self, prompt: str) -> str:
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
            raise LlmUnavailable(f"LLM 返回 {resp.status_code}（{self.url}）：{resp.text[:300]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise LlmUnavailable(f"LLM 的响应不是合法 JSON（{self.url}）：{exc}") from exc
        self.last_usage = _usage(body)
        return body

    def _parse(self, body: object) -> str:
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
            raise LlmUnavailable("`message.content` 是空白——端点回了空内容")
        return text
