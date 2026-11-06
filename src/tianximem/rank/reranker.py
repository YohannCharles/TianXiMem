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

DEFAULT_RERANK_ATTEMPTS: int = 4
DEFAULT_RERANK_BACKOFF_S: tuple[float, ...] = (0.2, 0.5, 1.0)


class RerankUnavailable(RuntimeError):
    pass


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, *, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class RemoteReranker:
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

        self.calls = 0
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        if self._envelope == "dashscope":
            return f"{self._base_url}/api/v1/services/rerank/text-rerank/text-rerank"
        return f"{self._base_url}/rerank"

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def score(self, *, query: str, documents: Sequence[str]) -> list[float]:
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
            except Exception as exc:  # noqa: BLE001
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
        if self._envelope == "dashscope":
            payload: dict[str, object] = {
                "model": self._model,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {"return_documents": False},
            }
        else:
            payload = {"model": self._model, "documents": list(documents)}
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
        except httpx.HTTPError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            raise RerankUnavailable(f"rerank 调用失败（{self.url}）：{reason}") from exc
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code != 200:
            raise RerankUnavailable(
                f"rerank 返回 {resp.status_code}（{self.url}）：{resp.text[:300]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise RerankUnavailable(f"rerank 的响应不是合法 JSON（{self.url}）：{exc}") from exc

    def _parse(self, body: object, *, expected: int) -> list[float]:
        results = body.get("results", body.get("data")) if isinstance(body, dict) else None
        if not isinstance(results, list) and isinstance(body, dict):
            output = body.get("output")
            if isinstance(output, dict):
                results = output.get("results")
        if not isinstance(results, list):
            raise RerankUnavailable(
                "响应形状不是 {'results': [...]}、{'data': [...]} 或 "
                f"{{'output': {{'results': [...]}}}}: {str(body)[:200]}"
            )

        scores: list[float | None] = [None] * expected
        for item in results:
            if not isinstance(item, dict):
                raise RerankUnavailable(f"`results` 里出现了非对象项：{str(item)[:120]}")
            index = item.get("index")
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
                raise RerankUnavailable(f"`score` 必须是有限数，收到 {raw_score!r}")
            if scores[index] is not None:
                raise RerankUnavailable(f"`index={index}` 出现了两次——reranker 返回了**重复候选**")
            scores[index] = float(raw_score)

        missing = [i for i, value in enumerate(scores) if value is None]
        if missing:
            raise RerankUnavailable(
                f"缺了 {len(missing)} 个候选的分数（前几个 index：{missing[:5]}）——"
                "reranker **静默丢了候选**；候选集合必须与输入一致"
            )
        return [value for value in scores if value is not None]
