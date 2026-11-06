from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import httpx
import numpy as np

from tianximem.common.render import TEMPLATE_VERSION
from tianximem.embed.truncate import truncate_for_embedding

__all__ = [
    "CachingEmbedder",
    "QUERY_CACHE_PREFIX",
    "DimNotKnownError",
    "DimensionMismatchError",
    "DiskVectorCache",
    "Embedder",
    "EmbeddingCoordinate",
    "EmbeddingError",
    "OpenAICompatEmbedder",
    "to_float32",
]

_CACHE_BUSY_TIMEOUT_MS: Final[int] = 5000

QUERY_CACHE_PREFIX: Final[str] = "query-adapter-v1:"


class EmbeddingError(RuntimeError):
    pass


class DimNotKnownError(RuntimeError):
    pass


class DimensionMismatchError(RuntimeError):
    pass


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_float32(values: Sequence[float]) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=np.float32)]


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingCoordinate:
    model: str
    render_template: str = TEMPLATE_VERSION
    dim: int | None = None

    def key(self) -> str:
        raw = json.dumps(
            {"model": self.model, "render_template": self.render_template, "dim": self.dim},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        suffix = "" if self.dim is None else f" dim={self.dim}"
        return f"{self.model} @ render={self.render_template}{suffix}"


class DiskVectorCache:
    def __init__(self, cache_dir: str | Path, coordinate: EmbeddingCoordinate) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._coordinate = coordinate
        self._path = self._dir / f"{coordinate.key()}.db"
        self._init_schema()
        self._verify_coordinate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None)
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {_CACHE_BUSY_TIMEOUT_MS}")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                " content_hash TEXT PRIMARY KEY,"
                " dim INTEGER NOT NULL,"
                " vec BLOB NOT NULL,"
                " created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta (k, v) VALUES ('coordinate', ?)",
                (self._coordinate.describe(),),
            )
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _verify_coordinate(self) -> None:
        with self.read() as conn:
            row = conn.execute("SELECT v FROM meta WHERE k = 'coordinate'").fetchone()
        if row is not None and row[0] != self._coordinate.describe():
            raise DimensionMismatchError(
                f"缓存文件的坐标系不符：文件里是 {row[0]!r}，当前是 "
                f"{self._coordinate.describe()!r}（{self._path}）"
            )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def coordinate(self) -> EmbeddingCoordinate:
        return self._coordinate

    def get(self, text: str) -> list[float] | None:
        with self.read() as conn:
            row = conn.execute(
                "SELECT dim, vec FROM vectors WHERE content_hash = ?", (_content_hash(text),)
            ).fetchone()
        if row is None:
            return None
        dim, blob = int(row[0]), bytes(row[1])
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.shape[0] != dim:  # pragma: no cover
            raise DimensionMismatchError(
                f"缓存条目自称 {dim} 维，实际 blob 是 {vec.shape[0]} 维（{self._path}）"
            )
        expected = self._coordinate.dim
        if expected is not None and dim != expected:
            raise DimensionMismatchError(
                f"缓存条目是 {dim} 维，但坐标系声明 {expected} 维"
                f"（{self._coordinate.describe()}；{self._path}）"
            )
        return [float(x) for x in vec]

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> int:
        now = int(time.time() * 1000)
        rows = []
        for text, vec in items:
            arr = np.asarray(vec, dtype=np.float32)
            rows.append((_content_hash(text), int(arr.shape[0]), arr.tobytes(), now))
        if not rows:
            return 0
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (content_hash, dim, vec, created_at)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return len(rows)

    def known_dims(self) -> set[int]:
        with self.read() as conn:
            return {int(r[0]) for r in conn.execute("SELECT DISTINCT dim FROM vectors")}

    def count(self) -> int:
        with self.read() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])


class CachingEmbedder:
    def __init__(self, inner: Embedder, cache: DiskVectorCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def inner(self) -> Embedder:
        return self._inner

    @property
    def cache(self) -> DiskVectorCache:
        return self._cache

    @property
    def dim(self) -> int:
        try:
            return self._inner.dim
        except DimNotKnownError:
            dims = self._cache.known_dims()
            if len(dims) == 1:
                return next(iter(dims))
            raise DimNotKnownError(
                "维度未知：内层尚未调用过，缓存里也没有（或有不只一种）维度"
            ) from None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, query=False)

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, query=True)

    def _encode(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []

        found: dict[str, list[float] | None] = {}
        misses: list[str] = []
        miss_keys: list[str] = []
        for text in texts:
            key = QUERY_CACHE_PREFIX + text if query else text
            if key in found:
                continue
            hit = self._cache.get(key)
            found[key] = hit
            if hit is None:
                misses.append(text)
                miss_keys.append(key)

        if misses:
            fresh = self._encode_missing(misses, query=query)
            if len(fresh) != len(misses):
                raise EmbeddingError(f"内层返回 {len(fresh)} 个向量，但请求了 {len(misses)} 条文本")
            prepared = [to_float32(v) for v in fresh]

            dims = {len(v) for v in prepared}
            if len(dims) > 1:
                raise DimensionMismatchError(f"同一次响应里出现了多种维度：{sorted(dims)}")
            new_dim = next(iter(dims))
            expected = self._cache.coordinate.dim
            if expected is not None and new_dim != expected:
                raise DimensionMismatchError(
                    f"Embedder 返回 {new_dim} 维，但配置声明 {expected} 维"
                    f"（`models.embed_dim`；坐标系 {self._cache.coordinate.describe()}）"
                )
            cached_dims = self._cache.known_dims()
            if cached_dims and cached_dims != {new_dim}:
                raise DimensionMismatchError(
                    f"新返回 {new_dim} 维，但缓存里是 {sorted(cached_dims)} 维"
                    f"（坐标系 {self._cache.coordinate.describe()}）"
                )

            self._cache.put_many(zip(miss_keys, prepared, strict=True))
            for key, vec in zip(miss_keys, prepared, strict=True):
                found[key] = vec

        out: list[list[float]] = []
        for text in texts:
            key = QUERY_CACHE_PREFIX + text if query else text
            vector = found.get(key)
            if vector is None:  # pragma: no cover
                raise EmbeddingError(f"内部错误：文本未取到向量（{text[:40]!r}…）")
            out.append(vector)
        return out

    def _encode_missing(self, misses: Sequence[str], *, query: bool) -> list[list[float]]:
        if query:
            encode_query = getattr(self._inner, "encode_query", None)
            if callable(encode_query):
                return encode_query(misses)
        return self._inner.encode(misses)


class OpenAICompatEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        batch_size: int = 64,
        max_query_tokens: int | None = None,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url 与 api_key 都不得为空")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client
        self._batch_size = batch_size
        self._max_query_tokens = max_query_tokens
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise DimNotKnownError(
                f"{self._model} 的维度尚未确定——必须先成功调用一次 encode()，"
                "维度只从真实返回里取，不从任何声明或常量里取"
            )
        return self._dim

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            out.extend(self._encode_chunk(texts[start : start + self._batch_size]))
        return out

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]:
        truncated = [
            truncate_for_embedding(text, max_tokens=self._max_query_tokens) for text in texts
        ]
        return self.encode(truncated)

    def _encode_chunk(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        payload = {"model": self._model, "input": list(texts)}
        try:
            resp = self._http().post(
                url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"{self._model} 调用失败（{url}）：{exc}") from exc
        if resp.status_code != 200:
            raise EmbeddingError(
                f"{self._model} 返回 {resp.status_code}（{url}）：{resp.text[:300]}"
            )
        return self._parse(resp.json(), expected=len(texts))

    def _parse(self, body: object, *, expected: int) -> list[list[float]]:
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise EmbeddingError(f"响应形状不是 OpenAI 兼容的 {{'data': [...]}}：{body!r:.200}")
        items = sorted(body["data"], key=lambda d: d.get("index", 0))
        vectors: list[list[float]] = []
        for item in items:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError(f"响应项缺少 embedding：{item!r:.200}")
            vectors.append(to_float32(vec))
        if len(vectors) != expected:
            raise EmbeddingError(f"请求 {expected} 条，返回 {len(vectors)} 条")
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            raise DimensionMismatchError(f"同一次响应里出现了多种维度：{sorted(dims)}")
        self._register_dim(next(iter(dims)))
        return vectors

    def _register_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
        elif self._dim != dim:
            raise DimensionMismatchError(
                f"{self._model} 先前返回 {self._dim} 维，这次返回 {dim} 维"
                "——同一坐标系下维度变化只能意味着模型被换掉了"
            )
