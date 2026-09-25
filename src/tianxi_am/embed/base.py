"""Embedder 协议 + 落盘向量缓存 + OpenAI 兼容的 HTTP 基类。

**§7.4 的接口是硬要求**：`dim` 由接口提供、**不能写死**；代码中不得出现任何依赖
具体模型输出结构的逻辑。Step 5 换模型时按新维度**重建集合**。

## 为什么 `dim` 是"第一次调用之后才知道的"

embed/README 明确 `dim` **不能写死**。而"不写死"不能靠"写个常量声明待会改"来实现——
那样等于把一个模型属性埋进代码。所以这里的 `dim` 是**只读属性**：

* `encode()` 成功返回后，`dim` 就等于**实际返回向量的长度**（来自真实响应，不是声明）
* 在此之前访问会抛 `DimNotKnownError`——**逼调用方先拿到真实维度再建集合**

集合的向量维度必须来自这个值（`store/qdrant_store.py` 的 `ensure_collection(dim)`）。

## 缓存的坐标系

缓存键 = **渲染后文本的内容哈希**，**不是 `memory_id`**——补全时内容变了而 `id` 不变，
用 `id` 当键会拿到**陈旧向量**（§7.2）。

坐标系（模型标识 + 渲染模板版本）**写进缓存文件本身**，而且**直接进文件名**：

    <cache_dir>/<coordinate_key>.db

这样"整体失效"是**结构性**的——Step 5 换了模型或改了模板 ⇒ 换一个坐标系键 ⇒
**换一个文件** ⇒ 旧缓存不可能被静默命中（§12.1 R1 对冲 1）。
旧文件可以随手删掉，不影响正确性。

## 一个刻意的表示统一

向量在**进入缓存之前**统一转成 `float32`，缓存与新鲜结果都走同一条转换。
否则"缓存命中"与"缓存未命中"会返回**精度不同**的向量 ⇒ **检索结果依赖缓存状态** ⇒
不可复现。这是本项目最不能接受的失败类型。
"""

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

from tianxi_am.common.render import TEMPLATE_VERSION

__all__ = [
    "CachingEmbedder",
    "DimNotKnownError",
    "DimensionMismatchError",
    "DiskVectorCache",
    "Embedder",
    "EmbeddingCoordinate",
    "EmbeddingError",
    "OpenAICompatEmbedder",
    "to_float32",
]

#: 缓存库连接上的 `busy_timeout`（毫秒，见 `DiskVectorCache._connect`）。
#:
#: ⚠ **与真源那条分开**（`storage.sqlite.busy_timeout_ms`）：两个**不同的库文件**、
#: 两条独立的争用路径，绑成一个值只会在调其中一个时意外改到另一个。
#: 取值与真源同档是因为面对的争用形状相同（WAL + 单写者）。
#:
#: ⚠ **它没有配置项，是有意的**（2026-09-25 从裸字面量提出来）：`config-reference.md`
#: 的纪律是"收入配置的键必须有消费方"，而眼下没有任何调用方需要调它——
#: **收一个没有消费方的键等于预留字段**。真要调时：加配置键 + 从这里改成注入。
_CACHE_BUSY_TIMEOUT_MS: Final[int] = 5000


class EmbeddingError(RuntimeError):
    """远程 embedding 调用失败（网络、鉴权、响应形状不符）。"""


class DimNotKnownError(RuntimeError):
    """在第一次 `encode()` 之前访问 `dim`。

    **这是有意的**：集合的向量维度必须来自接口的**真实返回**，不能来自任何声明或常量。
    """


class DimensionMismatchError(RuntimeError):
    """维度不一致。

    ⚠ 这条必须是**响亮的失败**：维度不同的向量混进同一个集合只会表现为
    "检索结果很差"，**不会报错**（embed/README 的原话）。
    """


def _content_hash(text: str) -> str:
    """缓存键 = **渲染后文本**的内容哈希（§7.2）。

    ⚠ 用文本而不是 `memory_id`：同一个 `id` 在补全后内容会变，
    用 `id` 当键会拿到陈旧向量；反过来，**相同文本、不同 `id` 必须能复用**。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_float32(values: Sequence[float]) -> list[float]:
    """把向量统一成 float32 再转回 `list[float]`。

    新鲜结果与缓存结果都经过这一步，**保证两者逐位相同**（见模块 docstring）。
    """
    return [float(x) for x in np.asarray(values, dtype=np.float32)]


@runtime_checkable
class Embedder(Protocol):
    """§7.4 的接口。**`dim` 由接口提供、不能写死。**"""

    @property
    def dim(self) -> int:
        """向量维度。**第一次 `encode()` 之前未定义**（抛 `DimNotKnownError`）。"""
        ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """把若干文本编码成向量。**每条文本恰好一个向量**，顺序与输入一致。"""
        ...


@dataclass(frozen=True, slots=True)
class EmbeddingCoordinate:
    """缓存的坐标系：**模型标识 + 渲染模板版本**。

    维度**不在这里**——它在第一次调用之后才知道，所以按条目校验（见 `DiskVectorCache`）。
    """

    model: str
    render_template: str = TEMPLATE_VERSION

    def key(self) -> str:
        raw = json.dumps(
            {"model": self.model, "render_template": self.render_template},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        return f"{self.model} @ render={self.render_template}"


class DiskVectorCache:
    """落盘的向量缓存。**键 = 渲染文本的内容哈希。**

    一个坐标系一个文件：`<cache_dir>/<coordinate_key>.db`。
    文件里也存一份坐标系（自描述），便于人工核对与排错。

    ## 连接模型：**短生命周期**（2026-09-24 改）

    和 `store/sqlite_store.py` 同一套模型：**本类不长期持有 `sqlite3.Connection`**，
    每次读/写各自 `connect → … → close`。

    ⚠ **为什么这条对本类同样致命**：缓存的调用方是 `CachingEmbedder`，而它跑在
    **FastAPI 的线程池**里（索引侧在 `Add` 请求线程、查询侧在 `Search` 请求线程）。
    `sqlite3` 的连接**只能被创建它的线程使用** ⇒ 之前那种"构造时建一个连接、之后长期复用"
    的写法，会在**第一个请求**就抛 `sqlite3.ProgrammingError`。
    测试没抓到它，只是因为并发用例当时被 `SqliteStore` 的那一份挡住了（同一个根因、
    同一个形状）。**两处的修法必须一致**，否则修好一处、另一处照样炸。
    """

    def __init__(self, cache_dir: str | Path, coordinate: EmbeddingCoordinate) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._coordinate = coordinate
        self._path = self._dir / f"{coordinate.key()}.db"
        self._init_schema()
        self._verify_coordinate()

    # ── 生命周期 ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """造一个配好必要 PRAGMA 的新连接。调用方负责关它。"""
        conn = sqlite3.connect(str(self._path), isolation_level=None)  # 自己管事务
        # 缓存**不是真源**：丢了只需重算一次向量，不会丢业务数据。
        # 所以用 NORMAL（WAL 下不会损坏，最坏丢最后几个已提交条目）而不是 FULL——
        # 索引侧会成批写，每条都 fsync 的代价落在最热的路径上，不划算。
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {_CACHE_BUSY_TIMEOUT_MS}")
        return conn

    def _init_schema(self) -> None:
        """建表 + 把 `journal_mode = WAL` 落一次（**数据库级**，不在每个连接上重复）。"""
        conn = self._connect()
        try:
            # WAL：多线程/多连接读写不互相阻塞。写进库文件，对后续连接持续生效。
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
        """一次**只读**操作的作用域（同类内部与测试都用它）。"""
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    # ── 自检 ───────────────────────────────────────────────────────────

    def _verify_coordinate(self) -> None:
        """文件里的坐标系必须与当前的一致。

        文件名已经保证了这件事；这一层是**第二道锁**——万一有人改名或复制文件，
        这里会响而不是静默用错缓存。
        """
        with self.read() as conn:
            row = conn.execute("SELECT v FROM meta WHERE k = 'coordinate'").fetchone()
        if row is not None and row[0] != self._coordinate.describe():
            raise DimensionMismatchError(
                f"缓存文件的坐标系不符：文件里是 {row[0]!r}，当前是 "
                f"{self._coordinate.describe()!r}（{self._path}）"
            )

    # ── 读写 ───────────────────────────────────────────────────────────

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
        if vec.shape[0] != dim:  # pragma: no cover — 文件损坏
            raise DimensionMismatchError(
                f"缓存条目自称 {dim} 维，实际 blob 是 {vec.shape[0]} 维（{self._path}）"
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
        # 一个写事务、一个连接：成批写要么全落要么全不落（不会留半个批次的条目）
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
            conn.rollback()  # 无事务可回滚时是 no-op（`execute("ROLLBACK")` 会抛）
            raise
        finally:
            conn.close()
        return len(rows)

    def known_dims(self) -> set[int]:
        """缓存里出现过的所有维度（用于与新鲜返回交叉校验）。"""
        with self.read() as conn:
            return {int(r[0]) for r in conn.execute("SELECT DISTINCT dim FROM vectors")}

    def count(self) -> int:
        with self.read() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])


class CachingEmbedder:
    """给任意 `Embedder` 套一层内容哈希缓存。

    **命中时绝不调用内层（远程）模型**——这是 §7.2 那条"缓存后即成为一次性成本、
    与迭代次数无关"的落点。
    """

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
        """优先信内层；内层还没调过就退回缓存里已出现过的维度（唯一一个才敢给）。"""
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
        texts = list(texts)
        if not texts:
            return []

        # 先全部查缓存，并顺手在批内去重（同一次 batch 里重复文本很常见）
        found: dict[str, list[float] | None] = {}
        misses: list[str] = []
        for text in texts:
            if text in found:
                continue
            hit = self._cache.get(text)
            found[text] = hit
            if hit is None:
                misses.append(text)

        if misses:
            fresh = self._inner.encode(misses)
            if len(fresh) != len(misses):
                raise EmbeddingError(f"内层返回 {len(fresh)} 个向量，但请求了 {len(misses)} 条文本")
            prepared = [to_float32(v) for v in fresh]

            # 维度自检：与内层已知维度、与缓存里既有维度都必须一致
            dims = {len(v) for v in prepared}
            if len(dims) > 1:
                raise DimensionMismatchError(f"同一次响应里出现了多种维度：{sorted(dims)}")
            new_dim = next(iter(dims))
            cached_dims = self._cache.known_dims()
            if cached_dims and cached_dims != {new_dim}:
                raise DimensionMismatchError(
                    f"新返回 {new_dim} 维，但缓存里是 {sorted(cached_dims)} 维"
                    f"（坐标系 {self._cache.coordinate.describe()}）"
                )

            self._cache.put_many(zip(misses, prepared, strict=True))
            for text, vec in zip(misses, prepared, strict=True):
                found[text] = vec

        out: list[list[float]] = []
        for text in texts:
            vector = found.get(text)
            if vector is None:  # pragma: no cover — 上面已保证填满
                raise EmbeddingError(f"内部错误：文本未取到向量（{text[:40]!r}…）")
            out.append(vector)
        return out


class OpenAICompatEmbedder:
    """OpenAI 兼容 `/embeddings` 的 HTTP 客户端基类。

    开发期的 `qwen3_embedding.Qwen3EmbeddingEmbedder` 与提交链路的 `text_embedding_v4` 都继承它——
    两者只是 base_url / key / model 不同。

    **`dim` 只有在第一次真实响应之后才有值**（见模块 docstring）。

    ⚠ 这里**不做重试**：重试策略属服务层（`service/`），而且 `Add` 的 32 次重试
    由 AML 从外部施加——在这一层再加一层重试会让失败路径变成两层叠加，难以归因。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        batch_size: int = 64,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url 与 api_key 都不得为空")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client
        self._batch_size = batch_size
        self._dim: int | None = None

    # ── 标识 ───────────────────────────────────────────────────────────

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

    # ── 调用 ───────────────────────────────────────────────────────────

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

    def _encode_chunk(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        payload = {"model": self._model, "input": list(texts)}
        try:
            resp = self._http().post(
                url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.HTTPError as exc:  # 网络层
            raise EmbeddingError(f"{self._model} 调用失败（{url}）：{exc}") from exc
        if resp.status_code != 200:
            raise EmbeddingError(
                f"{self._model} 返回 {resp.status_code}（{url}）：{resp.text[:300]}"
            )
        return self._parse(resp.json(), expected=len(texts))

    def _parse(self, body: object, *, expected: int) -> list[list[float]]:
        """从响应里取向量。**只认 OpenAI 兼容的 `data[].embedding`。**"""
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
        """记下**来自真实返回**的维度；与先前不一致就响亮地失败。"""
        if self._dim is None:
            self._dim = dim
        elif self._dim != dim:
            raise DimensionMismatchError(
                f"{self._model} 先前返回 {self._dim} 维，这次返回 {dim} 维"
                "——同一坐标系下维度变化只能意味着模型被换掉了"
            )
