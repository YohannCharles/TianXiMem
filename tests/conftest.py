"""单元测试的共用 fixture。

测试清单的权威来源是 [`../tests/CLAUDE.md`](../tests/CLAUDE.md)：
记忆块组合 / 幂等 / 契约 / 隔离 / 开关纯度 / 存储。

⚠ 本目录的测试**只用合成的 canonical 消息**（`role` / `content` / 可选 `timestamp`），
**不碰任何数据集文件**——LoCoMo 只作为实现完成后的 fixture 验证数据，
**不得反向影响核心数据模型**（D16）。

⚠ **测试绝不调用远程模型**：embedding 一律用 `FakeEmbedder`（确定性、可计数）。
需要 Qdrant 的用例在 Qdrant 不可达时**明确 skip**，不静默通过。
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import pytest

from tianximem.common.config import (
    ENV_QDRANT_URL,
    AppConfig,
    CacheConfig,
    EmbedCacheConfig,
    QdrantConfig,
    RerankConfig,
    SqliteConfig,
    StorageConfig,
)
from tianximem.common.render import render_pair
from tianximem.pairing.pairing import Message
from tianximem.retrieve import (
    DenseArm,
    EvidenceChecker,
    HybridRetriever,
    InMemoryCheckerInstrument,
)
from tianximem.service import build_services
from tianximem.service.pipeline import AddPipeline, SearchPipeline
from tianximem.store.qdrant_store import ScoredMemoryId
from tianximem.store.sqlite_store import SqliteStore, make_pair_id

# ── ① 的 fixture ────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path) -> Iterator[SqliteStore]:
    """一个建好表、用临时文件的真源（每个用例一份，互不干扰）。

    ⚠ **没有 `s.close()`**——`SqliteStore` 不持有连接（短生命周期模型）。
    """
    yield SqliteStore.open(tmp_path / "tianxi.db")


class FakeCounter:
    """确定性计数器：**1 个字符 = 1 个 token**（空白与换行也各算 1）。

    为什么不用真的 `o200k_base`：

    * **单元测试不该依赖 tiktoken 的 BPE 缓存文件**（首次使用要联网下载，
      `common/tokens.py` 的 `TokenCounterError` 就是为那条路径写的）
    * **字符计数让预算的算术精确可推**：`count(a) + count(sep) + count(b)`
      **恰好等于** `count(a + sep + b)`。BPE 没有这个恒等式（合并可以跨越拼接边界），
      所以 `packaging._fit_by_budget` 的决策用的是**上界**、报告用的才是真值——
      而上界与真值在字符计数下重合，于是"装了几个段"这件事在用例里是**算得出来的**。

    ⚠ 真分词器的接线（`load_counter` / `budget.tokenizer`）由 `test_config.py` 与
    `test_contract_preflight.py` 覆盖——**这里刻意不重叠**。
    """

    name: str = "fake-char"

    def count(self, text: str) -> int:
        return len(text)


@pytest.fixture
def counter() -> FakeCounter:
    return FakeCounter()


def rd(store: SqliteStore, method: Callable, /, *args, **kwargs):
    """在**一次短生命周期只读连接**里调用 `store` 的一个读取方法（测试用）。

    对应生产代码里的 `with store.read() as conn:`。测试里大量断言是"读一次、看结果"，
    每个都手写 `with` 只会淹没断言本身。**它绝不用于写**——写必须走 `store.transaction()`，
    否则 §6.5 的三步就散了。
    """
    with store.read() as conn:
        return method(conn, *args, **kwargs)


def run_parallel(targets: Sequence[Callable[[], None]], *, join_timeout: float = 20.0):
    """并发跑若干可调用对象，**返回它们抛出的异常**。

    ⚠ **返回异常列表是为了避免用例"空过"**：线程里的异常不会让 pytest 失败，
    所以"线程都结束了"这种断言在实现坏掉时也会通过。调用方必须 `assert errors == []`
    （或用 `pytest.raises` 显式检查列表内容）。
    """
    errors: list[BaseException] = []

    def _wrap(fn: Callable[[], None]) -> Callable[[], None]:
        def _inner() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 — 这里就是要收集全部
                errors.append(exc)

        return _inner

    threads = [threading.Thread(target=_wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=join_timeout)
    return errors


@pytest.fixture
def M() -> Callable[..., Message]:
    """消息构造器：`M("user", "你好")`、`M("assistant", "在", ts=1000)`。"""

    def _make(role: str, content: str, ts: int | None = None) -> Message:
        return Message(role=role, content=content, timestamp=ts)

    return _make


# ── ② 的 fixture ────────────────────────────────────────────────────────


class FakeEmbedder:
    """确定性、可计数的假 Embedder。**测试用它，绝不碰网络。**

    * 未 preset 的文本 → 按词的哈希做 bag-of-words 单位向量（同词 ⇒ 高余弦）
    * `preset(text, vec)` → 精确指定某段文本的向量，让 dense 与 bm25 两路
      **可以独立控制**（"只被 dense 召回"这类用例必须能构造）
    * `calls` 记录每一次真实调用，用来断言"缓存命中时没有重复调用"
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._preset: dict[str, list[float]] = {}
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def encoded_texts(self) -> list[str]:
        return [t for batch in self.calls for t in batch]

    def preset(self, text: str, vec: Sequence[float]) -> None:
        if len(vec) != self._dim:
            raise ValueError(f"preset 维度必须是 {self._dim}，收到 {len(vec)}")
        self._preset[text] = list(vec)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        self.calls.append(texts)
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        if text in self._preset:
            return list(self._preset[text])
        acc = [0.0] * self._dim
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            acc[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return [x / norm for x in acc]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def unit_vector(dim: int, index: int) -> list[float]:
    """第 `index` 个基向量——用来构造互不相干的 dense 候选。"""
    return [1.0 if i == index else 0.0 for i in range(dim)]


@pytest.fixture(scope="session")
def qdrant_client():
    """会话级 Qdrant 客户端；**不可达时整组用例 skip**（不静默通过）。"""
    from qdrant_client import QdrantClient

    url = os.environ.get(ENV_QDRANT_URL, "http://localhost:6333")
    client = QdrantClient(url=url, timeout=10.0)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 — 任何连接问题都跳过
        client.close()
        pytest.skip(f"Qdrant 不可达（{url}）：{exc}", allow_module_level=True)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def qdrant_store(qdrant_client):
    """一个**用完即删**的测试集合。

    ⚠ 刻意**不用**默认的 `memories`：那是开发/线上共用的集合，
    测试往里写点会让"检索结果为什么不对"变成一个无法回答的问题。
    """
    from tianximem.store.qdrant_store import QdrantStore

    name = f"memories_test_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(url="unused", collection=name, client=qdrant_client, hybrid=None)
    try:
        yield store
    finally:
        store.drop_collection()


# ── ③-c 的 fixture：服务层（Add / Search）─────────────────────────────────


@dataclass
class FakeQdrantSearch:
    """Search 用得到的假 Qdrant：`exists` / `hybrid_search`，外加 Add 要的 `index_pairs`。

    `by_user` 由用例指定每个用户"检索会返回什么"；分数固定为一个**刻意刺眼**的值，
    用来验证它绝不进入响应。

    ⚠ **`index_pairs` 不是可有可无的**：`wired` 夹具把它同时接到 `services.add` 上，
    否则 HTTP 级的 `/add` 用例会走**真的** `Qwen3EmbeddingEmbedder` 去打远程网关——
    而 `tests/CLAUDE.md` 的纪律是"**测试绝不调用远程模型**"。
    """

    by_user: dict[str, list[str]] = field(default_factory=dict)
    exists_flag: bool = True
    calls: list[dict] = field(default_factory=list)
    points: dict[str, str] = field(default_factory=dict)
    index_calls: list[list[str]] = field(default_factory=list)

    def exists(self) -> bool:
        return self.exists_flag

    def hybrid_search(self, *, user_id: str, query_text: str, dense_vector, top_k: int):
        self.calls.append(
            {
                "user_id": user_id,
                "query_text": query_text,
                "dense_vector": list(dense_vector),
                "top_k": top_k,
            }
        )
        ids = self.by_user.get(user_id, [])
        # ⚠ **分数必须严格递减**，与真 Qdrant 的契约一致（融合结果按分数降序返回）。
        #   给全部命中同一个分数会让"并列次序"进入下游——而 `retrieve/` 现在会按
        #   `(-score, memory_id)` 定死并列次序（**V13**），于是本夹具声明的顺序会被
        #   按 id 重排，用例断言的"返回的就是我给的顺序"就不成立了。
        #   ⇒ 递减分数既忠实于真实现，也让"顺序 = 我给的顺序"这条断言继续有效。
        #   ⚠ 仍然是**刻意刺眼**的值：它绝不允许出现在响应里（`test_contract.py` 钉着）。
        return [
            ScoredMemoryId(mid, FUSED_SCORE - index * FUSED_SCORE_STEP)
            for index, mid in enumerate(ids[:top_k])
        ]

    def index_pairs(self, pairs, embedder, *, renderer=render_pair, wait=True) -> int:
        """Add 侧只需要这一件事：渲染 → 嵌入（走假 embedder）→ 记下 point。"""
        records = list(pairs)
        texts = [renderer(p) for p in records]
        embedder.encode(texts)
        for pair, text in zip(records, texts, strict=True):
            self.points[pair.id] = text
        self.index_calls.append([p.id for p in records])
        return len(records)


#: 一个**刻意刺眼**的"融合分数"——它绝不允许出现在响应里
FUSED_SCORE = 0.987654321

#: 名次之间那一点递减量——只为让顺序**唯一确定**（见 `hybrid_search` 的注释）。
#: 取值不影响任何断言：用例只断言"这个刺眼的分数没有出现在响应里"。
FUSED_SCORE_STEP = 1e-6


@dataclass
class Wired:
    """③-c 的对象图：真 `SqliteStore` + 假 Qdrant + 真 `DenseArm`/checker。"""

    services: object
    qdrant: FakeQdrantSearch
    store: SqliteStore
    embedder: FakeEmbedder
    instrument: InMemoryCheckerInstrument
    retriever: HybridRetriever
    counter: FakeCounter
    budget_tokens: int

    def search(self, *, user_id: str = "u1", query: str = "q", top_k: int = 5):
        return self.services.search.run(user_id=user_id, query=query, top_k=top_k)


@pytest.fixture
def wired(tmp_path) -> Iterator[Wired]:
    """装配一条**不碰网络、不碰真 Qdrant** 的 Add + Search 链。

    ⚠ **`services.add` 与 `services.search` 都要换成假的**：`build_services` 造出来的
    `AddPipeline` 用的是**真的** `Qwen3EmbeddingEmbedder`——HTTP 级的 `/add` 用例若走它，
    就会去打远程网关（违反"测试绝不调用远程模型"），而且**在没有网络时会响亮失败**。
    """
    yield from _wire(tmp_path, inject_abs_time=False)


@pytest.fixture
def wired_dated(tmp_path) -> Iterator[Wired]:
    """同 `wired`，但**开着 T1 的"带日期"渲染变体**（`packaging.inject_abs_time=True`）。

    它存在是为了让"**T1 的两臂只差正文**"这件事可断言（§13 的开关纯度）——
    那条断言要同时跑两臂并逐项比，所以两臂都得能就地造出来。

    ⚠ **它用自己的子目录**：两臂若共用 `tmp_path`，它们会写**同一个 `tianxi.db`**，
    于是第二个 `wired` 落种子时直接撞 `UNIQUE(user_id, session_id, request_id, local_index)`。
    真实对照里两臂本来就该各有各的库与集合（见 `configs/CLAUDE.md` 的 `runs/`）。
    """
    own = tmp_path / "dated"
    own.mkdir()
    yield from _wire(own, inject_abs_time=True)


@pytest.fixture
def wired_annotated(tmp_path) -> Iterator[Wired]:
    """同 `wired`，但开着**相对时间注解**（`packaging.annotate_relatives=True`）。

    与 `wired_dated` 同一个理由：那条纯度断言（"开关只该改正文"）要**同时跑两臂**逐项比。
    ⚠ 它**不开** `inject_abs_time`——这两个开关各测各的：本 fixture 要证明的正是
    "**注解不进索引**"（与 T1 那条相反，T1 的正文**就是** embedding 输入）。
    """
    own = tmp_path / "annotated"
    own.mkdir()
    yield from _wire(own, inject_abs_time=False, annotate_relatives=True)


def _wire(tmp_path, *, inject_abs_time: bool, annotate_relatives: bool = False) -> Iterator[Wired]:
    """`wired` / `wired_dated` / `wired_annotated` 的**同一份**装配代码。**不要复制第二份。**"""
    config = AppConfig(
        # 不传 env ⇒ 不碰真实环境；只给必需的那几项，其余走 config.py 的内置默认值
        storage=StorageConfig(
            sqlite=SqliteConfig(path=str(tmp_path / "tianxi.db")),
            # 假 Qdrant 不看 url，但 config 会校验它非空
            qdrant=QdrantConfig(url="http://unused"),
        ),
        cache=CacheConfig(embed=EmbedCacheConfig(dir=str(tmp_path / "cache"))),
        embed_base_url="http://unused/v1",
        embed_api_key="k",
        # ⚠ **显式关掉精排**：这份装配造出来的 `search` 下面会被换成不带 reranker 的那条，
        # 而 `default.yaml` 的默认值是 `true`（提交口径）⇒ 不写这一行就会走
        # "打算精排、但端点为空"那条路，在**每个**用本夹具的用例里留下一条 WARNING。
        # 那不是这些用例要测的东西，而它看起来像"漏配了"。
        rerank=RerankConfig(enabled=False),
    )
    services = build_services(config)
    qdrant = FakeQdrantSearch()
    embedder = FakeEmbedder(dim=8)
    instrument = InMemoryCheckerInstrument()
    retriever = HybridRetriever(store=qdrant, dense=DenseArm(embedder))
    counter = FakeCounter()
    #: 预算**刻意给得极大**（字符计数下的 10 万），好让 `wired` 的用例测的是它自己
    #: 命名的那件事（契约形状 / 隔离 / 计数），而不是撞上预算提前停止。
    #: 预算本身的行为在 `test_packaging.py` 与 `test_neighbor.py` 里**单独**测。
    budget_tokens = 100_000
    services.search = SearchPipeline(
        store=services.store,
        qdrant=qdrant,
        retriever=retriever,
        checker=EvidenceChecker(instrument=instrument),
        counter=counter,
        budget_tokens=budget_tokens,
        inject_abs_time=inject_abs_time,
        annotate_relatives=annotate_relatives,
    )
    services.add = AddPipeline(
        store=services.store,
        qdrant=qdrant,
        embedder=embedder,
        inject_abs_time=inject_abs_time,
    )
    try:
        yield Wired(
            services,
            qdrant,
            services.store,
            embedder,
            instrument,
            retriever,
            counter,
            budget_tokens,
        )
    finally:
        services.close()


def seed_pair_in(
    store: SqliteStore,
    index: int,
    question: str | None = None,
    answer: str | None = None,
    *,
    user_id: str = "u1",
    session_id: str = "s1",
    event_time: int | None = None,
    request_id: str | None = None,
) -> str:
    """落一个块，返回它的 canonical `memory_id`（位置派生，见 ① 的 ID 纪律）。

    ⚠ **`index` 现在映射成 `(request_id=f"seed:{index}", local_index=0)`**（**D28**）：
    每个 `index` 是**一次独立的 Add**（各自一块）⇒ 它们之间**没有** `prev` / `next`，
    也不存在跨 Add 的序号。**要造"同一条链上相邻的几个块"请用 `seed_line`。**

    **全测试层唯一的一份**——`test_neighbor.py` / `test_reranker.py` 都从
    `tests.conftest import seed_pair_in as _seed` 取它，**别再各写一份**。

    * `question` / `answer` 省掉时按 `q{index}` / `a{index}` 补；
      传 `None`（显式）就是造**不完整块**（A-only / Q-only），它不进邻接链
    * `session_id` 可传——那两个文件要造"两个 session 各有一条候选"的形状
    * `event_time` 可传——`test_contract.py` 要造"有/无时间戳"两条
    * `request_id` 可传——要造"同一个 request_id 下两块"时用得上（配 `seed_in_add`）
    """
    with store.transaction() as conn:
        pair = store.insert_pair(
            conn,
            user_id=user_id,
            session_id=session_id,
            request_id=f"seed:{index}" if request_id is None else request_id,
            local_index=0,
            prev_memory_id=None,
            next_memory_id=None,
            question=question if question is not None else f"q{index}",
            answer=answer if answer is not None else f"a{index}",
            status="complete",
            event_time=event_time,
        )
    return pair.id


def seed_in_add(
    store: SqliteStore,
    entries: Sequence[tuple[str | None, str | None]],
    *,
    user_id: str = "u1",
    session_id: str = "s1",
    request_id: str = "seed:one-add",
    event_time: int | None = None,
) -> list[str]:
    """在**一次 Add 内**落下若干个块（`local_index` = 它们在 `entries` 里的下标）。

    邻接按**与 `pairing.link_blocks` 完全相同的规则**连：**只连完整 QA**，
    不完整的（`question` / `answer` 有一个是 `None`）两侧都是 `None`。

    ⚠ **必须一次写完**：`prev` / `next` 是**写下时**算好的，没有事后回填
    （D24 那条"块在写下那一刻就是最终形状"）。想造"中间缺一块"的形状，
    就**别落那一块**——它的邻居的指针仍然指着它，而它不在库里 ⇒ 段自然断开。
    """
    ids = [
        make_pair_id(user_id, session_id, request_id, index) for index in range(len(entries))
    ]
    complete = [i for i, (q, a) in enumerate(entries) if q is not None and a is not None]
    links: dict[int, tuple[int | None, int | None]] = {
        i: (None, None) for i in range(len(entries))
    }
    for position, index in enumerate(complete):
        links[index] = (
            complete[position - 1] if position else None,
            complete[position + 1] if position + 1 < len(complete) else None,
        )

    with store.transaction() as conn:
        for index, (question, answer) in enumerate(entries):
            prev_index, next_index = links[index]
            store.insert_pair(
                conn,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                local_index=index,
                prev_memory_id=None if prev_index is None else ids[prev_index],
                next_memory_id=None if next_index is None else ids[next_index],
                question=question,
                answer=answer,
                status="complete",
                event_time=event_time,
            )
    return ids


def seed_line(
    store: SqliteStore,
    positions: Sequence[int],
    *,
    user_id: str = "u1",
    session_id: str = "s1",
    qa: Callable[[int], tuple[str, str]] | None = None,
    request_id: str = "seed:line",
) -> list[str]:
    """在**一次 Add 内**落满 `0..max(positions)`，只返回 `positions` 那几行的 `memory_id`。

    **为什么需要它**：想造"这几个块被选中了、中间那几个没被选中"，光落被选中的那几行
    **不够**——段合并的判据是"前一条的 `next` 指着我"，而 `next` 是写下时按**整批**
    算好的：只落 0 和 2 的话，0 的 `next` 指着 1（那一行不在库里）⇒ 它们照样分开。
    ⇒ "对话里的洞"必须**真的少落一行**来表达，而那正是这张函数要的语义。

    ⚠ D28 起它们**必须在同一次 Add 里**（`request_id` 相同、`local_index` 0..n-1）——
    跨 Add 的块**永远不会**相邻，所以"落 0..n-1 号"在 D28 之后只有一个意思：
    **同一条链上的前 n 个块**。

    `qa`：可选，`(position) -> (question, answer)`；缺省是 `q{position}` / `a{position}`。
    """
    wanted = list(positions)
    if not wanted:
        return []
    entries = [
        (f"q{i}", f"a{i}") if qa is None else qa(i) for i in range(max(wanted) + 1)
    ]
    ids = seed_in_add(
        store, entries, user_id=user_id, session_id=session_id, request_id=request_id
    )
    return [ids[i] for i in wanted]


@pytest.fixture
def seed_pair() -> Callable[..., str]:
    """`seed_pair_in` 的 fixture 形态——**同一份实现**，用它的测试就不必自己 import。"""
    return seed_pair_in


@pytest.fixture
def seed_sparse() -> Callable[..., list[str]]:
    """`seed_line` 的 fixture 形态——"落满、只返回选中那几个"。"""
    return seed_line
