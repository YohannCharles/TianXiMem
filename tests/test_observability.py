"""`observability/` —— §14 指标的**出口**（**V12** 那条静默失败的口子）。

三类断言，各自挡一类不会再报错的错：

| 类 | 挡什么 |
| --- | --- |
| **聚合**（`SnapshotMetricsSink`） | 落盘那份与进程内那份**必须是同一个值**；0 请求的均值不是 `0` |
| **发射语义**（`SearchPipeline`） | 一条观测 = **一次请求**；计数是**增量**不是累计值 |
| **装配**（`build_metrics_sink`） | 没配路径 ⇒ `Null` 是**对**的；漏接则是**静默**的 |
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from eval.experiments import run as runner
from tests.conftest import Wired, seed_line

from tianximem.common.config import (
    AppConfig,
    CacheConfig,
    EmbedCacheConfig,
    QdrantConfig,
    SqliteConfig,
    StorageConfig,
)
from tianximem.observability import (
    MetricsSink,
    NullMetricsSink,
    SearchObservation,
    SnapshotMetricsSink,
)
from tianximem.retrieve import EvidenceChecker
from tianximem.service.app import build_metrics_sink, build_services
from tianximem.service.pipeline import SearchPipeline


def _obs(**overrides: object) -> SearchObservation:
    """一条观测，**默认值挑"什么都没发生"那一档**（只有耗时）。"""
    base: dict = {"latency_ms": 1.0, "rerank_calls": 0, "rerank_disabled": 0, "rerank_degraded": 0}
    return SearchObservation(**{**base, **overrides})  # type: ignore[arg-type]


class _FakeReranker:
    """最小 reranker 替身：**只要给出 `name` 与一个能让链路跑完的分数**。

    ⚠ 与 [`test_reranker.py`](./test_reranker.py) 的 `FakeReranker` 不是同一件事：
    那个要覆盖"恰好一次 / 降级 / 计数"的**行为**，这里只验**装配与转发**
    （名字与增量有没有走到快照里）。
    """

    @property
    def name(self) -> str:
        return "fake-reranker"

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        return [1.0] * len(documents)


# ══ 一、聚合：快照就是落盘的那一份 ═════════════════════════════════════


def test_snapshot_is_what_lands_on_disk(tmp_path: Path) -> None:
    """**父目录不存在也要能写**（服务不会替人建 var/）。"""
    path = tmp_path / "sub" / "metrics.json"
    sink = SnapshotMetricsSink(path=path)
    sink.record(_obs(latency_ms=10.0, rerank_calls=1, rerank_name="r1"))
    sink.record(_obs(latency_ms=30.0))
    assert json.loads(path.read_text(encoding="utf-8")) == sink.snapshot()


def test_latency_aggregates_count_mean_max(tmp_path: Path) -> None:
    sink = SnapshotMetricsSink(path=tmp_path / "m.json")
    for milliseconds in (10.0, 20.0, 60.0):
        sink.record(_obs(latency_ms=milliseconds))
    assert sink.snapshot()["latency_ms"] == {"count": 3, "mean": 30.0, "max": 60.0}


def test_zero_requests_reports_none_mean_not_zero(tmp_path: Path) -> None:
    """**0 请求的均值是 `None` 不是 `0`**——与 `run_record._accuracy` 同一条纪律：
    `0` 会被读成"每次都瞬时完成"，而事实是"没测到"。"""
    assert SnapshotMetricsSink(path=tmp_path / "m.json").snapshot()["latency_ms"]["mean"] is None


def test_a_write_failure_does_not_raise(tmp_path: Path) -> None:
    """**指标是诊断，写不进去不能让一次 Search 变成 500。**

    造法：把快照路径指到一个**已存在的目录** ⇒ `write_text` 必然 `IsADirectoryError`。
    """
    sink = SnapshotMetricsSink(path=tmp_path)  # 目录，不是文件
    sink.record(_obs())  # 不该抛
    assert sink.snapshot()["searches"] == 1  # 进程内的计数照旧累计


def test_null_sink_records_nothing() -> None:
    NullMetricsSink().record(_obs())  # 不抛、不留痕


# ══ 二、发射语义：一条观测 = 一次请求，计数是增量 ═══════════════════════


def _with_metrics(wired: Wired, path: Path, *, reranker: object | None = None) -> SearchPipeline:
    """把 `wired` 的 Search 换成**接了指标出口**的那一条，其余依赖原样复用。"""
    pipeline = SearchPipeline(
        store=wired.store,
        qdrant=wired.qdrant,
        retriever=wired.retriever,
        checker=EvidenceChecker(),
        counter=wired.counter,
        budget_tokens=wired.budget_tokens,
        reranker=reranker,  # type: ignore[arg-type]
        metrics=SnapshotMetricsSink(path=path),
    )
    wired.services.search = pipeline
    return pipeline


def test_one_observation_per_search(wired: Wired, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    _with_metrics(wired, path)
    for _ in range(3):
        wired.search()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["searches"] == 3
    assert snapshot["latency_ms"]["count"] == 3


def test_counts_are_per_request_not_cumulative(wired: Wired, tmp_path: Path) -> None:
    """⚠ **这条挡的是最容易犯的那个错**：把实例上的**累计值**原样发出去。

    没接 reranker ⇒ 每次 Search 记一次 `disabled`，实例上依次是 1 / 2 / 3。
    正确的和是 **3**；照抄累计值会得到 `1+2+3 = 6`——而它**只是"有点大"，不报错**，
    于是"346 次题里 60,000 次没精排"这种记录看起来还挺像回事。
    """
    path = tmp_path / "m.json"
    _with_metrics(wired, path)
    for _ in range(3):
        wired.search()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["rerank"]["disabled"] == 3  # **不是** 1+2+3
    assert snapshot["rerank"]["calls"] == 0


def test_concurrent_searches_report_only_their_own_rerank_increment(wired, tmp_path):
    barrier = Barrier(2)

    class ConcurrentReranker(_FakeReranker):
        def score(self, *, query, documents):
            barrier.wait(timeout=5)
            return super().score(query=query, documents=documents)

    path = tmp_path / "metrics.json"
    pipeline = _with_metrics(wired, path, reranker=ConcurrentReranker())
    wired.qdrant.by_user["u1"] = seed_line(wired.store, [0, 1, 2])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(pipeline.run, user_id="u1", query="question", top_k=100) for _ in range(2)
        ]
        assert all(f.result().items for f in futures)
    snapshot = json.loads(path.read_text())
    assert snapshot["searches"] == 2
    assert snapshot["rerank"]["calls"] == 2
    assert snapshot["grounded_evidence"]["outcomes"] == {"disabled": 2}


def test_concurrent_metrics_snapshot_does_not_lose_records(tmp_path):
    path = tmp_path / "metrics.json"
    sink = SnapshotMetricsSink(path=path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: sink.record(_obs(grounded_status="no_match")), range(80)))
    assert json.loads(path.read_text()) == sink.snapshot()
    assert sink.snapshot()["searches"] == 80
    assert sink.snapshot()["grounded_evidence"]["outcomes"] == {"no_match": 80}


def test_rerank_calls_and_name_reach_the_snapshot(wired: Wired, tmp_path: Path) -> None:
    """**V12 要的两样东西**：精排到底有没有在跑、跑的是哪个模型。

    ⚠ 必须先落种子：**没有候选时 `_maybe_rerank` 走的是 `disabled`**
    （"没有可排序的东西，调用是没有意义的"）——那一路与"接上了精排"长得不一样。
    """
    path = tmp_path / "m.json"
    _with_metrics(wired, path, reranker=_FakeReranker())
    wired.qdrant.by_user["u1"] = seed_line(wired.store, [0, 1, 2])
    wired.search()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["rerank"] == {
        "calls": 1,
        "disabled": 0,
        "degraded": 0,
        "name": "fake-reranker",
    }


# ══ 三、装配：漏接是静默的 ════════════════════════════════════════════


def test_no_configured_path_means_null_sink() -> None:
    """**没配就是不记**——缺省不是错误（与 reranker 同一套口径，D12）。"""
    assert isinstance(build_metrics_sink(AppConfig()), NullMetricsSink)


def test_configured_path_means_file_sink(tmp_path: Path) -> None:
    sink = build_metrics_sink(AppConfig(metrics_path=str(tmp_path / "m.json")))
    assert isinstance(sink, SnapshotMetricsSink)
    assert isinstance(sink, MetricsSink)  # 协议是 runtime_checkable 的


def test_build_services_wires_the_sink_into_the_pipeline(tmp_path: Path) -> None:
    """**装配漏传是静默的**：出口没接上时 Search 照常工作、计数照常累计，只是永远不写文件。

    与 `test_reranker.py` 的 `build_services` 那几条同一个理由——所以要有这一条断言，
    而不是靠"我传了"。
    """
    config = AppConfig(
        storage=StorageConfig(
            sqlite=SqliteConfig(path=str(tmp_path / "tianxi.db")),
            qdrant=QdrantConfig(url="http://unused"),
        ),
        cache=CacheConfig(embed=EmbedCacheConfig(dir=str(tmp_path / "cache"))),
        embed_base_url="http://unused/v1",
        embed_api_key="k",
        metrics_path=str(tmp_path / "m.json"),
    )
    services = build_services(config)
    try:
        assert isinstance(services.search.metrics, SnapshotMetricsSink)
    finally:
        services.close()


# ══ 四、runner 侧：读快照进 run record（`metrics=`）═══════════════════════


def _args(**overrides: object):
    """一个只带 `metrics` 的极简 args（`_read_service_metrics` 只读这一个字段）。"""
    return type("Args", (), {"metrics": None, **overrides})()


def test_runner_reads_the_snapshot_into_metrics(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"searches": 7, "rerank": {"degraded": 2}}), encoding="utf-8")
    assert runner._read_service_metrics(_args(metrics=str(path))) == {
        "searches": 7,
        "rerank": {"degraded": 2},
    }


def test_runner_without_a_path_reports_no_metrics() -> None:
    """没给路径 ⇒ 空 dict（不是 `None`、也不是编一个 0）——`metrics={}` 是**可见的**。"""
    assert runner._read_service_metrics(_args(metrics="")) == {}


def test_runner_warns_when_the_snapshot_is_missing(tmp_path: Path, capsys) -> None:
    """⚠ **读不到要说话**：空 `metrics` 有两种来源——"服务侧没配"与"这一轮真没精排"。"""
    assert runner._read_service_metrics(_args(metrics=str(tmp_path / "nope.json"))) == {}
    assert "指标快照不存在" in capsys.readouterr().err


def test_runner_warns_loudly_when_rerank_degraded(capsys) -> None:
    """**V12 的那一条**：端点挂了不会让别处变红，只有这一行会说。"""
    runner._warn_about_rerank({"searches": 346, "rerank": {"degraded": 346, "disabled": 0}})
    assert "精排降级 346 次" in capsys.readouterr().out


def test_runner_is_quiet_when_rerank_ran_clean(capsys) -> None:
    runner._warn_about_rerank({"searches": 346, "rerank": {"degraded": 0, "disabled": 0}})
    assert capsys.readouterr().out == ""
