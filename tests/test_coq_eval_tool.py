"""`tools/coq_eval.py` 的**纯函数**部分——不发请求、不读数据集。

## 为什么这个文件值当

**开关必须只影响它命名的那一件事**在这里有一个具体的落点：
四臂的配置是**从一份基线 `replace` 出来的**。多改一个字段（比如顺手把
`retrieval.grounded_evidence` 也关了），对照就不成立——
**而结果看起来完全正常，只是结论错了**。

所以这里逐个字段比对四臂与基线的差异，**把"只动了那四个"变成一条断言**。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import fields
from pathlib import Path

import pytest
from eval.datasets.preprocess import Message, Sample, Session
from eval.harness import SearchHit
from eval.harness.add_shape import MAX_MESSAGE_CHARS, shape_batch
from tests.test_router import _config  # noqa: PLC2701 — 复用同一套"真配置 + 临时路径"
from tools.coq_eval import (
    ARMS,
    _arm_config,
    _degraded_arms,
    _load_manifest,
    _require_source_rows,
    _reuse_hits,
    _write_hits,
)


def _arms() -> dict:
    return {arm.key: arm for arm in ARMS}


def test_every_arm_only_flips_its_own_switches(tmp_path) -> None:
    """⭐ 四臂与基线的差异**恰好**是那四个开关，一个不多一个不少。

    ⚠ 路径（SQLite / collection）是**隔离用的**，不算"开关"——但它也必须只差这一处，
    所以下面把它单独比对（否则"隔离跑到了开发库上"会静默发生）。
    """
    base = _config(tmp_path)
    sqlite_path = tmp_path / "coq.db"
    for arm in ARMS:
        config = _arm_config(
            base,
            arm,
            collection="coq-eval-test",
            sqlite_path=sqlite_path,
            metrics_path=tmp_path / "m.json",
            rerank=True,
        )
        # ── 四臂之间的差异 ────────────────────────────────────────────
        assert config.router.enabled is arm.router
        assert config.agentic.enabled is arm.agentic
        assert config.agentic.force is arm.force
        assert config.agentic.round_rerank_enabled is arm.round_rerank
        # ── 开关之外，逐字段与基线相同 ────────────────────────────────
        assert config.retrieval == base.retrieval
        assert config.neighbor == base.neighbor
        assert config.packaging == base.packaging
        assert config.budget == base.budget
        assert config.models == base.models
        assert config.llm_base_url == base.llm_base_url
        assert config.llm_model == base.llm_model
        # ── 路由器的**阈值**也必须原样（只有 `enabled` 才是开关）────────
        assert config.router.timeout_seconds == base.router.timeout_seconds
        assert config.router.max_output_tokens == base.router.max_output_tokens
        assert config.router.prompt_version == base.router.prompt_version
        # ── 多跳的**上限**同理：消融 A 与 E1 只差"每轮精不精排" ───────
        assert config.agentic.max_rounds == base.agentic.max_rounds
        assert config.agentic.per_round_top_k == base.agentic.per_round_top_k
        assert config.agentic.sufficiency_threshold == base.agentic.sufficiency_threshold
        assert config.agentic.final_rerank_query_mode == base.agentic.final_rerank_query_mode


def test_the_four_isolation_fixes_are_always_applied(tmp_path) -> None:
    """四处"不是开关但必须固定"的东西：**路径 / 采集 / 指标出口 / 精排口径**。

    ⚠ 每一条都对应一种**失败模式或不可逆污染**，见 `_arm_config` 的 docstring：
    · 路径不对 ⇒ 写进开发库；· 采集不关 ⇒ 自测流量混进官方采集原文；
    · 指标出口不设 ⇒ 指标段整段空着且不报错；
    · **精排不显式打开 ⇒ 四个臂都没有精排**（`.env` 的 `local` profile 关着它）
      ⇒ 那对对照**两个臂一模一样**，但分数照样出得来。
    """
    base = _config(tmp_path)
    config = _arm_config(
        base,
        _arms()["e1"],
        collection="coq-eval-test",
        sqlite_path=tmp_path / "coq.db",
        metrics_path=tmp_path / "m.json",
        rerank=True,
    )
    assert config.storage.sqlite.path == str(tmp_path / "coq.db")
    assert config.storage.qdrant.collection == "coq-eval-test"
    assert config.capture.enabled is False
    assert config.metrics_path == str(tmp_path / "m.json")
    assert config.rerank.enabled is True
    # ⚠ 关精排**只能**由显式开关做到（`--no-rerank`），profile 说了不算
    off = _arm_config(
        base,
        _arms()["e0"],
        collection="c",
        sqlite_path=tmp_path / "t.db",
        metrics_path=tmp_path / "m.json",
        rerank=False,
    )
    assert off.rerank.enabled is False


def test_the_paths_are_isolated_from_the_development_store(tmp_path) -> None:
    """⚠ 只改开关**不改路径**会让实验写进开发库——那是一次不可逆的污染。"""
    base = _config(tmp_path)
    sqlite_path = tmp_path / "coq.db"
    for arm in ARMS:
        config = _arm_config(
            base,
            arm,
            collection="coq-eval-test",
            sqlite_path=sqlite_path,
            metrics_path=tmp_path / "m.json",
            rerank=True,
        )
        assert config.storage.sqlite.path == str(sqlite_path) != base.storage.sqlite.path
        assert config.storage.qdrant.collection == "coq-eval-test"
        assert config.storage.qdrant.collection != base.storage.qdrant.collection


def test_the_baseline_arm_is_the_shipped_default_behaviour(tmp_path) -> None:
    """E0 = **当前装配出的那份配置**（两条开关都关）⇒ 它才配当基线。

    ⚠ 这一条挡的是"基线臂被顺手改了一下"——那样四条臂的对比全部失去参照点。
    """
    arm = _arms()["e0"]
    assert (arm.router, arm.agentic, arm.force) == (False, False, False)
    config = _arm_config(
        _config(tmp_path),
        arm,
        collection="c",
        sqlite_path=tmp_path / "t.db",
        metrics_path=tmp_path / "m.json",
        rerank=True,
    )
    assert config.router.enabled is False
    assert config.agentic.enabled is False


def test_the_forced_arm_really_bypasses_the_router(tmp_path) -> None:
    """E1 的**全部意义**是"绕过路由器" ⇒ `force` 开着、`router` 关着。"""
    arm = _arms()["e1"]
    assert arm.force is True and arm.router is False
    config = _arm_config(
        _config(tmp_path),
        arm,
        collection="c",
        sqlite_path=tmp_path / "t.db",
        metrics_path=tmp_path / "m.json",
        rerank=True,
    )
    assert config.agentic.enabled is True
    assert config.agentic.force is True
    assert config.router.enabled is False  # 没有路由器可绕


def test_the_ablation_arm_differs_from_e1_in_exactly_one_switch(tmp_path) -> None:
    """对照成立条件：A 与 B(=E1) **只差"每轮精不精排"**这一件事。

    ⚠ 差第二处 ⇒ 那个对照回答不了它要回答的问题，**而分数看起来照样有差**。
    """
    a, b = _arms()["a"], _arms()["e1"]
    base = _config(tmp_path)
    common = {
        "collection": "c",
        "sqlite_path": tmp_path / "t.db",
        "metrics_path": tmp_path / "m.json",
        "rerank": True,
    }
    config_a = _arm_config(base, a, **common)
    config_b = _arm_config(base, b, **common)
    differences = {
        f.name for f in fields(config_a) if getattr(config_a, f.name) != getattr(config_b, f.name)
    }
    assert differences == {"agentic"}
    assert config_a.agentic.round_rerank_enabled is False
    assert config_b.agentic.round_rerank_enabled is True


def test_the_manifest_reader_skips_comments_and_groups_by_user(tmp_path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_text(
        "# 注释\n"
        "\n"
        + json.dumps({"user_id": "conv-26", "question_id": "conv-26#q0003"})
        + "\n"
        + json.dumps({"user_id": "conv-26", "question_id": "conv-26#q0005"})
        + "\n"
        + json.dumps({"user_id": "conv-30", "question_id": "conv-30#q0003"})
        + "\n",
        encoding="utf-8",
    )
    assert _load_manifest(path) == {
        "conv-26": ["conv-26#q0003", "conv-26#q0005"],
        "conv-30": ["conv-30#q0003"],
    }


def test_an_empty_manifest_is_a_hard_failure(tmp_path) -> None:
    """⚠ **不能**"零题也算跑完"：那会产出一张全空的表，而它看起来像"跑过了"。"""
    path = tmp_path / "m.jsonl"
    path.write_text("# 只有注释\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load_manifest(path)


def test_the_shipped_manifest_lists_only_multi_hop_questions() -> None:
    """清单里的每一行都必须是 `category == "1"`——**只存 id，不存原文**。"""
    rows = [
        json.loads(line)
        for line in Path("eval/coq-multihop.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(rows) == 25
    assert {row["category"] for row in rows} == {"1"}
    assert {row["user_id"] for row in rows} == {"conv-26", "conv-30", "conv-41"}
    for row in rows:
        assert set(row) == {"dataset", "user_id", "question_id", "category", "expects_evidence"}
        # ⛔ 原文一个字都不许进清单
        assert "question" not in row and "answer" not in row


def _corpus_sample() -> Sample:
    return Sample(
        user_id="musique-test",
        dataset="musique",
        sessions=(
            Session(
                session_id="corpus",
                messages=(
                    Message("user", "Title: Same\nFirst chunk."),
                    Message("user", "Title: Same\nSecond chunk."),
                    Message("user", "Title: Long\n" + "words " * MAX_MESSAGE_CHARS),
                ),
            ),
        ),
        questions=(),
    )


def _source_rows(sample: Sample) -> list[tuple]:
    return [
        (sample.user_id, session.session_id, payload["content"].strip(), None)
        for session in sample.sessions
        for payload in shape_batch(
            session.messages, dataset=sample.dataset, speaker_names=sample.speaker_names
        )
    ]


def _write_corpus_store(path: Path, rows: list[tuple]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE qa_pairs(user_id, session_id, question, answer)")
        conn.executemany("INSERT INTO qa_pairs VALUES (?, ?, ?, ?)", rows)


def test_source_integrity_accepts_all_chunks_and_official_long_message_splits(tmp_path) -> None:
    sample = _corpus_sample()
    rows = _source_rows(sample)
    assert len(rows) > sample.message_count
    path = tmp_path / "complete.db"
    _write_corpus_store(path, rows)
    _require_source_rows([sample], path)


@pytest.mark.parametrize("damage", ["missing", "replaced", "extra", "wrong_session", "answer"])
def test_source_integrity_rejects_nonempty_but_incomplete_or_changed_corpus(
    tmp_path, damage
) -> None:
    """有行或总数相等不足以证明完整；每一块正文及其 session 都须匹配。"""
    sample = _corpus_sample()
    rows = _source_rows(sample)
    if damage == "missing":
        rows.pop(1)  # 同标题第二段缺失，但用户仍非空
    elif damage == "replaced":
        rows[1] = (*rows[1][:2], "Corpus: Title: Same\nUnrelated chunk.", None)
    elif damage == "extra":
        rows.append(rows[0])
    elif damage == "wrong_session":
        rows[1] = (rows[1][0], "old-corpus", *rows[1][2:])
    else:
        rows[1] = (*rows[1][:3], "unexpected answer")
    path = tmp_path / "damaged.db"
    _write_corpus_store(path, rows)
    with pytest.raises(SystemExit, match="真源语料不匹配"):
        _require_source_rows([sample], path)


# ── `--reuse-hits`：缺少正文的产物必须**响亮拒绝** ─────────────────────
#
# ⚠ 答案步要用正文拼 `retrieved_context`。复用产物若只含 `{id, created_at}`，就必须
#   重跑 Search，不能从 SQLite 反推一份"**看起来一样**"的正文。因此命中文件会保存正文，
#   并拒绝复用缺少正文的产物。


def _hits_row(**extra) -> dict:
    return {"id": "m1", "created_at": "2026-07-26", "score": 1.0, **extra}


def test_reuse_hits_refuses_the_old_format_instead_of_guessing_content(tmp_path: Path) -> None:
    """缺少正文的复用产物**不能复用**：报错并点名，不许从库里反推。"""
    (tmp_path / "e1-hits.json").write_text(
        json.dumps({"q1": [_hits_row()]}),
        encoding="utf-8",  # 没有 content
    )
    with pytest.raises(SystemExit, match="旧格式"):
        _reuse_hits(tmp_path, "e1")


def test_reuse_hits_round_trips_the_saved_content(tmp_path: Path) -> None:
    """新产物带正文 ⇒ 复用后拿到的 `SearchHit` 与当初写进去的逐字段相同。"""
    original = {"q1": [SearchHit(id="m1", content="Q: 正文", created_at="2026-07-26", score=0.5)]}
    _write_hits(tmp_path / "e1-hits.json", original)
    got = _reuse_hits(tmp_path, "e1")
    assert got is not None and got["q1"][0] == original["q1"][0]


def test_reuse_hits_without_a_file_falls_back_to_searching(tmp_path: Path) -> None:
    """没有产物 ⇒ 返回 `None`（调用方照常检索），**不是**报错。"""
    assert _reuse_hits(tmp_path, "e1") is None


# ── 降级必须可见：`llm_unavailable` 不许被当成"跑过了" ──────────────────


def test_a_degraded_arm_is_reported_as_degraded() -> None:
    """⭐ `gpt-4o-mini` 在首轮 `llm_unavailable` 时会退回**普通混合检索**，
    而**汇总表仍会给出一个看起来正常的准确率**。

    这一条钉的是"降级能被点名出来"——否则"降级后的 E1"会被读成"E1"，
    而那是个**不会报错的错结论**。
    """
    chosen = [a for a in ARMS if a.key in {"e1", "e0"}]
    report = {
        "e0": {"correct": 11, "judged": 50},  # 基线臂没有多跳循环，不该被点名
        "e1": {
            "correct": 11,
            "judged": 50,
            "agentic_stop_reasons": {"llm_unavailable": 45, "max_rounds": 5},
            "agentic_degraded": 45,
        },
    }
    assert _degraded_arms(report, chosen) == [
        ("E1 强制多跳（绕过路由器）", 45, {"llm_unavailable": 45, "max_rounds": 5})
    ]


def test_a_clean_arm_is_not_reported_as_degraded() -> None:
    """反例：Team 网关那轮（`llm_unavailable` 0 个）不许被点名——否则告警会变成噪音。"""
    chosen = [a for a in ARMS if a.key == "e1"]
    report = {
        "e1": {
            "agentic_stop_reasons": {"duplicate_query": 37, "sufficient": 10},
            "agentic_degraded": 0,
        }
    }
    assert _degraded_arms(report, chosen) == []


def test_the_single_hop_manifest_is_a_non_multi_hop_control() -> None:
    """⭐ **单跳对照清单**：`category` 不许有 `"1"`（那是多跳）、`expects_evidence` 必须是 1。

    它存在的意义是给多跳那条线一个"**本来就不该多跳**"的对照：
    同样的服务、同样的语料，每道题一次查找就该答对。
    ⛔ 若哪天有人把 `category=="1"` 的题塞进来，这份对照就废了——而**分数照常打印**。
    """
    rows = [
        json.loads(line)
        for line in Path("eval/coq-locomo-singlehop.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(rows) == 30
    # ⛔ 不是多跳（1）：也不要 adversarial（5，`answer=null`）
    assert {row["category"] for row in rows} == {"2", "3", "4"}
    # ⭐ 单跳的客观判据：证据恰好 1 条 dia-id
    assert {row["expects_evidence"] for row in rows} == {1}
    # 与多跳清单**同一批对话** ⇒ 两套题共用一个语料，可直接并排比
    assert {row["user_id"] for row in rows} == {"conv-26", "conv-30", "conv-41"}
    assert all(row["dataset"] == "locomo-refined" for row in rows)
    assert len({row["question_id"] for row in rows}) == 30  # 无重复
    for row in rows:
        assert set(row) == {"dataset", "user_id", "question_id", "category", "expects_evidence"}
        # ⛔ 原文一个字都不许进清单
        assert "question" not in row and "answer" not in row


# ── Plan 模式在本工具里的接线────────────────────────────────


def test_react_mode_keeps_the_historical_arm_keys() -> None:
    """⭐ **默认模式不带后缀**：`react` 的产物键就是臂键。

    ⛔ 加后缀会把 `e1` 变成 `e1-react`，破坏既有 `report.json` 的读取契约。
    """
    from tools.coq_eval import _arm_identity

    e1 = _arms()["e1"]
    assert _arm_identity(e1, "react") == ("e1", e1.label)
    key, label = _arm_identity(e1, "plan")
    assert key == "e1-plan"
    assert label.endswith("plan")


def test_the_plan_arm_only_ever_gets_its_execution_record() -> None:
    """⭐ **只有真跑过 Plan 的那些请求**才有执行记录可前置。

    ⛔ React 的调用记录**没有** `mode: plan` ⇒ 它拿到的字典是空的 ⇒
    `build_input_items` 不附加执行记录。
    """
    from tools.coq_eval import _plan_record_texts

    calls = [
        {"qid": "q1", "mode": "react", "stop_reason": "sufficient", "rounds": []},
        {
            "qid": "q2",
            "mode": "plan",
            "query": "orig?",
            "plan": [{"id": "s1", "query": "a", "depends_on": []}],
            "plan_history": [
                {
                    "step_id": "s1",
                    "executed_query": "a",
                    "answer": "A",
                    "evidence": ["m1"],
                    "status": "answered",
                    "stop_reason": "plan_complete",
                }
            ],
        },
    ]
    texts = _plan_record_texts(calls)
    assert set(texts) == {"q2"}
    text = texts["q2"]
    # ⛔ **必须自称是模型生成的执行记录**——不许伪装成记忆
    assert "MACHINE-GENERATED execution record" in text
    assert "NOT a memory record" in text
    # 计划与实际执行的问答都在
    assert "s1: a" in text
    assert "s1 question: a" in text
    assert "s1 answer: A" in text
    assert "s1 sources: m1" in text
    # ⛔ 不编造 memory id：出现的 id 只可能是记录里本来就有的那些
    assert "m9" not in text


def test_a_stalled_plan_step_says_so_in_the_record() -> None:
    """没解出来的那一步要在记录里**明说状态**：否则模型会把"没有下文"读成"计划就这么多"。"""
    from tools.coq_eval import _execution_record_text

    text = _execution_record_text(
        {
            "query": "orig?",
            "plan": [{"id": "s1", "query": "a", "depends_on": []}],
            "plan_history": [
                {
                    "step_id": "s1",
                    "executed_query": "a",
                    "answer": None,
                    "evidence": [],
                    "status": "unresolved",
                    "stop_reason": "dependency_unresolved",
                }
            ],
        }
    )
    assert "s1 answer: (none)" in text
    assert "s1 status: unresolved (dependency_unresolved)" in text
    assert "s1 sources: (none)" in text


def test_the_plan_summary_counts_what_the_report_must_say() -> None:
    """报告要的那几项**从 `plan_history` 里数出来**：计划几步 / 跑了几步 / 停在哪。"""
    from tools.coq_eval import _plan_summary

    calls = [
        {
            "qid": "q1",
            "mode": "plan",
            "plan_steps": 3,
            "steps_executed": 2,
            "dependencies_repaired": 2,
            "stop_reason": "dependency_unresolved",
            "plan_history": [
                {
                    "status": "answered",
                    "quotes_unverified": 0,
                    "documents": ["m1"],
                    "recalled": 20,
                },
                {
                    "status": "unresolved",
                    "quotes_unverified": 2,
                    "documents": ["m2"],
                    "recalled": 20,
                },
            ],
        }
    ]
    summary = _plan_summary(calls)
    assert summary["plan_steps"] == {3: 1}
    assert summary["steps_executed"] == {2: 1}
    assert summary["stop_reasons"] == {"dependency_unresolved": 1}
    assert summary["dependencies_repaired"] == 2
    assert summary["quotes_unverified"] == 2
    assert summary["answered_steps"] == 1
    assert summary["total_step_records"] == 2
    assert summary["mean_recalled_per_round"] == 20.0


def test_the_mode_is_recorded_in_the_run_switches(tmp_path) -> None:
    """⚠ 模式进 run record：它是这一轮的自变量，不记就没法归因。"""
    from tools.coq_eval import _arm_config

    from tianximem.common.config import AgenticConfig

    base = _config(tmp_path)
    assert base.agentic.mode == "react"  # 内置缺省
    plan = _arm_config(
        base,
        _arms()["e1"],
        collection="c",
        sqlite_path=tmp_path / "t.db",
        metrics_path=tmp_path / "m.json",
        rerank=True,
        mode="plan",
    )
    assert plan.agentic.mode == "plan"
    # ⛔ 不传 `mode` ⇒ **不覆盖**（无 mode 的配置逐字跑 React）
    keep = _arm_config(
        base,
        _arms()["e1"],
        collection="c",
        sqlite_path=tmp_path / "t.db",
        metrics_path=tmp_path / "m.json",
        rerank=True,
    )
    assert keep.agentic.mode == AgenticConfig().mode == "react"


# ── 恢复分支的答案输入────────────────────────────────────


def _plan_call(qid: str, *, entered: bool, status: str = "success") -> dict:
    """一条**最小可判**的 plan 调用记录（只放判据真正读到的字段）。"""
    return {
        "qid": qid,
        "mode": "plan",
        "query": "orig?",
        "plan": [{"id": "s1", "query": "a", "depends_on": []}],
        "plan_history": [
            {
                "step_id": "s1",
                "executed_query": "a",
                "answer": None if entered else "A",
                "evidence": [],
                "status": "unresolved" if entered else "answered",
                "stop_reason": "dependency_unresolved" if entered else "plan_complete",
                "quotes_unverified": 0,
                "documents": ["m1"],
                "recalled": 20,
            }
        ],
        "plan_steps": 1,
        "steps_executed": 1,
        "stop_reason": "dependency_unresolved" if entered else "plan_complete",
        "dependencies_repaired": 0,
        "recovery": {
            "entered": entered,
            # ⚠ 与真 recorder 一致：没进恢复分支时 `status` 就是 `not_triggered`。
            "status": status if entered else "not_triggered",
            "rerank_query_mode": "original" if entered else "joint",
        },
        "answer_input_has_record": not entered,
    }


def test_the_records_guard_excludes_recovery_questions_from_its_denominator() -> None:
    """⭐ 执行记录守卫的分母**不含恢复分支的题**。

    ⚠ ：LoCoMo 那 25 题里有 9 题 `unresolved` ⇒ 守卫报 `16/25` 并**中止整轮**
    （判分都没跑）。那是一条**假的**回归——那 9 题的答题输入**按设计**就不带执行记录。
    """
    from tools.coq_eval import _require_plan_records

    calls = [_plan_call("q1", entered=False), _plan_call("q2", entered=True)]
    assert set(_require_plan_records(calls, {"q1", "q2"})) == {"q1"}


def test_the_records_guard_still_fires_when_a_record_is_really_missing() -> None:
    """⚠ 但"真的缺"（`--reuse-hits` 那种，一条记录都没有）**仍要响亮失败**。"""
    from tools.coq_eval import _require_plan_records

    calls = [_plan_call("q1", entered=False)]
    with pytest.raises(SystemExit, match="执行记录"):
        _require_plan_records(calls, {"q1", "q2"})


def test_a_recovery_question_gets_no_execution_record_in_its_answer_input() -> None:
    """⭐ 进了恢复分支的题：**最终答题输入不折执行记录**（原题 ＋ 精排后的原文）。

    ⚠ 记录**没有丢**：它完整保存在 `<臂>-rounds.json` 里（`_execution_record_text`
    只是不会被**前置**到答案输入）。
    """
    from tools.coq_eval import _plan_record_texts, _recovery_calls

    calls = [_plan_call("q1", entered=False), _plan_call("q2", entered=True)]
    assert set(_plan_record_texts(calls)) == {"q1"}
    assert [c["qid"] for c in _recovery_calls(calls)] == ["q2"]


def test_the_pool_already_full_case_also_skips_the_record() -> None:
    """⚠ 判据是 `entered`，⛔ 不是 `status == "success"`：

    "触发但池子已经够了"（没真检索）的题**同样**不带执行记录——中间步骤失败，
    不代表原题答不出来（用户口径）。
    """
    from tools.coq_eval import _plan_record_texts, _recovery_calls

    calls = [_plan_call("q1", entered=True, status="skipped_pool_full")]
    assert _plan_record_texts(calls) == {}
    assert len(_recovery_calls(calls)) == 1


def test_the_plan_summary_separates_the_four_recovery_states() -> None:
    """⭐ **"触发但池子已经够了"与"真的跑了兜底"在汇总里分得开**。"""
    from tools.coq_eval import _plan_summary

    calls = [
        _plan_call("q1", entered=False),
        _plan_call("q2", entered=True, status="success"),
        _plan_call("q3", entered=True, status="skipped_pool_full"),
    ]
    summary = _plan_summary(calls)
    assert summary["recovery_statuses"] == {
        "not_triggered": 1,
        "success": 1,
        "skipped_pool_full": 1,
    }
    assert summary["rerank_query_modes"] == {"joint": 1, "original": 2}
    assert summary["answer_input_without_record"] == 2
    assert summary["answer_input_with_record"] == 1


def test_the_recorder_really_writes_the_recovery_fields() -> None:
    """⭐ **recorder 必须真的把 `recovery` 写进逐题记录**。

    ⚠ 下游汇总对缺失字段**有默认值**。字段未写入时会打印
    `recovery_statuses: {not_triggered: 50}`，**看起来像"一次都没触发"**。
    ⇒ 这条用例把"字段在不在"变成断言。
    """
    from dataclasses import replace

    from tools.coq_eval import _RecordingAgent

    from tianximem.agent import PlanRecovery, PlanResult, PlanStats

    inner_result = replace(
        PlanResult(
            candidates=(),
            rerank_query="q?",
            stats=PlanStats(
                rounds=(),
                stop_reason="dependency_unresolved",
                fallback_reason=None,
                candidate_count=0,
                unique_candidate_count=0,
                selected_evidence_count=0,
                joint_query_count=1,
                llm_calls=1,
                rerank_calls=0,
                latency_ms=1.0,
            ),
            plan=(),
            plan_history=(),
        ),
        recovery=PlanRecovery(
            entered=True,
            status="success",
            executed_rounds=1,
            target_top_k=100,
            pool_before=20,
            recalled=100,
            reranked="called",
            added_unique=80,
            latency_ms=12.5,
            rerank_query_mode="original",
        ),
    )

    class _Inner:
        def search(self, *, user_id: str, query: str, top_k: int):
            return inner_result

    recorder = _RecordingAgent(_Inner(), mode="plan")
    recorder.search(user_id="u", query="q?", top_k=100)
    record = recorder.calls[-1]
    assert record["recovery"]["entered"] is True
    assert record["recovery"]["status"] == "success"
    assert record["recovery"]["target_top_k"] == 100
    assert record["recovery"]["pool_before"] == 20
    assert record["recovery"]["reranked"] == "called"
    assert record["recovery"]["added_unique"] == 80
    assert record["recovery"]["rerank_query_mode"] == "original"
    # ⭐ 进去了恢复分支 ⇒ 答案输入**不折**执行记录。
    assert record["answer_input_has_record"] is False
    # ⚠ 缺字段时的默认值会**伪装**成"没触发"——所以这一条必须是硬断言。
    assert "recovery" in record
