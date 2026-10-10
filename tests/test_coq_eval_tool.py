"""`tools/coq_eval.py` 的**纯函数**部分——不发请求、不读数据集。

## 为什么这个文件值当

§13 的那条纪律（**开关必须只影响它命名的那一件事**）在这里有一个具体的落点：
四臂的配置是**从一份基线 `replace` 出来的**。多改一个字段（比如顺手把
`retrieval.grounded_evidence` 也关了），对照就不成立——
**而结果看起来完全正常，只是结论错了**。

所以这里逐个字段比对四臂与基线的差异，**把"只动了那四个"变成一条断言**。
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest
from tests.test_router import _config  # noqa: PLC2701 — 复用同一套"真配置 + 临时路径"
from tools.coq_eval import ARMS, _arm_config, _load_manifest


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
    """四处"不是开关但必须改"的东西：**路径 / 采集 / 指标出口 / 精排口径**。

    ⚠ 每一条都对应一次**踩过的坑或一次不可逆的污染**，见 `_arm_config` 的 docstring：
    · 路径不对 ⇒ 写进开发库；· 采集不关 ⇒ 自测流量混进 S6 的官方原文；
    · 指标出口不设 ⇒ §14 整段空着且不报错；
    · **精排不显式打开 ⇒ 四个臂都没有精排**（`.env` 的 `local` profile 关着它）
      ⇒ §12.4 那对对照**两个臂一模一样**，但分数照样出得来。
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
    """E0 = **今天装出来的那份配置**（两条开关都关）⇒ 它才配当基线。

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
    """E1 的**全部意义**是"绕过路由器"（§12.3）⇒ `force` 开着、`router` 关着。"""
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
    """§12.4 的对照成立条件：A 与 B(=E1) **只差"每轮精不精排"**这一件事。

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
        f.name
        for f in fields(config_a)
        if getattr(config_a, f.name) != getattr(config_b, f.name)
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
    """清单里的每一行都必须是 `category == "1"`——**只存 id，不存原文**（§12.5）。"""
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
