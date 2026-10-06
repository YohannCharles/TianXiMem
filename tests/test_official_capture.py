"""官方采集套件（`official-dataset-*`）的三层：**套件合成 / 加载层 / 分派 pipeline**。

**全部用合成 fixture**（`tmp_path`），不对采集目录或归档有任何依赖——
那份数据是 300+ MB 的线上流量、且不进 git（与 `test_datasets_extra.py` 同一条纪律）。

钉住的是四条**错了不会报错**的东西：

| 断言 | 不钉的后果 |
| --- | --- |
| `replay_plan` 按 ts 交错、add 先于同刻的 search | 先灌完再问 ⇒ 41% 的题**读到未来**，指标虚高 |
| 缺省只加载 `judge_kind != "none"` 的题 | 没金标的题也去调网关，白发钱且分数没变 |
| 套件把公开金标/extra 金标折叠成**统一的判分语义** | 判分器按错的 `judge_kind` 走，**静默**判错 |
| 分派 pipeline 的纯函数判分（MQuAKE 别名 / TempReason 串） | 逐题判分与 `extra_pipeline` 分家 |
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.datasets.official_capture import load_official_capture, replay_plan
from eval.harness.judge import build_official_items, pipeline_for

USER_A = "u_" + "a" * 64
USER_B = "u_" + "b" * 64


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question(seq: int, user: str, ts: str, **overrides) -> dict:
    row = {
        "ts": ts,
        "client": "1.2.3.4",
        "status": 200,
        "error": None,
        "latency_ms": 1.0,
        "user_id": user,
        "query": f"q{seq}?",
        "top_k": 100,
        "options_field": None,
        "question": f"q{seq}?",
        "options": [],
        "task": "open",
        "instruction": None,
        "prior_adds_for_user": 1,
        "seq": seq,
        "official_score": None,
        "gold_official_answer": null_,
        "public_gold": None,
    }
    row.update(overrides)
    return row


null_ = None


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    """一份**最小但形状齐全**的采集导出：2 个 user、3 道题、1 条 extra 金标。

    * `USER_A`：**交错**——a1 → s1 → a2 → s2（seq 1/3 早于最后一条 add）
    * `USER_B`：批式——a3 → s3，且 s3 **没有金标**（缺省不该被加载）
    """
    root = tmp_path / "capture"
    root.mkdir()
    _write(
        root / "official-eval-questions.jsonl",
        [
            _question(
                1,
                USER_A,
                "2026-09-29T10:00:01+00:00",
                public_gold={
                    "gold_locomo_answer": ["19 January, 2023"],
                    "gold_locomo_qa_id": "conv-30#q0000",
                },
            ),
            _question(
                3,
                USER_A,
                "2026-09-29T10:00:03+00:00",
                task="single_choice",
                instruction="…(X).",
                public_gold={
                    "gold_scriptmem_answer": "B. …",
                    "gold_scriptmem_answer_letters": ["B"],
                },
            ),
            _question(5, USER_B, "2026-09-29T10:00:05+00:00"),
        ],
    )
    _write(
        root / "official-adds.jsonl",
        [
            {
                "request_id": "r_a1",
                "user_id": USER_A,
                "session_id": "s_x",
                "ts": "2026-09-29T10:00:00+00:00",
                "messages": [{"role": "user", "content": "hello", "timestamp": 1}],
            },
            {
                "request_id": "r_a2",
                "user_id": USER_A,
                "session_id": "s_x",
                "ts": "2026-09-29T10:00:02+00:00",
                "messages": [{"role": "assistant", "content": "hi", "timestamp": 2}],
            },
            {
                "request_id": "r_b1",
                "user_id": USER_B,
                "session_id": "s_y",
                "ts": "2026-09-29T10:00:04+00:00",
                "messages": [{"role": "user", "content": "yo", "timestamp": 3}],
            },
        ],
    )
    events = [
        {
            "seq": 0,
            "ts": "2026-09-29T10:00:00+00:00",
            "kind": "add",
            "user_id": USER_A,
            "session_id": "s_x",
            "request_id": "r_a1",
            "n_messages": 1,
            "status": 200,
            "prior_adds_for_user": 0,
        },
        {
            "seq": 1,
            "ts": "2026-09-29T10:00:01+00:00",
            "kind": "search",
            "user_id": USER_A,
            "session_id": "s_x",
            "request_id": None,
            "n_messages": None,
            "status": 200,
            "prior_adds_for_user": 1,
        },
        {
            "seq": 2,
            "ts": "2026-09-29T10:00:02+00:00",
            "kind": "add",
            "user_id": USER_A,
            "session_id": "s_x",
            "request_id": "r_a2",
            "n_messages": 1,
            "status": 200,
            "prior_adds_for_user": 1,
        },
        {
            "seq": 3,
            "ts": "2026-09-29T10:00:03+00:00",
            "kind": "search",
            "user_id": USER_A,
            "session_id": "s_x",
            "request_id": None,
            "n_messages": None,
            "status": 200,
            "prior_adds_for_user": 2,
        },
        {
            "seq": 4,
            "ts": "2026-09-29T10:00:04+00:00",
            "kind": "add",
            "user_id": USER_B,
            "session_id": "s_y",
            "request_id": "r_b1",
            "n_messages": 1,
            "status": 200,
            "prior_adds_for_user": 0,
        },
        {
            "seq": 5,
            "ts": "2026-09-29T10:00:05+00:00",
            "kind": "search",
            "user_id": USER_B,
            "session_id": "s_y",
            "request_id": None,
            "n_messages": None,
            "status": 200,
            "prior_adds_for_user": 1,
        },
    ]
    _write(root / "official-timeline.jsonl", events)
    _write(
        root / "official-gold-extra.jsonl",
        [
            {
                "seq": 1,
                "ts": "2026-09-29T10:00:01+00:00",
                "user_id": USER_A,
                "question": "q1?",
                "source_dataset": "mquake-remastered",
                "source_locator": {"file": "a.parquet", "case_id": 1},
                "gold_kind": "exact",
                "gold_answers": ["1789"],
                "gold_rubric": None,
                "gold_native": None,
                "gold_judging": "MQuAKE：别名命中即对",
            }
        ],
    )
    # 套件（`official-eval-kit.jsonl`）是加载层的**唯一入口** ⇒ fixture 直接把它建出来，
    # 免得每个用例都先跑一遍 build（`test_kit_*` 那几个仍自己调一次，断言建出来的形状）。
    from tools.build_official_kit import build

    build(root)
    return root


# ── 套件合成 ───────────────────────────────────────────────────────────


def test_kit_folds_gold_into_one_judge_vocabulary(capture: Path) -> None:
    """公开金标与 extra 金标折成**同一套判分语义**；`judge_kind` 逐族不同。"""
    from tools.build_official_kit import build

    stats = build(capture)
    kit_path = capture / "official-eval-kit.jsonl"
    kit = [json.loads(line) for line in kit_path.open(encoding="utf-8")]
    by_seq = {row["seq"]: row for row in kit}

    assert stats["rows"] == 3
    assert stats["scorable"] == 2  # seq=5 没有金标
    # `official-gold-extra.jsonl` 后到 ⇒ 它赢（先到的赢只发生在同一个 seq 上）
    assert by_seq[1]["dataset"] == "mquake-remastered"
    assert by_seq[1]["gold_answers"] == ["1789"]
    assert by_seq[3]["dataset"] == "scriptmem"
    assert by_seq[3]["judge_kind"] == "letters"
    assert by_seq[3]["gold_answers"] == ["B"]
    assert by_seq[5]["judge_kind"] == "none"
    # 每一行都带得动"该用哪份 prompt"
    assert by_seq[3]["answer_prompt_ref"].startswith("dataset/.upstream/aml/pipeline_scriptmem.py")


def test_kit_attribution_is_per_user_and_records_basis(capture: Path) -> None:
    """归属表：一 user 一行，**依据分层记下来**（不是只写一个数据集名）。"""
    from tools.build_official_kit import build

    build(capture)
    rows = [
        json.loads(line) for line in (capture / "official-attribution.jsonl").open(encoding="utf-8")
    ]
    by_user = {row["user_id"]: row for row in rows}
    assert by_user[USER_A]["dataset"] == "mquake-remastered"
    assert by_user[USER_A]["basis"] == "labelled_rows"
    # 没有金标、也没有自标标签的 user **退到形态启发式**——`basis` 如实写成 `corpus`，
    # 别让"猜的"与"命中的"在表里长得一样。
    assert by_user[USER_B]["basis"] == "corpus"


# ── 加载层：timeline 交错 ──────────────────────────────────────────────


def test_replay_plan_interleaves_in_time_order(capture: Path) -> None:
    """**这一条是本文件最要紧的断言**：add 与 search 按 ts 交错，不是"先灌完再问"。"""
    plan = replay_plan(capture, only_users=[USER_A])[0]
    assert [event.kind for event in plan.events] == ["add", "search", "add", "search"]
    assert [event.seq for event in plan.events] == [0, 1, 2, 3]
    assert plan.add_count == 2
    assert [row["seq"] for row in plan.questions] == [1, 3]


def test_replay_plan_skips_unscorable_by_default(capture: Path) -> None:
    """没金标的题**不加载**（跑了也没有分数，而解答要花网关时间）。"""
    default = replay_plan(capture)
    assert sorted(row["seq"] for plan in default for row in plan.questions) == [1, 3]
    everything = replay_plan(capture, scorable_only=False)
    assert sorted(row["seq"] for plan in everything for row in plan.questions) == [1, 3, 5]


def test_replay_plan_orders_users_by_scoreable_questions(capture: Path) -> None:
    """`--users N` 取的是**产出最高的 N 个 user**，而不是"前 N 个"。

    ⚠ 一道可评分题都没有的 user **不进计划**（`USER_B` 就是这种）——放进来只会
    白白灌一遍语料、再判一堆没有分数的题。
    """
    plans = replay_plan(capture)
    assert [plan.user_id for plan in plans] == [USER_A]
    assert [len(plan.questions) for plan in plans] == [2]


def test_load_official_capture_keeps_payload_verbatim(capture: Path) -> None:
    """`Sample` 侧**不加工正文**（不加前缀、不 strip）——采集原文就是 payload。"""
    samples = load_official_capture(capture, limit=1)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.user_id == USER_A
    assert sample.dataset == "official-capture"
    assert [session.session_id for session in sample.sessions] == ["s_x", "s_x"]
    assert [question.qid for question in sample.questions] == ["1", "3"]
    assert [question.category for question in sample.questions] == [
        "mquake-remastered",
        "scriptmem",
    ]


# ── 输入项与分派 ───────────────────────────────────────────────────────


def test_build_official_items_carries_family_and_gold(capture: Path) -> None:
    """注入项的 `dataset` 是**题自己的归属**（一个 run 里混着 8 个数据集）。"""
    kit_path = capture / "official-eval-kit.jsonl"
    kit = [json.loads(line) for line in kit_path.open(encoding="utf-8")]
    items = build_official_items(kit, {"1": []}, date_mode="none")
    first = items[0]
    assert first["id"] == "1"
    assert first["dataset"] == "mquake-remastered"
    assert first["judge_kind"] == "exact"
    # `question` 是**采集原文**（检索用的那一串），`question_text` 才是解析后的题面
    assert first["question"] == "q1?"
    assert first["question_text"] == "q1?"
    assert first["gold_answer"] == ["1789"]
    assert "speaker_1_memories" not in first  # 平铺形状：走 `retrieved_context`
    assert first["retrieved_context"] == ""


def test_pipeline_for_resolves_official_capture() -> None:
    path = pipeline_for(Path("benchmark_data"), "official-capture")
    assert path.name == "official_capture_pipeline.py"
    assert path.exists()


def test_dispatcher_pure_judges_do_not_call_models() -> None:
    """分派层的**纯函数**判分（不碰网关）：MQuAKE 别名表命中、TempReason 串命中。"""
    from eval.harness.official_capture_pipeline import _judge

    mquake = {
        "id": "10",
        "dataset": "mquake-remastered",
        "judge_kind": "exact",
        "gold_answer": ["science fiction", "sci-fi"],
        "gold_answers": ["science fiction", "sci-fi"],
    }
    ok = _judge(mquake, "The genre is science fiction.", max_tokens=64)
    assert ok["is_correct"] is True
    assert "命中别名" in ok["judge_response"]

    bad = _judge(mquake, "Cannot determine from the memories.", max_tokens=64)
    assert bad["is_correct"] is False


def test_dispatcher_unknown_family_says_so_instead_of_scoring_zero() -> None:
    """没配判分的家族**响亮**（`JUDGE_ERROR`），不静默判 0 —— 静默判 0 会被读成"模型不行"。"""
    from eval.harness.official_capture_pipeline import _judge

    row = _judge(
        {"id": "11", "dataset": "some-new-set", "judge_kind": "weird", "gold_answers": ["x"]},
        "x",
        max_tokens=64,
    )
    assert row["label"] == "JUDGE_ERROR"
    assert "some-new-set" in row["judge_response"]


def test_dispatcher_answer_prompt_falls_back_for_unknown_family() -> None:
    """未知家族走通用 prompt——**新补一族金标时不需要改这个文件**（见 `_generic_judge`）。"""
    from eval.harness.official_capture_pipeline import _answer_prompt

    prompt = _answer_prompt(
        {"dataset": "halumem", "question": "Where does she live?", "retrieved_context": "ctx"}
    )
    assert "ctx" in prompt and "Where does she live?" in prompt


def test_feverous_verdict_is_not_alias_containment() -> None:
    """FEVEROUS 三分类：**顺序敏感**——`does not support` 不能被读成 `SUPPORTS`。

    这条钉的是"别把 FEVEROUS 塞进 `exact` 那一档"：别名包含会把 `NOT ENOUGH INFO`
    里出现的 `support` 反义句判成命中。
    """
    from eval.harness.official_capture_pipeline import _feverous_judge

    def judge(generated: str, gold: str) -> dict:
        return _feverous_judge(
            {"id": "1", "dataset": "feverous", "gold_answers": [gold]}, generated
        )

    assert judge("The claim is SUPPORTED by the evidence.", "SUPPORTS")["is_correct"] is True
    assert judge("This claim is REFUTED.", "SUPPORTS")["is_correct"] is False
    # ⚠ 反义表述：先命中 NOT ENOUGH INFO 那一档 ⇒ 不是 SUPPORTS
    no_info = judge("There is not enough information to support this.", "SUPPORTS")
    assert no_info["is_correct"] is False
    assert judge("The claim is REFUTES.", "REFUTES")["is_correct"] is True
    assert judge("I cannot tell.", "SUPPORTS")["is_correct"] is False


def test_halumem_judge_reads_upstream_prompt_verbatim() -> None:
    """HaluMem 的裁判 prompt **从归档的上游文件里读**（单数来源，不在本仓抄第二份）。"""

    from eval.datasets.layout import archive_file
    from eval.datasets.registry import benchmark_dir
    from eval.harness.official_capture_pipeline import (
        _HALU_TEMPLATE_FILE,
        _HALU_TEMPLATE_NAME,
        _parse_three_way,
        _upstream_template,
    )

    if not archive_file(benchmark_dir(), _HALU_TEMPLATE_FILE).exists():
        pytest.skip("HaluMem 归档不在（见 tools/recover_halumem.py）")
    template = _upstream_template(_HALU_TEMPLATE_FILE, _HALU_TEMPLATE_NAME)
    # 逐字规则在不在——少了它，判分就变成了我们自己编的一套
    assert "semantically equivalent" in template
    assert "{question}" in template and "{reference_answer}" in template
    assert "{key_memory_points}" in template and "{response}" in template
    # 三分类解析：只有 `Correct` 算对（`Omission` 是**另一个失败档**，不许并进正确率）
    assert _parse_three_way('{"evaluation_result": "Correct"}') == "Correct"
    assert _parse_three_way('{"evaluation_result": "Hallucination"}') == "Hallucination"
    assert _parse_three_way('{"evaluation_result": "Omission"}') == "Omission"
    assert _parse_three_way("no json here") is None


def test_every_family_gets_its_memories_in_the_answer_prompt() -> None:
    """**一条断言挡住"接错键"这一类**：每个族的答案 prompt 里都必须出现检索到的正文。

    ⚠ 这条是从一次真实事故里长出来的（2026-10-06）：CL-Bench 那一支一开始把我们的项
    直接交给官方那份 `build_answer_prompt`，而它读的是 `system_prompt` 与
    `retrieval.selected[*].text`——**我们的项里两个键都没有** ⇒ 渲染出 `(no memories)`
    ⇒ 模型**盲答**，74 题全 0 **且不报错**（分数看着只是"低"）。
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    from eval.datasets.registry import capture_dir
    from eval.harness import SearchHit
    from eval.harness.official_capture_pipeline import _answer_prompt

    kit_path = Path(capture_dir()) / "official-eval-kit.jsonl"
    if not kit_path.exists():
        pytest.skip("采集套件不在（见 tools/build_official_kit.py）")
    by_family: dict[str, list[dict]] = defaultdict(list)
    for line in kit_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["judge_kind"] != "none":
            by_family[row["dataset"]].append(row)

    marker = "MEMORY-MARKER-12345"
    hits = [SearchHit(id="m1", content=marker, created_at="2025-01-01", score=0.9)]
    assert by_family, "套件里一道可评分的题都没有？"
    for family, rows in sorted(by_family.items()):
        items = build_official_items(rows[:1], {str(rows[0]["seq"]): hits}, date_mode="none")
        prompt = _answer_prompt(items[0])
        assert marker in prompt, f"{family}：答案 prompt 里没有检索到的正文（接错键了？）"
