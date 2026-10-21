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


def test_official_items_use_each_dataset_contract_and_category() -> None:
    """每个数据集自己的答案契约/题类必须进入输入项——缺了就会退回通用 prompt。"""
    from eval.harness.judge_items import build_official_items

    rows = [
        {
            "seq": 1,
            "dataset": "musique",
            "judge_kind": "exact",
            "query": "q1",
            "question": "q1",
            "gold_native": {"answerable": True, "answer": "x", "answer_aliases": []},
        },
        {
            "seq": 2,
            "dataset": "hybridqa",
            "judge_kind": "exact",
            "query": "q2",
            "question": "q2",
            "gold_native": {"answer-text": "y"},
        },
        {
            "seq": 3,
            "dataset": "feverous",
            "judge_kind": "exact",
            "query": "q3",
            "question": "q3",
            "gold_native": {"label": "SUPPORTS", "evidence": []},
        },
        {
            "seq": 4,
            "dataset": "medmemorybench",
            "judge_kind": "exact",
            "query": "q4",
            "question": "q4",
            "gold_native": {"query_type": "multi_hop_clinical_deduction"},
        },
    ]
    items = build_official_items(rows, {})
    contracts = {item["dataset"]: item.get("answer_contract") for item in items}
    assert contracts["musique"] == "musique-local-qa-v1"
    assert contracts["hybridqa"] == "hybridqa-table-passages-v1"
    assert contracts["feverous"] == "feverous-claim-pages-evidence-v3"
    assert items[3]["category"] == "multi_hop_clinical_deduction"


def test_answer_prompt_uses_corpusqa_for_hybridqa() -> None:
    """HybridQA 必须用官方作答 prompt，不能退回通用那一份。"""
    from eval.harness.official_capture_pipeline import _answer_prompt

    hybrid = _answer_prompt(
        {
            "dataset": "hybridqa",
            "question": "Q",
            "retrieved_context": "CTX",
            "answer_contract": "hybridqa-table-passages-v1",
        }
    )
    assert "Return only the short answer" in hybrid
    assert "CTX" in hybrid


def test_feverous_capture_prompt_asks_only_for_a_label() -> None:
    """采集路径只评标签 ⇒ prompt 只要标签，不要证据 ID（判分也不用它们）。"""
    from eval.harness.official_capture_pipeline import _answer_prompt

    prompt = _answer_prompt(
        {
            "dataset": "feverous",
            "question": (
                "Verify the following claim using only the supplied corpus. "
                "Return one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO. "
                "Also return the native evidence IDs for a sufficient evidence set.\n\n"
                "Claim: The radio station KBQL is in Las Vegas, New Mexico."
            ),
            "retrieved_context": "CTX",
            "answer_contract": "feverous-claim-pages-evidence-v3",
        }
    )
    assert "SUPPORTS, REFUTES, or NOT ENOUGH INFO" in prompt
    assert "Claim: The radio station KBQL is in Las Vegas, New Mexico." in prompt
    assert "CTX" in prompt
    assert "native evidence IDs" not in prompt
    assert "Cannot determine from the memories" not in prompt


def test_answer_prompt_uses_native_musique_contract() -> None:
    """采集行没有 category ⇒ 一律原生 MuSiQue-Full（含拒答出口），不推断链式档。"""
    from eval.harness.official_capture_pipeline import _answer_prompt

    prompt = _answer_prompt(
        {
            "dataset": "musique",
            "question": "Q",
            "retrieved_context": "CTX",
            "answer_contract": "musique-local-qa-v1",
        }
    )
    assert "INSUFFICIENT_EVIDENCE" in prompt
    assert "Cannot determine from the memories" not in prompt


def test_dispatcher_routes_hybridqa_to_official_em(monkeypatch) -> None:
    """HybridQA 的正式分走 `corpusqa_pipeline.score_answer`（EM + F1 指标）。"""
    from eval.harness import corpusqa_pipeline
    from eval.harness.official_capture_pipeline import _judge

    seen: dict = {}

    def fake_score(item, generated):
        seen["item"] = item
        return {
            "id": item["id"],
            "is_correct": True,
            "label": "CORRECT",
            "judge_response": "upstream answer EM=1, token F1=1",
            "metrics": {"exact_match": 1.0, "f1": 1.0},
        }

    monkeypatch.setattr(corpusqa_pipeline, "score_answer", fake_score)
    row = _judge(
        {
            "id": "7",
            "dataset": "hybridqa",
            "judge_kind": "exact",
            "gold_answer": {"answer-text": "1991-92"},
        },
        "1991-92",
        max_tokens=8,
    )
    assert row["is_correct"] is True
    assert row["metrics"] == {"exact_match": 1.0, "f1": 1.0}
    assert seen["item"]["id"] == "7"


def test_dispatcher_routes_musique_to_its_own_judge(monkeypatch) -> None:
    """MuSiQue 走 `extra_pipeline.judge_one` 的规则判分，不借 TempReason 别名判分。"""
    from eval.harness import extra_pipeline as extra
    from eval.harness.official_capture_pipeline import _judge

    seen: dict = {}

    def fake_judge(dataset, qid, item, generated, **_kwargs):
        seen["args"] = (dataset, qid, generated)
        return {"id": qid, "is_correct": True, "label": "CORRECT", "judge_response": "rule"}

    monkeypatch.setattr(extra, "judge_one", fake_judge)
    row = _judge(
        {
            "id": "9",
            "dataset": "musique",
            "judge_kind": "exact",
            "gold_answer": {"answerable": True, "answer": "x"},
        },
        "x",
        max_tokens=8,
    )
    assert row["is_correct"] is True
    assert seen["args"] == ("musique", "9", "x")


def test_generic_exact_alias_is_diagnostic_only(monkeypatch) -> None:
    """没有专属分支的家族**不能**用 TempReason 别名包含进正式分数。"""
    from eval.harness import extra_pipeline as extra
    from eval.harness.official_capture_pipeline import _judge

    monkeypatch.setattr(extra, "judge_tempreason", lambda generated, gold: (True, "命中别名"))
    row = _judge(
        {"id": "1", "dataset": "some-new-set", "judge_kind": "exact", "gold_answers": ["x"]},
        "x",
        max_tokens=8,
    )
    assert row["label"] == "JUDGE_ERROR"
    assert row["is_correct"] is False
    assert row["diagnostics"]["alias_contains"] is True


@pytest.fixture
def personamem_archive():
    from eval.datasets.layout import upstream_aml_dir
    from eval.datasets.registry import benchmark_dir
    from eval.harness.official_capture_pipeline import _archived

    name = "pipeline_v2_personamem.py"
    if not (upstream_aml_dir(benchmark_dir(), name) / name).is_file():
        pytest.skip("PersonaMem 上游 pipeline 未准备（见 docs/benchmark-data.md）")
    return _archived("personamem-v2")


def test_personamem_capture_uses_official_generative_prompt(personamem_archive) -> None:
    """采集里的 PersonaMem 是开放题，必须复用官方 generative 消息构造。"""
    from eval.harness import extra_pipeline as extra
    from eval.harness.official_capture_pipeline import _answer_prompt

    prompt = _answer_prompt(
        {
            "dataset": "personamem-v2",
            "question": "What should I choose?",
            "retrieved_context": "CAPTURED-MEMORY-MARKER",
        }
    )

    assert "CAPTURED-MEMORY-MARKER" in prompt
    assert "What should I choose?" in prompt
    assert "Please recall my related preferences from our conversation history" in prompt
    assert "RAW-HISTORY" not in prompt
    assert "Cannot determine from the memories" not in prompt
    assert prompt != extra.ANSWER_PROMPT.format(
        memories="CAPTURED-MEMORY-MARKER", question="What should I choose?"
    )


def test_personamem_narrow_judge_uses_official_prompt_and_preference(
    monkeypatch, personamem_archive
) -> None:
    """PersonaMem 走**官方 narrow 口径**：`preference` + 官方 prompt → 0–1 分。

    ⚠ 三件事一起钉住（少任何一条，口径都会**静默**漂回自写裁判）：
    ① prompt 逐字来自归档那份（`JUDGE_PROMPT_NARROW_*`）；
    ② `preference` 从 `gold_answer` 取，且 `do not` 开头走 NEGATIVE 模板；
    ③ 0–1 分按我们定的阈值二值化，原始分进 `partial`。
    """
    from eval.harness import extra_pipeline as extra
    from eval.harness import official_capture_pipeline as capture

    module = personamem_archive
    seen: list[str] = []

    def fake_chat(_base, _key, _model, prompt, **_kwargs):
        seen.append(prompt)
        return "Reasoning … final score \\boxed{0.8}"

    monkeypatch.setattr(extra, "chat", fake_chat)
    item = {
        "id": "110",
        "dataset": "personamem-v2",
        "judge_kind": "narrow",
        "question": "What warm-up should I do?",
        "gold_answer": {"preference": "Practices yoga daily"},
        "gold_answers": ["reference answer"],
    }
    row = capture._judge(item, "Do yoga first.", max_tokens=64)
    assert row["is_correct"] is True
    assert row["partial"] == 0.8
    assert seen[0].startswith(module.JUDGE_PROMPT_NARROW_POSITIVE.split("{")[0])
    assert "Practices yoga daily" in seen[0]

    seen.clear()
    negative = capture._judge(
        {**item, "gold_answer": {"preference": "do not mention my surgery"}}, "ok", max_tokens=64
    )
    assert seen[0].startswith(module.JUDGE_PROMPT_NARROW_NEGATIVE.split("{")[0])
    assert negative["partial"] == 0.8


def test_personamem_narrow_judge_fails_loudly_without_preference(monkeypatch) -> None:
    """缺 `preference` **不许**静默退回"拿参考回答当标准"——那会悄悄换掉官方口径。"""
    from eval.harness import extra_pipeline as extra
    from eval.harness.official_capture_pipeline import _judge

    def boom(*_args, **_kwargs):
        raise AssertionError("没有 preference 就不该调裁判")

    monkeypatch.setattr(extra, "chat", boom)
    row = _judge(
        {
            "id": "110",
            "dataset": "personamem-v2",
            "judge_kind": "narrow",
            "question": "q",
            "gold_answer": None,
            "gold_answers": ["reference answer"],
        },
        "answer",
        max_tokens=64,
    )
    assert row["label"] == "JUDGE_ERROR"
    assert "preference" in row["judge_response"]


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

    ⚠ 官方 `build_answer_prompt` 读取 `system_prompt` 与
    `retrieval.selected[*].text`。若直接传入缺少这两个键的项，会渲染出 `(no memories)`，
    使模型盲答且不报错。
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


def test_gold_recovery_reads_corpus_rows_with_or_without_a_text_prefix(tmp_path):
    """`_corpus_index` 必须按 `Corpus:` 的位置切，不能要求正文以它开头。

    ⚠ 两个都会**静默**把改写/原始事实漏掉，进而把 MQuAKE 的 `gold_variant` 判反：
    ① 官方 add 的正文带 `[dataset][session][Dx:y][text] ` 前缀；
    ② 普通事实行是 `Corpus: {json}`（同一行），只取最后一行会得到非 JSON 结果。
    """
    from tools.recover_official_gold import _corpus_index

    adds = [
        {
            "ts": "2025-01-01T00:00:00+00:00",
            "user_id": USER_A,
            "messages": [
                {"content": 'Corpus: {"subject_id": "Q1", "relation_id": "P1", "object_id": "Q2"}'},
                {
                    "content": "[ds][session_1][D1:1][text] Corpus: "
                    '{"subject_id": "Q3", "relation_id": "P2", "object_id": "Q4"}'
                },
                {
                    "content": "[ds][session_2][D2:1][text] Corpus: UPDATE: replace the prior "
                    'value\n{"subject_id": "Q1", "relation_id": "P1", "object_id": "Q9"}'
                },
                {"content": "not a corpus row"},
                {"content": "Corpus: not-json"},
            ],
        }
    ]
    path = tmp_path / "adds.jsonl"
    _write(path, adds)

    index = _corpus_index(path, {USER_A})

    assert [triple for _ts, triple in index[USER_A]["fact"]] == [
        ("Q1", "P1", "Q2"),
        ("Q3", "P2", "Q4"),
    ]
    assert [triple for _ts, triple in index[USER_A]["update"]] == [("Q1", "P1", "Q9")]


def test_support_mirror_matches_every_real_kit_combination():
    """预检的 `supports()` 必须与 `_judge()` 的分派表**逐组合**一致。

    ⚠ 不钉住它，预检就会说"8,783 题都能跑"，而实际某一族全落 `JUDGE_ERROR`——
    那种错误只会在跑完 Add/Search 之后才出现，白花一整轮的钱。
    """
    from eval.datasets.registry import capture_dir
    from eval.harness.official_capture_pipeline import supports

    kit_path = Path(capture_dir()) / "official-eval-kit.jsonl"
    if not kit_path.exists():
        pytest.skip("采集套件不在（见 tools/build_official_kit.py）")
    combos = set()
    for line in kit_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["judge_kind"] != "none":
            combos.add((row["dataset"], row["judge_kind"]))
    assert combos
    assert {(d, k) for d, k in combos if not supports(d, k)} == set()
    # 反例：没有金标的行、以及 generic 分支不支持的 judge_kind 必须判 false。
    assert not supports("unknown", "none")
    assert not supports("hybridqa", "rubric")
    assert not supports("musique", "letters")


def test_kit_rejects_a_malformed_extra_row(capture: Path) -> None:
    """补金标行缺键/类型错时，**构建期**响亮失败——不能让它静默变成 `None`。"""
    from tools.build_official_kit import build

    extra_path = capture / "official-gold-extra.jsonl"
    rows = [json.loads(line) for line in extra_path.open(encoding="utf-8")]

    broken = dict(rows[0])
    broken.pop("gold_native")
    _write(extra_path, [broken])
    with pytest.raises(SystemExit, match="gold_native"):
        build(capture)

    broken = dict(rows[0])
    broken["gold_answers"] = "1789"  # 键名写成单数时的典型形状
    _write(extra_path, [broken])
    with pytest.raises(SystemExit, match="gold_answers"):
        build(capture)
