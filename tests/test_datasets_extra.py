"""`official-extra` 三份数据集的加载层与我们自写的 pipeline——**合成 fixture，不依赖归档**。

覆盖三件容易静默出错的东西：

1. **一条记忆 = 一个 session**（`mquake` / `corporatebench`）。写成"多条塞一个 session"不会报错，
   但在**旧规则**下**会被合并成一个块**（连续同 role 合并）⇒ 检索退化、分数虚高或虚低。
    ⚠ **D29 之后不会了**——配不上的消息各自独立成块，这条夹具的形状约定因此变成冗余（仍成立）。
2. **判分是纯函数**（MQuAKE 别名表 / CorporateBench 的标量·布尔·列表）——这里逐档钉死，
   包括 MQuAKE 那种长度 2 的别名（`de`）不能满屏假阳性。
3. **MemTrapBench 的均分与阈值分离**：四个原始分进 `judge_response`，换阈值不用重跑裁判。

> ⚠ 归档不在场时**这些用例照样跑**（全部用 tmp_path 造数据）——与 `test_benchmark_archive.py`
> 的 skip 纪律相反，因为这里验的是"我们自己的转换"，不是"归档里有什么"。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from eval.datasets.corporatebench import load_corporatebench
from eval.datasets.memtrapbench import load_memtrapbench
from eval.datasets.mquake import load_mquake
from eval.harness.extra_pipeline import (
    MEMTRAP_PASS_MEAN,
    judge_corporatebench,
    judge_memtrapbench_scores,
    judge_mquake,
)
from eval.harness.judge import EXTRA_DATASETS, _build_extra_items, pipeline_for

# ── 合成 fixture ────────────────────────────────────────────────────────────


def _case(case_id: int, question: str) -> dict:
    return {
        "case_id": case_id,
        "questions": [question, f"{question} (rephrased)"],
        "answer": "Old Answer",
        "answer_alias": ["old"],
        "new_answer": "New Answer",
        "new_answer_alias": ["new alias"],
        "orig_triples_labeled": [["Subject", "relation", "Old Answer"]],
        "requested_rewrite": [
            {"subject": "Subject", "relation_id": "P1", "target_new_str": "New Answer"},
        ],
    }


def _write_mquake(root: Path) -> None:
    data = root / "mquake-remastered" / "data"
    data.mkdir(parents=True)
    rows = [_case(1, "Q one?"), _case(2, "Q two?"), _case(3, "Q three?")]
    table = pa.Table.from_pylist(rows)
    for stem in ("CF3k", "T"):
        # 只写两份文件：加载器要求四份都在，缺一份要响亮失败（另有用例）
        pq.write_table(table, data / f"{stem}-00000-of-00001.parquet")
    for stem in ("CF6334", "CF9k"):
        pq.write_table(table, data / f"{stem}-00000-of-00001.parquet")


_MTB_ITEM = {
    "id": "0001",
    "context_history": [
        {"turn": 1, "role": "user", "content": "hello"},
        {"turn": 1, "role": "assistant", "content": "hi"},
    ],
    "final_trigger": "the trap question",
    "gold_standard": "do not do the bad thing",
    "test_type": "red",
}


def _write_memtrapbench(root: Path) -> None:
    """**四个场景全造**——少一个会让加载器响亮失败（见 `test_memtrapbench_missing_scenario_...`）。

    上游四个场景各有 350 / 200 / 350 / 150 题，**没有"合法的空场景"**。
    """
    base = root / "memtrapbench" / "memtrapbench"
    for scenario in ("cognitive_bias", "safety", "task_boundary", "trauma"):
        (base / scenario).mkdir(parents=True)
        (base / scenario / f"{scenario}_1.json").write_text(
            json.dumps([{**_MTB_ITEM, "id": f"{scenario}-0001"}]), encoding="utf-8"
        )
    # 种子文件必须被跳过（它们没有 context_history）
    (base / "trauma" / "hurt_seed.json").write_text(json.dumps([{"id": "s"}]), encoding="utf-8")


def _write_corporatebench(root: Path) -> None:
    base = root / "corporatebench"
    (base / "data" / "kb").mkdir(parents=True)
    email = "Message-ID: x\nFrom: A <a@zenithlabs.com>\nDate: 2024-01-23\nSubject: s\n\nBody text."
    with zipfile.ZipFile(base / "data" / "kb" / "zenith.kb", "w") as archive:
        archive.writestr("documents/email_a.txt", email)
        archive.writestr("documents/email_b.txt", "no date header here")
        archive.writestr("graph.nq", "<a> <b> <c> .")
    for qa, questions in (
        ("kb_qa", [{"id": 0, "question": "How many?", "answer": 1, "answer_type": "int"}]),
        (
            "topic_qa",
            [{"id": 0, "question": "Was it discussed?", "answer": True, "answer_type": "bool"}],
        ),
        (
            "integrated_qa",
            [{"id": 0, "question": "List them", "answer": ["a", "b"], "answer_type": "List[str]"}],
        ),
    ):
        folder = base / "data" / qa
        folder.mkdir(parents=True)
        (folder / "zenith_questions.json").write_text(
            json.dumps({"metadata": {}, "questions": questions}), encoding="utf-8"
        )


# ── 加载层：一条记忆 = 一个 session ─────────────────────────────────────────


def test_mquake_is_one_message_per_memory_in_one_session(tmp_path: Path) -> None:
    """**一条事实 = 一条消息**，而整个样本共用一个 `Session`。

    ⚠ 关键在**消息的条数与顺序**，"几条 session"只是运输方式：D29 之后没有非 user 跟随的
    连续 user **每条各自独立成块**，所以挤在一次 Add 里粒度不受影响（Add 次数反而少一个数量级）。
    """
    _write_mquake(tmp_path)
    samples = load_mquake(tmp_path, cases_per_user=2)
    # 四份文件 × ceil(3 个 case / 每 sample 2 个) = 4 × 2
    assert len(samples) == 8
    assert [s.user_id for s in samples[:2]] == ["mqk-CF3k-00001", "mqk-CF3k-00003"]
    first = samples[0]
    # 2 个 case × (1 条事实 + 1 条 UPDATE) = 4 条消息，全部在**一个** session 里
    assert len(first.sessions) == 1
    assert len(first.sessions[0].messages) == 4
    assert all(m.role == "user" for m in first.sessions[0].messages)
    texts = [m.content for m in first.sessions[0].messages]
    assert "Subject — relation — Old Answer" in texts[0]
    assert any("replaces the earlier value" in t for t in texts)


def test_mquake_gold_is_the_post_update_answer(tmp_path: Path) -> None:
    """记忆里同时有原始事实与 UPDATE ⇒ **金标一律取 `new_answer`**（别名表一起进 gold）。"""
    _write_mquake(tmp_path)
    question = load_mquake(tmp_path, cases_per_user=1)[0].questions[0]
    assert question.gold == ["New Answer", "new alias"]
    assert "Old Answer" not in question.gold


def test_mquake_missing_parquet_raises(tmp_path: Path) -> None:
    """少一份 parquet **响亮失败**——四份的题库不同，静默少读一份等于换了数据集。"""
    (tmp_path / "mquake-remastered" / "data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_mquake(tmp_path)


def test_memtrapbench_one_sample_per_item_and_seed_skipped(tmp_path: Path) -> None:
    """一条陷阱题一个 `Sample`；种子文件（没有 `context_history`）必须跳过。"""
    _write_memtrapbench(tmp_path)
    samples = load_memtrapbench(tmp_path)
    assert len(samples) == 4, "四个场景各一条（种子那条不算）"
    sample = samples[0]
    assert sample.message_count == 2, "context_history 原样投喂，不改形状"
    assert set(sample.questions[0].gold) == {
        "gold_standard",
        "test_type",
        "expected_failure_output",
    }


def test_memtrapbench_missing_scenario_is_a_loud_failure(tmp_path: Path) -> None:
    """⚠ **少一个场景目录必须响亮失败**（2026-10-04）——`glob` 对不存在的目录返回空迭代器、
    **一声不吭** ⇒ 少一个场景 = 静默少 150–350 题，而分数看着完全正常、**只是不可比**。

    这正是本仓最防的那一类（"分数看起来完全正常、却不可比"），而它在**跑一轮**时
    与"题目就这么多"**长得一模一样**——只有 `tests/test_experiments.py` 里那条
    钉住 `n_questions` 的用例会红，而那要人去跑测试才看得见。
    """
    _write_memtrapbench(tmp_path)
    (tmp_path / "memtrapbench" / "memtrapbench" / "trauma").rename(tmp_path / "搬走")
    with pytest.raises(FileNotFoundError, match="trauma"):
        load_memtrapbench(tmp_path)


def test_memtrapbench_seeds_are_skipped_by_content_not_filename(tmp_path: Path) -> None:
    """**判据是内容不是文件名**：种子条目没有 `context_history` ⇒ 跳过。

    ⚠ 这条是用真数据踩出来的：`trauma/hurt.json` **只装种子、名字里却没有 `_seed`**，
    按文件名过滤会在加载阶段直接炸（2026-09-30 冒烟）。
    """
    _write_memtrapbench(tmp_path)
    base = tmp_path / "memtrapbench" / "memtrapbench" / "trauma"
    seed = {"id": "seed_011", "domain": "Actuarial Science"}
    question = {**_MTB_ITEM, "id": "0002", "final_trigger": "the trap"}
    # 覆盖掉夹具给 trauma 的那条：这个文件名里**没有** `_seed`，但内容全是种子
    (base / "hurt.json").write_text(json.dumps([seed, question, seed]), encoding="utf-8")
    samples = load_memtrapbench(tmp_path)
    found = [s for s in samples if s.questions[0].question == "the trap"]
    assert len(found) == 1, "种子要跳过，题要留下"
    assert len(samples) == 5, "另外三个场景各一条 + trauma 这一条"


def test_memtrapbench_broken_question_still_raises(tmp_path: Path) -> None:
    """**有 `context_history`（是题）却缺判分要点** ⇒ 照样响亮失败——别把真坏的当成种子放过。"""
    _write_memtrapbench(tmp_path)
    base = tmp_path / "memtrapbench" / "memtrapbench" / "safety"
    (base / "poison_200.json").write_text(
        json.dumps([{"id": "1", "context_history": [{"turn": 1, "role": "user", "content": "x"}]}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gold_standard"):
        load_memtrapbench(tmp_path)


def test_corporatebench_documents_are_one_message_each(tmp_path: Path) -> None:
    """**一份文档 = 一条消息**（整份语料一个 session）；
    **时间戳从邮件头的 `Date:` 取**，取不到留空。
    """
    _write_corporatebench(tmp_path)
    samples = load_corporatebench(tmp_path, limit=1)
    assert len(samples) == 1, "limit 选的是 QA 子集"
    sample = samples[0]
    assert len(sample.sessions) == 1
    messages = sample.sessions[0].messages
    assert len(messages) == 2, "graph.nq 不喂"
    assert messages[0].timestamp_ms is not None  # email_a.txt 带 Date:
    assert messages[1].timestamp_ms is None  # email_b.txt 没有


def test_corporatebench_three_qa_types_share_the_corpus(tmp_path: Path) -> None:
    """三个 QA 子集**共用同一份语料**——语料不随 `limit` 裁（裁了就没法检索）。"""
    _write_corporatebench(tmp_path)
    samples = load_corporatebench(tmp_path)
    assert [s.user_id for s in samples] == ["corp-kb_qa", "corp-topic_qa", "corp-integrated_qa"]
    # ⚠ **语料同一份 ⇒ 逐字相同**，不是"session 数相同"那么弱：三家共用同一批消息。
    assert len({tuple(m.content for m in s.sessions[0].messages) for s in samples}) == 1
    assert samples[2].questions[0].gold == {"answer": ["a", "b"], "answer_type": "List[str]"}


# ── 判分：纯函数 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("Croatia", True),
        ("The answer is Croatia.", True),
        ("cro atia", False),
        ("Kolinda Grabar-Kitarović", True),
    ],
)
def test_mquake_alias_matching(generated: str, expected: bool) -> None:
    gold = ["Kolinda Grabar-Kitarović", "Croatia"]
    ok, _ = judge_mquake(generated, gold)
    assert ok is expected


def test_mquake_short_alias_does_not_match_inside_words() -> None:
    """长度 2 的别名（`de` = German）**必须按词边界**——否则 `democracy` 也算命中。"""
    ok, why = judge_mquake("It is a democracy.", ["de"])
    assert ok is False, why
    ok, _ = judge_mquake("Antwort: de", ["de"])
    assert ok is True


def test_mquake_empty_gold_is_not_a_pass() -> None:
    """金标别名表为空 ⇒ **判负**（不能让"没金标"变成"白送分"）。"""
    ok, why = judge_mquake("anything", [])
    assert ok is False and "为空" in why


@pytest.mark.parametrize(
    ("answer", "generated", "expected"),
    [
        ({"answer": 1, "answer_type": "int"}, "There is 1 department.", True),
        ({"answer": 1, "answer_type": "int"}, "There are 2 departments.", False),
        ({"answer": True, "answer_type": "bool"}, "Yes, it was discussed.", True),
        ({"answer": True, "answer_type": "bool"}, "No.", False),
        ({"answer": False, "answer_type": "bool"}, "No, it was not.", True),
        ({"answer": "Zenith Labs", "answer_type": "str"}, "It is zenith labs.", True),
        ({"answer": "Zenith Labs", "answer_type": "str"}, "It is not related.", False),
    ],
)
def test_corporatebench_scalar_judging(answer: dict, generated: str, expected: bool) -> None:
    ok, _ = judge_corporatebench(generated, answer)
    assert ok is expected


def test_corporatebench_list_needs_all_elements() -> None:
    """二值接口要求完全匹配；连续 set-F1 由专门的 CorporateBench 测试检查。"""
    gold = {"answer": ["alpha", "beta"], "answer_type": "List[str]"}
    assert judge_corporatebench("alpha and beta", gold)[0] is True
    assert judge_corporatebench("only alpha", gold)[0] is False


@pytest.mark.parametrize(
    ("raw", "expect_mean"),
    [
        (
            json.dumps(
                {
                    "dimension_1_factual_correctness": {"justification": "x", "score": 5},
                    "dimension_2_instruction_compliance": {"justification": "x", "score": 5},
                    "dimension_3_relevance_and_information_purity": {
                        "justification": "x",
                        "score": 5,
                    },
                    "dimension_4_delivery_efficiency_and_optimality": {
                        "justification": "x",
                        "score": 5,
                    },
                }
            ),
            5.0,
        ),
        ("not json at all", None),
        (json.dumps({"dimension_1_factual_correctness": {"score": 4}}), 4.0),
    ],
)
def test_memtrap_panel_parsing(raw: str, expect_mean: float | None) -> None:
    """四个原始分进 `judge_response`——**阈值 `MEMTRAP_PASS_MEAN` 与解析分离**，可事后重算。"""
    mean, _ = judge_memtrapbench_scores(raw)
    if expect_mean is None:
        assert mean is None
    else:
        assert mean == pytest.approx(expect_mean)
    assert MEMTRAP_PASS_MEAN == 4.0, "阈值改了要同步测试与文档"


# ── 接线：pipeline 路由与输入项 ─────────────────────────────────────────────


def test_pipeline_for_routes_extra_datasets_to_our_own_script(tmp_path: Path) -> None:
    """官方的在归档里、**我们自己写的在本目录**——两条路径都要能取到。"""
    for dataset in EXTRA_DATASETS:
        path = pipeline_for(tmp_path, dataset)
        assert path.exists(), f"{dataset}: {path}"
        assert path.name == "extra_pipeline.py"
    assert pipeline_for(tmp_path, "locomo-refined").name == "pipeline_locomo-refined.py"
    with pytest.raises(ValueError):
        pipeline_for(tmp_path, "nope")


def test_extra_items_carry_dataset_and_gold(tmp_path: Path) -> None:
    """输入项必须带 `dataset`（判分按它分派）与原始 gold；记忆走 `retrieved_context`。"""
    _write_mquake(tmp_path)
    sample = load_mquake(tmp_path, cases_per_user=1)[0]
    items = _build_extra_items(
        sample, {sample.questions[0].qid: []}, date_mode="none", annotate_mark="paren"
    )
    assert items[0]["dataset"] == "mquake-remastered"
    assert items[0]["gold_answer"] == ["New Answer", "new alias"]
    assert items[0]["retrieved_context"] == "", "没有命中就是空串，不是缺键"


def test_extra_pipeline_cli_takes_max_tokens_after_the_subcommand() -> None:
    """CLI 形状必须与归档 pipeline 一致：**`--max-tokens` 在子命令之后**。

    `run_judge` 按官方那几份的写法传参（`… answer --input … --output … --max-tokens 256`）。
    一开始我把这个选项挂在**顶层** parser 上 ⇒ 子解析器不认识它 ⇒ 整轮死在
    `unrecognized arguments`（2026-09-30 冒烟抓到的）。这条用例就是那个形状的看门狗。
    """
    from eval.harness.extra_pipeline import build_parser

    parser = build_parser()
    for argv in (
        ["answer", "--input", "i.jsonl", "--output", "o.jsonl", "--max-tokens", "256"],
        [
            "evaluate",
            "--input",
            "i.jsonl",
            "--answers",
            "a.jsonl",
            "--output",
            "l.jsonl",
            "--max-tokens",
            "256",
        ],
    ):
        args = parser.parse_args(argv)
        assert args.max_tokens == 256
    assert parser.parse_args(["answer", "--input", "i", "--output", "o"]).max_tokens == 512


# ── MedMemoryBench（zh）：检查点前缀 + 上游那套判分 ──────────────────────────
def _write_medmemorybench(root: Path) -> None:
    import pyarrow.parquet as pq

    def turn(persona: int, session: int, turn_no: int, role: str, text: str) -> dict:
        return {
            "persona_id": persona,
            "session_id": session,
            "turn": turn_no,
            "role": role,
            "content": text,
            "event_info": json.dumps({"event": "e", "type": "health", "date": "2024-01-05"}),
        }

    turns = [
        turn(1, 10, 1, "user", "十次问诊时说的话"),
        turn(1, 10, 2, "assistant", "医生回话"),
        turn(1, 20, 1, "user", "二十次问诊时说的话"),
        turn(1, 20, 2, "assistant", "医生回话"),
    ]
    base = root / "medmemorybench" / "data" / "zh"
    base.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(turns), base / "dialogues.parquet")

    def query(qid: str, session: int, qtype: str, answers: list[dict]) -> dict:
        return {
            "persona_id": 1,
            "query_id": qid,
            "session_id": session,
            "query_type": qtype,
            "question": f"{qtype} 的问题",
            "answers": json.dumps(answers, ensure_ascii=False),
            "source_key_points": "[]",
            "metadata": json.dumps(
                {"hop_count": 2, "reasoning_chain": [], "required_memory_nodes": []}
            ),
        }

    queries = [
        query(
            "q-s10-eem",
            10,
            "entity_exact_match",
            [{"content": "糖尿病", "is_correct": True, "explanation": "x"}],
        ),
        query(
            "q-s20-mc",
            20,
            "multiple_choice",
            [
                {"content": "A. 头孢", "is_correct": False},
                {"content": "B. 阿莫西林", "is_correct": True},
                {"content": "C. 克拉霉素", "is_correct": True},
            ],
        ),
    ]
    pq.write_table(pa.Table.from_pylist(queries), base / "queries.parquet")


def test_medmemorybench_checkpoint_never_reads_the_future(tmp_path: Path) -> None:
    """**这条是 loader 的全部要点**：检查点 10 的样本里**不许出现**第 20 次问诊的内容。

    读未来会让 `state_update` 那类题（"目前用药状态"）虚高——用全量语料问早期时点，
    答案自然就变了。所以记忆必须是**到该检查点为止的前缀**。
    """
    from eval.datasets import load_medmemorybench

    _write_medmemorybench(tmp_path)
    first, second = load_medmemorybench(tmp_path)
    assert first.user_id == "mmb-01-s010" and second.user_id == "mmb-01-s020"
    texts = [m.content for s in first.sessions for m in s.messages]
    assert "十次问诊时说的话" in texts
    assert "二十次问诊时说的话" not in texts, "检查点 10 的样本里混进了未来的对话"
    # 检查点 20 的样本则是**前缀**（含前 10 次）
    later = [m.content for s in second.sessions for m in s.messages]
    assert "十次问诊时说的话" in later and "二十次问诊时说的话" in later
    # 题只带自己检查点的那道（题号带 `mmb-{persona}-` 前缀，见下面那条唯一性用例）
    assert [q.qid for q in first.questions] == ["mmb-1-q-s10-eem"]
    assert [q.qid for q in second.questions] == ["mmb-1-q-s20-mc"]


def test_medmemorybench_spread_is_reachable(tmp_path: Path) -> None:
    """**`spread=True` 这条路径此前没有用例，于是它一直是崩的**（2026-10-01 才发现）。

    崩法是 `stratified_sample(..., key=lambda item: item[0][0])`——`ordered` 的元素
    已经是 `(persona, checkpoint)` 元组，再取一次下标就是**对 int 取下标** ⇒ `TypeError`。
    ⚠ **它不是"参数写错了"，是"这条路径从没被跑过"**：加载器只在 `spread` 且 `limit`
    小于总数时才会走到那个 lambda。**每加一个数据集，都要有一条用例真的把它的
    `spread` 路径走一遍**——静态看着对的代码不会自己证明自己。
    """
    from eval.datasets import load_medmemorybench

    _write_medmemorybench(tmp_path)
    plain = load_medmemorybench(tmp_path, limit=1)
    spread = load_medmemorybench(tmp_path, limit=1, spread=True)
    # 夹具只有 1 个 persona ⇒ 分不出层，两条路径拿到同一批；**要点是后者不抛**。
    assert [s.user_id for s in spread] == [s.user_id for s in plain]
    assert spread


def test_medmemorybench_answers_ride_along_in_gold(tmp_path: Path) -> None:
    """`answers`（含 `is_correct`）与 `metadata` 必须原样进 gold——判分要靠它们分流。"""
    from eval.datasets import load_medmemorybench

    _write_medmemorybench(tmp_path)
    question = load_medmemorybench(tmp_path)[1].questions[0]
    assert question.gold["query_type"] == "multiple_choice"
    assert [a["is_correct"] for a in question.gold["answers"]] == [False, True, True]


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("糖尿病", True),
        ("患者既往病史里提到过糖尿病。", True),
        ("糖 尿 病", True),  # 归一化会去掉空白
        ("高血压", False),
    ],
)
def test_mmb_string_contain(generated: str, expected: bool) -> None:
    """`entity_exact_match`：**每一条**正确项都要出现（多正确项时缺一即错）。"""
    from eval.harness.extra_pipeline import judge_mmb_string_contain

    gold = {"answers": [{"content": "糖尿病", "is_correct": True}]}
    assert judge_mmb_string_contain(generated, gold)[0] is expected


def test_mmb_string_contain_needs_every_correct_answer() -> None:
    from eval.harness.extra_pipeline import judge_mmb_string_contain

    gold = {
        "answers": [
            {"content": "糖尿病", "is_correct": True},
            {"content": "高血压", "is_correct": True},
        ]
    }
    assert judge_mmb_string_contain("有糖尿病和高血压", gold)[0] is True
    assert judge_mmb_string_contain("只有糖尿病", gold)[0] is False


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("答案为 B。", False),  # 正确项是 B 与 C —— 少选
        ("B、C 都正确", True),
        ("我觉得 B 和 C。", True),
        ("A、B、C 都对", False),  # 多选了一个错的
        ("没有相关信息", False),
    ],
)
def test_mmb_option_match_is_set_equality(generated: str, expected: bool) -> None:
    """`multiple_choice`：**集合完全相等**（实测 398 道里 296 道有 2–3 个正确项）。"""
    from eval.harness.extra_pipeline import judge_mmb_option_match

    gold = {
        "answers": [
            {"content": "A. 头孢", "is_correct": False},
            {"content": "B. 阿莫西林", "is_correct": True},
            {"content": "C. 克拉霉素", "is_correct": True},
        ]
    }
    assert judge_mmb_option_match(generated, gold)[0] is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"is_correct": true, "reason": "对"}', True),
        ('前面有废话 {"is_correct": false, "reason": "错"} 后面也有', False),
        ("完全不是 JSON", None),
        ('{"reason": "没有 is_correct 字段"}', None),
    ],
)
def test_mmb_llm_verdict_parsing(raw: str, expected: bool | None) -> None:
    """裁判 JSON 解析：取**第一个**带 `is_correct` 的对象；解不出就返回 `None`（→ JUDGE_ERROR）。"""
    from eval.harness.extra_pipeline import judge_mmb_llm_verdict

    assert judge_mmb_llm_verdict(raw)[0] is expected


def test_mmb_llm_verdict_handles_nested_json() -> None:
    r"""⚠ **多跳那一类的裁决是嵌套 JSON**（`node_validations` 里还有一层）。

    非贪婪正则 `\{.*?\}` 只会命中里层的节点对象、永远够不到外层的 `is_correct`
    ——2026-09-30 实测把一条**判对了的**回答记成了 `JUDGE_ERROR`。这条用例就是那个形状。
    """
    from eval.harness.extra_pipeline import judge_mmb_llm_verdict

    raw = (
        '{\n  "node_validations": [\n    {"node_id": 1, "mentioned": true, "note": "x"}\n  ],\n'
        '  "ncr_score": 0.0,\n  "is_correct": false,\n  "reason": "没有引用患者具体数据"\n}'
    )
    ok, why = judge_mmb_llm_verdict(raw)
    assert ok is False and "具体数据" in why


def test_mmb_prompt_comes_from_the_upstream_archive() -> None:
    """裁判 prompt **从归档里那份上游代码读**（不复制）——归档不在场时 skip。

    复制一份 prompt 就等于开了第二个家：上游改了、我们这份就悄悄旧了。
    """
    from eval.datasets.registry import benchmark_dir
    from eval.harness.extra_pipeline import build_mmb_judge_prompt

    if not (benchmark_dir() / "medmemorybench-code" / "utils" / "prompts_judge.py").exists():
        pytest.skip("上游代码不在归档里（见 docs/benchmark-data.md）")
    prompt = build_mmb_judge_prompt(
        "state_update",
        "他目前对恩格列净的用药状态是什么？",
        "已停用。",
        {
            "answers": [
                {"content": "已停用", "is_correct": True, "explanation": "第 40 次问诊改的"}
            ],
            "metadata": {},
        },
    )
    assert "他目前对恩格列净的用药状态是什么？" in prompt
    assert "已停用。" in prompt and "第 40 次问诊改的" in prompt
    assert "【评判标准】" in prompt, "拿到的应当是上游那份中文模板，不是我们编的"


# ── 分派表：**每个数据集的输入项字段都要对**（这条能拦住"分支没插进去"） ────────────
def test_every_dataset_builds_items_with_its_own_contract() -> None:
    """`build_input_items` 必须**按数据集**给出各自那份契约的字段。

    ⚠ 为什么值得一条用例：2026-09-30 加 ScriptMem / PersonaMem 的分支时，
    编辑被 `ruff format` 重排过的上下文"吃掉"了——分支**没插进去**，而症状是
    跑完 Add/Search 之后在判分那一步才炸（`unsupported qa_type: ''`）。
    一条"每个数据集都验字段"的用例能让它在**集合阶段**就红。
    """
    from eval.datasets.preprocess import Message, Question, Sample, Session
    from eval.harness.judge import _ADAPTER_PIPELINES, EXTRA_DATASETS, build_input_items

    # 只造 `Question`/`Sample`，不读真数据（归档缺席也要能跑）
    def sample_of(dataset: str) -> Sample:
        return Sample(
            user_id="u-x",
            dataset=dataset,
            sessions=(Session(session_id="s", messages=(Message(role="user", content="记忆"),)),),
            questions=(
                Question(
                    qid="q-1",
                    question="问？",
                    gold={"any": "gold"},
                    category="single_choice",
                ),
            ),
        )

    expected = {
        "clbench": {"idx", "question", "system_prompt", "rubrics", "retrieval"},
        "beam": {"id", "question", "context", "rubric", "question_type"},
        "personamem-v2": {
            "id",
            "question",
            "persona_id",
            "correct_answer",
            "incorrect_answers",
            "chat_history",
        },
        # 我们自写那一类（含 medmemorybench / tempreason）
        **{
            name: {"id", "dataset", "question", "gold_answer", "retrieved_context"}
            for name in EXTRA_DATASETS
        },
        # 通用路径（LoCoMo / LongMemEval）
        "locomo-refined": {
            "id",
            "question",
            "gold_answer",
            "speaker_1_memories",
            "speaker_2_memories",
        },
    }
    assert set(expected) >= set(_ADAPTER_PIPELINES) | EXTRA_DATASETS
    for dataset, fields in expected.items():
        item = build_input_items(sample_of(dataset), {})[0]
        missing = fields - set(item)
        assert not missing, f"{dataset}: 输入项缺 {sorted(missing)}（契约变了？）"


def test_question_ids_are_unique_across_every_sample() -> None:
    """**qid 必须唯一**——harness 每一环（`answers`/`labels`/`items` 字典）都按键取值。

    ⚠ 这条是从真数据里踩出来的：Doc-PP 的 `id` 既不全局唯一、也不 (doc_id, id) 唯一
    （同一份文档里两个 `id=0`）⇒ 作业项里出现重复 qid ⇒ **判分静默丢题**
    （`{id: item}` 把两条并成一条）。loader 现在改用行号，这条用例钉住"以后也不许重"。
    """
    from eval.datasets import (
        load_corporatebench,
        load_medmemorybench,
        load_memtrapbench,
        load_mquake,
        load_tempreason,
    )

    bench = Path("benchmark_data")
    if not (bench / "tempreason" / "test_l2.json").exists():
        pytest.skip("归档不在（见 docs/benchmark-data.md）")
    total = 0
    for loader in (
        load_mquake,
        load_memtrapbench,
        load_corporatebench,
        load_medmemorybench,
        load_tempreason,
    ):
        qids = [q.qid for sample in loader(bench, limit=2) for q in sample.questions]
        assert len(qids) == len(set(qids)), f"{loader.__name__}：qid 有重复"
        total += len(qids)
    assert total > 0


# ── ScriptMem / TempReason / PersonaMem / Doc-PP：加载层的合成 fixture ────────────
def _write_tempreason(root: Path) -> None:
    base = root / "tempreason"
    base.mkdir(parents=True)
    page = "Title line one\nSecond line.\nValparaiso University hired him in 1948."
    rows = [
        {
            "question": "Where did he work in 1948?",
            "date": "1948",
            "text_answers": {"text": ["Valparaiso University"]},
            "context": page,
        },
        {
            "question": "Where did he work in 1949?",
            "date": "1949",
            "text_answers": {"text": ["Valparaiso University"]},
            "context": page,
        },
    ]
    (base / "test_l2.json").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    (base / "test_l3.json").write_text("", encoding="utf-8")


def test_tempreason_splits_page_into_sentences(tmp_path: Path) -> None:
    """页 → 句（**换行也算切点**），同一页的题归到一个 `Sample`。"""
    from eval.datasets import load_tempreason

    _write_tempreason(tmp_path)
    samples = load_tempreason(tmp_path)
    assert len(samples) == 1, "两题共用同一页 ⇒ 一个 Sample"
    sample = samples[0]
    # 两个换行 + 句号 ⇒ 至少 3 句；**全部在同一个 session 里，一句一条消息**
    assert len(sample.sessions) == 1
    assert len(sample.sessions[0].messages) >= 3, "换行与句号都要切"
    assert len(sample.questions) == 2
    assert sample.questions[0].gold == ["Valparaiso University"]


def test_tempreason_l1_without_context_is_a_loud_failure(tmp_path: Path) -> None:
    """L1 那两档 `context` 是空的 ⇒ **响亮失败**，别让它跑成"没有记忆也能答"。"""
    from eval.datasets import load_tempreason

    base = tmp_path / "tempreason"
    base.mkdir(parents=True)
    (base / "test_l2.json").write_text(
        json.dumps({"question": "q", "date": "d", "text_answers": {"text": ["x"]}, "context": ""})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="没有 context"):
        load_tempreason(tmp_path)


def _write_personamem(root: Path) -> None:
    import csv as _csv

    base = root / "personamem-v2"
    (base / "data" / "chat_history_32k").mkdir(parents=True)
    (base / "benchmark" / "text").mkdir(parents=True)
    history = {
        "metadata": {"persona_id": 7},
        "chat_history": [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "我喜欢瑜伽"},
        ],
    }
    (base / "data" / "chat_history_32k" / "chat_persona7.json").write_text(
        json.dumps(history, ensure_ascii=False), encoding="utf-8"
    )
    with (base / "benchmark" / "text" / "benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=[
                "persona_id",
                "user_query",
                "correct_answer",
                "incorrect_answers",
                "chat_history_32k_link",
                "topic_query",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "persona_id": "7",
                "user_query": "{'role': 'user', 'content': '我早上该做什么？'}",
                "correct_answer": "继续瑜伽",
                "incorrect_answers": '["跑步", "游泳"]',
                "chat_history_32k_link": "data/chat_history_32k/chat_persona7.json",
                "topic_query": "Health",
            }
        )


def test_personamem_item_contract_matches_the_official_pipeline(tmp_path: Path) -> None:
    """`user_query` 是 repr 串、要剥出 content；gold 要带官方认的那两个键。"""
    from eval.datasets import load_personamem
    from eval.harness.judge import build_input_items

    _write_personamem(tmp_path)
    sample = load_personamem(tmp_path)[0]
    assert sample.user_id == "pm-7" and sample.message_count == 2
    assert sample.questions[0].question == "我早上该做什么？"

    item = build_input_items(sample, {})[0]
    assert item["correct_answer"] == "继续瑜伽"
    assert item["incorrect_answers"] == ["跑步", "游泳"]
    assert [m["content"] for m in item["chat_history"]] == ["persona", "我喜欢瑜伽"]


# ── TempReason：`fact_context` 必须进记忆（2026-10-03）─────────────────
def test_tempreason_ingests_the_fact_context():
    """⛔ **`fact_context` 是这份数据集的记忆本体**，不是"每题一份的泄题"。

    一度不收它（旧理由："几乎每题一份，不是共享的记忆"）。实测推翻：

    * 同一页的各题**共用同一份事实集**（60 个多题页全部成立）；
    * 正文页**往往没有时间区间**，而问题问的是**某一天的雇主是谁**
      ⇒ 事实句是唯一能推出答案的材料；
    * 那份事实集里**同时含 gold 与 `neg_answers`** —— 要挑对区间才答得出，**不是泄题**。

    不收的后果实测（332 题）：gold 只在 `fact_context` 里、不在正文页的 **239/332**，
    错题里 **259/310 是逐字拒答** ⇒ overall 0.066 量的是"我们没给那份事实"。
    修好后 gold 进记忆的比例 **91/332 → 323/332**。
    """
    from eval.datasets.tempreason import load_tempreason

    from tianximem.common.tokens import load_counter

    bench = Path("benchmark_data")
    if not (bench / "tempreason" / "test_l2.json").exists():
        pytest.skip("归档不在（见 docs/benchmark-data.md）")

    samples = load_tempreason(bench, limit=8, spread=True)
    hits = total = 0
    for sample in samples:
        body = "\n".join(m.content for s in sample.sessions for m in s.messages).lower()
        for question in sample.questions:
            total += 1
            if any(str(g).lower() in body for g in question.gold):
                hits += 1
    assert total > 0
    # 8 页的小样本上也应当**几乎全中**；留一点余量给表面形式不同的（如带重音的 gold）
    assert hits / total >= 0.9, f"gold 只在 {hits}/{total} 的记忆里——fact_context 没进语料？"
    load_counter()  # noqa: B018 —— 顺带证明这条路径不依赖任何外部服务


def test_facts_first_dedupes_and_keeps_order():
    """`_facts_first`：同一句在记忆里**只许出现一次**（同页几题共用一份事实集）。"""
    from eval.datasets.tempreason import _facts_first

    got = _facts_first(
        [
            "A works for X from Jan, 1949 to Jan, 1953.",
            "A works for X from Jan, 1949 to Jan, 1953.",  # ← 与上一条同句
            "A works for Y from Jan, 1953 to Jan, 1962.",
        ]
    )
    assert got == [
        "A works for X from Jan, 1949 to Jan, 1953.",
        "A works for Y from Jan, 1953 to Jan, 1962.",
    ]


# ── MedMemoryBench：题号必须**全局**唯一（2026-10-04）──────────────────────
def _write_medmemorybench_two_personas(root: Path) -> None:
    """两个 persona **用同一个 `query_id`**——parquet 里就是这么写的。"""
    base = root / "medmemorybench" / "data" / "zh"
    base.mkdir(parents=True)
    turns = [
        {
            "persona_id": persona,
            "session_id": 10,
            "turn": 1,
            "role": "user",
            "content": f"persona {persona} 的话",
            "event_info": json.dumps({"date": "2024-01-05"}),
        }
        for persona in (1, 2)
    ]
    pq.write_table(pa.Table.from_pylist(turns), base / "dialogues.parquet")
    queries = [
        {
            "persona_id": persona,
            "query_id": "session_10_eem_1",  # ★ 两个 persona 一模一样
            "session_id": 10,
            "query_type": "entity_exact_match",
            "question": "问题",
            "answers": json.dumps([{"content": "答案", "is_correct": True}]),
            "source_key_points": "[]",
            "metadata": "{}",
        }
        for persona in (1, 2)
    ]
    pq.write_table(pa.Table.from_pylist(queries), base / "queries.parquet")


def test_medmemorybench_question_ids_are_unique_across_personas(tmp_path: Path) -> None:
    """⚠ **parquet 里的 `query_id` 只在 persona 内唯一**（实测 1,939 题只有 **100** 个唯一值）。

    这份数据集是**唯一一个**把原始 id 原样当题号用的（其余各份都前缀了来源单元：
    `pm-{persona}-{index}` / `{stem}-{case_id}-{index}` …）⇒ 跨 persona 撞车。

    **不撞车才不报错的地方**：产物按 `sample.user_id` 分目录，批内恰好不撞，
    所以比分不受影响——于是这个缺陷**一声不吭**地活着；
    直到有下游按 id 跨批配对（诊断工具读整个 run 目录 ⇒ 172 行压成 20 个键，
    报出"答案只有 20/172 题、跑批中断？"——**那个信号是假的**）。
    """
    from eval.datasets import load_medmemorybench

    _write_medmemorybench_two_personas(tmp_path)
    samples = load_medmemorybench(tmp_path)
    assert [s.user_id for s in samples] == ["mmb-01-s010", "mmb-02-s010"]
    ids = [q.qid for s in samples for q in s.questions]
    assert len(ids) == len(set(ids)) == 2, f"题号跨 persona 撞车：{ids}"


# ── MedMemoryBench 判分：**多跳那一类要自己的预算**（2026-10-04）──────────
def test_mmb_mcd_judge_gets_its_own_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """`multi_hop_clinical_deduction` 的输出**比其余三类长一个量级**（要逐节点回 `note`）。

    2026-10-03 的 `base-mmb` 实测：**17 道 mcd 里 15 道**的裁判输出被砍在 618–778 字符处，
    JSON 没收尾 ⇒ 全记 `JUDGE_ERROR` ⇒ 看起来是"这一类能力为 0"，其实是**判分链路在丢题**。
    上游对它单独给 `max_tokens=2000`（其余三类 500）。

    按 2000 实测 9 道：输出 1,286–1,801 字符、**全部正常收尾并解析**。
    """
    from eval.harness import extra_pipeline as ep

    seen: list[int] = []

    def fake_call(judge, prompt, *, max_tokens):  # noqa: ANN001, ARG001
        seen.append(max_tokens)
        return '{"is_correct": true, "reason": "ok"}'

    monkeypatch.setattr(ep, "_call_judge", fake_call)
    judge = ("base", "key", "model")
    item = {"question": "?", "gold_answer": {"query_type": "multi_hop_clinical_deduction"}}
    ep._judge_medmemorybench("q1", item, "模型答案", judge=judge, max_tokens=1024)
    assert seen == [ep.MMB_MCD_MAX_TOKENS]
    # 其余三类**照旧**用 harness 给的那个数（别顺手把它们也抬上去）
    item["gold_answer"] = {"query_type": "state_update"}
    ep._judge_medmemorybench("q2", item, "模型答案", judge=judge, max_tokens=1024)
    assert seen[-1] == 1024


@pytest.mark.parametrize(
    "raw",
    [
        # 全角开引号 + 半角闭引号：那个裸 `"` 把 JSON 字符串提前截断 ⇒ 整份非法
        '{"is_correct": false, "reason": "未引用患者“终身禁用 NSAIDs"这一禁忌症"}',
        # 两头都用半角
        '{"is_correct": true, "reason": "患者"A"与"B"都要覆盖"}',
        # 带前后散文
        '判断如下：\n{"is_correct": true, "reason": "他"说"不行"}\n以上。',
    ],
)
def test_mmb_judge_recovers_from_unescaped_quotes(raw: str) -> None:
    """裁判写中文理由时**混用全角开引号与半角闭引号**，而那个半角 `"` 会截断字符串。

    上游的括号计数法在这里**同样失效**（它靠 `"` 翻转 `in_string`，撞上之后后面的 `}`
    被当成字符串内容）⇒ 这不是"照上游实现"能解决的事，得有一步**转义修复**兜底。
    ⚠ 它是**兜底**：严格解析成功时一步都不走。
    """
    from eval.harness.extra_pipeline import judge_mmb_llm_verdict

    ok, why = judge_mmb_llm_verdict(raw)
    assert ok is not None, f"没修回来：{why}"
    assert "未转义引号" in why, "修复过就得**看得见**，别静默"


def test_mmb_judge_does_not_guess_a_non_boolean() -> None:
    """⚠ `bool("false")` **为真**——裁判偶尔把布尔写成字符串，不归一化就会把**判负**记成**判对**。"""
    from eval.harness.extra_pipeline import judge_mmb_llm_verdict

    assert judge_mmb_llm_verdict('{"is_correct": true}')[0] is True
    assert judge_mmb_llm_verdict('{"is_correct": "true"}')[0] is True
    assert judge_mmb_llm_verdict('{"is_correct": "false"}')[0] is False
    # 读不出来的**交回 None（⇒ JUDGE_ERROR）**，不要瞎猜成 False 或 True
    assert judge_mmb_llm_verdict('{"is_correct": "nope"}')[0] is None
    assert judge_mmb_llm_verdict("裁判说这道题算过。")[0] is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # ★ 本用例的由来：**宾语是足球俱乐部，名字以缩写结尾** ⇒ 区间被拦腰切断
        (
            "Raúl Servín plays for Atlas F.C. from Jan, 1991 to Jan, 1992.",
            ["Raúl Servín plays for Atlas F.C. from Jan, 1991 to Jan, 1992."],
        ),
        (
            "Anton Schumacher plays for Oldham Athletic A.F.C. from Jan, 1991 to Jan, 1991.",
            ["Anton Schumacher plays for Oldham Athletic A.F.C. from Jan, 1991 to Jan, 1991."],
        ),
        # 其它形态的缩写（有元音的那一类走词表）
        (
            "David E. Finley, Jr. is the chair from Jan, 1950 to Jan, 1963.",
            ["David E. Finley, Jr. is the chair from Jan, 1950 to Jan, 1963."],
        ),
        (
            "He attended the St. Petersburg Academy from Jan, 1758 to Jan, 1761.",
            ["He attended the St. Petersburg Academy from Jan, 1758 to Jan, 1761."],
        ),
        (
            "Moxie Marlinspike works for Twitter, Inc. from Jan, 2011 to Jan, 2013.",
            ["Moxie Marlinspike works for Twitter, Inc. from Jan, 2011 to Jan, 2013."],
        ),
        # **真句界一个都不能少**（缩写的例外不能把它一起放过）
        (
            "He moved to the U.S. in 1948. He later left.",
            [
                "He moved to the U.S. in 1948.",
                "He later left.",
            ],
        ),
        ("He earned 8.5 points. Then quit.", ["He earned 8.5 points.", "Then quit."]),
        ("Title line one\nSecond line.", ["Title line one", "Second line."]),
    ],
)
def test_tempreason_sentence_split_survives_abbreviations(text: str, expected: list[str]) -> None:
    """⚠ **只在句末标点后切句，会把以缩写结尾的宾语拦腰切断。**

    ```text
    Raúl Servín plays for Atlas F.C. from Jan, 1991 to Jan, 1992.
    ```

    切成 `…plays for Atlas F.C.` 与 **`from Jan, 1991 to Jan, 1992.`**，而**后半句自成
    一条记忆块**。问 "May, 1991 他在哪支队"时，检索带回来的是**没有区间的**那半句
    ——答案本来躺在被切走的那一半里，模型只能拒答。

    实测（2026-10-04）：L2 的事实行被切碎 **7,061/44,168（16.0%）**、L3
    **5,255/38,432（13.7%）**；到题目层是 539/5,397（10%）与 360/4,426（8%）的题，
    gold 恰好落在被切碎的那一行上。补上缩写例外后降到 **0.14% / 0.10%**。

    ⚠ 剩下的那一点是德语俱乐部的序数（`1. FC Köln`）——**故意不收**：想用
    "数字 + 点 + 全大写短词"保住它，但实测页面正文里 `数字. 大写词` 有 58,834 处，
    其中 1,684 处的下一个词是全大写且 ≤4 字符，**而 `1909. A year later` 正在其中**
    ⇒ 会把真句界合并掉，代价远大于收益。
    """
    from eval.datasets.tempreason import _sentences

    assert _sentences(text) == expected


def test_tempreason_gets_its_own_answer_prompt() -> None:
    """⚠ **通用 `ANSWER_PROMPT` 的第 2 条会把它逼成一律拒答**。

    那条写的是「记忆里**没有**答案就逐字回 `Cannot determine from the memories.`」——
    而 TempReason 的答案**从来不在记忆里字面出现**：记忆给**区间**
    （`… plays for Atlas F.C. from Jan, 1991 to Jan, 1992.`），问题问**一个时点**
    （`… in May, 1991?`）。9B 的模型把 "contain" 读成字面匹配 ⇒ 拒答 93/332。

    ⚠ 与 memtrapbench 那条**正好相反**：那份是**故意**不给答案，这份是答案恒在、
    只是要推一步。⇒ 两份 prompt 取向相反，**都不能用通用的那份**。

    定版依据（76 条拒答题，同条件三候选）：B1 44 条新对 / B2 43 / **B3 52**。
    """
    from eval.harness.extra_pipeline import ANSWER_PROMPT, TEMPR_ANSWER_PROMPT

    assert "Cannot determine from the memories" in ANSWER_PROMPT
    assert "Cannot determine from the memories" not in TEMPR_ANSWER_PROMPT
    assert "always answerable" in TEMPR_ANSWER_PROMPT
