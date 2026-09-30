"""`eval/datasets/` —— schema 落差预处理层（§12.2 / §12.3 第 9 条 / D16）。

**大部分用例是合成 fixture**：归档不入库（`benchmark_data/` 被 gitignore），
所以逻辑用例不能让"归档在不在"决定跑不跑。真数据的事实另开一组，归档缺席时 skip。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eval.datasets import (
    benchmark_dir,
    data_fingerprint,
    load_locomo,
    load_longmemeval,
)
from eval.datasets.locomo import CONVERSATIONS_JSONL, QUESTIONS_JSONL, REFINED_JSON
from eval.datasets.longmemeval import LME_JSON
from eval.datasets.preprocess import (
    normalize_content,
    parse_lme_time,
    parse_locomo_time,
    to_epoch_ms,
)

ARCHIVE = Path("benchmark_data")
needs_archive = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="归档不在（见 docs/benchmark-data.md）"
)


# ── 合成 fixture ───────────────────────────────────────────────────────────
def _locomo_questions() -> list[dict]:
    return [
        {
            "qa_id": "conv-1#q0000",
            "sample_id": "conv-1",
            "qa_index": 0,
            "question": "Where did Sam go?",
            # 字符串 category——**这是 questions.jsonl 的真实类型**（locomo_refined.json 是整数）
            "category": "2",
            "answer": ["Kyoto"],
            "evidence": ["D1:2"],
        },
        {
            "qa_id": "conv-1#q0001",
            "sample_id": "conv-1",
            "qa_index": 1,
            "question": "What did Sam buy?",
            "category": "4",
            "answer": ["a bike", "a helmet"],
            "evidence": ["D1:3"],
        },
    ]


def _conversations_row() -> dict:
    """开发路径的源：每条 message **自带 `role`**。"""
    return {
        "sample_id": "conv-1",
        "speaker_a": "Sam",
        "speaker_b": "Rae",
        "sessions": [
            {
                "session_index": 1,
                "date_time": "1:56 pm on 8 May, 2023",
                "messages": [
                    {"session_index": 1, "message_index": 1, "role": "user", "text": "Hi Rae!"},
                    {
                        "session_index": 1,
                        "message_index": 2,
                        "role": "assistant",
                        "text": "Hi Sam!",
                    },
                ],
            },
            {
                "session_index": 2,
                "date_time": "2:30 am on 9 May, 2023",
                "messages": [
                    {
                        "session_index": 2,
                        "message_index": 1,
                        "role": "assistant",
                        "text": "back again",
                    },
                ],
            },
        ],
    }


def _refined_row() -> dict:
    """归档路径的源：**没有 `role`**，只有 `speaker`；且 text **带首尾空白**（真实存在 209 条）。"""
    return {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Sam",
            "speaker_b": "Rae",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Sam", "dia_id": "D1:1", "text": "  Hi Rae!  "},
                {"speaker": "Rae", "dia_id": "D1:2", "text": "Hi Sam!\n"},
            ],
            "session_2_date_time": "2:30 am on 9 May, 2023",
            "session_2": [{"speaker": "Rae", "dia_id": "D2:1", "text": "back again"}],
        },
    }


def _write_locomo(tmp: Path, *, layout: str) -> Path:
    bench = tmp / layout
    bench.mkdir(parents=True, exist_ok=True)
    (bench / QUESTIONS_JSONL).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _locomo_questions()), encoding="utf-8"
    )
    if layout == "dev":
        (bench / CONVERSATIONS_JSONL).write_text(json.dumps(_conversations_row()), encoding="utf-8")
    else:
        (bench / REFINED_JSON).write_text(json.dumps([_refined_row()]), encoding="utf-8")
    return bench


# ── 时间解析 ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1:56 pm on 8 May, 2023", datetime(2023, 5, 8, 13, 56, tzinfo=UTC)),
        ("12:10 am on 11 August, 2023", datetime(2023, 8, 11, 0, 10, tzinfo=UTC)),
        ("12:09 am on 13 September, 2023", datetime(2023, 9, 13, 0, 9, tzinfo=UTC)),
        ("11:54 am on 2 May, 2022", datetime(2022, 5, 2, 11, 54, tzinfo=UTC)),
        ("8:30 pm on 1 January, 2023", datetime(2023, 1, 1, 20, 30, tzinfo=UTC)),
    ],
)
def test_parse_locomo_time(text, expected):
    """**`12:xx am` 必须落到 0 点、`12:xx pm` 到 12 点**——极值正是 `%12` 最容易写错的地方。"""
    assert parse_locomo_time(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2023/05/20 (Sat) 02:21", datetime(2023, 5, 20, 2, 21, tzinfo=UTC)),
        ("2023/05/30 (Tue) 23:40", datetime(2023, 5, 30, 23, 40, tzinfo=UTC)),
    ],
)
def test_parse_lme_time(text, expected):
    assert parse_lme_time(text) == expected


def test_lme_time_does_not_validate_weekday():
    """格式里的 `(Sat)` **故意不校验**——它与日期不总是一致，校验只会把能跑的变成跑不了的。"""
    assert parse_lme_time("2023/05/20 (Mon) 02:21") == datetime(2023, 5, 20, 2, 21, tzinfo=UTC)


@pytest.mark.parametrize("bad", ["", "yesterday", "2023-05-20 02:21", "1:56 pm 8 May 2023"])
def test_time_parsers_raise_instead_of_returning_none(bad):
    """**解析失败必须抛。** 静默退化成 `None` 会让 `event_time` 全 NULL 而没有任何报错。"""
    with pytest.raises(ValueError):
        parse_locomo_time(bad)
    with pytest.raises(ValueError):
        parse_lme_time(bad)


def test_to_epoch_ms_is_milliseconds():
    """§2.1 的 `timestamp` 是**毫秒**——发成秒会让时间退回到 1970 年附近。"""
    moment = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    assert to_epoch_ms(moment) == 1_683_554_160_000


# ── content 归一化 ─────────────────────────────────────────────────────────
def test_normalize_content_strips_but_keeps_inner_whitespace():
    """只去首尾——**内部空白是正文的一部分**，动它就是在改数据。"""
    assert normalize_content("  hello  world\n", where="t") == "hello  world"


def test_normalize_content_rejects_empty():
    with pytest.raises(ValueError, match="content 为空"):
        normalize_content("   \n\t ", where="conv-1#s1:3")


# ── LoCoMo：两条来源路径必须产出同一结果 ───────────────────────────────────
def test_locomo_two_source_layouts_are_identical(tmp_path):
    """**本文件最重要的一条**（D16 的推论）。

    归档没有 `conversations.jsonl`，只有 `locomo_refined.json`。而后者每条 text **带首尾空白**
    （真实数据里 209 条）。断言两条路径产出**逐字相同的 `Message` 流**——
    于是"归档单独就够用，不必依赖 `eval/datasets/` 的 clone"这句话是可执行的，不是声明。
    """
    dev = load_locomo(_write_locomo(tmp_path, layout="dev"))
    archive = load_locomo(_write_locomo(tmp_path, layout="archive"))

    assert dev == archive
    assert [m.content for s in dev[0].sessions for m in s.messages] == [
        "Hi Rae!",
        "Hi Sam!",
        "back again",
    ]


def test_locomo_derives_role_from_speaker(tmp_path):
    """`speaker_a` → `user`、`speaker_b` → `assistant`。

    缺 `role` 时 `pairing/` **不会报错**，它会把整个 session 归成一个对——
    所以这一步错了是静默的，必须靠断言钉住。
    """
    sample = load_locomo(_write_locomo(tmp_path, layout="archive"))[0]
    assert [m.role for s in sample.sessions for m in s.messages] == [
        "user",
        "assistant",
        "assistant",
    ]


def test_locomo_session_ids_are_unique_per_user_and_ordered(tmp_path):
    sample = load_locomo(_write_locomo(tmp_path, layout="archive"))[0]
    assert [s.session_id for s in sample.sessions] == ["conv-1#s1", "conv-1#s2"]


def test_locomo_unknown_speaker_raises(tmp_path):
    """说话人既不是 a 也不是 b 时**必须抛**——猜一个 role 会让配对判据静默走错分支。"""
    bench = _write_locomo(tmp_path, layout="archive")
    payload = json.loads((bench / REFINED_JSON).read_text(encoding="utf-8"))
    payload[0]["conversation"]["session_1"][0]["speaker"] = "Ghost"
    (bench / REFINED_JSON).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="未知说话人"):
        load_locomo(bench)


def test_locomo_category_is_normalized_to_str(tmp_path):
    """`questions.jsonl` 里是 `"4"`、`locomo_refined.json` 里是 `4`。

    不归一化会**静默筛出 0 条**（§12.3）。
    """
    sample = load_locomo(_write_locomo(tmp_path, layout="dev"))[0]
    assert [q.category for q in sample.questions] == ["2", "4"]


def test_locomo_gold_keeps_original_list(tmp_path):
    """`gold` **不预先拼接**——归档的 `gold_answer()` 走 `memory_text()`，列表由它自己拼。"""
    sample = load_locomo(_write_locomo(tmp_path, layout="dev"))[0]
    assert sample.questions[1].gold == ["a bike", "a helmet"]


def test_locomo_missing_conversation_file_raises(tmp_path):
    bench = tmp_path / "empty"
    bench.mkdir()
    (bench / QUESTIONS_JSONL).write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="对话全文无处可取"):
        load_locomo(bench)


def test_locomo_question_with_empty_gold_raises(tmp_path):
    bench = _write_locomo(tmp_path, layout="dev")
    rows = _locomo_questions()
    rows[0]["answer"] = []
    (bench / QUESTIONS_JSONL).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="gold 为空"):
        load_locomo(bench)


# ── LongMemEval ────────────────────────────────────────────────────────────
def _write_lme(tmp: Path, *, n_sessions: int = 2, n_questions: int = 1) -> Path:
    bench = tmp / "lme"
    bench.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n_questions):
        entries.append(
            {
                "question_id": f"q{i}" + ("_abs" if i == 1 else ""),
                "question_type": "multi-session",
                "question": f"Question {i}?",
                "question_date": "2023/05/30 (Tue) 23:40",
                "answer": f"Answer {i}",
                "answer_session_ids": [f"s{i}_0"],
                "haystack_dates": [f"2023/05/2{d} (Sat) 02:2{d}" for d in range(n_sessions)],
                "haystack_session_ids": [f"s{i}_{d}" for d in range(n_sessions)],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": f"u{i}{d}", "has_answer": True},
                        {"role": "assistant", "content": f"a{i}{d}"},
                    ]
                    for d in range(n_sessions)
                ],
            }
        )
    (bench / LME_JSON).write_text(json.dumps(entries), encoding="utf-8")
    return bench


def test_lme_derives_user_id_per_question(tmp_path):
    """**一题一个 `user_id`**——LME 每题自带 haystack（与 LoCoMo 的 N 题 : 1 对话相反）。"""
    samples = load_longmemeval(_write_lme(tmp_path, n_questions=2))
    assert [s.user_id for s in samples] == ["lme-q0", "lme-q1_abs"]
    assert all(len(s.questions) == 1 for s in samples)


def test_lme_synthesizes_per_message_timestamp(tmp_path):
    """时间在 session 级 ⇒ 同 session 内**所有消息同一个时间戳**（数据集的真实属性，非近似）。"""
    sample = load_longmemeval(_write_lme(tmp_path, n_sessions=2))[0]
    stamps = [[m.timestamp_ms for m in s.messages] for s in sample.sessions]
    assert all(len(set(s)) == 1 for s in stamps)
    assert stamps[0][0] != stamps[1][0]
    assert stamps[0][0] == to_epoch_ms(parse_lme_time("2023/05/20 (Sat) 02:20"))


def test_lme_parallel_arrays_must_match(tmp_path):
    """三条平行数组长度不等 ⇒ 取错了文件，**必须抛**而不是截断到最短。"""
    bench = _write_lme(tmp_path)
    entries = json.loads((bench / LME_JSON).read_text(encoding="utf-8"))
    entries[0]["haystack_dates"] = entries[0]["haystack_dates"][:1]
    (bench / LME_JSON).write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="平行数组长度不等"):
        load_longmemeval(bench)


def test_lme_skips_blank_turns_and_says_so(tmp_path, capsys):
    """**V14**：`lme_s_cleaned.json` 里真有一条空正文的消息（`sharegpt_ADHo6Ob_0:9`）。

    照旧抛 ⇒ **全量 500 题根本跑不起来**（Step 5 的大跑批直接崩）；
    静默放行 ⇒ 空正文进 `join`、相邻两项粘连。⇒ **跳过 + 告警**。
    """
    bench = _write_lme(tmp_path, n_sessions=1)
    entries = json.loads((bench / LME_JSON).read_text(encoding="utf-8"))
    turns = entries[0]["haystack_sessions"][0]
    turns.append({"role": "user", "content": ""})  # 空串
    turns.append({"role": "assistant", "content": "   "})  # 纯空白——口径是 `strip()` 之后
    (bench / LME_JSON).write_text(json.dumps(entries), encoding="utf-8")

    messages = load_longmemeval(bench)[0].sessions[0].messages
    assert [m.content for m in messages] == ["u00", "a00"]  # 两条空正文都被剔掉
    assert capsys.readouterr().err.count("content 为空") == 2


def test_lme_abstention_flag(tmp_path):
    """拒答题是**横切标记**（`question_id` 以 `_abs` 结尾），不是第 7 类（§12.2）。"""
    samples = load_longmemeval(_write_lme(tmp_path, n_questions=2))
    assert [q.is_abstention for s in samples for q in s.questions] == [False, True]
    assert samples[1].questions[0].category == "multi-session"


def test_lme_limit_and_evidence(tmp_path):
    samples = load_longmemeval(_write_lme(tmp_path, n_questions=3), limit=2)
    assert len(samples) == 2
    assert samples[0].questions[0].evidence == ("s0_0",)


# ── 路径口径（D16）──
def test_benchmark_dir_reads_env_only():
    """**唯一读取点**：默认归档，`.env` 可改指（本地开发指到 `eval/datasets/...`）。"""
    assert benchmark_dir({}) == Path("benchmark_data")
    assert benchmark_dir({"TIANXIMEM_BENCHMARK_DIR": "/tmp/x"}) == Path("/tmp/x")


def test_loaders_take_path_from_argument_not_cwd(tmp_path, monkeypatch):
    """D16：**加载器只认传进来的 `Path`**，不自己去猜数据在哪。

    行为断言而不是扫源码：换个 cwd、并把 `TIANXIMEM_BENCHMARK_DIR` 指到别处，
    传绝对路径的结果必须**一个字节都不变**。硬编码路径出错时不会报错，
    只会让"换台机器就指错数据"变成静默行为——所以只能这么测。
    """
    bench = _write_locomo(tmp_path / "real", layout="dev")
    empty = tmp_path / "empty"
    empty.mkdir()

    monkeypatch.setenv("TIANXIMEM_BENCHMARK_DIR", str(empty))
    monkeypatch.chdir(empty)

    assert len(load_locomo(bench)) == 1
    # 反过来：不传路径时**没有**任何隐式回退——环境变量不参与
    with pytest.raises(FileNotFoundError):
        load_locomo(Path("."))


def test_data_fingerprint_records_batching_and_files(tmp_path):
    """§13 要求数据指纹含 **数据集 + 版本 + 切批口径**——少一样，两次 run 就不可比。"""
    bench = _write_lme(tmp_path, n_questions=2)
    fp = data_fingerprint(bench, "longmemeval-s", n_samples=2, n_questions=2)
    assert fp["dataset"] == "longmemeval-s"
    assert [f["name"] for f in fp["files"]] == [LME_JSON]
    assert len(fp["files"][0]["sha256"]) == 16
    # 切批口径必须写明"本地复现"——S2 没清掉之前，线上真实口径未知（§17.1）
    assert "local repro only" in fp["batching"]


def test_data_fingerprint_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="未知数据集"):
        data_fingerprint(tmp_path, "beam", n_samples=0, n_questions=0)


# ── 真数据（归档缺席时 skip）──
@needs_archive
def test_archive_locomo_counts():
    samples = load_locomo(ARCHIVE)
    assert len(samples) == 10
    assert sum(len(s.questions) for s in samples) == 1382
    assert sum(s.message_count for s in samples) == 5882
    # adversarial（category 5，answer=null）**不在 questions.jsonl 里**——只有 1..4
    assert {q.category for s in samples for q in s.questions} == {"1", "2", "3", "4"}


@needs_archive
def test_archive_locomo_matches_conversations_jsonl():
    """归档与 clone **只在首尾空白上不同**（209 条）——两个来源必须逐字等价。

    只在 clone 也在场时跑：归档不收录 `conversations.jsonl`（D16 / 出处链）。
    """
    clone = Path("eval/datasets/LoCoMo-Refined/data/public")
    if not (clone / CONVERSATIONS_JSONL).exists():
        pytest.skip("开发 clone 不在场——归档单独跑即可")
    assert load_locomo(ARCHIVE) == load_locomo(clone)


@needs_archive
def test_archive_lme_head_shape():
    """真文件的形状（`limit=` 避免解析 277 MB——全量解析留给正式 run，不留给单测）。"""
    samples = load_longmemeval(ARCHIVE, limit=3)
    assert len(samples) == 3
    assert all(s.questions[0].category for s in samples)
    assert all(s.message_count > 0 for s in samples)


# ── LongMemEval 的**分层抽样**（2026-09-26）──────────────────────────────
def _write_grouped_lme(tmp: Path) -> Path:
    """按类型**分块**排的 fixture——**这正是真文件的样子**（实测 7 个连续块）。"""
    bench = tmp / "lme-grouped"
    bench.mkdir(parents=True, exist_ok=True)
    entries = []
    groups = (("single-session-user", 8), ("multi-session", 8), ("temporal-reasoning", 4))
    for kind, count in groups:
        for i in range(count):
            entries.append(
                {
                    "question_id": f"{kind}-{i}",
                    "question_type": kind,
                    "question": "Q?",
                    "question_date": "2023/05/30 (Tue) 23:40",
                    "answer": "A",
                    "answer_session_ids": ["s0"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:20"],
                    "haystack_session_ids": ["s0"],
                    "haystack_sessions": [[{"role": "user", "content": "u", "has_answer": True}]],
                }
            )
    (bench / LME_JSON).write_text(json.dumps(entries), encoding="utf-8")
    return bench


def test_lme_limit_alone_takes_one_single_type(tmp_path):
    """**这条是"为什么需要 spread"的证据**：文件按类型分块 ⇒ `[:limit]` 只取到一类。"""
    from eval.datasets.sampling import stratified_sample

    bench = _write_grouped_lme(tmp_path)
    entries = json.loads((bench / LME_JSON).read_text(encoding="utf-8"))
    naive = entries[:6]
    assert {e["question_type"] for e in naive} == {"single-session-user"}  # ← 单一类型

    spread = stratified_sample(entries, 6, key=lambda e: str(e["question_type"]))
    assert {e["question_type"] for e in spread} == {
        "single-session-user",
        "multi-session",
        "temporal-reasoning",
    }


def test_lme_spread_is_deterministic_and_ordered(tmp_path):
    """分层抽样**不随机**：同一份文件两次给同一批题，且保持文件原顺序。"""
    from eval.datasets.sampling import stratified_sample

    entries = json.loads((_write_grouped_lme(tmp_path) / LME_JSON).read_text(encoding="utf-8"))
    key = lambda e: str(e["question_type"])  # noqa: E731 —— 测试里就地用
    first, second = stratified_sample(entries, 5, key=key), stratified_sample(entries, 5, key=key)

    assert [e["question_id"] for e in first] == [e["question_id"] for e in second]
    order = [e["question_id"] for e in entries]
    picked = [e["question_id"] for e in first]
    assert picked == sorted(picked, key=order.index)  # 原顺序


def test_lme_spread_noop_when_limit_covers_everything(tmp_path):
    """`limit` 不小于总数 ⇒ 原样返回（不抽样、不重排）。"""
    from eval.datasets.sampling import stratified_sample

    entries = json.loads((_write_grouped_lme(tmp_path) / LME_JSON).read_text(encoding="utf-8"))
    assert [
        e["question_id"]
        for e in stratified_sample(entries, len(entries), key=lambda e: str(e["question_type"]))
    ] == [e["question_id"] for e in entries]


# ── CL-Bench 的加载层（2026-09-27）──────────────────────────────────────
def _write_clbench(tmp: Path) -> Path:
    """按真实形状造一份：`system` + 装着文档与任务的 `user` + rubrics + metadata。"""
    bench = tmp / "clb"
    bench.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, n in (("Rule System Application", 4), ("Domain Knowledge Reasoning", 4)):
        for i in range(n):
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": f"SYSTEM {kind}"},
                        {
                            "role": "user",
                            "content": (
                                f"DOC {kind} {i}\n" + "filler " * 50 + f"\n\nTask {i}: what is X?"
                            ),
                        },
                    ],
                    "rubrics": [f"r{i}-1", f"r{i}-2"],
                    "metadata": {
                        "task_id": f"{kind}-{i}",
                        "context_id": f"ctx-{kind}",
                        "context_category": kind,
                        "sub_category": "sub",
                    },
                }
            )
    (bench / "clbench.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    return bench


def test_clbench_one_sample_per_task(tmp_path):
    """一个 `task_id` 一个 Sample，`user_id` 带 `clb-` 前缀——**每题自带自己的记忆**。"""
    from eval.datasets import load_clbench

    samples = load_clbench(_write_clbench(tmp_path))
    assert [s.user_id for s in samples][:2] == [
        "clb-Rule System Application-0",
        "clb-Rule System Application-1",
    ]
    assert all(len(s.questions) == 1 for s in samples)
    # rubrics 原样进 gold（`clb_pipeline` 的 `official_rubrics()` 认它）
    assert samples[0].questions[0].gold == ["r0-1", "r0-2"]
    assert samples[0].questions[0].category == "Rule System Application / sub"


def test_clbench_task_text_is_the_tail_window(tmp_path):
    """**任务文本 = 末条 user 的尾部窗口**——CL-Bench 没有独立的 question 字段。

    实测规律：任务指令写在末条 user 的**最末尾**（`…what do Sightings Cards do?`）；
    而末段中位只有 434 字符、最长的却有 13.4 万（整本手册塞成一段）⇒ 取尾部固定窗口。
    """
    from eval.datasets import load_clbench
    from eval.datasets.clbench import TASK_TAIL_CHARS

    sample = load_clbench(_write_clbench(tmp_path))[0]
    question = sample.questions[0].question
    assert question.endswith("Task 0: what is X?"), question[-60:]
    assert len(question) <= TASK_TAIL_CHARS


def test_clbench_messages_keep_roles_and_have_no_timestamp(tmp_path):
    """role 原样保留（`system` 那条要在），**时间戳为空**——这份数据本来就没有。"""
    from eval.datasets import load_clbench

    sample = load_clbench(_write_clbench(tmp_path))[0]
    roles = [m.role for m in sample.sessions[0].messages]
    assert roles == ["system", "user"]
    assert all(m.timestamp_ms is None for m in sample.sessions[0].messages)


def test_clbench_rejects_a_record_without_rubrics(tmp_path):
    """**没有 rubrics 就判不了分** ⇒ 必须抛，不能静默跑出一堆 0。"""
    import pytest
    from eval.datasets import load_clbench

    bench = _write_clbench(tmp_path)
    (bench / "clbench.jsonl").write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "x"}],
                "rubrics": [],
                "metadata": {"task_id": "t"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rubrics"):
        load_clbench(bench)


def test_clbench_reader_survives_u2028(tmp_path):
    """**这份文件里真的有 `U+2028`** ⇒ 必须按 `"\\n"` 切，不能用 `splitlines()`。

    实测：`clbench.jsonl` 用 `splitlines()` 读会在 char 22463 处 `JSONDecodeError`。
    """
    from eval.datasets import load_clbench

    bench = _write_clbench(tmp_path)
    record = {
        "messages": [{"role": "user", "content": "doc end\n\nTask?"}],
        "rubrics": ["r"],
        "metadata": {"task_id": "u2028", "context_category": "C", "sub_category": "s"},
    }
    with (bench / "clbench.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    samples = load_clbench(bench)
    assert any(s.user_id == "clb-u2028" for s in samples)
