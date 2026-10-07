"""`eval/datasets/` —— schema 落差预处理层（§12.2 / §12.3 第 9 条 / D16）。

**大部分用例是合成 fixture**：归档不入库（`dataset/` 被 gitignore），
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
from eval.datasets.layout import archive_file
from eval.datasets.locomo import CONVERSATIONS_JSONL, QUESTIONS_JSONL, REFINED_JSON
from eval.datasets.longmemeval import LME_JSON
from eval.datasets.preprocess import (
    normalize_content,
    parse_lme_time,
    parse_locomo_time,
    to_epoch_ms,
)

ARCHIVE = benchmark_dir({})
needs_archive = pytest.mark.skipif(
    not archive_file(ARCHIVE, QUESTIONS_JSONL).is_file(),
    reason="归档不在（见 docs/benchmark-data.md）",
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
    assert benchmark_dir({}) == Path("dataset")
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
    # ⚠ 例子要真是**没登记**的名字，且**别用真实存在的数据集名**——这条已经踩过两次：
    #   先是 `beam`、后是 `personamem-v2`，两个后来都真的接上了，用例随即变红。
    #   ⇒ 用一个明确不存在的名字，别再让它随别的改动一起翻。
    with pytest.raises(ValueError, match="未知数据集"):
        data_fingerprint(tmp_path, "no-such-dataset", n_samples=0, n_questions=0)


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


# ── BEAM（`official-extra` 那份、但裁判用官方 pipeline）──────────────────────
def _beam_row(conversation_id: str = "1") -> dict:
    """一行 BEAM——**`time_anchor` 只挂在每个 session 的第一条上**（实测形状）。"""
    import pyarrow as pa

    def message(text: str, role: str, anchor: str | None, index: str) -> dict:
        return {
            "content": text,
            "id": index,
            "index": index,
            "question_type": "main_question",
            "role": role,
            "time_anchor": anchor,
        }

    probing = {
        "abstention": [
            {
                "question": "Did I mention the missing detail?",
                "rubric": ["Say there is no information about the missing detail."],
                "difficulty": "medium",
            }
        ],
        "temporal_reasoning": [
            {
                "question": "How many weeks do I have?",
                "rubric": ["State 4 weeks.", "Show the subtraction."],
                "difficulty": "easy",
            }
        ],
    }
    return {
        "conversation_id": conversation_id,
        "conversation_seed": {"category": "Coding", "id": 1},
        # `user_profile` 里那句 `Name:` 是**官方正文前缀**（`Christina Baker: …`）的来源 ⇒
        # 加载器要拿它填 `speaker_names[0]`，缺了会响亮失败。
        "user_profile": repr(
            {"user_info": "USER PROFILE:\n    • Name: Craig Baker\n    • Age: 49"}
        ),
        "chat": [
            [
                message("hello", "user", "March-15-2024", "1,1"),
                message("hi", "assistant", None, "1,2"),
            ],
            [message("second session, no anchor", "user", None, "2,1")],
        ],
        "probing_questions": repr(probing),
        "_pa": pa.binary(),  # 占位，避免 import 提示
    }


def _write_beam(root: Path, *, rows: list[dict] | None = None) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    payload = [{k: v for k, v in row.items() if k != "_pa"} for row in (rows or [_beam_row()])]
    path = root / "beam" / "data"
    path.mkdir(parents=True, exist_ok=True)
    target = path / "100K-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(payload), target)
    return target


def test_beam_time_anchor_spreads_within_a_session(tmp_path: Path) -> None:
    """`time_anchor` 只在 session 第一条上 ⇒ **要传播给同 session 的其余消息**。

    不传播的后果是静默的：`event_time` 全空、`created_at` 一律发空串。
    而**没有 anchor 的 session 保持 `None`**——不要拿上一个 session 的日期去填。
    """
    from eval.datasets import load_beam

    _write_beam(tmp_path)
    sample = load_beam(tmp_path)[0]
    first, second = sample.sessions
    assert [m.timestamp_ms for m in first.messages] == [1710460800000, 1710460800000]
    assert all(m.timestamp_ms is None for m in second.messages)


def test_beam_speaker_name_is_the_persona_name(tmp_path: Path) -> None:
    """`speaker_names[0]` 必须是 persona 名——**官方正文前缀用的就是它**
    （canary `Christina Baker: …`），而 `add_shape` 的 `@speaker` 规则直接读这个字段。"""
    from eval.datasets import load_beam

    _write_beam(tmp_path)
    assert load_beam(tmp_path)[0].speaker_names == ("Craig Baker", "Assistant")


def test_beam_without_a_persona_name_is_a_loud_failure(tmp_path: Path) -> None:
    """解不出名字**不许静默**：少一个说话人名会让检索看不见"谁在说"，
    而屏幕上看不出任何异常（`add_shape._label` 那边也会抛）。"""
    from eval.datasets import load_beam

    row = _beam_row()
    del row["user_profile"]
    _write_beam(tmp_path, rows=[row])
    with pytest.raises(ValueError, match="persona 名"):
        load_beam(tmp_path)


def test_beam_gold_is_the_rubric_and_abstention_is_flagged(tmp_path: Path) -> None:
    """金标取 `rubric`（**每组都有**；`answer` 只有 5 组有），拒答组单独标记。"""
    from eval.datasets import load_beam

    _write_beam(tmp_path)
    questions = load_beam(tmp_path)[0].questions
    assert [q.category for q in questions] == ["abstention", "temporal_reasoning"]
    assert questions[0].gold == ["Say there is no information about the missing detail."]
    assert [q.is_abstention for q in questions] == [True, False]
    assert [q.qid for q in questions] == ["1-abstention-0", "1-temporal_reasoning-0"]


def test_beam_missing_parquet_raises(tmp_path: Path) -> None:
    """缺 parquet **响亮失败**——别让它退化成一格空的记忆。"""
    from eval.datasets import load_beam

    with pytest.raises(FileNotFoundError):
        load_beam(tmp_path)


def test_beam_group_order_is_fixed_not_dict_order(tmp_path: Path) -> None:
    """题的顺序由 `GROUPS` 常量决定，**不跟着 dict 的插入序走**——题号一变，历史分数就不可比。"""
    from eval.datasets import load_beam
    from eval.datasets.beam import GROUPS

    row = _beam_row()
    # 故意把 dict 的键序倒过来
    row["probing_questions"] = repr(
        {
            group: __import__("ast").literal_eval(row["probing_questions"])[group]
            for group in reversed(GROUPS)
            if group in ("abstention", "temporal_reasoning")
        }
    )
    _write_beam(tmp_path, rows=[row])
    questions = load_beam(tmp_path)[0].questions
    assert [q.category for q in questions] == ["abstention", "temporal_reasoning"]


# ── 门禁：喂给嵌入的块不能超过提交口径的窗口 ─────────────────────────────
#
# ⚠ **这条守的是一个线上不会报错的东西**：`text-embedding-v4` 超限时**静默截断**
# （官方 FAQ：`Content exceeding this limit is truncated before embedding`）——
# 模型只会看到前 8,192 token，而库里那条记忆**看起来是完整的**。
# 本地的开发网关同窗口但会响亮 400，所以本地这个"红"是门禁，**不是故障**。
#
# 它由打包预算（`eval/harness/batching.py` 的 20 条 / 2,000 词）间接保证，
# 但"间接保证"是**测出来的余量**，不是断言——改打包参数就能悄悄捅破。


def _max_block_tokens(
    bench_dir: Path, dataset: str, limit: int, *, max_words: int | None = None
) -> int:
    """跑一遍"加载 → 渲染 → 打包 → 按 D24 组块"，返回最大的那一块的 token 数。

    **用的是真的 `pairing` 与真的 `render`**（不是近似）：这两个决定了模型最终读到什么，
    自己写一份复制品只会让门禁守着一个不存在的形状。
    """
    from eval.experiments.run import _load
    from eval.harness.add_shape import shape_batch
    from eval.harness.batching import batches

    from tianximem.common.render import render_pair
    from tianximem.common.tokens import load_counter
    from tianximem.pairing.pairing import Message as PairingMessage
    from tianximem.pairing.pairing import compose_memory_blocks

    counter = load_counter()
    worst = 0
    for sample in _load(dataset, bench_dir, limit, spread=True):
        for session in sample.sessions:
            payloads = shape_batch(
                session.messages, dataset=sample.dataset, speaker_names=sample.speaker_names
            )
            budget = {} if max_words is None else {"max_words": max_words}
            for batch in batches(payloads, **budget):
                blocks = compose_memory_blocks(
                    [PairingMessage(role=p["role"], content=p["content"]) for p in batch]
                )
                for block in blocks:
                    # `render_pair` 收的是**鸭子类型的"一个对"**（要 `question` / `answer` /
                    # `event_time` 三个属性），而 `MemoryBlock` 正好有 ⇒ 直接传它。
                    worst = max(worst, counter.count(render_pair(block)))
    return worst


@needs_archive
@pytest.mark.parametrize(
    ("dataset", "limit"),
    [
        ("locomo-refined", 2),
        ("longmemeval-s", 2),
        # ⚠ clbench 必须扫**够大**：出事的块在**第 7 与第 52 个样本**上，
        # `limit=3` 只扫前 3 个 ⇒ 门禁是真的、却从来没盖到会失败的地方
        # （2026-10-03 实测：`limit=300` 下有两个块 38,270 / 9,824 token）。
        # ⇒ 这一格是**真门禁**：它盖的是冻结口径的 **299 个样本**，别调小。
        ("clbench", 300),
        ("beam", 3),
        ("personamem-v2", 2),
        ("mquake-remastered", 2),
        ("memtrapbench", 4),
        ("corporatebench", 2),
        ("medmemorybench", 12),
        ("tempreason", 3),
    ],
)
def test_no_block_exceeds_the_submit_embedding_window(dataset: str, limit: int) -> None:
    """**跑真数据**：任何一条记忆块都不得超过 `text-embedding-v4` 的 8,192 token。

    超了线上**不会报错**——它把尾巴砍掉再嵌入，而返回给 AML 的 `content` 仍是完整的
    ⇒ 一条"检索得到、但向量只代表前一半"的记忆，**没有任何信号**。

    ⚠ **采样要够大**：这条一开始给 clbench 只扫 `limit=3`，而出事的块在**第 7 与第 52 个
    样本**上——门禁是真的、却从来没盖到会失败的地方（2026-10-03 才发现）。
    "检查能失败"与"检查盖到了会失败的地方"是两件事。
    """
    from tianximem.common.tokens import EMBED_MAX_TOKENS

    worst = _max_block_tokens(ARCHIVE, dataset, limit)
    assert worst <= EMBED_MAX_TOKENS, (
        f"{dataset}: 最大块 {worst} token > 提交窗口 {EMBED_MAX_TOKENS}"
        "——线上会**静默截断**，本地的开发网关会 400"
    )


def test_the_embedding_window_gate_can_actually_fail(tmp_path: Path) -> None:
    """**门禁必须能失败**（[`CLAUDE.md`](./CLAUDE.md) 的那条纪律），**并且失败点要指对**。

    ⚠ 这条改过**三次**，每次都是被实现教会的：

    1. 第一版塞了一条**全 user** 的超长语料，没红——它被**打包预算**摊成多批救回来了；
    2. 改配对规则之后（同 role 不再合并），**全 user 的批根本不会合并** ⇒ 同一个夹具
       更救得回来（最大块 674 token）；
    3. **D32 之后**（一段 user 只配最后一条），"超长 user + 回答"也不再撞破
       ——user 那一侧被 8,000 字符切分**逐片摊开**了。

    ⇒ **能撞破窗口的只剩一个形状：答案侧**——`U` 后面跟着**一长串连续的非 user**。
    它们在 ① 里被并成**一个 RoleBlock**，而 D32 **没有**限制这一侧（只限了 question）
    ⇒ 整段一起进同一个块。**这是 D32 的已知残留**：全量 clbench 上实测最大 5,541 token
    （安全），但**结构上并没有封死**。
    ⇒ 夹具用这个形状，**关掉预算必须超限**；而默认预算下它被摊进多批，不超。
    """
    from tianximem.common.tokens import EMBED_MAX_TOKENS

    long_text = " ".join(["overflowing"] * 3000)
    row = _beam_row()
    row["chat"] = [
        [
            {
                "content": "short question",
                "id": "1,0",
                "index": "1,0",
                "question_type": "main_question",
                "role": "user",
                "time_anchor": None,
            },
            *[
                {
                    "content": long_text,
                    "id": f"1,{i}",
                    "index": f"1,{i}",
                    "question_type": "main_question",
                    "role": "assistant",
                    "time_anchor": None,
                }
                for i in range(1, 7)
            ],
        ]
    ]
    _write_beam(tmp_path, rows=[row])

    # ① 预算拿掉 ⇒ 6 条 assistant 并成一段、整体挂在那条 user 上 ⇒ 巨块 ⇒ 门禁必须报出来
    assert _max_block_tokens(tmp_path, "beam", 1, max_words=10**9) > EMBED_MAX_TOKENS
    # ② 默认预算 ⇒ 同一份语料被摊成多批，每批都在窗口内
    assert _max_block_tokens(tmp_path, "beam", 1) <= EMBED_MAX_TOKENS
