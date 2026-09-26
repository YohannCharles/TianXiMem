"""`eval/harness/` —— 切批 / HTTP 驱动 / 注入 / 裁判包装 / run record（§13）。

**这些用例一律不碰网络、不碰归档 pipeline**：
驱动用 `httpx.MockTransport`，裁判包装用 `tmp_path` 里的**桩 pipeline**
（它模仿归档脚本的 CLI 契约，并且**真的去 `import api_config`**——那是 PYTHONPATH
注入这条机制的端到端验证）。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest
from eval.datasets import Message, Question, Sample, Session
from eval.harness import (
    MAX_MESSAGES_PER_BATCH,
    ServiceClient,
    batches,
    build_input_items,
    build_record,
    config_fingerprint,
    render_memories,
    request_id_for,
    run_judge,
    summarize,
)
from eval.harness.driver import SearchHit
from eval.harness.judge import JudgeResult
from eval.reports.schema import DIMENSIONS, RunRecord


# ── fixture ────────────────────────────────────────────────────────────────
def _messages(n: int) -> tuple[Message, ...]:
    return tuple(
        Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}", timestamp_ms=1000 + i)
        for i in range(n)
    )


def _sample(n_messages: int = 3, n_questions: int = 2) -> Sample:
    return Sample(
        user_id="conv-1",
        dataset="locomo-refined",
        sessions=(Session("conv-1#s1", _messages(n_messages)),),
        questions=tuple(
            Question(qid=f"conv-1#q{i:04d}", question=f"Q{i}?", gold=[f"A{i}"], category="4")
            for i in range(n_questions)
        ),
        speaker_names=("Sam", "Rae"),
    )


class _Recorder:
    """记录发出去的请求，并按需要给出响应。"""

    def __init__(self, search_response: dict | None = None) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.search_response = search_response or {"data": []}

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/add":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    **{k: body[k] for k in ("request_id", "user_id", "session_id")},
                },
            )
        return httpx.Response(200, json=self.search_response)

    def client(self) -> ServiceClient:
        return ServiceClient(
            "http://stub", client=httpx.Client(transport=httpx.MockTransport(self.handler))
        )


# ── 切批（§6.5）──
def test_batches_are_20_and_keep_source_order():
    """**源序就是 `pair_idx` 的最终依据**——重排会静默改变邻域，绝不能做。"""
    got = batches(_messages(45))
    assert [len(b) for b in got] == [20, 20, 5]
    assert [m.content for b in got for m in b] == [f"m{i}" for i in range(45)]
    assert MAX_MESSAGES_PER_BATCH == 20


def test_batches_empty_input_gives_no_batches():
    """空 session 产出 **0 批**，不是一个空批——空批在服务端是响亮失败（422）。"""
    assert batches(()) == []


def test_batches_rejects_nonpositive_limit():
    with pytest.raises(ValueError):
        batches(_messages(3), max_messages=0)


def test_request_id_is_deterministic():
    """§2.2 规定重试沿用同一个 `request_id` ——随机 id 会让"重跑一遍"变成"写第二遍"。"""
    assert request_id_for("u", "s", 0) == request_id_for("u", "s", 0)
    assert request_id_for("u", "s", 0) != request_id_for("u", "s", 1)
    assert request_id_for("u", "s", 0) != request_id_for("u", "s2", 0)


# ── HTTP 驱动 ──────────────────────────────────────────────────────────────
def test_add_payload_matches_contract_shape():
    """§2.1：`request_id` / `user_id` / `session_id` / `messages[]`；
    message 只有 `role` / `content` / 可选 `timestamp`（D16 的 canonical 形状）。"""
    recorder = _Recorder()
    recorder.client().add(request_id="rid", user_id="u1", session_id="s1", messages=_messages(2))
    path, body = recorder.requests[0]
    assert path == "/add"
    assert set(body) == {"request_id", "user_id", "session_id", "messages"}
    assert body["messages"][0] == {"role": "user", "content": "m0", "timestamp": 1000}
    assert body["request_id"] == "rid"


def test_message_without_timestamp_omits_the_key():
    """`timestamp` 缺省时**不能发 `null`**——`event_time` 为 NULL 时 `created_at` 发 `""` 是
    §11.3 那条**有定义的降级路径**，而 `null` 走的是另一条。"""
    recorder = _Recorder()
    recorder.client().add(
        request_id="r", user_id="u", session_id="s", messages=(Message(role="user", content="x"),)
    )
    assert "timestamp" not in recorder.requests[0][1]["messages"][0]


def test_search_over_limit_is_a_loud_failure():
    """**§2.2 的核心那条**：返回超过 `top_k` 是**契约错误**，**AML 不会被静默截断**。

    本地必须抓死：等到 Smoke 那 30 次配额里才发现，代价就是配额本身。
    """
    recorder = _Recorder(
        {"data": [{"id": "1", "content": "c", "created_at": "", "score": 1.0}] * 3}
    )
    with pytest.raises(AssertionError, match="契约错误"):
        recorder.client().search(user_id="u", query="q", top_k=2)


def test_search_null_data_is_rejected():
    """空结果必须是 `[]` 而不是 `null`（§2.1）。`null` 会让下游静默变成"没有记忆"。"""
    recorder = _Recorder({"data": None})
    with pytest.raises(AssertionError, match="必须是数组"):
        recorder.client().search(user_id="u", query="q", top_k=10)


def test_search_returns_hits_verbatim():
    recorder = _Recorder(
        {"data": [{"id": "i", "content": "c", "created_at": "2023-05-08", "score": 0.5}]}
    )
    hits = recorder.client().search(user_id="u", query="q", top_k=10)
    assert hits == [SearchHit(id="i", content="c", created_at="2023-05-08", score=0.5)]


def test_ingest_sends_one_batch_per_20_messages():
    """21 条消息 ⇒ **两批**，且两批的 `request_id` 分别是 0 与 1。"""
    recorder = _Recorder()
    sample = _sample(n_messages=21)
    assert recorder.client().ingest(sample) == 2
    assert [b["request_id"] for _, b in recorder.requests] == [
        "conv-1|conv-1#s1|0",
        "conv-1|conv-1#s1|1",
    ]
    assert [len(b["messages"]) for _, b in recorder.requests] == [20, 1]


# ── 注入（V3 / S1 的落点）──
def test_memories_go_to_speaker_1_and_speaker_2_stays_empty():
    """键名 `speaker_1_memories` 由 pipeline 源码确定；**怎么分是未知的**（见 judge docstring）。

    这条断言钉的是**当前那个显式的代理假设**——它变了就是一次有意改动，不是静默漂移。
    """
    hits = [
        SearchHit(id="a", content="first", created_at="", score=1.0),
        SearchHit(id="b", content="second", created_at="", score=0.5),
    ]
    items = build_input_items(_sample(), {"conv-1#q0000": hits, "conv-1#q0001": []})
    assert items[0]["speaker_1_memories"] == "first\nsecond"
    assert items[0]["speaker_2_memories"] == ""
    assert items[1]["speaker_1_memories"] == ""


def test_input_item_fields_follow_pipeline_not_readme():
    """字段名**以 pipeline 代码为准**：stage 间规范字段是 `generated_answer`，
    输入要带 `id` / `question` / gold 四键之一 / `speaker_*_name`；
    readme 写的 `predicted_answer` / `hypothesis` **没有 pipeline 读**。"""
    items = build_input_items(_sample(), {"conv-1#q0000": [], "conv-1#q0001": []})
    assert set(items[0]) == {
        "id",
        "question",
        "gold_answer",
        "speaker_1_name",
        "speaker_2_name",
        "speaker_2_memories",
        "speaker_1_memories",
    }
    assert items[0]["id"] == "conv-1#q0000"
    # gold 保留**原始 list**——归档的 `gold_answer()` 走 `memory_text()`，列表由它自己拼
    assert items[0]["gold_answer"] == ["A0"]
    assert "hypothesis" not in items[0] and "predicted_answer" not in items[0]


def test_render_memories_uses_newline_like_the_pipeline_does():
    """归档的 `memory_text()` 对列表正是 `"\\n".join(...)`——所以两种拼法**逐字相同**。"""
    hits = [
        SearchHit(id="a", content="x", created_at="", score=1.0),
        SearchHit(id="b", content="y", created_at="", score=0.5),
    ]
    assert render_memories(hits) == "\n".join(["x", "y"])


# ── 裁判包装（subprocess + PYTHONPATH 注入）──
_STUB_PIPELINE = '''\
"""模仿归档 pipeline 的 CLI 契约；并证明 `api_config` 在 PYTHONPATH 上可 import。"""
import argparse, json, sys
from api_config import ANSWER_MODEL, JUDGE_MODEL, JUDGE_VERSION  # noqa: F401

def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("answer")
    for name in ("input", "output"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--max-tokens", type=int, default=256)
    e = sub.add_parser("evaluate")
    for name in ("input", "answers", "output"):
        e.add_argument(f"--{name}", required=True)
    e.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    if args.command == "answer":
        with open(args.output, "a", encoding="utf-8") as fh:
            for item in rows(args.input):
                # 把注入的记忆原样当作"答案"，方便断言注入确实到了模型侧
                fh.write(json.dumps({"id": item["id"],
                                     "generated_answer": item["speaker_1_memories"]}) + "\\n")
        return
    with open(args.output, "w", encoding="utf-8") as fh:
        for item in rows(args.input):
            fh.write(json.dumps({"id": item["id"], "label": "CORRECT", "is_correct": True,
                                 "judge_response": '{"label": "CORRECT"}'}) + "\\n")

main()
'''


@pytest.fixture
def stub_pipeline(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline_stub.py"
    path.write_text(_STUB_PIPELINE, encoding="utf-8")
    return path


def test_run_judge_wires_through_and_parses(stub_pipeline, tmp_path):
    """端到端（无网络）：回答步 → 判分步 → 解析。

    **桩脚本里那行 `import api_config` 就是 PYTHONPATH 注入的验证**——
    归档的 7 个 pipeline 今天全部 import 失败，这条路径必须真的通。
    """
    hits = [SearchHit(id="a", content="memory text", created_at="", score=1.0)]
    items = build_input_items(_sample(), {"conv-1#q0000": hits, "conv-1#q0001": []})
    results = run_judge(stub_pipeline, items, tmp_path / "out")

    assert [r.qid for r in results] == ["conv-1#q0000", "conv-1#q0001"]
    assert all(r.is_correct and r.label == "CORRECT" for r in results)
    # 注入的记忆确实到了脚本侧（桩脚本把它当成 generated_answer 回写）
    assert results[0].generated_answer == "memory text"


def test_run_judge_creates_output_dir(stub_pipeline, tmp_path):
    """**`out_dir` 必须由调用方建**——`pipeline_locomo-refined.py` 不会 `mkdir(parents=True)`。"""
    target = tmp_path / "deep" / "nested"
    run_judge(stub_pipeline, build_input_items(_sample(), {}), target)
    assert (target / "input.jsonl").exists()


def test_run_judge_surfaces_pipeline_failure(tmp_path):
    """pipeline 非 0 退出必须**带上 stderr 一起抛**——否则只剩一句"跑失败了"。"""
    broken = tmp_path / "broken.py"
    broken.write_text("import sys; sys.stderr.write('boom\\n'); sys.exit(3)", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exit 3"):
        run_judge(broken, build_input_items(_sample(), {}), tmp_path / "out")


# ── run record（§13）──
def _results(*flags: bool) -> list[JudgeResult]:
    return [
        JudgeResult(
            qid=f"conv-1#q{i:04d}",
            is_correct=flag,
            label="",
            judge_response="",
            generated_answer="",
        )
        for i, flag in enumerate(flags)
    ]


def test_summarize_keeps_abstention_out_of_the_total():
    """**拒答题行为与其他题不同**（正确答案是拒答），混进分类会失真（§12.2）。"""
    sample = Sample(
        user_id="u",
        dataset="longmemeval-s",
        sessions=(),
        questions=(
            Question("q0", "?", "g", "multi-session"),
            Question("q1_abs", "?", "g", "multi-session", is_abstention=True),
        ),
    )
    results = [
        JudgeResult("q0", is_correct=True, label="", judge_response="", generated_answer=""),
        JudgeResult("q1_abs", is_correct=False, label="", judge_response="", generated_answer=""),
    ]
    summary = summarize(results, [sample])
    assert summary["overall"] == 0.5  # 拒答题**算进总分**，只是不混进分类
    assert summary["breakdown"]["multi-session"] == {"accuracy": 1.0, "n": 1}
    assert summary["abstention"] == {"accuracy": 0.0, "n": 1}


def test_summarize_empty_is_none_not_zero():
    """空集合必须是 `None`——`0.0` 会被读成"全错"，而它其实是"没测到"。"""
    assert summarize([], [])["overall"] is None


def test_config_fingerprint_hash_is_order_insensitive(tmp_path):
    """指纹对字典顺序不敏感——否则"同一个配置"会因为书写顺序不同而看起来变过。"""
    a = config_fingerprint("local", {"rerank": False, "agent": False}, configs_dir=tmp_path)
    b = config_fingerprint("local", {"agent": False, "rerank": False}, configs_dir=tmp_path)
    assert a["switches_hash"] == b["switches_hash"]
    assert config_fingerprint("local", configs_dir=tmp_path)["switches_hash"] != a["switches_hash"]


def test_config_fingerprint_hashes_the_config_snapshot(tmp_path):
    """③-d 之后**开关的家是 `configs/<profile>.yaml`**——指纹必须覆盖它。

    harness 不解析 yaml（那会与 `common/config.py` 抢知识），只逐字节哈希；
    **"改了配置但没人注意到"靠哈希就够发现了**。
    """
    (tmp_path / "default.yaml").write_text("retrieval:\n  prefetch_limit: 200\n", encoding="utf-8")
    (tmp_path / "local.yaml").write_text(
        "storage:\n  qdrant:\n    collection: memories_dev\n", encoding="utf-8"
    )

    fingerprint = config_fingerprint("local", configs_dir=tmp_path)
    assert fingerprint["snapshot_files"] == ["default.yaml", "local.yaml"]
    assert len(fingerprint["snapshot_hashes"]["local.yaml"]) == 16
    assert "消费方尚未接线" in fingerprint["note"]

    # 改了 yaml ⇒ 指纹变（哪怕 switches 一个字没动）
    (tmp_path / "local.yaml").write_text(
        "storage:\n  qdrant:\n    collection: memories_other\n", encoding="utf-8"
    )
    assert (
        config_fingerprint("local", configs_dir=tmp_path)["snapshot_hashes"]
        != fingerprint["snapshot_hashes"]
    )


def test_config_fingerprint_says_loudly_when_configs_are_missing(tmp_path):
    """找不到配置文件必须**说出来**——一个"看着齐全"的空指纹比没有更糟。"""
    assert "不完整" in config_fingerprint("local", configs_dir=tmp_path / "nope")["note"]


def test_build_record_validates_and_states_why_dimensions_are_empty(tmp_path):
    fp = {"dataset": "locomo-refined", "files": [{"name": "questions.jsonl"}]}
    record = build_record(
        run_id="r1",
        step="Step 1",
        profile="local",
        bench_dir=tmp_path,
        samples=[_sample()],
        results=_results(True, False),
        data_fingerprint=fp,
        models={"embedder": "Qwen/Qwen3-Embedding-8B"},
    )
    assert set(DIMENSIONS) <= set(record.scores)
    assert all(record.scores[d] is None for d in DIMENSIONS)
    # **空值必须带理由**——否则与"跑了但没分"无法区分（§3.2）
    assert record.scores["by_dimension_note"].startswith("代理评测的七个维度子分")
    assert "observability/ 未实现" in record.counters["note"]
    assert record.counters["pending_orphaned_real"] is None
    # 第 4/7 维的证据指向机制与单测，不是一个编出来的数
    assert "test_idempotency" in record.dimension_mechanism["Memory governance"]


def test_build_record_merges_override_counters(tmp_path):
    """接上聚合端之后，调用方可以覆盖计数器——但**两个来源仍必须都在**。"""
    with pytest.raises(ValueError, match="pending_orphaned_real"):
        build_record(
            run_id="r",
            step="Step 1",
            profile="local",
            bench_dir=tmp_path,
            samples=[_sample()],
            results=_results(True),
            data_fingerprint={"d": 1},
            models={"m": "x"},
            counters={"pending_created": 3},
        )


def test_run_record_rejects_missing_dimension():
    """七个维度必须**逐维记录**（§3.2 / reports/CLAUDE.md）。"""
    with pytest.raises(ValueError, match="缺维度"):
        RunRecord(
            run_id="r",
            step="Step 1",
            profile="local",
            config_fingerprint={"a": 1},
            data_fingerprint={"b": 2},
            models={"c": "3"},
            scores={"overall": 1.0, "Explicit fact recall": 1.0},
            counters={"pending_orphaned_real": 0, "pending_orphaned_misjudged": 0},
        ).validate()


def test_run_record_rejects_empty_dimension_without_reason():
    with pytest.raises(ValueError, match="没说理由"):
        RunRecord(
            run_id="r",
            step="Step 1",
            profile="local",
            config_fingerprint={"a": 1},
            data_fingerprint={"b": 2},
            models={"c": "3"},
            scores={"overall": 1.0, **{d: None for d in DIMENSIONS}},
            counters={"pending_orphaned_real": 0, "pending_orphaned_misjudged": 0},
        ).validate()


def test_run_record_rejects_missing_fingerprint():
    """三个指纹缺一个，两次 run 就不可比（§13）。"""
    with pytest.raises(ValueError, match="models 为空"):
        RunRecord(
            run_id="r",
            step="Step 1",
            profile="local",
            config_fingerprint={"a": 1},
            data_fingerprint={"b": 2},
            models={},
            scores={"overall": 1.0, **{d: None for d in DIMENSIONS}, "by_dimension_note": "n/a"},
            counters={"pending_orphaned_real": 0, "pending_orphaned_misjudged": 0},
        ).validate()


def test_run_record_scope_carries_the_extrapolation_warning(tmp_path):
    """§12.4 / P3：代理评测**不可线性外推**——这句话跟着每一个数字走。"""
    record = build_record(
        run_id="r",
        step="Step 1",
        profile="local",
        bench_dir=tmp_path,
        samples=[_sample()],
        results=_results(True),
        data_fingerprint={"d": 1},
        models={"m": "x"},
    )
    assert "relative comparisons only" in record.scope


def test_write_record_lands_in_runs_dir(tmp_path):
    record = build_record(
        run_id="r9",
        step="Step 1",
        profile="local",
        bench_dir=tmp_path,
        samples=[_sample()],
        results=_results(True),
        data_fingerprint={"d": 1},
        models={"m": "x"},
    )
    from eval.harness import write_record

    path = write_record(record, tmp_path / "reports")
    assert path == tmp_path / "reports" / "runs" / "r9.json"
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "r9"


def test_harness_does_not_import_src():
    """**全目录最硬的一条边界**：harness 打 HTTP，**不 import `src/tianxi_am`**。

    走进程内调用会让 B1（ReFind）变成特例、两条基线不可比，而且**碰不到契约层**
    （[`../../eval/CLAUDE.md`](../../eval/CLAUDE.md)）。

    用 AST 扫 **import 语句**而不是扫文本——docstring 里提到 `src/tianxi_am`
    （比如 `api_config` 解释"embedding 配置只属于哪一边"）是说明，不是依赖。
    """
    harness = Path(__file__).resolve().parents[1] / "eval" / "harness"
    offenders = []
    for path in harness.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [
                f"{path.name}:{node.lineno}: {name}" for name in names if "tianxi_am" in name
            ]
    assert not offenders, "harness 依赖了 src/：\n" + "\n".join(offenders)


def test_harness_sources_have_no_src_path_hack():
    """`sys.path` 里塞 `src/` 与直接 import 等价——两条都算越界。"""
    harness = Path(__file__).resolve().parents[1] / "eval" / "harness"
    for path in harness.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "insert":
                text = ast.unparse(node)
                assert "src" not in text, f"{path.name}:{node.lineno}: {text}"


# ── 判分步骤的**瞬时故障重试**（2026-09-26 事故）────────────────────────
def test_run_retries_a_transient_failure(monkeypatch, tmp_path):
    """瞬时失败要重试，**成功即返回**——不把"抖了一下"升级成"整轮作废"。

    动机是一次真实事故：LongMemEval 那轮跑到第 47 题，判分侧一次 TLS 握手失败就把
    **两小时的跑批整个打死**。两个子命令都幂等（`answer` 追加并跳过已完成、
    `evaluate` 覆盖），所以重试是安全的。
    """
    from eval.harness import judge

    calls = {"n": 0}

    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Completed()
        return type("C", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(judge.subprocess, "run", fake_run)
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)  # 别真睡

    judge._run(tmp_path / "p.py", ["answer"], timeout=None)
    assert calls["n"] == 2  # 第一次失败、第二次成功


def test_run_gives_up_after_the_attempt_budget(monkeypatch, tmp_path):
    """**重试要有界**：真 bug 重试 3 次还是失败 ⇒ 照样抛，错误信息原样带出去。

    "一直失败"绝不能被伪装成"在重试"。
    """
    from eval.harness import judge

    calls = {"n": 0}

    class _Completed:
        returncode = 2
        stdout = "out"
        stderr = "err"

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        return _Completed()

    monkeypatch.setattr(judge.subprocess, "run", fake_run)
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="exit 2"):
        judge._run(tmp_path / "p.py", ["evaluate"], timeout=None)
    assert calls["n"] == 3


def test_sanitize_jsonl_tmp_path(tmp_path):
    """末尾被截断的半行要被丢掉，**完整行一行不动**。

    事故见 `judge._sanitize_jsonl` 的 docstring：半行会让 `answer` 步
    `JSONDecodeError`，而表现是"**卡在同一道题**"（看起来像网络问题）。
    """
    from eval.harness import judge

    path = tmp_path / "answers.jsonl"
    good = '{"id": "a", "generated_answer": "x"}'
    path.write_text(f'{good}\n{good}\n{{"id": "b", "generated', encoding="utf-8")

    assert judge._sanitize_jsonl(path) == 1
    assert path.read_text(encoding="utf-8") == f"{good}\n{good}\n"


def test_sanitize_jsonl_keeps_middle_lines(tmp_path):
    """**只在末尾删**：中途的坏行说明别的问题，不该被静默吞掉。"""
    from eval.harness import judge

    path = tmp_path / "answers.jsonl"
    path.write_text('{"id": "a"}\n{坏行\n{"id": "c"}\n', encoding="utf-8")

    assert judge._sanitize_jsonl(path) == 0
    assert "坏行" in path.read_text(encoding="utf-8")


def test_sanitize_jsonl_noop_without_file(tmp_path):
    from eval.harness import judge

    assert judge._sanitize_jsonl(tmp_path / "nope.jsonl") == 0


def test_jsonl_line_survives_the_archives_splitlines_reader(tmp_path):
    """正文里的 `U+2028` 不能让那一行被劈开——**归档是按 `splitlines()` 读的**。

    事故见 `judge._jsonl_line` 的 docstring：LongMemEval 那轮有 1 个席位的正文含
    `U+2028`，`json.dumps(ensure_ascii=False)` 不转义它、而 `splitlines()` 照断 ⇒
    那一行被劈成两半 ⇒ `JSONDecodeError: Unterminated string` ⇒ **卡在同一道题**，
    却看起来像网络问题（实测耗掉一小时，11 次续跑全死在同一题）。
    """
    from eval.harness import judge

    for sep in (" ", " ", "\x85"):
        item = {"id": "q1", "speaker_1_memories": f"before{sep}after", "gold_answer": "x"}
        path = tmp_path / "input.jsonl"
        path.write_text(judge._jsonl_line(item), encoding="utf-8")

        # ① 用**归档的读法**读回来：一行一项，且语义一字不变
        parsed = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert parsed == [item], repr(sep)

        # ② 反面证据：不转义就会被劈开 —— 这条断言钉的就是那个 bug 本身
        assert len(json.dumps(item, ensure_ascii=False).splitlines()) == 2, repr(sep)


def test_readers_split_on_newline_not_splitlines(tmp_path):
    """**按写它的方式读**：`answers.jsonl` 里出现 `U+2028` 时不能被 `splitlines()` 劈开。

    这个文件是**归档**写的（`json.dumps(..., ensure_ascii=False) + "\\n"`），我们只读。
    ⇒ 读法必须与写法一致，否则一条记录会被看成两条，而**不会报错**——
    表现是"答案对不上号"。
    """

    answer = "He said: first;  second"
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "generated_answer": answer}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert len(path.read_text(encoding="utf-8").splitlines()) == 3  # ← 反面证据：劈成 3 行
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    assert rows == [{"id": "q1", "generated_answer": answer}]


def test_sanitize_jsonl_treats_a_u2028_line_as_one_record(tmp_path):
    """一条**完整**记录不能因为含 `U+2028` 就被当成"坏的"丢掉——那会白重生成一次。

    正确处置是**就地转义**（语义不变），而不是删。⚠ 它自己也得按 `"\n"` 切：
    用 `splitlines()` 的话，这一条会被看成两条坏的。
    """
    from eval.harness import judge

    path = tmp_path / "answers.jsonl"
    line = json.dumps({"id": "a", "generated_answer": "x y"}, ensure_ascii=False)
    path.write_text(f"{line}\n", encoding="utf-8")

    assert judge._sanitize_jsonl(path) == 1  # 转义 1 行（**不是**丢掉）
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "id": "a",
        "generated_answer": "x y",
    }

def test_sanitize_jsonl_escapes_a_line_that_would_be_split(tmp_path):
    """**归档自己写的行**里若带 `U+2028`，也要在它读之前转义掉。

    这是我们控制不了写入侧时的唯一办法：`answers.jsonl` 由归档写、又由它自己的
    `rows()`（`splitlines()`）读 ⇒ 一旦**模型生成的答案**里带 `U+2028`，
    **归档会在下一次续跑时崩在自己写的文件上**，而"丢半行"救不了（那行解析得通）。
    """
    from eval.harness import judge

    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "generated_answer": "a b"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert judge._sanitize_jsonl(path) == 1  # 改了 1 行
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 1, "还是会被劈行"
    assert json.loads(text) == {"id": "q1", "generated_answer": "a b"}  # 语义一字不变


def test_sanitize_jsonl_is_a_noop_on_a_clean_file(tmp_path):
    from eval.harness import judge

    path = tmp_path / "answers.jsonl"
    good = json.dumps({"id": "a", "generated_answer": "x"}, ensure_ascii=False) + "\n"
    path.write_text(good, encoding="utf-8")

    assert judge._sanitize_jsonl(path) == 0
    assert path.read_text(encoding="utf-8") == good


# ── 超限兜底（2026-09-26，只为 B1）──────────────────────────────────────
def _client(base_url, *, status, body=None):
    """一个用 `MockTransport` 造出来的客户端：**所有请求都回同一个状态码**。"""
    from eval.harness import ServiceClient

    def handler(request):
        if status >= 400:
            return httpx.Response(status, json={"detail": "boom"})
        return httpx.Response(200, json=body or {"data": []})

    return ServiceClient(
        base_url,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_search_falls_back_only_on_server_errors():
    """5xx ⇒ 兜底到第二个实例；**4xx 不兜底**（那是我们自己的 bug，兜底会把 bug 藏起来）。"""

    hit = {"id": "x", "content": "c", "created_at": "", "score": 1.0}
    ok = _client("http://main", status=200, body={"data": [hit]})
    primary = _client("http://fallback", status=500)
    primary._fallback = ok  # 主实例 5xx ⇒ 兜底

    assert primary.search_raw(user_id="u", query="q", top_k=1)["data"][0]["id"] == "x"
    assert primary.fallback_used == 1

    client_4xx = _client("http://main", status=400)
    client_4xx._fallback = ok
    with pytest.raises(httpx.HTTPStatusError):
        client_4xx.search_raw(user_id="u", query="q", top_k=1)
    assert client_4xx.fallback_used == 0  # ← 4xx 不该兜底

    no_fallback = _client("http://main", status=500)
    with pytest.raises(httpx.HTTPStatusError):
        no_fallback.search_raw(user_id="u", query="q", top_k=1)  # 没配兜底 ⇒ 照旧抛


def test_injection_is_capped_at_the_platform_prefix(monkeypatch):
    """**平台的答案阶段是"取 117,760 token 前缀"**——harness 要自己模拟。

    为什么：我们自己的服务守预算，**基线不一定**。实测 ReFind 在 LongMemEval 上有一题
    返回 **897,838 字符**（≈22–30 万 token）⇒ 判分提示爆 128k ⇒ 网关 400 ⇒ 整轮被打死。
    而平台会替它截断 ⇒ 正确的模拟是"截前缀"，不是"让整轮炸掉"。
    """
    from eval.harness import judge

    small = "hello world"
    assert judge.truncate_to_platform_prefix(small) == (small, False)

    huge = "word " * 200_000  # 远超 117,760 token
    text, cut = judge.truncate_to_platform_prefix(huge)
    assert cut is True
    assert judge._encoder().encode(text).__len__() <= judge.PLATFORM_TOKEN_PREFIX
    assert huge.startswith(text)  # ★ 是**前缀**，不是重排、不是摘要


def test_build_input_items_reports_truncation(monkeypatch, capsys):
    """截断要在跑批时**可见**（打印一行），否则"这一题的上下文少了一截"没人知道。"""
    from eval.harness import judge
    from eval.harness.driver import SearchHit

    monkeypatch.setattr(
        judge, "truncate_to_platform_prefix", lambda text: (text[:10], True)
    )
    hit = SearchHit(id="a", content="Q: x\nA: y" * 100, created_at="", score=1.0)
    question = type("Q", (), {"qid": "q1", "question": "?", "gold": "g"})()
    sample = type("S", (), {"questions": [question], "speaker_names": ("A", "B")})()
    judge.build_input_items(sample, {"q1": [hit]})
    assert "被截到平台前缀" in capsys.readouterr().out
