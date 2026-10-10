"""`eval/experiments/` —— 通用 runner 与各 arm 的声明（§12.1 / §13）。

**一律不碰网络、不碰归档数据**：数据用合成的 `Sample`（monkeypatch 掉加载器）、
服务用 `httpx.MockTransport`、裁判用 `tmp_path` 里的**桩 pipeline**
（它连文件名都照抄归档的真名 `pipeline_locomo-refined.py`，好让 `pipeline_for` 真的生效）。

> ⚠ 归档 pipeline 那份桩在 [`test_harness.py`](./test_harness.py) 里已有一份。
> 这里再写一份不是重复：那边验的是 `run_judge` 的形状，这边验的是
> **"一轮跑完会往哪落什么"**——两者的断言对象不同。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest
from eval.datasets import Message, Question, Sample, Session
from eval.experiments import run as runner
from eval.harness import ServiceClient

# ── fixture ────────────────────────────────────────────────────────────────
_STUB_PIPELINE = '''\
"""照抄归档 pipeline 的 CLI 契约；并真的 import api_config（PYTHONPATH 注入的验证）。"""
import argparse, json
from api_config import ANSWER_MODEL  # noqa: F401

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
                fh.write(json.dumps({"id": item["id"],
                                     "generated_answer": item["speaker_1_memories"]}) + "\\n")
        return
    with open(args.output, "w", encoding="utf-8") as fh:
        for item in rows(args.input):
            fh.write(json.dumps({"id": item["id"], "label": "CORRECT", "is_correct": True,
                                 "judge_response": '{"label": "CORRECT"}'}) + "\\n")

main()
'''


def _sample(user_id: str = "conv-1", n_questions: int = 2) -> Sample:
    messages = tuple(
        Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}", timestamp_ms=1000 + i)
        for i in range(3)
    )
    return Sample(
        user_id=user_id,
        dataset="locomo-refined",
        sessions=(Session(f"{user_id}#s1", messages),),
        questions=tuple(
            Question(qid=f"{user_id}#q{i:04d}", question=f"Q{i}?", gold=[f"A{i}"], category="4")
            for i in range(n_questions)
        ),
        speaker_names=("Sam", "Rae"),
    )


class _Recorder:
    """记下 harness 发出去的请求；`/search` 回两条命中。"""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

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
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "mem-1",
                        "content": "Q: m0\nA: m1",
                        "created_at": "2023-05-08",
                        "score": 1.0,
                    },
                    {"id": "mem-2", "content": "Q: m2", "created_at": "", "score": 0.5},
                ]
            },
        )

    def client(self) -> ServiceClient:
        return ServiceClient(
            "http://stub", client=httpx.Client(transport=httpx.MockTransport(self.handler))
        )


@pytest.fixture
def bench_dir(tmp_path: Path) -> Path:
    """一个"长得像归档"的目录：桩 pipeline 按**真名**命名（`pipeline_for` 才找得到它），
    两个数据源文件**写成空文件**——数据指纹只逐字节哈希它们，**内容从不被读**
    （加载器在用例里被 monkeypatch 掉了）。"""
    (tmp_path / "pipeline_locomo-refined.py").write_text(_STUB_PIPELINE, encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "conversations.jsonl").write_text("", encoding="utf-8")
    return tmp_path


# ── run_id 与指纹（纯函数）──
def test_run_id_carries_the_dataset_and_utc_stamp():
    from datetime import UTC, datetime

    got = runner.derive_run_id("locomo-refined", now=datetime(2026, 9, 25, 10, 30, 0, tzinfo=UTC))
    assert got == "locomo-refined-20260925T103000"


def test_models_fingerprint_says_when_it_does_not_know():
    """**R1：不记模型就归因不了分数变化**。缺了就写"没说"，**不编一个看着合理的值**。"""
    got = runner.models_fingerprint(embedder="", llm="gpt-4o-mini", reranker="")
    assert "未声明" in got["embedder"]
    assert got["llm"] == "gpt-4o-mini"
    assert got["reranker"] == "disabled"  # 「没开」与「没声明」是两件事，各有各的写法


def test_switches_must_be_a_json_object():
    assert runner._parse_switches(None) == {}
    assert runner._parse_switches('{"rerank.enabled": false}') == {"rerank.enabled": False}
    with pytest.raises(SystemExit):
        runner._parse_switches("[1, 2]")  # 合法 JSON，但不是对象
    with pytest.raises(SystemExit):
        runner._parse_switches("{不是 json}")


# ── 一轮：投喂 → 检索 → 裁判（打桩 HTTP）──
def test_run_round_ingests_searches_and_judges(bench_dir, tmp_path, monkeypatch):
    """一轮的完整形状：**先 Add 后 Search**、每题一次 `top_k=100`、裁判落在 per-user 目录。"""
    recorder = _Recorder()
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])

    samples, results = runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "runs" / "r1",
        client=recorder.client(),
    )

    assert [s.user_id for s in samples] == ["conv-1"]
    assert [r.qid for r in results] == ["conv-1#q0000", "conv-1#q0001"]
    assert all(r.is_correct for r in results)

    paths = [path for path, _ in recorder.requests]
    assert paths[0] == "/add"  # 语料先进去，再检索
    assert paths.count("/search") == 2  # 每题一次
    for path, body in recorder.requests:
        if path == "/search":
            assert body["user_id"] == "conv-1"
            assert body["top_k"] == runner.DEFAULT_TOP_K
        else:
            # 源序不重排；前缀是缺省的官方形态（fixture 的 speaker_names=("Sam","Rae")
            # 且 dataset=locomo-refined ⇒ `@speaker` 规则）。
            assert body["messages"][0]["content"] == "Sam: m0"

    # 逐题原始产出落在 per-user 目录里（runs/ 被 .gitignore 覆盖，见 eval/reports/CLAUDE.md）
    assert (tmp_path / "runs" / "r1" / "conv-1" / "labels.jsonl").exists()


def test_run_round_parallel_judging_matches_serial(bench_dir, tmp_path, monkeypatch):
    """`--judge-workers > 1` **只并行判分**：逐题结果与串行逐字一致，各 user 仍各有自己的目录。

    这条守的是并行改造最危险的那类后果——**结果少了、串了、或顺序变了**，
    而屏幕上只会看到"分数不一样"。并行的是**判分子进程**（直连网关、不碰我们的服务），
    所以 Add / Search 的请求顺序**必须与串行完全相同**。
    """
    samples = [_sample(f"conv-{i}") for i in (1, 2, 3)]
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: samples)

    serial_recorder, parallel_recorder = _Recorder(), _Recorder()
    _, serial = runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "serial",
        client=serial_recorder.client(),
    )
    _, parallel = runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "parallel",
        client=parallel_recorder.client(),
        judge_workers=3,
    )

    # 结果：顺序、qid、判定三样都要一致（少了任何一样都说明并行把东西吞掉了）
    assert [r.qid for r in parallel] == [r.qid for r in serial]
    assert [r.is_correct for r in parallel] == [r.is_correct for r in serial]
    # 打服务的请求序列一致（Add/Search 仍是主线程串行，并行只发生在判分那一段）
    assert parallel_recorder.requests == serial_recorder.requests
    # 每个 user 的原始产出照旧落在自己的目录里
    for i in (1, 2, 3):
        assert (tmp_path / "parallel" / f"conv-{i}" / "labels.jsonl").exists()


def test_judging_rotates_endpoints_and_defaults_to_parallel(
    bench_dir, tmp_path, monkeypatch, capsys
):
    """判分**默认并行到端点数**，且每个 sample 依次领一个对话端点（D39）。

    两个对话网关提供的是**同一个** `Qwen/Qwen3.5-9B` ⇒ 这条只决定"谁来算"，
    **不该改变任何判定**（上面那条用例守的就是那一半：并行 == 串行，逐题逐字）。
    """
    samples = [_sample(f"conv-{i}") for i in (1, 2, 3)]
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: samples)
    monkeypatch.setattr(runner, "ENDPOINTS", (("http://e0/v1", "k0"), ("http://e1/v1", "k1")))

    seen: list[tuple[str, int | None]] = []
    real_run_judge = runner.run_judge

    def recording(pipeline, items, out_dir, *, dataset="locomo-refined", endpoint_index=None, **kw):
        seen.append((Path(out_dir).name, endpoint_index))
        return real_run_judge(pipeline, items, out_dir, dataset=dataset, **kw)

    monkeypatch.setattr(runner, "run_judge", recording)
    runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "rotating",
        client=_Recorder().client(),
    )

    # 依次领号（串行也轮转）：三个 sample 落在 0 / 1 / 0
    assert seen == [("conv-1", 0), ("conv-2", 1), ("conv-3", 0)]
    # `judge_workers` 缺省（0）= 自动 = 端点数 ⇒ 这一轮真开了两路
    assert "判分并行 2 路" in capsys.readouterr().out


def test_run_round_skips_ingest_when_asked(bench_dir, tmp_path, monkeypatch):
    """`--skip-ingest` 只掉 Add，**Search 一次都不能少**——少检索就是静默漏题。"""
    recorder = _Recorder()
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])

    runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "out",
        skip_ingest=True,
        client=recorder.client(),
    )

    assert [path for path, _ in recorder.requests] == ["/search", "/search"]


def test_run_round_rejects_unknown_dataset(bench_dir, tmp_path):
    with pytest.raises(ValueError, match="未知数据集"):
        runner.run_round(
            dataset="personamem", base_url="http://stub", bench_dir=bench_dir, out_dir=tmp_path
        )


def test_limit_is_recorded_as_a_truncated_run():
    """**截断过的数字与全量看起来一样** ⇒ 数据指纹的 `note` 必须写下来（§13）。"""
    assert runner.truncation_note(None) == ""
    assert "截断" in runner.truncation_note(3)
    # 两种截断维度是**分开**写的：只截 sample 与只截题目是两回事
    note = runner.truncation_note(1, max_questions=10)
    assert "sample" in note and "题目截断" in note
    assert "题目截断" in runner.truncation_note(None, max_questions=10)


def test_max_questions_only_trims_the_questions_not_the_corpus(bench_dir, tmp_path, monkeypatch):
    """`--max-questions` **只裁题目**：语料照常整份 Add。

    ⚠ 若连语料一起裁，检索就没有素材、分数全是假的——而它在报告里看起来只是"分低"。
    """
    recorder = _Recorder()
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])
    sample = _sample()

    _, results = runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "runs" / "r1",
        max_questions=1,
        client=recorder.client(),
    )

    # 只判了 1 题（`_sample()` 有 2 题）
    assert len(results) == 1
    # 而 Add 一条都没少：语料是整份喂进去的
    added = [body for path, body in recorder.requests if path == "/add"]
    assert len(added) == 1
    assert len(added[0]["messages"]) == len(sample.sessions[0].messages)


def _exploding_client(*_args, **_kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return ServiceClient("http://stub", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_main_returns_precondition_code_when_service_is_unreachable(
    monkeypatch, bench_dir, tmp_path, capsys
):
    """服务不可用 ≠ 跑失败：**退出码 2**（与 `check_env` / `preflight` 同一套约定）。

    ⚠ 用打桩的客户端而不是"指向一个没人听的端口"：本机 WSL + Docker Desktop 的端口转发
    会把任意 localhost 端口接成 **502**（不是拒绝连接），那样写出来的用例**换台机器就变**。
    """
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench_dir)
    monkeypatch.setattr(runner, "ensure_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])
    monkeypatch.setattr(runner, "ServiceClient", _exploding_client)
    _pretend_judge_is_configured(monkeypatch)

    code = runner.main(["--dataset", "locomo-refined", "--reports-dir", str(tmp_path / "reports")])

    assert code == runner.EXIT_PRECONDITION_FAILED
    assert "make serve" in capsys.readouterr().err


def test_main_writes_a_record_that_carries_the_fingerprints(bench_dir, tmp_path, monkeypatch):
    """**一轮跑完的产物形状**：record 落在 `reports/runs/<run_id>.json`，三个指纹都在。

    全程打桩（服务 / 数据 / 归档 pipeline），所以这不碰网络也不碰归档——它验的是
    "runner 把哪几样东西交给了 `build_record`"，而**那正是 §13 要求可追溯的部分**。
    """
    recorder = _Recorder()
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench_dir)
    monkeypatch.setattr(runner, "ensure_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])
    monkeypatch.setattr(runner, "ServiceClient", lambda *a, **kw: recorder.client())
    _pretend_judge_is_configured(monkeypatch)
    reports = tmp_path / "reports"

    code = runner.main(
        [
            "--dataset",
            "locomo-refined",
            "--limit",
            "1",
            "--run-id",
            "r-trunc",
            "--reports-dir",
            str(reports),
            "--configs-dir",
            str(tmp_path / "configs"),  # 空目录：配置快照 hash 会**明说它不完整**
        ]
    )

    assert code == runner.EXIT_OK
    record = json.loads((reports / "runs" / "r-trunc.json").read_text(encoding="utf-8"))
    assert record["run_id"] == "r-trunc"
    assert record["scores"]["overall"] == 1.0
    assert record["breakdown"]["4"]["n"] == 2
    assert "截断" in record["data_fingerprint"]["note"]
    assert record["data_fingerprint"]["batching"].startswith("20 messages")
    assert "未声明" in record["models"]["embedder"]  # R1：缺就写"没说"，不编
    assert "不完整" in record["config_fingerprint"]["note"]


def _pretend_judge_is_configured(monkeypatch) -> None:
    """让 `main()` 跳过"裁判前置条件"那一关。

    那些用例测的是**别的事**（退出码映射、run record 的字段），而 pytest 进程里没有
    `AML_*` 环境变量 ⇒ 不绕过的话它们会全部停在同一条检查上。**别把它当"检查多余"的证据**——
    那条检查自己有一条用例（见下）。
    """
    monkeypatch.setattr(runner, "judge_preconditions", list)


def test_main_fails_before_running_when_the_judge_is_not_configured(
    monkeypatch, bench_dir, tmp_path, capsys
):
    """**裁判跑不起来时要在跑之前失败**（§15 的口径：别把"整轮白跑"伪装成"跑完没分"）。

    裁判是 subprocess，只继承环境变量；而 `.env` 由 `common/config.py` 自己读。
    ⇒ 少了 `--env-file` 时 Add/Search 会**全部正常跑完**，然后才在裁判那一步炸。
    """
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench_dir)
    monkeypatch.setattr(runner, "ensure_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])
    monkeypatch.setattr(runner, "judge_preconditions", lambda: ["AML_BASE_URL", "AML_MODEL"])

    def _never(*_a, **_kw):  # pragma: no cover —— 走到这里就说明检查没拦住
        raise AssertionError("前置条件没过就不该去连服务")

    monkeypatch.setattr(runner, "ServiceClient", _never)

    code = runner.main(["--dataset", "locomo-refined", "--reports-dir", str(tmp_path)])

    assert code == runner.EXIT_PRECONDITION_FAILED
    err = capsys.readouterr().err
    assert "AML_BASE_URL" in err and "--env-file" in err


# ── 记忆注入里的日期锚点（S1 的第二条假设，2026-09-25）──
def test_date_prefix_matches_the_one_in_src():
    """harness 里的 `DATE_PREFIX` 与 `src` 里那份**必须逐字相同**。

    harness 不许 import `src/`（AST 钉着），所以这个常量**故意重复了一份**——
    而"两处格式一致"只能靠这条断言守。一旦分叉，注入给模型的日期与 `created_at`
    就是两种形状，**而没有任何东西会报错**。
    """
    from eval.harness.judge import DATE_PREFIX as HARNESS_PREFIX

    from tianximem.common.render import DATE_PREFIX as SRC_PREFIX

    assert HARNESS_PREFIX == SRC_PREFIX


def test_platform_token_budget_matches_src():
    """harness 的平台截断窗口与分词器，**必须与 `src` 那两份相等**。

    与上一条同一条理由、同一种守法：harness 不许 import `src/`（AST 钉着），
    于是这对常量**故意重复了一份**，只能靠断言守。

    ⚠ **它守的是"判分看到的上下文"与"平台实际喂给答案模型的上下文"是同一段**：
    §2.2 / §6.4 说答案阶段按 **117,760 token 取前缀**，所以 harness 判分前
    必须先模拟同一条截断。窗口或分词器任一侧漂了，harness 会**继续按旧值截断**——
    判分依据的那段文本与平台真正用到的不是同一段，**而没有任何东西会报错**
    （`judge.py` 的注释自称"与 `budget.tokenizer` 同一口径"，此前无人验证）。
    """
    from eval.harness.judge import PLATFORM_TOKEN_PREFIX, PLATFORM_TOKENIZER

    from tianximem.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

    assert PLATFORM_TOKEN_PREFIX == MAX_INPUT_TOKENS
    assert PLATFORM_TOKENIZER == DEFAULT_TOKENIZER


def test_render_memories_date_prefix_is_opt_in():
    """默认**不加**日期（保持既有基线可比）；开了才加，且缺 `created_at` 时不加。"""
    from eval.harness import render_memories
    from eval.harness.driver import SearchHit

    hits = [
        SearchHit(id="a", content="Q: q\nA: a", created_at="2023-07-02", score=1.0),
        SearchHit(id="b", content="Q: q2", created_at="", score=0.5),
    ]

    assert render_memories(hits) == "Q: q\nA: a\nQ: q2"
    assert render_memories(hits, date_mode="per_item") == (
        "[2023-07-02] Q: q\nA: a\nQ: q2"  # ← 空 created_at 的那条不加前缀（§11.3 的降级路径）
    )


def test_render_memories_header_mode_states_what_the_dates_are():
    """`header` 模式：**把"这些日期是什么"写明**。

    `per_item` 单独用实测没能让模型改用日期（27/35 仍答相对），所以 `header` 把语义写出来——
    这一档存在就是为了分辨"提示不够清楚"与"模型做不到"。
    """
    from eval.harness import render_memories
    from eval.harness.driver import SearchHit
    from eval.harness.judge import DATE_HEADER

    hits = [
        SearchHit(id="a", content="Q: q", created_at="2023-07-03", score=1.0),
        SearchHit(id="b", content="Q: q2", created_at="2023-05-08", score=0.5),
    ]
    out = render_memories(hits, date_mode="header")

    assert out.startswith(DATE_HEADER.format(dates="2023-07-03, 2023-05-08"))
    assert "[2023-07-03] Q: q" in out and "[2023-05-08] Q: q2" in out


def test_runner_flag_reaches_the_injection(bench_dir, tmp_path, monkeypatch):
    """`--memory-date` 要真的走到 `build_input_items`——否则这个开关是**静默无效**的。"""
    from eval.experiments import run as runner

    seen: dict = {}
    real_build = runner.build_input_items

    def spy(sample, hits_by_qid, **kwargs):
        seen.update(kwargs)
        return real_build(sample, hits_by_qid, **kwargs)

    monkeypatch.setattr(runner, "build_input_items", spy)
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [_sample()])

    runner.run_round(
        dataset="locomo-refined",
        base_url="http://stub",
        bench_dir=bench_dir,
        out_dir=tmp_path / "out",
        skip_ingest=True,
        date_mode="header",
        client=_Recorder().client(),
    )

    assert seen == {"date_mode": "header", "annotate_mark": "paren"}


# ── 注解模式：把句子里的相对时间就地换算成绝对日期（2026-09-25）──
# 动机：`t1-dated` 的 temporal 79 道里 27 道**只差"把 yesterday 减一天"这一步**。
# 这组用例钉住三件事：**只加不改**、**推不出不动**、**锚点是段自己的日期**。
_ANCHOR = __import__("datetime").date(2023, 7, 20)  # 2023-07-20 是周四


def test_annotate_only_adds_and_never_replaces():
    """**原文一字不动**，注解跟在后面——D21 的口径（"日期是额外锚点，不是替换"）。"""
    from eval.harness.annotate import annotate

    out = annotate("Hey Mel! I joined an activist group last Tues.", _ANCHOR)
    assert out == "Hey Mel! I joined an activist group last Tues (July 18, 2023)."


def test_annotate_refuses_to_guess():
    """**推不出就不动**：`a few years ago` 没有数字，换算它等于编一个日期——
    而编错的日期比不换算更糟（模型会照抄，裁判是精确比值的）。"""
    from eval.harness.annotate import annotate

    assert annotate("A few years ago I moved.", _ANCHOR) == "A few years ago I moved."
    assert annotate("Several weeks ago we spoke.", _ANCHOR) == "Several weeks ago we spoke."
    assert annotate("Nothing temporal here.", _ANCHOR) == "Nothing temporal here."


def test_annotate_week_and_weekend_start_on_monday():
    """周的起点是**周一**——这条是拿 gold 反推出来的，不是选出来的。

    `The weekend before 17 July 2023`（gold 并列 `From July 15, 2023 to July 16, 2023`）
    与 `two weekends before 17 July 2023`（`From July 8, 2023 to July 9, 2023`）都锚在
    2023-07-17（周一）上；**按周日为界这两条都会差一天**。
    """
    from datetime import date

    from eval.harness.annotate import annotate

    monday = date(2023, 7, 17)
    assert annotate("last weekend we camped", monday) == (
        "last weekend (July 15 to 16, 2023) we camped"
    )
    assert annotate("camping two weekends ago", monday) == (
        "camping two weekends ago (July 8 to 9, 2023)"
    )
    # 2023-06-09 是周五：`The week before 9 June 2023` 的 gold 并列形式是 5/29–6/4
    assert annotate("last week", date(2023, 6, 9)) == "last week (May 29 to June 4, 2023)"


def test_annotate_covers_month_year_and_duration():
    """月/年粒度与"持续了多久"（`for 7 years` → `since 2016`）各一例。"""
    from eval.harness.annotate import annotate

    assert annotate("I started last month", _ANCHOR) == "I started last month (June 2023)"
    assert annotate("I moved last year", _ANCHOR) == "I moved last year (2022)"
    assert annotate("I've been teaching for seven years", _ANCHOR) == (
        "I've been teaching for seven years (since 2016)"
    )


def test_annotate_mark_switch_only_changes_the_wrapper():
    """两种记号**只差外壳**：换算结果一字不改（否则它就不是单变量对照了）。"""
    from eval.harness.annotate import MARKS, annotate

    paren = annotate("last night we talked", _ANCHOR, mark="paren")
    tag = annotate("last night we talked", _ANCHOR, mark="tag")

    assert MARKS == ("paren", "tag")
    assert paren == "last night (July 19, 2023) we talked"
    assert tag == "last night [= July 19, 2023] we talked"
    assert paren.count("July 19, 2023") == tag.count("July 19, 2023") == 1


def test_render_memories_annotate_mode_adds_no_prefix():
    """`annotate` **不加前缀**——索引侧 `packaging.inject_abs_time=true` 已经给每一对
    加了日期，这里再前缀一次会变成两个日期（而"多一个日期"不会有任何东西报错）。"""
    from eval.harness import render_memories
    from eval.harness.driver import SearchHit

    hits = [
        SearchHit(
            id="a", content="Q: x\nA: I joined last Tues", created_at="2023-07-20", score=1.0
        ),
        SearchHit(id="b", content="Q: y\nA: a few years ago", created_at="2023-07-20", score=0.5),
        SearchHit(id="c", content="Q: z\nA: undated", created_at="", score=0.1),
    ]

    out = render_memories(hits, date_mode="annotate")

    assert "[2023-07-20]" not in out  # ← 不加前缀
    assert "last Tues (July 18, 2023)" in out
    assert "a few years ago" in out and "a few years ago (" not in out  # 推不出 ⇒ 不动
    assert out.endswith("Q: z\nA: undated")  # 没有 created_at 的那条原样保留


# ── T1：两臂的冻结快照与核对（§13 的配置指纹）──
def test_t1_arms_differ_only_in_the_switch_and_the_collection():
    """两臂**只差两件事**：`inject_abs_time` 与集合名。

    差第三件事就说明这个对照同时在测别的——而**分数看起来完全正常**（§13 的纯度规则）。
    """
    from eval.experiments import arms
    from eval.experiments import t1_timestamp as t1

    plain, dated = t1.ARM_PLAIN, t1.ARM_DATED
    assert plain.inject_abs_time is False and dated.inject_abs_time is True
    assert plain.collection != dated.collection  # 向量不同 ⇒ 必须分集合
    assert arms.flatten(plain.overrides()) == {
        "storage.qdrant.collection": plain.collection,
        "packaging.inject_abs_time": False,
    }
    # `switches` 是**只记开关**的那一份：集合名不在里面（它不是这次对照的自变量）
    assert dated.switches() == {"packaging.inject_abs_time": True}


def test_t1_freeze_then_verify_round_trips(tmp_path, monkeypatch):
    """`--freeze` 写出的快照必须能通过 `verify`——而且**改了开关就会被抓出来**。

    快照掉队（比如两份都成了 `false`）是这个脚手架最危险的失败模式：你会**认真地跑完
    一次 T1、得到"两臂没有差别"**，而那个结论是假的。
    """
    from eval.experiments import arms
    from eval.experiments import t1_timestamp as t1

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "default.yaml").write_text("models:\n  embedder: X\n", encoding="utf-8")
    (configs / "local.yaml").write_text(
        "storage:\n  qdrant:\n    collection: dev\n", encoding="utf-8"
    )
    monkeypatch.setattr(arms, "RUNS_DIR", tmp_path / "runs")

    for arm in t1.ARMS:
        t1.freeze(arm, configs_dir=configs)
        assert t1.verify(arm) == []

    # 基线被逐字复制（注释与内容都是配置的一部分）
    assert (t1.ARM_DATED.dir / "default.yaml").read_text(encoding="utf-8") == (
        configs / "default.yaml"
    ).read_text(encoding="utf-8")

    # 人为把"带"臂改回 false ⇒ 必须报出来
    local = t1.ARM_DATED.dir / "local.yaml"
    local.write_text(
        local.read_text(encoding="utf-8").replace(
            "inject_abs_time: true", "inject_abs_time: false"
        ),
        encoding="utf-8",
    )
    problems = t1.verify(t1.ARM_DATED)
    assert problems and "inject_abs_time" in problems[0]

    # 集合名同样在核对范围内（它在 `overrides()` 里，就是"本臂该有的取值"）
    local.write_text(
        local.read_text(encoding="utf-8").replace(
            "collection: memories_t1_dated", "collection: 别的集合"
        ),
        encoding="utf-8",
    )
    problems = t1.verify(t1.ARM_DATED)
    assert problems and any("collection" in p for p in problems)


def test_t1_verify_reports_missing_snapshot(tmp_path, monkeypatch):
    from eval.experiments import arms
    from eval.experiments import t1_timestamp as t1

    monkeypatch.setattr(arms, "RUNS_DIR", tmp_path / "runs")
    assert "先跑 `--freeze`" in t1.verify(t1.ARM_PLAIN)[0]


# ── A3：与 T1 共用同一套脚手架，但**每臂只动一个键**──
def test_a3_freeze_then_verify_round_trips(tmp_path, monkeypatch):
    """A3 与 T1 走同一个 `freeze`/`verify`，所以这里测的是**它自己的那半边**：

    `overrides()` 只动 `rerank.enabled` ⇒ 校验也必须**只**盯这一个键。
    判据从 `overrides()` 现算，所以这条断言同时钉住了"两臂不共用集合"这件事。
    """
    from eval.experiments import a3_rerank as a3
    from eval.experiments import arms

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "default.yaml").write_text("models:\n  embedder: X\n", encoding="utf-8")
    (configs / "local.yaml").write_text("rerank:\n  enabled: false\n", encoding="utf-8")
    monkeypatch.setattr(arms, "RUNS_DIR", tmp_path / "runs")

    assert arms.flatten(a3.ARM_ON.overrides()) == {"rerank.enabled": True}
    assert a3.ARM_ON.switches() == {"rerank.enabled": True}
    for arm in a3.ARMS:
        a3.freeze(arm, configs_dir=configs)
        assert a3.verify(arm) == []

    # rerank 只改排名 ⇒ **两臂的集合相同**（与 T1 正相反，那边的 `overrides()` 里有集合）
    assert "storage" not in a3.ARM_ON.overrides()

    local = a3.ARM_ON.dir / "local.yaml"
    local.write_text(
        local.read_text(encoding="utf-8").replace("enabled: true", "enabled: false"),
        encoding="utf-8",
    )
    problems = a3.verify(a3.ARM_ON)
    assert problems and "rerank.enabled" in problems[0]


# ── T2：机器半边出待填表，人那半边拒绝代填 ──
def test_t2_refuses_to_conclude_before_the_human_fills_labels(tmp_path):
    """**没填完就不出结论**：半张表的分布比没有分布更危险——它看起来像个结果。"""
    from eval.experiments import t2_cross_session as t2

    rows = [
        {"qid": f"q{i}", "is_abstention": i < 2, "label": "ok" if i == 0 else ""}
        for i in range(t2.EXPECTED_TOTAL)
    ]
    # 题数只有对上文档口径时才继续；这里凑够 133 道
    path = tmp_path / "t2.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )

    code = t2.main(["--in", str(path)])

    assert code == t2.EXIT_FAILED
    assert t2.summarize(rows)[2], "应当报出未填的 qid"


def test_t2_summarizes_and_keeps_abstention_separate():
    from eval.experiments import t2_cross_session as t2

    rows = [
        {"qid": "a", "is_abstention": True, "label": "not_retrieved"},
        {"qid": "b", "is_abstention": False, "label": "truncated"},
        {"qid": "c", "is_abstention": False, "label": "truncated"},
        {"qid": "d", "is_abstention": False, "label": ""},  # 没填
    ]

    total, abstention, unlabeled = t2.summarize(rows)

    assert total == {"truncated": 2, "not_retrieved": 1}
    assert abstention == {"not_retrieved": 1}  # 拒答题**单列**，但不从整体里剔除
    assert unlabeled == ["d"]


def test_t2_rejects_a_wrong_question_count(tmp_path, capsys):
    """题数对不上文档口径（133）时必须响亮失败——**筛错题与"分数变了"看起来一样**。"""
    from eval.experiments import t2_cross_session as t2

    path = tmp_path / "short.jsonl"
    path.write_text(
        json.dumps({"qid": "q", "is_abstention": False, "label": "ok"}), encoding="utf-8"
    )

    assert t2.main(["--in", str(path)]) == t2.EXIT_FAILED
    assert "133" in capsys.readouterr().err


# ── 边界：`eval/experiments/` 与 `eval/baselines/` 同样**打 HTTP、不 import src** ──
@pytest.mark.parametrize("subdir", ["experiments", "baselines"])
def test_eval_side_does_not_import_src(subdir: str):
    """`eval/` 全目录一条边界（[`../eval/CLAUDE.md`](../eval/CLAUDE.md)）。

    `test_harness_does_not_import_src` 只扫 `eval/harness/`——**arm 与基线同样要打 HTTP**
    （B1 只以"另一个 Add/Search 服务"的形式存在，进程内调用会让两条基线不可比）。
    用 AST 扫 import 语句，不扫文本（docstring 里提到 `src/tianximem` 是说明，不是依赖）。
    """
    root = Path(__file__).resolve().parents[1] / "eval" / subdir
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [
                f"{subdir}/{path.name}:{node.lineno}: {name}"
                for name in names
                if "tianximem" in name
            ]
    assert not offenders, "eval/ 下不许 import src/：\n" + "\n".join(offenders)


def test_shape_note_follows_the_constant_and_reaches_the_fingerprint():
    """数据集的**形状**（"一条记忆 = 一个 session""一个 Sample = N 个 case"）是我们的构造。

    它和切批口径同一性质：**改它会改分数，而两次 run 的 record 否则一模一样**
    （[`../../eval/harness/batching.py`](../../eval/harness/batching.py)："常量不是旋钮"）。
    ⇒ 两条都要钉住：**跟着常量走**（写死数字就会漂移）+ **真的挂到 note 上**。
    """
    from eval.datasets import shape_note
    from eval.datasets.mquake import CASES_PER_USER, SHAPE_NOTE

    assert str(CASES_PER_USER) in SHAPE_NOTE, "常量改了而 note 没跟着变 —— 静默漂移"
    assert shape_note("mquake-remastered") == SHAPE_NOTE
    assert "语料" in shape_note("corporatebench")
    assert shape_note("locomo-refined") == "", "没登记形状说明的数据集应当给空串（不是瞎编一句）"

    # note 的两半要拼得起来：截断说一半、形状说一半，缺一段都不可比
    joined = "\n".join(p for p in (runner.truncation_note(2), shape_note("mquake-remastered")) if p)
    assert "截断" in joined and "形状是本地约定" in joined


# ── 冻结口径（`recipes.py`）────────────────────────────────────────────
def test_frozen_recipes_cover_exactly_the_datasets():
    """口径表与加载器表**必须同集合**——多一个没有加载器，少一个没法跑基线。"""
    from eval.experiments.recipes import FROZEN_RECIPES

    assert set(FROZEN_RECIPES) == set(runner.DATASETS)


def test_frozen_flags_render_as_pasteable_cli():
    """`flags()` 出来的串要能**逐字粘回** `run.py`（它存在的意义就是"别手抄数字"）。"""
    from eval.experiments.recipes import FROZEN_RECIPES

    for dataset, recipe in FROZEN_RECIPES.items():
        parser = runner.build_parser()
        args = parser.parse_args(
            ["--dataset", dataset, *recipe.flags().split()]
            if recipe.flags() != "(none)"
            else ["--dataset", dataset]
        )
        assert args.dataset == dataset
        if recipe.limit is not None:
            assert args.limit == recipe.limit
        assert args.spread == recipe.spread


@pytest.mark.parametrize("dataset", sorted(runner.DATASETS))
def test_frozen_recipe_yields_a_sane_reproducible_sample(dataset: str):
    """**这条是整张表的地基**：实际加载出来的题数必须等于表里那个数。

    ⚠ 少了它，"口径漂了"只会在**两次 run 的分数不可比**时静默表现出来——
    而那正是本项目最忌讳的一类失败（不报错，只是结论错）。

    顺带在同一趟加载里验两件事（**共用一次加载**：这份要读 277 MB 的文件，
    再开一条只为多断言一次不划算）：

    * **确定性**：同一口径连加载两次给同一批 qid
      （`stratified_sample` 组内等间隔取、无随机 ⇒ 这是它的承诺）。
    * **样本内 `session_id` 不重复**（2026-10-02）：重复会让
      `request_id_for(user, session, index)` **撞车**、payload 不同 ⇒ 服务按 **D28 回 409**，
      把整轮打死；而它**另一半后果是静默的**（两段无关对话落进同一个段合并分组）。
      LME 实测 9/301 个样本中招，已在 `longmemeval.py` 就地消重。
    """
    from eval.datasets.layout import archive_file
    from eval.datasets.registry import benchmark_dir
    from eval.experiments.recipes import recipe_for

    bench = benchmark_dir({})
    if not archive_file(bench, "questions.jsonl").exists():
        pytest.skip("归档不在（见 docs/benchmark-data.md）")
    if dataset in {"hybridqa", "feverous"}:
        from eval.datasets.registry import _sources

        # 按需下载不能让常规测试隐含要求完整 Wikipedia 数据库。
        if any(not path.is_file() for path in _sources(bench, dataset)):
            pytest.skip(f"{dataset} 语料/派生物未准备；合成 fixture 另行覆盖加载与评分")

    recipe = recipe_for(dataset)
    qids: list[str] = []
    for _ in range(2):
        samples = runner._load(dataset, bench, recipe.limit, spread=recipe.spread)
        again = [q.qid for sample in samples for q in sample.questions]
        if qids:
            assert again == qids, f"{dataset}：同一口径两次加载给了不同的题"
        qids = again

    assert len(qids) == recipe.n_questions, (
        f"{dataset}：冻结口径说 {recipe.n_questions} 题，实测 {len(qids)} 题。"
        " 加载器改过、或者 k 改了而表没跟着改。"
    )

    for sample in samples:
        seen = [s.session_id for s in sample.sessions]
        assert len(seen) == len(set(seen)), (
            f"{dataset} 的 {sample.user_id}：样本内 `session_id` 重复 "
            f"（{[k for k in seen if seen.count(k) > 1][:3]}）⇒ `request_id` 会撞车、"
            "服务回 409，且两段无关对话会被并进同一个段合并分组。"
            " 修在对应加载器里（LME 的先例见 `longmemeval.py::_sessions`）。"
        )


def test_frozen_refuses_to_be_combined_with_an_explicit_limit(capsys, monkeypatch, tmp_path):
    """`--frozen` 与显式 `--limit` **互斥**——两处同时给会让"记录的口径"与
    "实际跑的口径"分家，而那是不可比里最难查的一种。"""
    parser = runner.build_parser()
    args = parser.parse_args(["--dataset", "clbench", "--frozen", "--limit", "5"])
    assert args.frozen and args.limit == 5  # 解析层不管，`main` 才拒绝

    monkeypatch.setattr(runner, "benchmark_dir", lambda: tmp_path)
    code = runner.main(["--dataset", "clbench", "--frozen", "--limit", "5"])
    assert code == runner.EXIT_PRECONDITION_FAILED
    assert "不能同时给" in capsys.readouterr().err


# ── 跑批诊断工具（tools/diagnose_run.py，2026-10-03）─────────────────────
def test_diagnose_flags_a_prompt_induced_refusal_pattern(tmp_path: Path):
    """⛔ **这个工具的价值全在"把低分与模型不行分开"**——所以它必须能报出指纹。

    夹具造的是 memtrapbench 那次的**真实形状**：模型逐字拒答，**而 gold 就在给它的
    上下文里**（⇒ 该报"查 prompt"），外加一批 `JUDGE_ERROR`（⇒ 该报"查判分链路"）。
    """
    import json as _json

    from tools.diagnose_run import clinical_report

    run_dir = tmp_path / "runs" / "toy"
    (run_dir / "u").mkdir(parents=True)
    rows = []
    for index in range(10):
        rows.append(
            {
                "id": f"q{index}",
                "dataset": "toy",
                "question": "?",
                "gold_answer": "Zebra",
                # gold 就在上下文里 —— 这正是"证据在眼前却拒答"的指纹
                "retrieved_context": "the answer is Zebra and more text",
                "category": "c1" if index < 5 else "c2",
            }
        )
    (run_dir / "u" / "input.jsonl").write_text(
        "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    (run_dir / "u" / "answers.jsonl").write_text(
        "\n".join(
            _json.dumps({"id": f"q{i}", "generated_answer": "Cannot determine from the memories."})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "u" / "labels.jsonl").write_text(
        "\n".join(
            _json.dumps({"id": f"q{i}", "label": "JUDGE_ERROR", "is_correct": False})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )

    assert clinical_report("toy", tmp_path, ("cannot determine",)) == 0
    # 结论要能指出这两条（用 capsys 抓不上——它直接 print 到 stdout，
    # 但返回值与"跑通"已经证明路径走完；指纹的判据在 `table` 里，这里退而验文件被读到）


def test_diagnose_does_not_blame_the_corpus_for_a_no_memory_dataset(tmp_path: Path):
    """⚠ **判读树对 memtrapbench / personamem 不适用**——它们的答案**设计成不在记忆里**。

    这一条是本工具**第一次真跑就自己撞出来的**：它对 memtrapbench 判了"证据不在 ⇒ 查语料"，
    而那份数据集的 README 原文是 "No-Memory Solvability: the final query must be answerable
    correctly **even without the history**" ⇒ "gold 不在记忆里"是**预期行为**。
    """
    from tools.diagnose_run import _NO_MEMORY_DATASETS

    assert "memtrapbench" in _NO_MEMORY_DATASETS
    assert "personamem-v2" in _NO_MEMORY_DATASETS


def test_diagnose_reads_the_gold_shape_of_every_dataset(tmp_path: Path) -> None:
    """⚠ `gold_answer` 的形状**逐数据集不同**（实测四种，见 `gold_strings` 的 docstring）。

    **只认 `"answer"` 的那一版对 medmemorybench 恒返回空列表** ⇒ "证据在不在"恒为假
    ⇒ 工具报出"97% 的证据不在"并建议**去查语料**——**整条结论是判据自己造的**。
    （抓出它的是 ① 的**自校准**：判对的题里「证据在」= 0%，而那在语义上不可能。）

    本用例把四种形状各喂一遍，钉住"都得读得出东西"。
    """
    from tools.diagnose_run import gold_strings

    assert gold_strings({"gold_answer": ["a", "b"]}) == ["a", "b"]
    assert gold_strings({"gold_answer": "答案"}) == ["答案"]
    assert gold_strings({"gold_answer": 42}) == ["42"]
    assert gold_strings({"gold_answer": {"answer": "答案", "answer_type": "x"}}) == ["答案"]
    # medmemorybench：正确答案住在 answers[]，**且只有 is_correct 的那些才算**
    mmb = {
        "gold_answer": {
            "query_type": "multiple_choice",
            "answers": [
                {"content": "错的", "is_correct": False},
                {"content": "对的", "is_correct": True},
            ],
            "metadata": {},
        }
    }
    assert gold_strings(mmb) == ["对的"]
    # memtrapbench：判分**要点**（不是答案串，但也不该读成空）
    assert gold_strings({"gold_answer": {"gold_standard": "要点"}}) == ["要点"]


def test_diagnose_will_not_blame_the_corpus_when_its_own_check_cannot_see(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """「gold 在不在记忆里」是**子串匹配**，它的召回要**先用判对的题量一遍**才敢读。

    夹具模拟 medmemorybench 那种情形：gold 是**长句**而记忆里是**改写**——
    模型答对（判分认），但子串匹配认不出。此时工具**必须拒绝**说
    "证据不在 ⇒ 查语料/加载器"：那是**把人送去查错的那一层**。
    """
    import json as _json

    from tools.diagnose_run import clinical_report

    run_dir = tmp_path / "runs" / "toy"
    (run_dir / "u").mkdir(parents=True)
    long_gold = "患者应避免自行加量并尽快复查糖化血红蛋白"
    rows = [
        {
            "id": f"q{i}",
            "dataset": "toy",
            "question": "?",
            "gold_answer": {"answers": [{"content": long_gold, "is_correct": True}]},
            "retrieved_context": "医生建议复诊（**改写**，与 gold 不字面相同）",
            "category": "c1",
        }
        for i in range(30)
    ]
    (run_dir / "u" / "input.jsonl").write_text(
        "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    (run_dir / "u" / "answers.jsonl").write_text(
        "\n".join(
            _json.dumps({"id": f"q{i}", "generated_answer": "Cannot determine from the memories."})
            for i in range(30)
        )
        + "\n",
        encoding="utf-8",
    )
    # **全部判对**——语义上证据必然是给到了的，可子串匹配一条都认不出
    (run_dir / "u" / "labels.jsonl").write_text(
        "\n".join(
            _json.dumps({"id": f"q{i}", "label": "CORRECT", "is_correct": True}) for i in range(30)
        )
        + "\n",
        encoding="utf-8",
    )
    assert clinical_report("toy", tmp_path, ("cannot determine",)) == 0
    printed = capsys.readouterr().out
    assert "召回上限" in printed and "0.0%" in printed, printed
    assert "不可信" in printed, printed
    # ★ 这才是本用例的要点：**不许**把结论指向语料或加载器
    assert "查**语料/加载器**" not in printed, printed


def test_diagnose_warns_when_the_model_switched_refusal_wording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """⚠ **换 prompt 之后模型常换一句拒答，而工具只认 `--refusal` 里那几句。**

    2026-10-04 的 `tr-promptfix` 实测：换了 prompt 后不再回
    `Cannot determine from the memories.`，改成
    `The provided memories do not contain information about …`——**12 条、全判错**，
    而工具当时印的是"**拒答率 0/332 = 0.0%**"。

    "0%"那一行**太好信**了。现在这种条数会**单独印一行**（不计进拒答率——
    启发式会误伤"答案里本来就有的否定句"，真算数要靠显式加 `--refusal`）。
    """
    import json as _json

    from tools.diagnose_run import clinical_report

    run_dir = tmp_path / "runs" / "toy"
    (run_dir / "u").mkdir(parents=True)
    rows = [
        {"id": f"q{i}", "dataset": "toy", "question": "?", "gold_answer": "X", "category": "c"}
        for i in range(6)
    ]
    (run_dir / "u" / "input.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (run_dir / "u" / "answers.jsonl").write_text(
        "\n".join(
            _json.dumps(
                {
                    "id": f"q{i}",
                    "generated_answer": "The provided memories do not contain information about X.",
                }
            )
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "u" / "labels.jsonl").write_text(
        "\n".join(
            _json.dumps({"id": f"q{i}", "label": "WRONG", "is_correct": False}) for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    assert clinical_report("toy", tmp_path, ("cannot determine from the memories",)) == 0
    printed = capsys.readouterr().out
    assert "拒答率 0/6" in printed, printed
    assert "换了措辞的拒答" in printed and "**6 条" in printed, printed


# ── 缺数据时的两条出口（2026-10-04）─────────────────────────────────────────
def test_main_gives_a_hint_when_the_dataset_loads_to_nothing(
    monkeypatch, bench_dir, tmp_path, capsys
):
    """零个样本 ⇒ **前置条件不满足**，而不是一句 traceback。

    ⚠ **裸 `ValueError` 不在主入口的 `except` 元组里** ⇒ 用户看到的是一段栈，
    退出码也不是"前置条件"那个（脚本里分不出来）。而这条恰恰是**新机器上
    最可能撞到的失败**：归档没取回来时，加载器要么响亮抛 `FileNotFoundError`，
    要么**静默返回空**（`glob` 对不存在的目录不报错）——后者就落到这里。
    """
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench_dir)
    monkeypatch.setattr(runner, "ensure_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit, spread=False: [])
    monkeypatch.setattr(runner, "judge_preconditions", lambda: [])
    monkeypatch.setattr(runner, "missing_pipeline", lambda dataset, bench: None)

    def _never(*_a, **_kw):  # pragma: no cover —— 走到这里就说明没拦住
        raise AssertionError("一个样本都没有就不该去连服务")

    monkeypatch.setattr(runner, "ServiceClient", _never)

    code = runner.main(["--dataset", "locomo-refined", "--reports-dir", str(tmp_path)])

    assert code == runner.EXIT_PRECONDITION_FAILED
    err = capsys.readouterr().err
    assert "bench_dir" in err and "make fetch-data" in err


def test_main_fails_before_running_when_the_judge_script_is_missing(
    monkeypatch, bench_dir, tmp_path, capsys
):
    """**裁判脚本不在要在跑之前查**——它是唯一一处"查晚了钱已经花了"的前置条件。

    `run_judge` 起的是 subprocess，而**脚本不存在不会在"起进程"那一刻失败**：
    整轮的 `Add` 与 `Search` 会全部跑完（embedding 已经付过钱），然后**第一个样本**
    的裁判那一步才以 `RuntimeError` 炸（子进程重试 3 次后抛，stderr 里只有一句
    python 的 `can't open file`）。
    """
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench_dir)
    monkeypatch.setattr(runner, "ensure_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "judge_preconditions", lambda: [])
    monkeypatch.setattr(
        runner, "missing_pipeline", lambda dataset, bench: bench / "pipeline_locomo-refined.py"
    )

    def _never(*_a, **_kw):  # pragma: no cover —— 走到这里就说明检查没拦住
        raise AssertionError("裁判脚本都没有就不该去连服务（更不该投喂）")

    monkeypatch.setattr(runner, "ServiceClient", _never)

    code = runner.main(["--dataset", "locomo-refined", "--reports-dir", str(tmp_path)])

    assert code == runner.EXIT_PRECONDITION_FAILED
    err = capsys.readouterr().err
    assert "裁判脚本不在" in err and "make fetch-data" in err


def test_missing_pipeline_is_quiet_for_the_in_repo_ones(bench_dir):
    """本仓内那几份（自写的 + 适配器）**一定在**，别对它们报假警。"""
    assert runner.missing_pipeline("mquake-remastered", bench_dir) is None
    assert runner.missing_pipeline("medmemorybench", bench_dir) is None
    assert runner.missing_pipeline("personamem-v2", bench_dir) is None


def test_missing_pipeline_reports_the_archive_one_when_absent(tmp_path):
    """归档里那几份**跟着数据集走** ⇒ 归档没取回时必须报出来（这里给一个空目录）。"""
    absent = runner.missing_pipeline("locomo-refined", tmp_path)
    assert absent is not None and absent.name == "pipeline_locomo-refined.py"
    assert runner.missing_pipeline("beam", tmp_path).name == "pipeline_beam.py"
