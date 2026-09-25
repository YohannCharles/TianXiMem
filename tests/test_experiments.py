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
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit: [_sample()])

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
            assert body["messages"][0]["content"] == "m0"  # 源序不重排

    # 逐题原始产出落在 per-user 目录里（runs/ 被 .gitignore 覆盖，见 eval/reports/CLAUDE.md）
    assert (tmp_path / "runs" / "r1" / "conv-1" / "labels.jsonl").exists()


def test_run_round_skips_ingest_when_asked(bench_dir, tmp_path, monkeypatch):
    """`--skip-ingest` 只掉 Add，**Search 一次都不能少**——少检索就是静默漏题。"""
    recorder = _Recorder()
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit: [_sample()])

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
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit: [_sample()])
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
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit: [_sample()])
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
    monkeypatch.setattr(runner, "_load", lambda dataset, bench, limit: [_sample()])
    monkeypatch.setattr(runner, "judge_preconditions", lambda: ["AML_BASE_URL", "AML_MODEL"])

    def _never(*_a, **_kw):  # pragma: no cover —— 走到这里就说明检查没拦住
        raise AssertionError("前置条件没过就不该去连服务")

    monkeypatch.setattr(runner, "ServiceClient", _never)

    code = runner.main(["--dataset", "locomo-refined", "--reports-dir", str(tmp_path)])

    assert code == runner.EXIT_PRECONDITION_FAILED
    err = capsys.readouterr().err
    assert "AML_BASE_URL" in err and "--env-file" in err


# ── T1：两臂的冻结快照与核对（§13 的配置指纹）──
def test_t1_arms_differ_only_in_the_switch_and_the_collection():
    """两臂**只差两件事**：`inject_abs_time` 与集合名。

    差第三件事就说明这个对照同时在测别的——而**分数看起来完全正常**（§13 的纯度规则）。
    """
    from eval.experiments import t1_timestamp as t1

    plain, dated = t1.ARM_PLAIN, t1.ARM_DATED
    assert plain.inject_abs_time is False and dated.inject_abs_time is True
    assert plain.collection != dated.collection  # 向量不同 ⇒ 必须分集合
    assert _flatten(plain.overrides()) == {
        "storage.qdrant.collection": plain.collection,
        "packaging.inject_abs_time": False,
    }
    assert t1._switches(dated) == {"packaging.inject_abs_time": True}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        out.update(_flatten(value, f"{path}.") if isinstance(value, dict) else {path: value})
    return out


def test_t1_freeze_then_verify_round_trips(tmp_path, monkeypatch):
    """`--freeze` 写出的快照必须能通过 `verify`——而且**改了开关就会被抓出来**。

    快照掉队（比如两份都成了 `false`）是这个脚手架最危险的失败模式：你会**认真地跑完
    一次 T1、得到"两臂没有差别"**，而那个结论是假的。
    """
    from eval.experiments import t1_timestamp as t1

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "default.yaml").write_text("models:\n  embedder: X\n", encoding="utf-8")
    (configs / "local.yaml").write_text(
        "storage:\n  qdrant:\n    collection: dev\n", encoding="utf-8"
    )
    monkeypatch.setattr(t1, "RUNS_DIR", tmp_path / "runs")

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


def test_t1_verify_reports_missing_snapshot(tmp_path, monkeypatch):
    from eval.experiments import t1_timestamp as t1

    monkeypatch.setattr(t1, "RUNS_DIR", tmp_path / "runs")
    assert "先跑 `--freeze`" in t1.verify(t1.ARM_PLAIN)[0]


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
    用 AST 扫 import 语句，不扫文本（docstring 里提到 `src/tianxi_am` 是说明，不是依赖）。
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
                if "tianxi_am" in name
            ]
    assert not offenders, "eval/ 下不许 import src/：\n" + "\n".join(offenders)
