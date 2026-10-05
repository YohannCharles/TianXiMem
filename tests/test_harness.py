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
    ADD_SHAPES,
    DEFAULT_ADD_SHAPE,
    LABEL_RULES,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES_PER_BATCH,
    ServiceClient,
    batches,
    build_input_items,
    build_record,
    config_fingerprint,
    render_memories,
    request_id_for,
    run_judge,
    shape_batch,
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
    """**源序切批、不重排**——`request_id` 里带着批序号（服务端位置 = `(request_id, local_index)`），
    重排会让同一 `request_id` 对应另一批消息。"""
    # ⚠ 输入是**渲染后的 payload**（`batches` 要按 `content` 长度卡 per-add 预算）。
    got = batches(_shaped("locomo-refined", _messages(45)))
    assert [len(b) for b in got] == [20, 20, 5]
    # `_messages` 是 user/assistant 交替 ⇒ 标签跟着在 Sam/Rae 之间换（`@speaker` 规则）。
    assert [m["content"] for b in got for m in b] == [
        f"{'Sam' if i % 2 == 0 else 'Rae'}: m{i}" for i in range(45)
    ]
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
    message 只有 `role` / `content` / 可选 `timestamp`（D16 的 canonical 形状）。

    ⚠ 正文带前缀是**缺省的官方形态**（[`add_shape.py`](../eval/harness/add_shape.py)）——
    这里没传 `dataset`，落到缺省规则 `user:` / `assistant:`（全量占比 82.9% 的那条）。
    """
    recorder = _Recorder()
    recorder.client().add(
        request_id="rid", user_id="u1", session_id="s1", messages=_shaped("", _messages(2))
    )
    path, body = recorder.requests[0]
    assert path == "/add"
    assert set(body) == {"request_id", "user_id", "session_id", "messages"}
    assert body["messages"][0] == {"role": "user", "content": "user: m0", "timestamp": 1000}
    assert body["request_id"] == "rid"


def test_add_payload_native_shape_is_the_datasets_own():
    """`native` = 改之前的形态（数据集给什么 role 就发什么、正文不加前缀）——**只作对照**。"""
    recorder = _Recorder()
    client = ServiceClient(
        "http://stub",
        client=httpx.Client(transport=httpx.MockTransport(recorder.handler)),
        add_shape="native",
    )
    client.add(
        request_id="rid",
        user_id="u1",
        session_id="s1",
        messages=_shaped("locomo-refined", _messages(2), shape="native"),
    )
    assert recorder.requests[0][1]["messages"][0] == {
        "role": "user",
        "content": "m0",
        "timestamp": 1000,
    }


def test_message_without_timestamp_omits_the_key():
    """`timestamp` 缺省时**不能发 `null`**——`event_time` 为 NULL 时 `created_at` 发 `""` 是
    §11.3 那条**有定义的降级路径**，而 `null` 走的是另一条。"""
    recorder = _Recorder()
    recorder.client().add(
        request_id="r",
        user_id="u",
        session_id="s",
        messages=_shaped("locomo-refined", (Message(role="user", content="x"),)),
    )
    assert "timestamp" not in recorder.requests[0][1]["messages"][0]


# ── 官方 add 形态（`add_shape.py`）────────────────────────────────────────
#
# ⚠ **这些断言里的每一个标签都取自官方流量的原文**，不是我们编的——见
# [`eval/harness/add_shape.py`](../eval/harness/add_shape.py) 的实测表。
def _shaped(dataset: str, messages, *, speaker_names=("Sam", "Rae"), shape=DEFAULT_ADD_SHAPE):
    return shape_batch(messages, dataset=dataset, speaker_names=speaker_names, shape=shape)


def test_official_is_the_default_shape():
    """线上就是官方形态 ⇒ 缺省必须与线上一致。改这个缺省等于换输入。"""
    assert DEFAULT_ADD_SHAPE == "official"
    assert "official" in ADD_SHAPES and "native" in ADD_SHAPES


@pytest.mark.parametrize(
    ("dataset", "role", "label"),
    [
        ("locomo-refined", "user", "Sam"),  # canary `Caroline: …`
        ("locomo-refined", "assistant", "Rae"),  # canary `Melanie: …`
        ("longmemeval-s", "user", "User"),  # 正文逐字命中
        ("longmemeval-s", "assistant", "Assistant"),
        ("clbench", "user", "user"),  # canary
        ("clbench", "assistant", "assistant"),
        ("beam", "user", "Sam"),  # canary `Christina Baker: …`（名字由加载器从 persona 取）
        ("beam", "assistant", "Assistant"),
        ("personamem-v2", "user", "User"),  # canary
        ("personamem-v2", "assistant", "Assistant"),
        ("mquake-remastered", "user", "Corpus"),  # canary `Corpus: {json}`
        ("mquake-remastered", "assistant", "Corpus"),
        ("memtrapbench", "user", "user"),  # canary
        ("memtrapbench", "assistant", "assistant"),
        ("corporatebench", "user", "document"),  # 内容归属（zenithlabs 邮件）
        ("corporatebench", "assistant", "document"),
        ("medmemorybench", "user", "user"),  # 内容归属（中文医患，占全量 73%）
        ("medmemorybench", "assistant", "assistant"),
        ("tempreason", "user", "source"),  # canary `source: …`
        ("tempreason", "assistant", "source"),
    ],
)
def test_official_label_rules_match_the_measured_traffic(dataset, role, label):
    """逐数据集的标签**对着官方原文钉住**——漂了就是"输入变了"，而分数看不出这件事。"""
    got = _shaped(dataset, (Message(role=role, content="x"),))
    assert got[0]["content"] == f"{label}: x"
    assert got[0]["role"] == role


def test_every_dataset_with_a_loader_has_a_label_rule():
    """**新加数据集必须显式登记标签**——落进缺省规则不会报错，只会静默换个输入。"""
    from eval.experiments.run import DATASETS

    assert set(DATASETS) <= set(LABEL_RULES), sorted(set(DATASETS) - set(LABEL_RULES))


def test_system_is_folded_into_the_user_side():
    """**逐字节证据**：`clbench.jsonl:24` 的 raw `messages[0]` 是 `role="system"`、
    正文 `You are a Chemical Engineer…`，而官方发的是 `role="user"` + `user: You are …`。

    ⇒ 官方的 role 取值域**只有 user / assistant**，`system` 是被折进 user 侧的。
    这条**不是排版**：折叠把 `system` 折成 `user`，会改变"配得上对"的位置（D29）。
    """
    got = _shaped("clbench", (Message(role="system", content="You are a Chemical Engineer"),))
    assert got[0] == {"role": "user", "content": "user: You are a Chemical Engineer"}

    got = _shaped("personamem-v2", (Message(role="system", content="persona"),))
    assert got[0] == {"role": "user", "content": "User: persona"}


def test_alluser_collapses_every_role():
    """判分池的形态（23/23 个判分 user 全是 `role: user`）：说话人只活在正文标签里。"""
    got = _shaped(
        "locomo-refined",
        (Message(role="user", content="a"), Message(role="assistant", content="b")),
        shape="alluser",
    )
    assert [m["role"] for m in got] == ["user", "user"]
    assert [m["content"] for m in got] == ["Sam: a", "Rae: b"]


def test_native_is_the_untouched_dataset_shape():
    got = _shaped("locomo-refined", _messages(2), shape="native")
    assert got == [m.to_add_payload() for m in _messages(2)]


def test_locomo_first_message_matches_the_official_canary_verbatim():
    """**逐字钉住**：数据集里的 `{"speaker": "Caroline", "dia_id": "D1:1",
    "text": "Hey Mel! Good to see you! How have you been?"}`
    ——官方发出去的正是同一条 canary：

        [locomo_refined][session_1][D1:1][text] Caroline: Hey Mel! Good to see you!
        How have you been?

    （`[<dataset>][session_N][D<k>:<i>][text] ` 那一段是**那 132 条 canary 专有的**，
    主干流量没有；两处都带的只有 `Caroline: `。）

    ⚠ 数据集自己**不带**这个名字——`locomo_refined.json` 的 `text` 里没有它 ⇒
    前缀是**加出来的**，也就是这个用例钉的那一步。
    """
    got = _shaped(
        "locomo-refined",
        (Message(role="user", content="Hey Mel! Good to see you! How have you been?"),),
        speaker_names=("Caroline", "Melanie"),
    )
    assert got[0]["content"] == "Caroline: Hey Mel! Good to see you! How have you been?"


def test_long_message_is_split_at_the_official_cap():
    """**官方单条硬上限 ≈8,000 字符**（实测 2,514 条正好 8,000、0 条超过）。

    切出来的片是**独立的、同 role 的消息**，**每片重新带 `<标签>: ` 前缀**——
    主干流量里逐字见过切点落在半个 token 上（`…"row_span": "` → `Corpus: 1"…`）。
    """
    text = "x" * 20_000
    got = _shaped("mquake-remastered", (Message(role="user", content=text),))
    assert len(got) == 3
    assert all(len(m["content"]) <= MAX_MESSAGE_CHARS for m in got)
    assert all(m["role"] == "user" for m in got)
    assert all(m["content"].startswith("Corpus: ") for m in got)
    # 去掉每片的前缀，拼回去必须与原文**逐字相同**——是切分，不是截断。
    assert "".join(m["content"][len("Corpus: ") :] for m in got) == text


def test_the_cap_counts_the_label_too():
    """上限是**连前缀一起**算的：切的是 `<标签>: <正文>`，不是正文。

    算错这一处，每片都会超出 `len(前缀)` 个字符——而超出的那几字符正好撞嵌入上限。
    """
    got = _shaped("corporatebench", (Message(role="user", content="y" * 20_000),))
    assert all(len(m["content"]) <= MAX_MESSAGE_CHARS for m in got)
    # 片长应当**贴着**上限（前缀短，正文刚好填满），而不是留出前缀那份余量。
    assert max(len(m["content"]) for m in got) == MAX_MESSAGE_CHARS


def test_native_shape_does_not_split():
    """`native` 是"数据集给的形状"——切分是**平台行为**，所以这一档不切。"""
    got = _shaped(
        "mquake-remastered", (Message(role="user", content="x" * 20_000),), shape="native"
    )
    assert len(got) == 1
    assert len(got[0]["content"]) == 20_000


def test_splitting_happens_before_batching():
    """**先 shape、再切批**——否则批界会落在错的条数上。

    一条 20,000 字符的消息切完是 3 条 ⇒ 单独就占满一批的三格（`20` 条上限内）。
    """
    messages = (Message(role="user", content="x" * 20_000),)
    payloads = _shaped("mquake-remastered", messages)
    assert len(payloads) == 3  # 一条 20,000 字符被切成 3 片
    assert len(batches(payloads)) == 1  # 3 片 × 8,000 = 24,000 < 32,000 ⇒ 仍在一批内

    # 而更多片就装不下了 ⇒ 会摊成多批（这正是官方对超长正文的做法：
    # 片落进**不同的 Add**，否则同 role 的片按 D24 会合并成一个超限的块）。
    # `"x" * 40_000` 空白分词只有 1 个"词"，所以这里显式喂词块来触发预算。
    many = tuple(
        Message(role="user", content=" ".join(["w"] * 900)) for _ in range(6)
    )  # 每条 ~900 词 ⇒ 一批最多 2 条（1,800 ≤ 2,000）
    payloads = _shaped("mquake-remastered", many)
    assert [len(b) for b in batches(payloads)] == [2, 2, 2]


def test_unknown_dataset_falls_back_to_the_dominant_rule():
    got = _shaped("no-such-dataset", _messages(2))
    assert [m["content"] for m in got] == ["user: m0", "assistant: m1"]


def test_unknown_shape_is_a_loud_failure():
    with pytest.raises(ValueError, match="未知的 add 形态"):
        _shaped("locomo-refined", _messages(1), shape="yolo")


def test_speaker_label_without_a_name_is_a_loud_failure():
    """`@speaker` 缺名字**不许静默**——少一个说话人名会让检索看不见"谁说的"，
    而屏幕上与"名字丢了"是同一个样子。"""
    with pytest.raises(ValueError, match="缺名字"):
        shape_batch(
            (Message(role="user", content="x"),),
            dataset="locomo-refined",
            speaker_names=("", "Rae"),
            shape="official",
        )


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


#: BEAM 那套 CLI 的桩——**`evaluate` 只认 `--judge-max-tokens`**（与 `pipeline_beam.py` 一致）。
#: 于是"`run_judge` 给 evaluate 传了 `--max-tokens`"会直接变成 argparse 报错 ⇒ **用例会红**。
_BEAM_STUB_PIPELINE = '''\
"""模仿 pipeline_beam.py 的 CLI 契约：answer 收 --max-tokens，evaluate 收 --judge-max-tokens。"""
import argparse, json

def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="command", required=True)
a = sub.add_parser("answer")
a.add_argument("--input", required=True)
a.add_argument("--output", required=True)
a.add_argument("--max-tokens", type=int, default=512)
e = sub.add_parser("evaluate")
e.add_argument("--input", required=True)
e.add_argument("--answers", required=True)
e.add_argument("--output", required=True)
e.add_argument("--judge-max-tokens", type=int, default=1024)
args = p.parse_args()

if args.command == "answer":
    with open(args.output, "a", encoding="utf-8") as fh:
        for item in rows(args.input):
            fh.write(json.dumps({"id": item["id"], "generated_answer": item["context"]}) + "\\n")
else:
    with open(args.output, "a", encoding="utf-8") as fh:
        for item in rows(args.input):
            fh.write(json.dumps({
                "id": item["id"],
                "question_type": item["question_type"],
                "llm_judge_score": 1.0 if len(item["rubric"]) == 1 else 0.5,
                "rubric_scores": [
                    {"rubric": r, "score": 1.0, "reason": "ok"} for r in item["rubric"]
                ],
                "judge_response": "{}",
            }) + "\\n")
'''


def test_run_judge_beam_uses_the_official_cli_and_maps_scores(tmp_path):
    """BEAM 走**官方 pipeline 的 CLI 契约**：`answer` 收 `--max-tokens`、`evaluate` **不收**。

    两条断言各自有牙齿：
    * 桩脚本的 `evaluate` 只认 `--judge-max-tokens` ⇒ 传错参数会 argparse 报错（用例红）；
    * 二值化口径：只有均分 1.0 才算 CORRECT，**真分原样留在 `label`/`judge_response` 里**
      （换阈值不用重跑裁判）。
    """
    path = tmp_path / "beam_stub.py"
    path.write_text(_BEAM_STUB_PIPELINE, encoding="utf-8")
    sample = Sample(
        user_id="beam-1",
        dataset="beam",
        sessions=(Session(session_id="1-s1", messages=(Message(role="user", content="hi"),)),),
        questions=(
            Question(
                qid="1-abstention-0", question="q1", gold=["only rubric"], category="abstention"
            ),
            Question(
                qid="1-temporal-0",
                question="q2",
                gold=["r1", "r2"],
                category="temporal_reasoning",
                is_abstention=False,
            ),
        ),
    )
    items = build_input_items(sample, {})
    assert items[0]["context"] == "", "没命中就是空串"
    assert items[1]["rubric"] == ["r1", "r2"], "金标要以 `rubric` 的形式进项"
    assert items[1]["question_type"] == "temporal_reasoning"

    results = run_judge(path, items, tmp_path / "out", dataset="beam")
    assert [r.qid for r in results] == ["1-abstention-0", "1-temporal-0"]
    assert results[0].is_correct is True and results[0].label == "SCORE=1.00"
    assert results[1].is_correct is False and results[1].label == "SCORE=0.50"


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
    # ⚠ 这个量**已不存在**（D24）⇒ `None` 必须**显式说明**理由，
    #   否则读的人会以为是漏填。
    assert "D24" in record.counters["note"]
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
    """**全目录最硬的一条边界**：harness 打 HTTP，**不 import `src/tianximem`**。

    走进程内调用会让 B1（ReFind）变成特例、两条基线不可比，而且**碰不到契约层**
    （[`../../eval/CLAUDE.md`](../../eval/CLAUDE.md)）。

    用 AST 扫 **import 语句**而不是扫文本——docstring 里提到 `src/tianximem`
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
                f"{path.name}:{node.lineno}: {name}" for name in names if "tianximem" in name
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
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
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

    monkeypatch.setattr(judge, "truncate_to_platform_prefix", lambda text: (text[:10], True))
    hit = SearchHit(id="a", content="Q: x\nA: y" * 100, created_at="", score=1.0)
    question = type("Q", (), {"qid": "q1", "question": "?", "gold": "g"})()
    sample = type(
        "S", (), {"questions": [question], "speaker_names": ("A", "B"), "dataset": "locomo-refined"}
    )()
    judge.build_input_items(sample, {"q1": [hit]})
    assert "被截到平台前缀" in capsys.readouterr().out


# ── `/add` 的**有界重试**（2026-10-02，实测被网关一次抖动打死过一整轮）──────
def _flaky_add(*, fail_times, status=500, exc=None):
    """前 `fail_times` 次失败、之后成功。返回 `(client, 每次请求的 body)`。

    ⚠ 记录的是 **body 原文**（不是"调了几次"）——本组用例的重点之一就是
    **重投的 payload 必须逐字相同**，只数次数是测不出来的。
    """
    from eval.harness import ServiceClient

    calls: list[object] = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) <= fail_times:
            if exc is not None:
                raise exc
            return httpx.Response(status, json={"detail": "boom"})
        return httpx.Response(200, json={"applied": True})

    return (
        ServiceClient("http://x", client=httpx.Client(transport=httpx.MockTransport(handler))),
        calls,
    )


def _add(client, **kw):
    return client.add(
        request_id=kw.get("request_id", "r1"),
        user_id="u",
        session_id="s",
        messages=[{"role": "user", "content": "x"}],
    )


def test_add_retries_server_errors_with_a_byte_identical_payload(monkeypatch):
    """5xx ⇒ 重投**同一 `request_id` + 同一 payload**（AML 也是这么做的，§2.2 的 32 次）。

    ⚠ **payload 变了就是 409**（D28 的响亮冲突）⇒ 所以比的是整个 body，不是调用次数。
    """
    monkeypatch.setattr("eval.harness.driver.ADD_RETRY_BACKOFF_S", 0.0)
    client, calls = _flaky_add(fail_times=2)

    assert _add(client) == {"applied": True}

    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]  # ← 逐字相同，409 不会发生


def test_add_retries_network_errors():
    """网络层失败（实测那条是 `peer closed connection without sending complete message body`）
    **也要重试**——它不是 4xx，是我方无从预防的瞬时故障。"""
    client, calls = _flaky_add(
        fail_times=1, exc=httpx.RemoteProtocolError("peer closed connection")
    )

    assert _add(client) == {"applied": True}
    assert len(calls) == 2


def test_add_does_not_retry_client_errors(monkeypatch):
    """⛔ **4xx 一律不重试**——那是我方 bug，重试只会把它藏起来。

    ⚠ **409 是这里最要紧的一格**：它的含义正是"同 `request_id` 不同 payload"（D28）,
    重试它等于**把整轮跑批建在一个已经分叉的批次表上**，而且一次都不会红。
    """
    monkeypatch.setattr("eval.harness.driver.ADD_RETRY_BACKOFF_S", 0.0)

    for status in (400, 409, 422):
        client, calls = _flaky_add(fail_times=99, status=status)
        with pytest.raises(httpx.HTTPStatusError):
            _add(client)
        assert len(calls) == 1, f"{status} 被重试了——那会把冲突藏起来"


def test_add_retries_are_bounded(monkeypatch):
    """重试**有界**：一直 5xx 就必须抛出去，不能永远转下去。

    （同 `judge._run(attempts=3)` 的口径——那里也是"幂等 + 有界"。）
    """
    monkeypatch.setattr("eval.harness.driver.ADD_RETRY_BACKOFF_S", 0.0)
    client, calls = _flaky_add(fail_times=99, status=503)

    with pytest.raises(httpx.HTTPStatusError):
        _add(client)
    assert len(calls) == 3

    from eval.harness.driver import ADD_RETRY_ATTEMPTS

    assert len(calls) == ADD_RETRY_ATTEMPTS


def test_token_counting_survives_literal_special_tokens():
    """⛔ 正文里出现**字面的** `<|endoftext|>` 不能让整轮跑批崩掉。

    2026-10-02 实测：LongMemEval 的 haystack 与 CorporateBench 的邮件正文里都有它，
    而 `o200k_base` 默认把这种字面量当**特殊 token** 并抛 `ValueError`
    ⇒ **两条链各自死在 `build_input_items` 里**，而且**重跑一次还死在同一题上**
    （数据没变）——现象像"这一题特别慢"，**极易被误判成网关问题**。

    ⚠ `src` 那份（`tokens.py::O20kCounter.count`）**一直是传** `disallowed_special=()`
    的，但 harness 不许 import `src/` ⇒ 这个字面量是**故意重复的第二份**，
    本用例是它唯一的守护。
    """
    from eval.harness.judge import _encode, _encoder

    try:
        encoder = _encoder()
    except Exception as exc:  # noqa: BLE001 — tiktoken 首次使用要联网取 BPE 文件
        pytest.skip(f"tiktoken 不可用（首次使用要联网预热）：{exc}")

    # 逐字包含那两个串——默认参数下这两行都会抛 ValueError
    for text in ("Q: 前文\nA: <|endoftext|> 正文里就带着它", "A: <|endofprompt|> 另一个"):
        assert len(_encode(encoder, text)) > 0

    # 端到端那一层也得过（它是真正会崩的那个函数）
    from eval.harness.judge import truncate_to_platform_prefix

    text = "Q: q\nA: <|endoftext|>"
    out, cut = truncate_to_platform_prefix(text)
    assert out == text and cut is False


# ── 部分分（`partial`）：**不进 overall** 的第二根尺（2026-10-03）──────────
def _clb_sample(*qids: str) -> Sample:
    return Sample(
        user_id="u",
        dataset="clbench",
        sessions=(),
        questions=tuple(Question(q, "?", "g", "DKR") for q in qids),
    )


def test_partial_credit_is_aggregated_but_never_touches_overall():
    """⛔ **`partial` 不许改 `overall`**——榜分是二值那一列，它只是本地仪器。

    它存在的理由是**分辨率**：CL-Bench 官方分全有全无，一道题从 0 翻到 1 要每一条
    rubric 都满足 ⇒ 中间进展看不见。实测有题 `score=0` 而 `ratio=0.50`（14 条里满足 7 条）
    ——二值分下与"一条都没满足"完全一样。
    """
    results = [
        # 两题都没过（全有全无），但部分分分别是 0.5 与 0.0
        JudgeResult(
            "q0",
            is_correct=False,
            label="WRONG",
            judge_response="",
            generated_answer="",
            partial=0.5,
        ),
        JudgeResult(
            "q1",
            is_correct=False,
            label="WRONG",
            judge_response="",
            generated_answer="",
            partial=0.0,
        ),
        # 一题过了
        JudgeResult(
            "q2",
            is_correct=True,
            label="CORRECT",
            judge_response="",
            generated_answer="",
            partial=1.0,
        ),
    ]
    summary = summarize(results, [_clb_sample("q0", "q1", "q2")])

    assert summary["overall"] == round(1 / 3, 6)  # ← 只数二值那一列，`partial` 一点没参与
    assert summary["partial_credit"] == {"mean": 0.5, "n": 3}
    # 逐类也要有——判断改动时看的就是逐类
    assert summary["breakdown"]["DKR"]["partial"] == 0.5
    assert summary["breakdown"]["DKR"]["accuracy"] == round(1 / 3, 6)


def test_partial_credit_is_absent_for_datasets_that_do_not_provide_it():
    """给不出部分分的数据集 ⇒ `mean=None, n=0`，**不是 0.0**（同 `_accuracy` 的纪律）。

    空集合写 `0.0` 会被读成"全错"，而它其实是"没测到"。
    """
    results = [
        JudgeResult("q0", is_correct=True, label="", judge_response="", generated_answer=""),
    ]
    summary = summarize(results, [_clb_sample("q0")])
    assert summary["partial_credit"] == {"mean": None, "n": 0}
    assert "partial" not in summary["breakdown"]["DKR"]  # ← 键**不出现**，不是 None


def test_clbench_labels_carry_the_requirement_ratio_as_partial(tmp_path: Path):
    """归档裁判写的 `rubric_clbench_requirement_ratio` **必须落到 `partial` 上**。

    ⚠ 这是最容易**静默**断的一环：`ratio` 一直在 `labels.jsonl` 里，
    而二值分照样算得出来——断了也不报错，只是本地又变回"看不见中间进展"。
    """
    from eval.harness.judge import _read_clbench_labels

    answers = tmp_path / "answers.jsonl"
    labels = tmp_path / "labels.jsonl"
    answers.write_text(
        json.dumps({"idx": "clb-0", "model_output": "答案"}) + "\n", encoding="utf-8"
    )
    labels.write_text(
        json.dumps(
            {
                "idx": "clb-0",
                "rubric_clbench_score": 0.0,
                "rubric_clbench_rationale": "差一条",
                "rubric_clbench_requirement_status": ["yes"] * 7 + ["no"] * 7,
                "rubric_clbench_requirement_ratio": 0.5,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    got = _read_clbench_labels(answers, labels)
    assert len(got) == 1
    assert got[0].is_correct is False  # 二值那列照旧
    assert got[0].partial == 0.5  # ← 14 条里满足 7 条


# ── 两个"接错了数据集"的修复（2026-10-03）──────────────────────────────
def test_memtrapbench_gets_its_own_answer_prompt():
    """⛔ **memtrapbench 不能用通用答案 prompt**——两者的取向正好相反。

    通用那份写着「**只用记忆里的信息**、记忆里没有就回 `Cannot determine from the memories.`」，
    而 MemTrapBench 的设计是（官方 README 原文）**"No-Memory Solvability: the final query
    must be answerable correctly even without the history"**、gold 是
    **"the correct answer that ignores the misleading memory"**。

    用错的后果实测（250 题）：**160/250 逐字拒答、通过 0**；而 `poison` 那类会被
    "不许用外部知识"逼着照抄毒记忆。⇒ 这条用例钉住**那份 prompt 不许再被"统一化"掉**。
    """
    from eval.harness.extra_pipeline import ANSWER_PROMPT, MEMTRAP_ANSWER_PROMPT

    assert MEMTRAP_ANSWER_PROMPT != ANSWER_PROMPT
    # 通用的那两条禁令**一个字都不许出现在** memtrapbench 那份里
    assert "Do not use outside knowledge" not in MEMTRAP_ANSWER_PROMPT
    assert "Cannot determine from the memories" not in MEMTRAP_ANSWER_PROMPT
    # 官方那份的原文特征
    assert "when needed" in MEMTRAP_ANSWER_PROMPT
    assert "{memories}" in MEMTRAP_ANSWER_PROMPT and "{question}" in MEMTRAP_ANSWER_PROMPT


def test_judge_output_cap_is_big_enough_for_a_four_dimension_reply():
    """裁判自己的回答不能被输出上限截断——**截断的后果是整条记为错**。

    2026-10-03 实测：`max_tokens=256` 时 MemTrapBench **129/250 条的 JSON 断在中间**
    （129/129 根括号未闭合），而解析失败一律记错 ⇒ **一半的分数是截断的产物**。
    四维各带 justification 的回答本来就长（实测被切的那批 p50 已到 250）。
    """
    import inspect

    from eval.harness.judge import run_judge

    default = inspect.signature(run_judge).parameters["max_tokens"].default
    assert default >= 1024, f"裁判输出上限退化到 {default}——四维回答会被截断成 JUDGE_ERROR"


def test_mquake_gets_its_own_answer_prompt():
    """⛔ **mquake 也不能用通用答案 prompt**——那两条规则对它有害。

    · 「只用记忆、不用外部知识」——而 **MQuAKE-CF 的"未编辑跳"本来就要求世界知识**；
    · 「记忆里没有就逐字拒答」——记忆里旧值与「…is X (this replaces the earlier value)」
      **并存**，模型一看到冲突就引用这条拒答。

    实测（768 题，只重答 201 条拒答、其余不动的同条件对照）：
    B0 现状 0.3997 / B1 只删拒答条款 0.4857 / B2 +改写优先 0.5143 / **B3 本份 0.5208**。
    """
    from eval.harness.extra_pipeline import ANSWER_PROMPT, MQUAKE_ANSWER_PROMPT

    assert MQUAKE_ANSWER_PROMPT != ANSWER_PROMPT
    # 那两条禁令**一个字都不许出现**
    assert "Do not use outside knowledge" not in MQUAKE_ANSWER_PROMPT
    assert "Cannot determine from the memories" not in MQUAKE_ANSWER_PROMPT
    # 必须给的两条：改写优先 + 允许补世界知识
    # ⚠ **先把空白压平再查**：源码里是隐式拼接、断行位置会变，查原文会假红
    flat = " ".join(MQUAKE_ANSWER_PROMPT.split())
    assert "replacement is the current truth" in flat
    assert "fill it with what you already know" in flat
