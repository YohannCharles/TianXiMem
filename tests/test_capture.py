"""`Add` / `Search` 的**原文采集**：配置开关、记录形状、以及"它不改变任何行为"。

本文件保护的是两件事，都不是功能：

* **S6**：平台的 `request_id` 到底长什么样——`chunk_ordinal` 记的就是"服务此刻能不能
  从它里面解析出序号"，而 `null` 正是要看的那一眼（`docs/open-questions.md` 的 S6）。
* **它绝不改变行为**：抄一份原文，不改变下游看到的 body、不吞异常、写盘失败不影响响应。

⚠ **HTTP 那几条用例走真 `create_app` + `TestClient`**（与 `test_contract.py` 同一套 `wired`
夹具），因为要证明的正是"中间件挂在真 app 上不发生任何事"——那种事只在真链路上看得见。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.conftest import Wired, rd

from tianxi_am.common.config import ConfigError, load_config
from tianxi_am.service.app import build_capture, create_app
from tianxi_am.service.capture import CAPTURED_PATHS, CaptureMiddleware, RequestCapture

_MIN_ENV = {"AML_EMB_BASE_URL": "http://unused/v1", "AML_EMB_API_KEY": "k"}

_ADD_PAYLOAD: dict[str, Any] = {
    # ⚠ 用本仓 harness 的形态（`<user>|<session>|<n>`）：默认正则两个形态都认，
    #   而"平台实发的是哪一种"正是 S6 未决的那件事。
    "request_id": "u1|s1|0",
    "user_id": "u1",
    "session_id": "s1",
    "messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}],
}


# ── 夹具与助手 ─────────────────────────────────────────────────────────


def _mini_config_dir(tmp_path: Path, capture_block: str = "") -> Path:
    """一份**最小可加载**的配置目录：只有我们写进去的那个段，其余走内置默认值。

    `default.yaml` 存在但不写任何键是合法的（`_read_yaml(required=True)` 只要求它存在
    且是个映射）——这样每条用例只面对**它自己**声明的那几个键。
    """
    where = tmp_path / "configs"
    where.mkdir(exist_ok=True)
    (where / "default.yaml").write_text(capture_block, encoding="utf-8")
    return where


def _load(tmp_path: Path, capture_block: str = "", **env: str):
    return load_config({**_MIN_ENV, **env}, config_dir=_mini_config_dir(tmp_path, capture_block))


@pytest.fixture
def capture(tmp_path: Path) -> Iterator[RequestCapture]:
    """一个开着、写临时文件、上限 1 MiB 的采集器。"""
    cap = RequestCapture(path=tmp_path / "requests.jsonl", max_bytes=1 << 20)
    assert cap.open() is True, "夹具自己先失败了——后面的断言都会成为空过"
    yield cap
    cap.close()


def _lines(cap: RequestCapture) -> list[dict[str, Any]]:
    """把文件读成记录列表（**逐行解析**：一行坏掉就该在这里响亮失败）。"""
    text = cap.path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _requests(cap: RequestCapture) -> list[dict[str, Any]]:
    return [entry for entry in _lines(cap) if entry["kind"] == "req"]


def _app_with_capture(wired: Wired, cap: RequestCapture):
    """把采集器挂到**真装配链**上（`wired` 的 services + 真 `create_app`）。"""
    wired.services.capture = cap  # type: ignore[attr-defined]
    return create_app(wired.services)  # type: ignore[arg-type]


# ── 1. 开关与配置：一个键只有一个家 ────────────────────────────────────


def test_capture_is_off_by_default(tmp_path: Path) -> None:
    """默认关——它按设计只在"要看清官方发了什么"时打开（S6）。"""
    cfg = _load(tmp_path)
    assert cfg.capture.enabled is False
    assert cfg.capture.max_bytes == 64 * 1024 * 1024
    assert cfg.capture.path == "var/capture/requests.jsonl"
    # 关 ⇒ 连中间件都不装（`build_capture` 返回 None，`create_app` 里那一段不会执行）
    assert build_capture(cfg) is None


def test_enabled_comes_from_yaml_and_path_from_env(tmp_path: Path) -> None:
    """开关归 yaml、**路径归 env**——与 `rerank.enabled` + `TIANXI_RERANKER_*` 同一个拆法。"""
    cfg = _load(
        tmp_path,
        "capture:\n  enabled: true\n  max_bytes: 4096\n",
        TIANXI_CAPTURE_PATH="/data/capture/requests.jsonl",
    )
    assert cfg.capture.enabled is True
    assert cfg.capture.max_bytes == 4096
    assert cfg.capture.path == "/data/capture/requests.jsonl"


def test_enabled_is_not_overridable_from_env(tmp_path: Path) -> None:
    """**yaml 说了算**：环境里放一个同名的开关也没用（"每个键只有一个家"的守门人）。"""
    cfg = _load(tmp_path, "capture:\n  enabled: false\n", TIANXI_CAPTURE_ENABLED="true")
    assert cfg.capture.enabled is False


def test_path_in_yaml_is_rejected(tmp_path: Path) -> None:
    """路径是 env 拥有的键，写进 yaml 必须**直接报错**（而不是被静默忽略）。"""
    with pytest.raises(ConfigError, match=r"capture"):
        _load(tmp_path, "capture:\n  path: var/other.jsonl\n")


def test_unknown_key_under_capture_is_rejected(tmp_path: Path) -> None:
    """拼错的键必须响亮失败——`capure:` 被静默忽略的话，跑的是默认值而没人发现。"""
    with pytest.raises(ConfigError, match="capture"):
        _load(tmp_path, "capture:\n  enabld: true\n")


@pytest.mark.parametrize("block", ['capture:\n  enabled: "yes"\n', "capture:\n  enabled: 1\n"])
def test_non_bool_enabled_is_rejected(tmp_path: Path, block: str) -> None:
    """`enabled` 必须是**真布尔**。

    ⚠ PyYAML 走 YAML 1.1：**裸** `yes` 会被解析成 `True`（那是合法的布尔写法），
    而**带引号**的 `"yes"` 是字符串、`1` 是整数——后两者会让开关"看起来设了、其实没生效"。
    """
    with pytest.raises(ConfigError, match=r"capture\.enabled"):
        _load(tmp_path, block)


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_max_bytes_is_rejected(tmp_path: Path, value: int) -> None:
    """上限是**护栏**：非正值等于"记 0 行"，而它看起来像是开着。"""
    with pytest.raises(ConfigError, match=r"capture\.max_bytes"):
        _load(tmp_path, f"capture:\n  max_bytes: {value}\n")


# ── 2. 一行记录里有什么 ────────────────────────────────────────────────


def test_record_carries_the_raw_body_and_the_parsed_chunk_ordinal(
    capture: RequestCapture,
) -> None:
    """`body` = 原样收到的对象；派生列给出 `request_id` 与**解析出的 chunk 序号**。"""
    payload = {
        "request_id": "eval:run1:locomo_refined:conv-0:chunk-3",
        "user_id": "u1",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "你好，world"}],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    capture.record(method="POST", path="/add", body=body, status=200, latency_ms=12.34)

    (entry,) = _requests(capture)
    assert entry["body"] == payload  # 逐字段原样（含中文）
    assert entry["body_bytes"] == len(body)
    assert entry["request_id"] == payload["request_id"]
    assert entry["chunk_ordinal"] == 3  # ← S6 要看的那一列
    assert entry["status"] == 200
    assert entry["kind"] == "req"


def test_meta_line_describes_the_file(capture: RequestCapture) -> None:
    """第一行是 `meta`：文件要能自己说明自己（几个月后打开它的人不必翻代码）。"""
    meta = _lines(capture)[0]
    assert meta["kind"] == "meta"
    assert "chunk_ordinal" in meta["note"]


def test_unparsable_chunk_ordinal_is_recorded_as_null(capture: RequestCapture) -> None:
    """**取不出序号 ⇒ 记 `null`，而不是抛异常。**

    这正是那次冒烟失败的形状（`...:chunk-0-1790669310`）：id 后面多粘了一段，
    服务端解析不出来 ⇒ 500。采集必须**照样把它记下来**，否则要核验的东西就没了。
    """
    payload = {**_ADD_PAYLOAD, "request_id": "u1|s1|0-1790669310"}

    capture.record(
        method="POST",
        path="/add",
        body=json.dumps(payload).encode("utf-8"),
        status=None,
        latency_ms=1.0,
        error="ValueError",
    )

    (entry,) = _requests(capture)
    assert entry["request_id"] == "u1|s1|0-1790669310"
    assert entry["chunk_ordinal"] is None
    assert entry["error"] == "ValueError"


def test_non_json_body_is_kept_verbatim(capture: RequestCapture) -> None:
    """解析不了就**原样留字符串**——"官方发的不是 JSON"本身就是要核验的事实。"""
    capture.record(method="POST", path="/add", body=b"not json at all", status=422, latency_ms=1.0)

    (entry,) = _requests(capture)
    assert entry["body_raw"] == "not json at all"
    assert "body_parse_error" in entry
    assert "body" not in entry


def test_authorization_value_never_lands_in_the_file(capture: RequestCapture) -> None:
    """**只记"在不在"，不记值**——密钥不许落盘（`.env` 的纪律）。"""
    capture.record(
        method="POST",
        path="/search",
        body=b'{"user_id":"u1","query":"q","top_k":1}',
        status=200,
        latency_ms=1.0,
        has_authorization=True,
    )

    text = capture.path.read_text(encoding="utf-8")
    (entry,) = _requests(capture)
    assert entry["authorization"] == "present"
    assert "secret" not in text.lower()


def test_max_bytes_stops_recording_with_a_marker(tmp_path: Path) -> None:
    """写满 ⇒ 一行 `truncated` 标记 + **停止记录**——"忘了关"的代价必须有界。"""
    cap = RequestCapture(path=tmp_path / "r.jsonl", max_bytes=2000)
    assert cap.open() is True

    for _ in range(50):
        cap.record(method="POST", path="/add", body=b"x" * 200, status=200, latency_ms=1.0)

    kinds = [entry["kind"] for entry in _lines(cap)]
    assert kinds.count("truncated") == 1  # 标记只有一行
    assert kinds[-1] == "truncated"  # 而且是**最后一行**
    assert 0 < kinds.count("req") < 50  # 记了一些，但远不是全部
    # 标记之后一个字节都不再写（否则"停止记录"只是句话）
    assert kinds.index("truncated") == len(kinds) - 1
    assert _lines(cap)[-1]["written_bytes"] <= 2000
    assert cap.enabled is False
    cap.close()


def test_write_failure_only_warns_and_never_raises(
    capture: RequestCapture, caplog: pytest.LogCaptureFixture
) -> None:
    """**写盘失败不影响任何东西**——先例是 `SnapshotMetricsSink`（诊断不该变成故障）。"""

    class _Broken:
        def write(self, _: str) -> int:
            raise OSError("模拟磁盘满")

        def flush(self) -> None:
            raise OSError("模拟磁盘满")

        def close(self) -> None:
            pass

    with caplog.at_level("WARNING"):
        capture._fh = _Broken()  # type: ignore[assignment]  # 直接造出"写不进去"
        capture.record(method="POST", path="/add", body=b"{}", status=200, latency_ms=1.0)

    assert capture.enabled is False  # 就地关掉，不每个请求刷一行日志
    assert any("写盘失败" in record.getMessage() for record in caplog.records)


# ── 3. 真 HTTP：它记下来了，而且**什么都没改变** ───────────────────────


def test_add_request_is_recorded_verbatim(capture: RequestCapture, wired: Wired) -> None:
    """一次成功的 `/add`：记下的 body **逐字段等于**发出去的那一份。"""
    with TestClient(_app_with_capture(wired, capture)) as client:
        resp = client.post("/add", json=_ADD_PAYLOAD)

    assert resp.status_code == 200
    (entry,) = _requests(capture)
    assert entry["method"] == "POST"
    assert entry["path"] == "/add"
    assert entry["status"] == 200
    assert entry["content_type"] == "application/json"
    assert entry["body"] == _ADD_PAYLOAD
    assert entry["chunk_ordinal"] == 0


def test_boundary_422_is_recorded_before_the_route(capture: RequestCapture, wired: Wired) -> None:
    """**不合 schema 的请求也记得到**——这正是"中间件而不是路由函数"的理由。

    路由函数只在请求**通过**校验后才被调用，所以那一类在路由里拿不到。
    """
    with TestClient(_app_with_capture(wired, capture)) as client:
        resp = client.post("/search", json={"user_id": "u1", "query": "q", "top_k": 0})

    assert resp.status_code == 422
    (entry,) = _requests(capture)
    assert entry["status"] == 422
    assert entry["body"] == {"user_id": "u1", "query": "q", "top_k": 0}


def test_unhandled_exception_is_recorded_with_its_type(
    capture: RequestCapture, wired: Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**冒烟失败的那条路径**：`request_id` 取不出 chunk 序号 ⇒ 异常 ⇒ 非 200。

    断言三件事，它们合起来就是"线上出了 500 时，这个文件能不能回答'是什么炸的'"：

    * `error` 记下**异常类型**（500 的响应体在这一层**外面**生成，见 `capture.py`）
    * `status` 是 `None`（异常穿过本层时还没有状态码）
    * 请求原文**照旧在**（要核验的就是它）
    """
    bad = {
        **_ADD_PAYLOAD,
        # 那次冒烟失败的形状：`chunk-0` 后面**还粘了一段运行时间戳**
        "request_id": "aml-local:probe:addtest:__selftest__:s1:chunk-0-1790669310",
    }

    def _boom(_batch: object) -> None:
        raise ValueError("模拟 pairing.parse_chunk_ordinal 取不到序号")

    monkeypatch.setattr(wired.services.add, "apply", _boom)  # type: ignore[attr-defined]

    # ⚠ `raise_server_exceptions=False`：Starlette 的 `ServerErrorMiddleware` 在**发完 500
    #    之后仍然会把异常重新抛出**（好让服务器记日志），而 `TestClient` 默认会把它
    #    在测试里再抛一次——那正是"这个 500 真的发生了"的证据，但这里要的是**响应与记录**。
    app = _app_with_capture(wired, capture)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/add", json=bad)

    assert resp.status_code == 500  # `errors.py` 的通用处理器
    (entry,) = _requests(capture)
    assert entry["error"] == "ValueError"
    assert entry["status"] is None
    assert entry["request_id"] == bad["request_id"]
    assert entry["chunk_ordinal"] is None


def test_health_is_not_recorded(capture: RequestCapture, wired: Wired) -> None:
    """平台探活会不停打 `/health`——它零核验价值，**不进文件**。"""
    with TestClient(_app_with_capture(wired, capture)) as client:
        assert client.get("/health").status_code == 200

    assert _requests(capture) == []
    assert "/health" not in CAPTURED_PATHS


def test_capture_changes_neither_the_response_nor_the_truth_source(
    capture: RequestCapture, wired: Wired
) -> None:
    """**【最重要的一条】** 开 / 关采集，对同一个请求的**响应与真源完全一样**。

    ⚠ 它抓的是中间件最经典的写错方式：把 body 读掉却没原样转发（下游拿到空 body
    ⇒ 422）、或少转发一条（`more_body` 丢了 ⇒ 截断）——**那两种症状都长得像"服务坏了"**，
    与本模块毫无表面关联。

    做法：同一个 payload 发两次（第二次走幂等守卫的修复路径），断言响应逐字相同、
    且库里**只有一行**——采集开着那次也一样。
    """
    with TestClient(_app_with_capture(wired, capture)) as client:
        on = client.post("/add", json=_ADD_PAYLOAD)

    wired.services.capture = None  # type: ignore[attr-defined]
    with TestClient(create_app(wired.services)) as client:  # type: ignore[arg-type]
        off = client.post("/add", json=_ADD_PAYLOAD)

    assert on.status_code == off.status_code == 200
    assert on.json() == off.json()
    rows = rd(wired.store, wired.store.iter_pairs)
    assert len(rows) == 1  # 两次都没多写（D25 的守卫 + 位置派生）
    assert rows[0].question == "Q1"
    assert _requests(capture) != []


def test_concurrent_writes_produce_whole_lines(capture: RequestCapture) -> None:
    """并发下**每行都是完整 JSON**。

    ⚠ 是在测**行不会交错**，不是"写得多快"——交错的行是一份读不出来的日志，
    而它只在并发下出现。写入点在生产里是**响应边界**（事件循环 + FastAPI 的线程池），
    所以这条直接打 `record`：那是唯一碰文件的地方，也是锁守着的地方。
    """
    errors: list[BaseException] = []

    def record(i: int) -> None:
        try:
            capture.record(
                method="POST",
                path="/add",
                body=json.dumps({**_ADD_PAYLOAD, "session_id": f"s{i}"}).encode("utf-8"),
                status=200,
                latency_ms=1.0,
            )
        except BaseException as exc:  # noqa: BLE001 — 线程里的异常会让用例空过，要收上来断言
            errors.append(exc)

    threads = [threading.Thread(target=record, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    entries = _requests(capture)
    assert len(entries) == 8
    assert {entry["body"]["session_id"] for entry in entries} == {f"s{i}" for i in range(8)}


def test_middleware_is_only_installed_when_a_capture_is_present(wired: Wired) -> None:
    """默认装配里**没有**这个中间件——关着的开关不该在请求路径上留下任何东西。"""
    app = create_app(wired.services)  # type: ignore[arg-type]
    assert not any(m.cls is CaptureMiddleware for m in app.user_middleware)
