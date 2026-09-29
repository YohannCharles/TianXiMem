"""`Add` / `Search` 的**原文采集**——把官方发来的请求**原样**抄一份落盘。

## 它为什么存在

**不是为了功能，是为了"看见"**，两件事：

1. **S6**：平台的 `request_id` 到底长什么样（是否真带 `chunk-<n>`、序号在不在末尾）
   ——我们**从没见过**它，而 D25 的位置模型整个押在这个假设上
   （[`../../../docs/open-questions.md`](../../../docs/open-questions.md) 的 S6 记着
   "来源是团队告知、不是一手文档"）。这个文件是**唯一能直接看到**它的地方
   （另一条路是问主办方）。
2. **契约核验**：官方实发请求的形状——字段名、`timestamp` 的单位、`top_k` 的值、
   有没有未知字段。这些在本地只能猜（harness 是我们自己写的）。

## 四条纪律（每一条都很好违反，所以写在这里）

1. **在解析之前抄** ⇒ 所以是 **ASGI 中间件**，不是路由函数。
   不合 schema 的请求（422）**进不了路由函数**，在那里读 `Request.body()` 拿不到那一类
   ——而"官方的请求不合我们的设定"恰恰是最要看的东西。
2. **不改变下游看到的 body**：`receive` 只是**复制**再原样转发，
   不是"读掉再重放"——这两者的差别在这里就是全部。
3. **不吞异常**：异常照旧往上抛（[`errors.py`](./errors.py) 的纪律），这里只记类型名。
4. **写盘失败不影响响应**：只留一行 WARNING。先例是
   [`../observability/metrics.py`](../observability/metrics.py) 的 `SnapshotMetricsSink`
   ——"指标是诊断，不能因为它写不进去就让一次 Search 变成 500"。

## 为什么是 JSONL

一行一次请求，追加写、随时 `tail` / `jq`；写坏只坏一行，不会毁掉整份。
每行都带 `kind`，让文件自己说明自己（`meta` / `req` / `truncated`）。

## ⚠ 一个必须知道的边界

那一行在**响应发出前**写（`flush`，**不 fsync**）。若进程在响应前崩溃/被杀，
这次请求不落盘——AML 会重试（§2.2 最多 32 次），重试那次会记到。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TextIO

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tianxi_am.common.config import DEFAULT_CHUNK_ORDINAL_PATTERN
from tianxi_am.pairing.pairing import parse_chunk_ordinal

__all__ = ["CAPTURED_PATHS", "CaptureMiddleware", "RequestCapture"]

logger = logging.getLogger(__name__)

#: 只抄这两个业务端点。`/health` **刻意不记**：平台探活会把它刷爆，而它零核验价值
#: （它连下游都不碰，见 `routes.py`）。
CAPTURED_PATHS: Final[frozenset[str]] = frozenset({"/add", "/search"})

#: 非 2xx 响应体的记录上限（字节）。2xx 的响应体是三个回显字段，没有核验价值；
#: 失败的 `detail` 才是要看的（例如"本批未被确认"那句）。
_MAX_RESPONSE_BODY: Final[int] = 8192


def _now_iso() -> str:
    """本地时区的 ISO8601（带偏移）——读日志的人看的是墙钟，不是 UTC。"""
    return datetime.now(UTC).astimezone().isoformat(timespec="milliseconds")


def _try_parse_chunk_ordinal(request_id: str, pattern: str) -> int | None:
    """**只读地**用当前正则试解析；取不到返回 `None`，**绝不抛**。

    ⚠ 这不是校验，是**观测**：解析失败正是 S6 要看的那一个信号，所以它绝不能在
    这里变成异常。真正的失败判定在 `pairing.parse_chunk_ordinal`——那是请求路径上的事，
    本模块是**旁路**。
    """
    try:
        return parse_chunk_ordinal(request_id, pattern)
    except (ValueError, TypeError):
        return None


class RequestCapture:
    """把一次请求写成一行 JSONL。**没打开成功时是一个 no-op 对象。**

    ## 线程安全

    `--workers 1`（§15）⇒ 只有一个进程，但 FastAPI 的 `def` 路由跑在**线程池**里，
    而本类的写入点在**响应边界**上（事件循环 + 线程池都可能碰到它）⇒ 用一把锁把
    "判断是否超限 + 写 + 计数"做成一个原子段。跨进程不需要（单 worker）。
    """

    def __init__(
        self,
        *,
        path: str | Path,
        max_bytes: int,
        chunk_ordinal_pattern: str = DEFAULT_CHUNK_ORDINAL_PATTERN,
    ) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        #: ⚠ 与请求路径**同一个**模式（`ingest.chunk_ordinal_pattern`）——
        #: 记录里的 `chunk_ordinal` 必须反映"服务此刻会怎么解析"，不是另一套口径。
        self._pattern = chunk_ordinal_pattern
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        self._written = 0
        self._lines = 0

    # ── 生命周期 ────────────────────────────────────────────────────────

    def open(self) -> bool:
        """打开文件（追加）并写一行 `meta`。**返回是否可用**——不复用异常。

        ⚠ **失败不抛**：采集是诊断，不该有能力让服务起不来（那会变成"没进场"，
        是最坏的失败）。但**必须响亮**——调用方（`service/app.py` 的 `build_capture`）
        会把 ERROR 打出来，否则"开了却没记到"就成了一次静默失败。
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fh = self._path.open("a", encoding="utf-8", newline="\n")
        except OSError as exc:
            logger.error(
                "请求采集**打不开文件**，本次运行将不记录任何请求：%s（%s）", self._path, exc
            )
            return False
        self._fh = fh
        self._emit(
            {
                "kind": "meta",
                "ts": _now_iso(),
                "path": str(self._path),
                "max_bytes": self._max_bytes,
                "chunk_ordinal_pattern": self._pattern,
                "note": (
                    "TianXi_AM 的请求原文采集。每一行 `kind=req` 是一次 `/add` 或 `/search`："
                    "`body` 是**原样**收到的请求体。`chunk_ordinal` = 服务用当前的 "
                    "`ingest.chunk_ordinal_pattern` 解析 `request_id` 的结果，"
                    "**null 表示解析不出来**（那正是要看的东西）。"
                    "⚠ 只在 `capture.enabled=true` 时才有这个文件。"
                ),
            }
        )
        return True

    def close(self) -> None:
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:  # pragma: no cover — 关闭失败没有可做的事
                logger.warning("请求采集关闭文件失败：%s", self._path, exc_info=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def enabled(self) -> bool:
        """此刻**真的在记**吗（打开成功且还没写满）。"""
        return self._fh is not None

    @property
    def lines(self) -> int:
        """已写出的行数（含 `meta` / `truncated` 标记）——测试与诊断用。"""
        return self._lines

    # ── 记录 ────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        method: str,
        path: str,
        body: bytes,
        status: int | None,
        latency_ms: float,
        error: str | None = None,
        client: str | None = None,
        content_type: str | None = None,
        has_authorization: bool = False,
        response_body: bytes = b"",
    ) -> None:
        """写一行。**任何失败都只留 WARNING**（纪律 4）。

        `error` = 未被处理的异常类型名（那时 `status` 仍是 `None`：异常要穿过本层才被
        `errors.py` 的通用处理器转成 500，见 `capture` 的说明）。
        """
        entry: dict[str, Any] = {
            "kind": "req",
            "ts": _now_iso(),
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "client": client,
            "content_type": content_type,
            # ⚠ **只记"在不在"，绝不记值**——密钥不许落盘（`.env` 的纪律）。
            "authorization": "present" if has_authorization else "absent",
            "body_bytes": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "error": error,
        }

        text = body.decode("utf-8", errors="replace")
        payload: Any = None
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                # 解析不了就**原样留字符串**——那本身就是要核验的东西（官方发的可能不是 JSON）。
                entry["body_raw"] = text
                entry["body_parse_error"] = str(exc)
            else:
                entry["body"] = payload
        if isinstance(payload, dict):
            request_id = payload.get("request_id")
            if isinstance(request_id, str):
                entry["request_id"] = request_id
                entry["chunk_ordinal"] = _try_parse_chunk_ordinal(request_id, self._pattern)
        if response_body:
            entry["response_body"] = response_body.decode("utf-8", errors="replace")

        self._emit(entry)

    # ── 写 ──────────────────────────────────────────────────────────────

    def _emit(self, entry: dict[str, Any]) -> None:
        """把一条记录写出去。**唯一碰文件的地方**（截断判断也在这里）。"""
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        size = len(line.encode("utf-8"))
        with self._lock:
            fh = self._fh
            if fh is None:
                return
            if self._written + size > self._max_bytes:
                # ⚠ 护栏：`capture.enabled` 忘了关时，不该由它把 `/data` 卷写满
                #   （Full run 连跑 0.5–2 天，§2.2）。写一行截断标记后**停止记录**。
                self._fh = None
                self._write_marker(fh)
                return
            try:
                fh.write(line)
                # flush 而不是 fsync：这是诊断日志，不需要崩溃持久性，
                # 而 fsync 会在**每个请求**的关键路径上等一次磁盘。
                fh.flush()
            except OSError:
                logger.warning(
                    "请求采集写盘失败，**本次运行不再记录**：%s", self._path, exc_info=True
                )
                self._fh = None
                return
            self._written += size
            self._lines += 1

    def _write_marker(self, fh: TextIO) -> None:
        """写"已写满、停止记录"那一行，然后关掉文件。"""
        marker = json.dumps(
            {
                "kind": "truncated",
                "ts": _now_iso(),
                "written_bytes": self._written,
                "max_bytes": self._max_bytes,
                "note": (
                    "已达 `capture.max_bytes` ⇒ **从这里开始不再记录**。"
                    "把 `capture.enabled` 关掉，或调大 `capture.max_bytes` 后重跑。"
                ),
            },
            ensure_ascii=False,
        )
        try:
            fh.write(marker + "\n")
            fh.flush()
            fh.close()
        except OSError:  # pragma: no cover — 连标记都写不进去时没有可做的事
            logger.warning("请求采集写截断标记失败：%s", self._path, exc_info=True)
        self._lines += 1
        logger.warning(
            "请求采集已达上限 %d 字节 ⇒ **停止记录**（文件：%s）。"
            "后续请求照常处理，只是不再进这个文件。",
            self._max_bytes,
            self._path,
        )


class CaptureMiddleware:
    """把 `/add`、`/search` 的**原始请求体**抄一份——纯 ASGI，不是 `BaseHTTPMiddleware`。

    **为什么是纯 ASGI**：要在 body 被解析之前拿到它，唯一的做法是包住 `receive`。
    `BaseHTTPMiddleware` 也能做，但它自带一套 stream/task 语义，而这里要的语义很简单：
    **原样转发、顺手复制一份**。三十行把它写清楚，比借一层间接更容易守住纪律 2。

    ⚠ **只包住这两个端点**（`CAPTURED_PATHS`）：其余路径（含 `/health`）**零开销**直通。
    """

    def __init__(self, app: ASGIApp, *, capture: RequestCapture) -> None:
        self._app = app
        self._capture = capture

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("path") not in CAPTURED_PATHS:
            await self._app(scope, receive, send)
            return

        chunks: list[bytes] = []
        status: int | None = None
        error: str | None = None
        response_chunks: list[bytes] = []
        response_bytes = 0

        async def receive_capture() -> Message:
            # ⚠ **转发的是原消息本身**，我们只是"顺手把 body 复制进 chunks"——
            #   不是"读掉、缓存、再重放"。后者一旦写错（少转发一条、丢了 more_body、
            #   或没处理 http.disconnect）下游就会**拿到一个不同的 body**，
            #   而症状是"某些请求莫名 422/500"，与本模块毫无表面关联。
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"")
                if chunk:
                    chunks.append(chunk)
            return message

        async def send_capture(message: Message) -> None:
            nonlocal status, response_bytes
            if message["type"] == "http.response.start":
                status = int(message["status"])
            elif message["type"] == "http.response.body" and status is not None and status >= 300:
                chunk = message.get("body", b"")
                if response_bytes < _MAX_RESPONSE_BODY:
                    response_chunks.append(chunk)
                    response_bytes += len(chunk)
            await send(message)

        started = time.perf_counter()
        try:
            await self._app(scope, receive_capture, send_capture)
        except BaseException as exc:
            # ⚠ **记完照抛**（纪律 3）：通用 500 处理器在这一层**外面**
            #   （`ServerErrorMiddleware` 是最外层，而 user middleware 在它里面）
            #   ⇒ 异常会穿过本层，`status` 那时仍是 None，所以这里记类型名。
            error = type(exc).__name__
            raise
        finally:
            host = scope.get("client")
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            content_type = headers.get(b"content-type", b"").decode("latin-1") or None
            self._capture.record(
                method=str(scope.get("method", "")),
                path=str(scope.get("path", "")),
                body=b"".join(chunks),
                status=status,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=error,
                client=(host[0] if host else None),
                content_type=content_type,
                has_authorization=b"authorization" in headers,
                response_body=b"".join(response_chunks),
            )
