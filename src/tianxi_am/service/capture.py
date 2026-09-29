"""`Add` / `Search` 的**原文采集**——把官方发来的请求**原样**抄一份落盘。

## 它为什么存在

**不是为了功能，是为了"看见"**：

1. **它已经干掉过一次事故（S6，2026-09-29）**：官方的 `request_id` 到底长什么样，
   我们此前只听过转述，而 D25 的位置模型整个押在那个转述上。它抓到的真实请求是
   `r_3115…` 这种**不透明 id** ⇒ 解析必然失败 ⇒ **Add 全挂** ⇒ 直接催生了 **D28**
   （`request_id` 回到 opaque string）。
2. **契约核验**：官方实发请求的形状——字段名、`timestamp` 的单位、`top_k` 的值、
   有没有未知字段。这些在本地只能猜（harness 是我们自己写的）。
   ⚠ **D28 起它不再解析 `request_id`**（那个解析器已经删了）——它只把**原文**记下来，
   形状的判断留给读的人（见 `deploy/CLAUDE.md` §0.6 的判据表）。

**记哪些**：`/add` 与 `/search`（`CAPTURED_PATHS`）——**任何形状都记**，
包括不合 schema 的 422 与内部异常导致的 500（它跑在路由与校验**之前**）。
`/health` **刻意不记**（平台探活会刷屏，零核验价值）。

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

## 为什么是 JSONL，以及**为什么要轮转**

一行一次请求，追加写、随时 `tail` / `jq`；写坏只坏一行，不会毁掉整份。
每行都带 `kind`，让文件自己说明自己（`meta` / `req` / `truncated`）。

⚠ **一轮 Full 的请求原文按 ~1.5 GB 估**（§2.2 的 0.5–2 天），而**单个 1.5 GB 的 JSONL
根本打不开**——诊断产物打不开就等于没记。⇒ **每 `capture.max_bytes`（默认 50 MiB）
换一份文件**，文件名见 `RequestCapture._file_for`。
**没有总量上限**：写满一块盘是"忘关"的代价，由人盯着（跑完记得 `capture.enabled: false`）。

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


class RequestCapture:
    """把一次请求写成一行 JSONL，**写满一份就换下一份**（轮转）。

    没打开成功时是一个 no-op 对象（`enabled` 为 `False`，所有写入静默跳过）。

    ## 文件怎么分

    | 文件 | 什么时候 |
    | --- | --- |
    | `<path>`（配置里那个） | 第 1 份 |
    | `<stem>.part2<suffix>`、`.part3`… | 第 2、3… 份（写满即转） |

    每份的 `meta` 行带着 `file_seq` ⇒ **光看文件自己就知道它是第几份**。

    ## 线程安全

    `--workers 1`（§15）⇒ 只有一个进程，但 FastAPI 的 `def` 路由跑在**线程池**里，
    而本类的写入点在**响应边界**上（事件循环 + 线程池都可能碰到它）⇒ 用一把锁把
    "判断该不该转 + 写 + 计数"做成一个原子段。跨进程不需要（单 worker）。
    """

    def __init__(self, *, path: str | Path, max_bytes: int) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        #: 当前是第几份（1 = 配置里那个路径）。
        self._seq = 0
        #: **当前文件**已写的字节数 / 行数。
        self._written = 0
        self._lines_in_file = 0
        #: 累计写出的行数（跨所有份，含 `meta` / `truncated`）——测试与诊断用。
        self._lines = 0

    # ── 生命周期 ────────────────────────────────────────────────────────

    def _file_for(self, seq: int) -> Path:
        """第 `seq` 份的路径。**第 1 份就是配置里写的那个路径**（文档与命令都指着它）。"""
        if seq <= 1:
            return self._path
        return self._path.with_name(f"{self._path.stem}.part{seq}{self._path.suffix}")

    def open(self) -> bool:
        """打开第 1 份（追加）并写一行 `meta`。**返回是否可用**——不复用异常。

        ⚠ **失败不抛**：采集是诊断，不该有能力让服务起不来（那会变成"没进场"，
        是最坏的一类失败）。但**必须响亮**——调用方（`service/app.py` 的 `build_capture`）
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
        with self._lock:
            self._fh = fh
            self._seq = 1
            self._written = 0
            self._lines_in_file = 0
            self._write_meta_locked()
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
        """**第 1 份**的路径（= 配置里那个）。此刻在写哪一份见 `current_path`。"""
        return self._path

    @property
    def current_path(self) -> Path:
        """**此刻正在写**的那一份。"""
        return self._file_for(self._seq)

    @property
    def files_written(self) -> int:
        """已经开过几份（1 = 还没转过）。"""
        return self._seq

    @property
    def enabled(self) -> bool:
        """此刻**真的在记**吗（打开成功、且还没到总量上限）。"""
        return self._fh is not None

    @property
    def lines(self) -> int:
        """累计写出的行数（含 `meta` / `truncated` 标记）——测试与诊断用。"""
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
        `errors.py` 的通用处理器转成 500，见本模块 docstring）。
        """
        entry: dict[str, Any] = {
            "kind": "req",
            "ts": _now_iso(),
            "file_seq": self._seq,
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
            # ⚠ **只把 `request_id` 抄出来，不解析它**（D28）：它是不透明字符串，
            #   这里取出来只是为了让人**一眼看见它长什么样**。
            request_id = payload.get("request_id")
            if isinstance(request_id, str):
                entry["request_id"] = request_id
        if response_body:
            entry["response_body"] = response_body.decode("utf-8", errors="replace")

        self._emit(entry)

    # ── 写（**本类唯一碰文件的地方**）────────────────────────────────────

    def _emit(self, entry: dict[str, Any]) -> None:
        """写一条记录；必要时**先转下一份**。"""
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        size = len(line.encode("utf-8"))
        with self._lock:
            if self._fh is None:
                return
            # ⚠ `_lines_in_file > 0`：**别让 meta 行自己触发轮转**——它在每份文件的开头，
            #   而"新文件刚开就被判定写满"会让轮转在一个文件里无限递归。
            if (
                self._lines_in_file > 0
                and self._written + size > self._max_bytes
                and not self._rotate_locked()  # ← 短路的最后一环：只有前两条成立才会转
            ):
                self._stop_locked()
                return
            self._append_locked(line)

    def _append_locked(self, line: str) -> bool:
        """把一行追加到当前文件（**已持锁**）。写失败 ⇒ 停止记录并返回 `False`。"""
        fh = self._fh
        if fh is None:
            return False
        try:
            fh.write(line)
            # flush 而不是 fsync：这是诊断日志，不需要崩溃持久性，
            # 而 fsync 会在**每个请求**的关键路径上等一次磁盘。
            fh.flush()
        except OSError:
            logger.warning(
                "请求采集写盘失败，**本次运行不再记录**：%s", self.current_path, exc_info=True
            )
            self._fh = None
            return False
        size = len(line.encode("utf-8"))
        self._written += size
        self._lines += 1
        self._lines_in_file += 1
        return True

    def _rotate_locked(self) -> bool:
        """换下一份（**已持锁**）。**开不了新文件**（盘满 / 权限）⇒ `False`。"""
        nxt = self._seq + 1
        new_path = self._file_for(nxt)
        try:
            fh = new_path.open("a", encoding="utf-8", newline="\n")
        except OSError:
            logger.warning("请求采集**开不了下一份**（%s）⇒ 停止记录", new_path, exc_info=True)
            return False
        old, self._fh = self._fh, fh
        self._seq = nxt
        self._written = 0
        self._lines_in_file = 0
        if old is not None:
            try:
                old.close()
            except OSError:  # pragma: no cover
                logger.warning("请求采集关闭上一份失败：%s", new_path, exc_info=True)
        logger.warning(
            "请求采集写满一份（%d 字节）⇒ 已转到第 %d 份：%s", self._max_bytes, nxt, new_path
        )
        self._write_meta_locked()
        return True

    def _write_meta_locked(self) -> None:
        """每份文件的头一行——**它就是"这份是什么、第几份"的唯一说明**。"""
        self._append_locked(
            json.dumps(
                {
                    "kind": "meta",
                    "ts": _now_iso(),
                    "file_seq": self._seq,
                    "path": str(self.current_path),
                    "max_bytes": self._max_bytes,
                    "note": (
                        "TianXi_AM 的请求原文采集。每一行 `kind=req` 是一次 `/add` 或 "
                        "`/search`：`body` 是**原样**收到的请求体，`request_id` 是从里面"
                        "取出来的那一个字段（**不做任何解析**——D28 之后它是不透明字符串）。"
                        f"本份是第 {self._seq} 份；写满 {self._max_bytes} 字节就换下一份"
                        "（文件名形如 `<name>.part2.jsonl`）。"
                        "⚠ 只在 `capture.enabled=true` 时才有这些文件。"
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def _stop_locked(self) -> None:
        """**转不到下一份**（盘满 / 权限）：写一行标记、关掉文件、停止记录（已持锁）。"""
        fh = self._fh
        self._append_locked(
            json.dumps(
                {
                    "kind": "truncated",
                    "ts": _now_iso(),
                    "file_seq": self._seq,
                    "written_bytes": self._written,
                    "max_bytes": self._max_bytes,
                    "note": (
                        "**开不了下一份文件**（盘满 / 权限？）⇒ 从这里开始不再记录"
                        "（后续请求照常处理，只是不进文件）。"
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._fh = None
        if fh is not None:
            try:
                fh.close()
            except OSError:  # pragma: no cover
                logger.warning("请求采集关闭文件失败：%s", self.current_path, exc_info=True)


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
