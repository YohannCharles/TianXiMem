"""契约预检 —— **在花掉任何一次 Smoke 之前**，把 [`contract.md`](../../docs/contract.md) §4 的清单
在本地的真服务上自动跑一遍。

```bash
make contract-check          # 或 uv run python eval/smoke/preflight.py
uv run python eval/smoke/preflight.py --base-url http://127.0.0.1:8000   # 打已在跑的服务
```

## 它是什么、不是什么

| 是 | 不是 |
| --- | --- |
| **契约级**自检：打**真 HTTP**，看**真响应** | 质量评测（那是 [`../harness/`](../harness/) 的事） |
| 跑在**本地服务 + 本地 Qdrant + 自建网关**上 | 不碰 AML 任何端点 ⇒ **不消耗 Smoke 配额** |
| **Smoke 之前的最后一道门** | 不能替代 Smoke（S1–S3 只能靠 Smoke 消除，§17.1） |

## 四条设计决定

**1. 它自己拉起服务，而不是要求你先起一个。**
`make contract-check` 必须是一条命令、一个退出码。所以缺省自己 `subprocess` 起
`uvicorn --factory --workers 1`，跑完关掉。（`--base-url` 用于打已经在跑的那一个。）

**2. 它不 `import tianximem`。**
`eval/` 的纪律是"**打 HTTP、不 import `src/`**"（[`../CLAUDE.md`](../CLAUDE.md)）——
理由有两条，两条都适用于本文件：契约层**只有打 HTTP 才碰得到**（进程内调用根本看不到
`/add` 的 422 与 `data` 的序列化）。顺带还有一条更微妙的：

> **用我们自己的 `common.render` 去算出"期望的 content"，是一个循环论证。**
> 它只能证明"服务调用了 render"，证明不了"render 本身对"。
> ⇒ 所以本文件断言的是 content 的**性质**（含原文、有行首标记、首尾无空白、无绝对时间戳），
> 这些性质**独立于我们的实现**。

**3. 它不碰开发期的集合。**
自启时用一份**临时 configs/**（`TIANXIMEM_CONFIG_DIR` 指过去），把集合换成
`memories_preflight`，跑完 drop 掉。否则预检写进去的记忆会**永久留在 `memories_dev` 里**，
污染之后的每一次检索——而那正是 `open-questions.md` **V9** 的形状。

**4. 它不共用 [`../harness/driver.py`](../harness/driver.py) 的 `ServiceClient`。**
两者都打 `/add` 与 `/search`，请求形状也确实长得一样——但**失败模型不同，不能硬合并**：
`ServiceClient` 遇到契约违规**直接抛**（它服务一次 run，早失败早好）；本文件则把每条
断言**收集成一份报告**、逐条 FAIL（它服务的是"把契约清单一次跑完、好知道**哪几条**坏了"）。
合并后总有一方要拿到错的那半。

> 这处重复**是可检测的**，不是静默风险：两边对线上形状的认知分别由
> [`../../tests/test_harness.py`](../../tests/test_harness.py) 与
> [`../../tests/test_contract_preflight.py`](../../tests/test_contract_preflight.py) 钉住，
> 且任何一处漂了、真打服务时都会立刻红。

## 前置条件（不满足 ⇒ exit 2，不是"跳过"）

| 条件 | 为什么必须 |
| --- | --- |
| Qdrant 可达 | 检索是契约的一半；不可达时 `/search` 会 500 |
| 自建网关的 embedding 端点可达 | `Add` 要真嵌入才写得进 Qdrant |

**不满足时明确失败，不静默跳过**——门禁静默放行比门禁不存在更危险。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ── 退出码 ─────────────────────────────────────────────────────────────
# 分开是为了让"没跑成"与"跑了但没过"在 CI / make 里可区分。
EXIT_OK = 0
EXIT_CHECKS_FAILED = 1
EXIT_PRECONDITION_FAILED = 2

_REPO = Path(__file__).resolve().parents[2]

#: ⚠ 这两个名字的**单一声明**在 `src/tianximem/common/config.py` 的 `ENV_*` 表。
#: 本文件不能 import 它（`eval/` 不 import `src/`），所以这里是一处**有意的重复**——
#: 由 `tests/test_contract_preflight.py::test_env_names_match_the_single_declaration` 钉住相等。
_ENV_QDRANT_URL = "TIANXIMEM_QDRANT_URL"
_ENV_EMBED_BASE_URL = "AML_EMB_BASE_URL"
_ENV_EMBED_API_KEY = "AML_EMB_API_KEY"
_ENV_CONFIG_DIR = "TIANXIMEM_CONFIG_DIR"
_ENV_PROFILE = "TIANXIMEM_PROFILE"

#: `TIANXIMEM_QDRANT_URL` 没设时的回落值（与 `config.py` 的内置默认同值）。
_DEFAULT_QDRANT_URL = "http://localhost:6333"
_ENV_SQLITE_PATH = "TIANXIMEM_SQLITE_PATH"
_ENV_EMBED_CACHE_DIR = "TIANXIMEM_EMBED_CACHE_DIR"
_ENV_WORKERS = "TIANXIMEM_WORKERS"

#: 预检专用的集合名。**与开发期的 `memories_dev` / 提交期的 `memories` 都不同**。
_PREFLIGHT_COLLECTION = "memories_preflight"

#: 预检用的 profile 名（临时 configs/ 里的那一个）。
_PREFLIGHT_PROFILE = "preflight"

#: `score` 的量级必须是**名次倒数**，不是融合分数。
#:
#: 融合分数（RRF，k=61、两路权重 1.0）的量级是 `1/61 ≈ 0.016`，两位小数下几乎是 0；
#: 而 `1/(rank+1)` 在 rank=0 处是 `1.0`。两者的量级差约 30 倍（D5 实测过 0.0328 vs 0.5），
#: 所以只要断言"最小的那个 score 也不像融合分数"就能把它们分开。
_MIN_PLAUSIBLE_SCORE = 0.05

#: `created_at` 的合法形状：日粒度 `YYYY-MM-DD`，或**空串**（`event_time` 为 NULL，§11.3）。
_DAY_GRAIN_RE = r"^\d{4}-\d{2}-\d{2}$"


# ── 检查框架（很小，够用）──────────────────────────────────────────────


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(Result(name, ok, detail))

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if not r.ok]

    def render(self) -> str:
        width = max((len(r.name) for r in self.results), default=0)
        lines = []
        for r in self.results:
            mark = "PASS" if r.ok else "FAIL"
            line = f"  [{mark}] {r.name.ljust(width)}"
            if r.detail:
                line += f"  {r.detail}"
            lines.append(line)
        return "\n".join(lines)


class CheckFailed(AssertionError):
    """一条检查不通过。**消息里必须带上实际收到的值**——否则排错要重新跑一遍。"""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


# ── 服务进程 ───────────────────────────────────────────────────────────


def _force_utf8_stdio() -> None:
    """把 stdout/stderr 切到 UTF-8 + `errors="replace"`。

    ⚠ **这不是洁癖，是一个实测到的崩溃**：本文件**通篇是中文**，
    而 Windows 的默认控制台是 GBK（cp936），`sys.stdout` 的默认 error handler 是 **`strict`**。
    于是"缺密钥"那条中文报错在**准备打印的时候**抛 `UnicodeEncodeError`，
    预检以 **exit 1** 退出——**正好在它要报告问题的时刻崩掉**，是最坏的一种失败。

    （`sys.stderr` 默认是 `backslashreplace`，所以子进程那边的 traceback 不受影响；
    这也是为什么只有 stdout 需要修。）
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # 被重定向到非文本流时 reconfigure 会抛
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_temp_configs(root: Path) -> Path:
    """造一份临时 `configs/`：照抄基线的 yaml，再加一个只改集合名的 profile。

    ⚠ **只覆盖集合名**——其余一切继承 `default.yaml`。预检要验的是**真配置下的真行为**，
    把别的键也改掉会让"预检通过"与"服务能跑"变成两件事。
    """
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_REPO / "configs" / "default.yaml", configs / "default.yaml")
    (configs / f"{_PREFLIGHT_PROFILE}.yaml").write_text(
        "# 预检专用：只把集合换成独立的那个，跑完 drop 掉（见 eval/smoke/preflight.py）\n"
        "storage:\n"
        "  qdrant:\n"
        f"    collection: {_PREFLIGHT_COLLECTION}\n",
        encoding="utf-8",
    )
    return configs


def _drop_collection(qdrant_url: str) -> None:
    """删掉预检集合（开始前先清一次，跑完再清一次）。404 是正常的。"""
    # 清不掉不该让预检失败——真正的门禁在后面的检查项上
    with contextlib.suppress(httpx.HTTPError):
        httpx.delete(f"{qdrant_url.rstrip('/')}/collections/{_PREFLIGHT_COLLECTION}", timeout=10.0)


def _rmtree_best_effort(path: Path, *, attempts: int = 5) -> None:
    """删掉临时目录。Windows 上句柄释放是**异步**的，所以重试几次。

    ⚠ **删不掉要说出来。** 用 `ignore_errors=True` 会把"留下了垃圾"也一起吞掉，
    而静默留下垃圾正是本项目最不想要的那一类失败——实测过：它一次性留下了 7 个临时目录，
    而屏幕上什么都没说。
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.2 * (attempt + 1))
    print(f"⚠ 临时目录没能删掉（多半还被占着）：{path}", file=sys.stderr)


@dataclass
class ServiceHandle:
    """一个在跑的服务（自己起的或用户给的）。"""

    base_url: str
    proc: subprocess.Popen[str] | None = None
    workdir: Path | None = None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover — 卡住才走到
                self.proc.kill()
                self.proc.wait(timeout=15)
        if self.workdir is not None:
            _rmtree_best_effort(self.workdir)


def _launch_service(*, timeout: float, verbose: bool) -> ServiceHandle:
    """起一个**隔离**的服务实例：临时库 + 临时缓存 + 临时集合。

    ⚠ **配置合不合规由服务自己说了算**——预检**不**在这里重复一遍"密钥填了吗"的检查。
    理由：服务读的是 `.env` + 真实环境变量的合并视图，而预检只能看到 `os.environ`。
    在这里再查一遍就会得到一个**与服务不一致**的判断（`.env` 里填了、shell 里没导出 ⇒
    预检说"没配"，而服务其实能起来）。⇒ **配置校验交给 `load_config()`，那是它的单一实现。**
    服务起不来时它的 stderr 会被这里捕获并原样转述（见下）。
    """
    workdir = Path(tempfile.mkdtemp(prefix="tianxi-preflight-"))
    configs = _write_temp_configs(workdir)
    port = _free_port()
    log_path = workdir / "service.log"

    env = dict(os.environ)
    env.update(
        {
            _ENV_CONFIG_DIR: str(configs),
            _ENV_PROFILE: _PREFLIGHT_PROFILE,
            _ENV_SQLITE_PATH: str(workdir / "preflight.db"),
            _ENV_EMBED_CACHE_DIR: str(workdir / "embed_cache"),
            _ENV_WORKERS: "1",  # 显式写死：有它 + assert_single_process 两道防线
            # 子进程的日志要按 UTF-8 写（它的报错也是中文）——否则读回来全是乱码。
            # 默认用的是**本机 locale**（这里 GBK），而我们按 UTF-8 读。
            "PYTHONIOENCODING": "utf-8",
        }
    )

    def _tail(lines: int = 12) -> str:
        """服务日志的尾部——起不来时它就是**最有用的一句话**。"""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:  # pragma: no cover
            return "（没有日志）"
        return "\n".join(text.splitlines()[-lines:]) or "（日志是空的）"

    # 子进程的日志写文件而不是 PIPE：没人读的 PIPE 会被写满并卡住子进程。
    if verbose:
        print(f"（服务日志：{log_path}）")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "tianximem.service.app:create_app_from_env",
                "--factory",
                "--workers",
                "1",
                "--port",
                str(port),
                "--log-level",
                "warning" if not verbose else "info",
            ],
            env=env,
            cwd=str(_REPO),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    # ⚠ 这里**立刻**关掉我们那份句柄：子进程拿的是自己的副本（Windows 上是
    # `STARTF_USESTDHANDLES` 给的独立 handle），所以它照旧能写；而父进程不再占着文件
    # ⇒ 后面删临时目录时不会被自己挡住。（`_tail()` 按**路径**重读，不靠这个句柄。）
    handle = ServiceHandle(base_url=f"http://127.0.0.1:{port}", proc=proc, workdir=workdir)
    _await_ready(handle, timeout=timeout, tail=_tail)
    return handle


def _await_ready(handle: ServiceHandle, *, timeout: float, tail: Callable[[], str]) -> None:
    """轮询 `/openapi.json` 直到 200；**起不来时把服务日志的尾部一起带出去**。

    ⚠ 两条失败路径都必须**先取日志再 `stop()`**：`stop()` 会把整个临时目录删掉，
    而日志就在里面——顺序反了就永远读不到那句最有用的话。
    """
    deadline = time.monotonic() + timeout
    last: str = "还没收到任何响应"
    while time.monotonic() < deadline:
        if handle.proc.poll() is not None:
            exit_code = handle.proc.returncode
            log_tail = tail()
            handle.stop()
            raise PreconditionError(
                f"服务进程启动即退出（exit={exit_code}）。它自己说的是：\n"
                f"----\n{log_tail}\n----\n"
                "（配置校验由 load_config() 负责，上面的报错就是它给的。）"
            )
        try:
            resp = httpx.get(f"{handle.base_url}/openapi.json", timeout=2.0)
            if resp.status_code == 200:
                return
            last = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(0.3)

    log_tail = tail()  # 同上：先取日志再删目录
    handle.stop()
    raise PreconditionError(
        f"服务在 {timeout:.0f}s 内没起来（最后一次：{last}）。日志尾部：\n{log_tail}"
    )


class PreconditionError(RuntimeError):
    """跑不起来（不是检查不过）。"""


# ── 预检本体 ───────────────────────────────────────────────────────────


class Preflight:
    """把契约清单逐条落成检查。**每条检查自己造数据**，不依赖别的检查留下的状态。

    `client` 可注入——**这不是为了方便，是为了让"检查真的会失败"可被证明**：
    `tests/test_contract_preflight.py` 塞一个故意违规的假服务进去，
    逐条确认对应的检查**确实报 FAIL**。一个永远不会 FAIL 的门禁不是门禁。
    """

    def __init__(self, base_url: str, *, nonce: str, client: httpx.Client | None = None) -> None:
        self._client = (
            client if client is not None else httpx.Client(base_url=base_url, timeout=60.0)
        )
        self._owns_client = client is None
        self._nonce = nonce
        self.report = Report()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ── 原语 ───────────────────────────────────────────────────────────

    def _add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, object]],
    ) -> httpx.Response:
        return self._client.post(
            "/add",
            json={
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "messages": messages,
            },
        )

    def _search(self, *, user_id: str, query: str, top_k: int) -> httpx.Response:
        return self._client.post(
            "/search", json={"user_id": user_id, "query": query, "top_k": top_k}
        )

    def _data(self, resp: httpx.Response) -> list[dict[str, object]]:
        """取 `data`，并**顺便断言它的类型**——`[]` 与 `null` 的区别是契约的一部分。"""
        expect(resp.status_code == 200, f"期望 200，收到 {resp.status_code}：{resp.text[:200]}")
        body = resp.json()
        expect(isinstance(body, dict), f"响应体不是对象：{body!r}")
        expect("data" in body, f"响应里没有 `data` 字段：{sorted(body)}")
        data = body["data"]
        expect(isinstance(data, list), f"`data` 必须是数组（空结果要 `[]` 不是 `null`）：{data!r}")
        return data

    def _seed(
        self,
        user_id: str,
        session_id: str,
        tags: list[str],
        *,
        separate_sessions: bool = False,
    ) -> list[tuple[str, str]]:
        """写入若干**可识别**的记忆，返回 `[(question, answer)]`。

        ⚠ 每个 `request_id` 与每段正文都带本次运行的 nonce ⇒ **预检之间不会互相命中**，
        也不会命中开发期残留的数据。

        ⚠ **`separate_sessions=True` 时每个 tag 各占一个 `session_id`**。
        这不是洁癖：位置相邻的记忆会被合并成**同一个 Context Segment**，
        于是响应里**只有一项**——凡是"要多条才能验"的检查（`score` 的单调性、
        `top_k` 真的会截断）都必须让它们分属不同 session，**否则检查会静默变空过**。

        ⚠ **`request_id` 只是"唯一 + 原样回显"**（§2.1）——服务端**不解析它**（D28）。
        这里仍然把 `chunk-<n>` 拼进去，纯粹是为了让 id 在日志里**可读**；
        它**不是**格式要求，服务端认任何形状（`abc` / UUID / `foo:bar` 都行）。
        """
        made: list[tuple[str, str]] = []
        for ordinal, tag in enumerate(tags):
            question = f"{self._nonce} {tag} question"
            answer = f"{self._nonce} {tag} answer"
            resp = self._add(
                request_id=f"{self._nonce}-{user_id}-{tag}-chunk-{ordinal}",
                user_id=user_id,
                session_id=f"{session_id}-{tag}" if separate_sessions else session_id,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            )
            expect(
                resp.status_code == 200,
                f"写入 {tag} 失败：期望 200，收到 {resp.status_code}：{resp.text[:200]}",
            )
            made.append((question, answer))
        return made

    # ── 检查 ───────────────────────────────────────────────────────────

    def check_add_echoes_the_three_fields(self) -> None:
        """§2.1：`200` + `success: true` + 三个字段**逐字原样回显**。"""
        rid, uid, sid = f"{self._nonce}-echo-chunk-0", f"{self._nonce}-u-echo", "s-echo-中文-ü"
        resp = self._add(
            request_id=rid,
            user_id=uid,
            session_id=sid,
            messages=[{"role": "user", "content": f"{self._nonce} echo"}],
        )
        expect(resp.status_code == 200, f"期望 200，收到 {resp.status_code}：{resp.text[:200]}")
        body = resp.json()
        expect(body.get("success") is True, f"`success` 必须是 true：{body!r}")
        expect(
            body.get("request_id") == rid, f"`request_id` 没逐字回显：{body.get('request_id')!r}"
        )
        expect(body.get("user_id") == uid, f"`user_id` 没逐字回显：{body.get('user_id')!r}")
        expect(
            body.get("session_id") == sid, f"`session_id` 没逐字回显：{body.get('session_id')!r}"
        )

    def check_add_rejects_illegal_requests(self) -> None:
        """非法请求必须是**明确的 4xx**（不是 500、也不是静默接受）。

        ⚠ 非 200 的行为契约**未定义**（§15 ⇒ 假设 AML 会重试），所以 `Add` 侧真正要守的是
        "**合法的必须 200**"；但**边界校验落在 4xx 而不是 5xx** 仍要钉住——
        5xx 意味着我们的代码崩了，那是另一类问题。
        """
        good = {
            "request_id": "r",
            "user_id": "u",
            "session_id": "s",
            "messages": [{"role": "user", "content": "x"}],
        }
        cases: list[tuple[str, dict[str, object]]] = [
            ("messages 为空数组", {**good, "messages": []}),
            ("缺 request_id", {k: v for k, v in good.items() if k != "request_id"}),
            ("缺 user_id", {k: v for k, v in good.items() if k != "user_id"}),
            ("缺 session_id", {k: v for k, v in good.items() if k != "session_id"}),
            ("role 为空串", {**good, "messages": [{"role": "", "content": "x"}]}),
            ("messages 不是数组", {**good, "messages": "nope"}),
        ]
        for name, payload in cases:
            resp = self._client.post("/add", json=payload)
            expect(
                400 <= resp.status_code < 500,
                f"{name}：期望 4xx，收到 {resp.status_code}",
            )

    def check_search_returns_a_list_of_exactly_four_fields(self) -> None:
        """§2.1：`data` 是数组；每项**恰好** `id` / `content` / `created_at` / `score`。"""
        user = f"{self._nonce}-u-shape"
        self._seed(user, "s-shape", ["shape"])
        data = self._data(
            self._search(user_id=user, query=f"{self._nonce} shape question", top_k=10)
        )
        expect(len(data) >= 1, "刚写入的记忆检索不到——先看 Add→Search 那条")

        for item in data:
            expect(isinstance(item, dict), f"`data` 的项不是对象：{item!r}")
            expect(
                set(item) == {"id", "content", "created_at", "score"},
                f"字段集合必须恰好是 4 个，收到 {sorted(item)}",
            )
            expect(
                isinstance(item["id"], str) and item["id"], f"`id` 必须是非空字符串：{item['id']!r}"
            )
            expect(
                isinstance(item["content"], str), f"`content` 必须是字符串：{type(item['content'])}"
            )
            expect(
                isinstance(item["created_at"], str),
                f"`created_at` 必须是字符串（日粒度或空串）：{item['created_at']!r}",
            )
            expect(isinstance(item["score"], (int, float)), f"`score` 必须是数：{item['score']!r}")

    def check_created_at_is_day_grain_or_empty(self) -> None:
        """§11.3：`created_at` **始终存在**，且是 `YYYY-MM-DD` 或空串。

        ⚠ 空串是**有定义的降级路径**（消息没带 `timestamp` ⇒ `event_time` 为 NULL）。
        本条用**不带 `timestamp`** 的消息写入，所以这里期望的就是空串——
        顺带证明了我们没有拿"Add 的到达时间"兜底（那会给模型错误信息）。
        """
        import re

        user = f"{self._nonce}-u-createdat"
        self._seed(user, "s-createdat", ["no-ts"])  # 不带 timestamp
        data = self._data(self._search(user_id=user, query=f"{self._nonce} no-ts", top_k=10))
        expect(len(data) >= 1, "刚写入的记忆检索不到")

        for item in data:
            value = item["created_at"]
            expect(
                value == "" or re.match(_DAY_GRAIN_RE, str(value)) is not None,
                f"`created_at` 必须是 `YYYY-MM-DD` 或空串，收到 {value!r}",
            )

        # 另一侧：**带了** timestamp 的记忆必须给出日粒度日期
        user_ts = f"{self._nonce}-u-createdat-ts"
        resp = self._add(
            request_id=f"{self._nonce}-ts-chunk-0",
            user_id=user_ts,
            session_id="s-ts",
            messages=[
                # 2023-05-08T23:30Z：UTC 口径下是 05-08（东八区会是 05-09）——跨日点
                {
                    "role": "user",
                    "content": f"{self._nonce} ts question",
                    "timestamp": 1683588600000,
                },
                {
                    "role": "assistant",
                    "content": f"{self._nonce} ts answer",
                    "timestamp": 1683588600000,
                },
            ],
        )
        expect(resp.status_code == 200, f"带 timestamp 的 Add 失败：{resp.status_code}")
        data_ts = self._data(
            self._search(user_id=user_ts, query=f"{self._nonce} ts question", top_k=10)
        )
        expect(len(data_ts) >= 1, "带 timestamp 的记忆检索不到")
        expect(
            data_ts[0]["created_at"] == "2023-05-08",
            f"带 timestamp 的记忆应当给出 UTC 日粒度 2023-05-08（无旋钮），"
            f"收到 {data_ts[0]['created_at']!r}",
        )

    def check_content_properties(self) -> None:
        """§11.3：`content` 的**性质**（不比对我们的 render —— 那是循环论证）。

        四条：含原文（证明是**证据**而不是生成的答案）· 有 `Q:` / `A:` 行首标记 ·
        首尾无空白（AML 只做 `"\\n".join`，不插分隔符）· **无绝对时间戳前缀**。
        """
        import re

        user = f"{self._nonce}-u-content"
        [(question, answer)] = self._seed(user, "s-content", ["content"])
        data = self._data(self._search(user_id=user, query=question, top_k=10))
        expect(len(data) >= 1, "刚写入的记忆检索不到")

        content = str(data[0]["content"])
        expect(
            question in content,
            f"`content` 里没有我们写入的问题原文——它可能是生成出来的：{content[:200]!r}",
        )
        expect(answer in content, f"`content` 里没有我们写入的回答原文：{content[:200]!r}")
        expect(
            "Q:" in content, f"`content` 缺 `Q:` 行首标记（相邻项会被粘成一段）：{content[:200]!r}"
        )
        expect(
            content == content.strip(),
            f"`content` 首尾有空白：{content[:80]!r} … {content[-80:]!r}",
        )
        expect(
            not re.search(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", content),
            f"`content` 里有绝对时间戳（会同时踩中裁判的两条 TIME 规则）：{content[:200]!r}",
        )

    def check_score_is_rank_derived(self) -> None:
        """§11.3：`score` **单调递减**、`rank=0` 就是 `1.0`，且**不是**原始 RRF 分数。

        ⚠ 三条记忆**分属三个 session**：同一个 session 里位置相邻的记忆会被
        合并成**一个** Context Segment，那样响应里只有一项，单调性就**没东西可验**了
        （检查会静默空过——它只说"≥ 2 条"，不够，得真的拿到多条）。
        """
        user = f"{self._nonce}-u-score"
        self._seed(user, "s-score", ["score-a", "score-b", "score-c"], separate_sessions=True)
        data = self._data(self._search(user_id=user, query=f"{self._nonce} score", top_k=10))
        expect(len(data) >= 2, f"这条要用多条的响应来验单调性，只拿到 {len(data)} 条")

        scores = [float(item["score"]) for item in data]
        expect(scores[0] == 1.0, f"rank=0 的 score 必须是 1.0，收到 {scores[0]!r}")
        for i in range(1, len(scores)):
            expect(
                scores[i] < scores[i - 1],
                f"score 必须**严格**单调递减（AML 若按它重排，顺序不能变）：{scores}",
            )
        expect(
            all(s >= _MIN_PLAUSIBLE_SCORE for s in scores),
            f"score 的量级像融合分数（RRF k=61 约 0.016）而不是名次倒数：{scores}",
        )
        expect(
            scores == [1.0 / (i + 1) for i in range(len(scores))],
            f"score 应当是 `1/(rank+1)`（config-reference 的 `packaging.score_mode`）：{scores}",
        )

    def check_count_is_within_top_k(self) -> None:
        """§2.2：`len(data) <= top_k` **精确成立**，且 `top_k` 真的被用上了。

        ⚠ **`<= top_k` 单独是空过的**：写死 `limit=100` 也能满足它。
        所以先用 `top_k=1` 验"**真的会截断**"——库里明明有 3 条，只许回 1 条。

        ⚠ 三条记忆**分属三个 session**：同一个 session 里相邻的三条会合并成**一个**
        Context Segment，`top_k=1` 返回 1 条就成了**必然**，
        这条检查也就跟着空过了（它要验的是"截断"，不是"合并"）。
        """
        user = f"{self._nonce}-u-count"
        self._seed(user, "s-count", ["count-a", "count-b", "count-c"], separate_sessions=True)
        query = f"{self._nonce} count"

        for top_k in (1, 10, 100):
            data = self._data(self._search(user_id=user, query=query, top_k=top_k))
            expect(
                len(data) <= top_k,
                f"top_k={top_k} 时返回了 {len(data)} 条 —— **返回超过 top_k 是契约错误**",
            )

        # 非空过的那一半：库里 3 条（3 个 session ⇒ 3 个段），top_k=1 必须**恰好** 1 条
        one = self._data(self._search(user_id=user, query=query, top_k=1))
        expect(len(one) == 1, f"库里有 3 段，top_k=1 却返回 {len(one)} 条——`top_k` 没被用上")

    def check_empty_result_is_a_list(self) -> None:
        """§2.1：空结果是 `[]`，**不是 `null`**。"""
        user = f"{self._nonce}-u-empty"  # 这个 user 从没写过任何东西
        resp = self._search(user_id=user, query="never written anything", top_k=10)
        expect(
            resp.status_code == 200, f"空库检索应当 200，收到 {resp.status_code}：{resp.text[:200]}"
        )
        expect(
            resp.json() == {"data": []},
            f"空结果必须是 `{{'data': []}}`（不是 null、也不能缺字段）：{resp.json()!r}",
        )

    def check_add_then_search_immediately(self) -> None:
        """§2.1：**Add 返回 200 时，那条记忆已经可检索。**

        "不允许异步建索引"是契约原文。这条是它的唯一可观测形式：
        Add 的响应一回来就查，**中间不等任何东西**。
        """
        user = f"{self._nonce}-u-e2e"
        [(question, _answer)] = self._seed(user, "s-e2e", ["e2e"])

        data = self._data(self._search(user_id=user, query=question, top_k=10))
        expect(
            len(data) >= 1,
            "Add 已经返回 200，但紧接着的 Search 什么都没召回到 —— "
            "违反『响应前必须持久化完成且立即可搜索』（§2.1）",
        )
        expect(
            question in str(data[0]["content"]),
            f"召回的第一条不是刚写进去的那条：{str(data[0]['content'])[:200]!r}",
        )

    def check_isolation_between_users(self) -> None:
        """§2.2：`user_id` 是**唯一**的隔离字段——A 写的，B 绝不能看到。"""
        alice = f"{self._nonce}-u-alice"
        bob = f"{self._nonce}-u-bob"
        self._seed(alice, "s-iso", ["alice-secret"])
        self._seed(bob, "s-iso", ["bob-secret"])

        query = f"{self._nonce} secret"
        a_data = self._data(self._search(user_id=alice, query=query, top_k=100))
        b_data = self._data(self._search(user_id=bob, query=query, top_k=100))

        # 先证明**检索本身是有效的**（否则"B 看不到"可能只是因为谁都没召回）
        expect(len(a_data) >= 1, "alice 检索不到自己写的记忆——先看 Add→Search 那条")
        expect(len(b_data) >= 1, "bob 检索不到自己写的记忆")

        a_ids = {str(item["id"]) for item in a_data}
        b_ids = {str(item["id"]) for item in b_data}
        expect(a_ids.isdisjoint(b_ids), f"两个 user 返回了同一个 id：{sorted(a_ids & b_ids)}")
        expect(
            all("alice-secret" not in str(item["content"]) for item in b_data),
            "bob 的检索结果里出现了 alice 的内容——**跨 user 检索被禁止**（§2.2）",
        )
        expect(
            all("bob-secret" not in str(item["content"]) for item in a_data),
            "alice 的检索结果里出现了 bob 的内容",
        )

    def check_session_id_is_not_a_filter(self) -> None:
        """§2.2：`session_id` 只是分组字段，**不是 Search 的过滤器**。

        同一 `user_id` 下的另外一个 session 写的记忆，必须能被检索到。
        （结构上 `SearchRequest` 里根本没有 `session_id` 字段——本条是它的**行为**证明。）
        """
        user = f"{self._nonce}-u-session"
        self._seed(user, "s-first", ["cross-session"])

        data = self._data(
            self._search(user_id=user, query=f"{self._nonce} cross-session", top_k=10)
        )
        expect(
            len(data) >= 1,
            "同一 user 另一个 session 的记忆检索不到——`session_id` 被当成了过滤器（§2.2）",
        )

    def check_replay_does_not_write_again(self) -> None:
        """§2.2：同一 `request_id` 重复 POST（payload 不变）⇒ **库里没有新增行**。

        **怎么在 HTTP 层看见"没有新增行"**：**两次检索返回的 id 集合与条数完全一致**。
        ⚠ 位置是请求的纯函数 ⇒ 重放算出同一位置 ⇒ **没有守卫时**重放会撞 `UNIQUE`、
        整批**非 200**（响亮），而不是静默多一行。
        ⇒ 所以这条检查验的是**"守卫把正常重试从 500 里救回来"**。

        ⚠ **另一半在段模型下才看得出来**：若有行被重复写入，新行与旧行在同一个 session 里
        **相邻** ⇒ 它们会被合并进**同一个** Context Segment。于是重复写入**不再表现为多一项**，
        而是**同一段变宽**（同样的问答在 `content` 里出现两次）。
        所以只比 `id` 与条数会**漏掉**它——必须连 `content` 一起比。
        """
        user = f"{self._nonce}-u-replay"
        question = f"{self._nonce} replay question"
        answer = f"{self._nonce} replay answer"
        payload = {
            "request_id": f"{self._nonce}-replay-chunk-0",
            "user_id": user,
            "session_id": "s-replay",
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

        first = self._client.post("/add", json=payload)
        expect(first.status_code == 200, f"第一次 Add 失败：{first.status_code}")
        before = self._data(self._search(user_id=user, query=question, top_k=100))
        expect(len(before) >= 1, "第一次 Add 后检索不到")

        # payload **逐字不变**地重放（AML 重试就是这样）
        for attempt in range(2, 4):
            replay = self._client.post("/add", json=payload)
            expect(
                replay.status_code == 200,
                f"第 {attempt} 次重放应当也返回 200（重试必须能拿到成功），"
                f"收到 {replay.status_code}",
            )

        after = self._data(self._search(user_id=user, query=question, top_k=100))
        expect(
            [str(i["id"]) for i in after] == [str(i["id"]) for i in before],
            f"重放后检索到的 id 变了 —— 说明**位置被重新分配**、库里多了行**（幂等破了）**：\n"
            f"      之前 {[str(i['id'])[:12] for i in before]}\n"
            f"      之后 {[str(i['id'])[:12] for i in after]}",
        )
        expect(
            len(after) == len(before),
            f"重放后条目数从 {len(before)} 变成 {len(after)} —— 重复写入了",
        )
        # ★ 段模型下的那一半：多一行不会多一项，只会让**同一段**里多一份相同的原文
        expect(
            [str(i["content"]) for i in after] == [str(i["content"]) for i in before],
            "重放后正文变了 —— 同一段里多出了一份相同的记忆（重复写入）",
        )
        expect(
            sum(str(i["content"]).count(question) for i in after) == 1,
            f"同一段里出现了两次相同的问题原文 —— 库里有重复行：{after[0]['content'][:200]!r}",
        )

    def check_search_rejects_illegal_requests(self) -> None:
        """`Search` 的边界：`top_k <= 0`、空白 `query`、缺字段都必须是 4xx。"""
        cases: list[tuple[str, dict[str, object]]] = [
            ("top_k = 0", {"user_id": "u", "query": "q", "top_k": 0}),
            ("top_k = -1", {"user_id": "u", "query": "q", "top_k": -1}),
            ("query 是空白", {"user_id": "u", "query": "   ", "top_k": 10}),
            ("query 是空串", {"user_id": "u", "query": "", "top_k": 10}),
            ("缺 user_id", {"query": "q", "top_k": 10}),
            ("缺 top_k", {"user_id": "u", "query": "q"}),
        ]
        for name, payload in cases:
            resp = self._client.post("/search", json=payload)
            expect(400 <= resp.status_code < 500, f"{name}：期望 4xx，收到 {resp.status_code}")

    def check_unknown_fields_are_tolerated(self) -> None:
        """**未知字段一律忽略**（不做 `extra="forbid"`）。

        理由在 [`../../src/tianximem/service/schemas.py`](../../src/tianximem/service/schemas.py)：
        AML 将来加字段不该让我们 422——那会把它送进重试。
        """
        resp = self._client.post(
            "/search",
            json={"user_id": f"{self._nonce}-u-extra", "query": "q", "top_k": 3, "future_field": 1},
        )
        expect(resp.status_code == 200, f"带未知字段的请求应当被容忍，收到 {resp.status_code}")

    # ── 跑 ─────────────────────────────────────────────────────────────

    def run(self) -> Report:
        """按**依赖顺序**跑（前面的失败会让后面的失去意义，但仍全部跑完并汇报）。"""
        checks: list[tuple[str, Callable[[], None]]] = [
            ("Add · 200 且逐字回显三个字段", self.check_add_echoes_the_three_fields),
            ("Add · 非法请求是 4xx", self.check_add_rejects_illegal_requests),
            ("Add → Search · 200 后立即可检索", self.check_add_then_search_immediately),
            (
                "Search · data 是数组、每项恰好四个字段",
                self.check_search_returns_a_list_of_exactly_four_fields,
            ),
            ("Search · created_at 是日粒度或空串", self.check_created_at_is_day_grain_or_empty),
            ("Search · content 的四条性质", self.check_content_properties),
            ("Search · score 由名次生成、不是融合分数", self.check_score_is_rank_derived),
            ("Search · 精确计数 ≤ top_k（且 top_k 真的生效）", self.check_count_is_within_top_k),
            ("Search · 空结果是 []", self.check_empty_result_is_a_list),
            (
                "Search · 非法请求是 4xx / 未知字段被忽略",
                self.check_search_rejects_illegal_requests,
            ),
            ("Search · 未知字段被容忍", self.check_unknown_fields_are_tolerated),
            ("隔离 · user_id 是唯一隔离字段", self.check_isolation_between_users),
            ("隔离 · session_id 不是过滤器", self.check_session_id_is_not_a_filter),
            ("幂等 · 重复 request_id 不重复写入", self.check_replay_does_not_write_again),
        ]

        for name, fn in checks:
            try:
                fn()
            except CheckFailed as exc:
                self.report.add(name, False, str(exc))
            except Exception as exc:  # noqa: BLE001 — 任何异常都算这条没过，但要记类型
                self.report.add(name, False, f"{type(exc).__name__}: {exc}")
            else:
                self.report.add(name, True)
        return self.report


# ── 入口 ───────────────────────────────────────────────────────────────


def _check_qdrant(env: dict[str, str]) -> None:
    """Qdrant 必须可达。**明确失败，不静默跳过。**

    ⚠ **这里只查 Qdrant，不查 embedding 密钥**——后者由服务自己校验
    （`load_config()` 是它唯一的实现，见 `_launch_service` 的说明）。
    预检需要 Qdrant 是因为**它自己要 drop 那个临时集合**，而不只是为了转述服务的问题。
    """
    url = (env.get(_ENV_QDRANT_URL) or _DEFAULT_QDRANT_URL).rstrip("/")
    try:
        resp = httpx.get(f"{url}/collections", timeout=10.0)
    except httpx.HTTPError as exc:
        raise PreconditionError(
            f"Qdrant 不可达（{url}）：{type(exc).__name__}。"
            "它是必需的前置（检索是契约的一半，而且预检要自己建/删临时集合）——"
            "`make qdrant-up`，或确认 Docker 在跑。"
        ) from exc
    if resp.status_code != 200:
        raise PreconditionError(f"Qdrant 返回 HTTP {resp.status_code}（{url}）")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()  # ⚠ 必须在任何 print 之前（见它的 docstring）
    parser = argparse.ArgumentParser(
        prog="preflight",
        description="AML 契约预检：在真服务上跑 contract.md §4 的清单（本地，不耗 Smoke 配额）",
    )
    parser.add_argument("--base-url", default=None, help="打已经在跑的服务（不给就自己起一个）")
    parser.add_argument("--timeout", type=float, default=60.0, help="等服务起来的上限秒数")
    parser.add_argument("--keep", action="store_true", help="不清理临时目录（排错用）")
    parser.add_argument("--verbose", action="store_true", help="把服务子进程的日志打到屏幕上")
    args = parser.parse_args(argv)

    print("AML 契约预检（本地，不消耗 Smoke 配额）")
    print("=" * 68)

    try:
        _check_qdrant(dict(os.environ))
    except PreconditionError as exc:
        print(f"\n前置条件不满足，预检没有开始：\n  {exc}\n")
        return EXIT_PRECONDITION_FAILED

    handle: ServiceHandle | None = None
    preflight: Preflight | None = None
    try:
        try:
            if args.base_url:
                handle = ServiceHandle(base_url=args.base_url.rstrip("/"))
                print(f"服务：{handle.base_url}（外部）")
            else:
                handle = _launch_service(timeout=args.timeout, verbose=args.verbose)
                print(
                    f"服务：{handle.base_url}"
                    f"（本次自启 · 集合 {_PREFLIGHT_COLLECTION} · 临时库与缓存）"
                )
        except PreconditionError as exc:
            # ⚠ 起不来也是"**没跑成**"（exit 2），不是"跑了但没过"（exit 1）。
            # 两者的处置完全不同：前者要你去修环境，后者要你去修代码。
            print(f"\n前置条件不满足，预检没有开始：\n{exc}\n")
            return EXIT_PRECONDITION_FAILED

        qdrant_url = (os.environ.get(_ENV_QDRANT_URL) or _DEFAULT_QDRANT_URL).rstrip("/")
        print(f"Qdrant：{qdrant_url}")
        print(f"profile：{_PREFLIGHT_PROFILE}\n")

        preflight = Preflight(handle.base_url, nonce=f"pf-{uuid.uuid4().hex[:8]}")
        report = preflight.run()
        print(report.render())

        print("\n" + "=" * 68)
        if report.failed:
            print(
                f"**{len(report.failed)} / {len(report.results)} 条不通过** —— "
                "契约不合规，不要跑 Smoke。"
            )
            print("\n过一遍上表里 FAIL 的项；`contract.md` §4 是清单的唯一声明处。")
            return EXIT_CHECKS_FAILED
        print(
            f"全部 {len(report.results)} 条通过。契约合规"
            "（仅就本地可验证的部分——S1/S2/S3 仍需 Smoke）。"
        )
        return EXIT_OK
    finally:
        if preflight is not None:
            preflight.close()
        if handle is not None:
            if args.keep and handle.workdir is not None:
                print(f"\n（--keep：临时目录留在 {handle.workdir}）")
                handle.proc and handle.proc.terminate()
            else:
                handle.stop()
            if not args.base_url:
                _drop_collection(
                    (os.environ.get(_ENV_QDRANT_URL) or _DEFAULT_QDRANT_URL).rstrip("/")
                )


if __name__ == "__main__":
    sys.exit(main())
