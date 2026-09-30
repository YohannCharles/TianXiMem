"""位置模型的**性质探针**：**乱序 + 并发投喂**与**顺序投喂**得到逐字相同的真源。

这是"位置是请求的纯函数"这条性质真正要买的东西（D25 起，**D28 之后更强**：
位置里连"顺序"这个概念都不再有了，跨 Add 既不比较先后、也不建立邻接）。
两条臂各起一个干净服务（各自的库与集合）：

  A. 顺序：harness 的 driver 按源序逐批 Add（= 线上正常情况下 AML 的形态）
  B. 乱序 + 并发：同一批 Add 打乱顺序、多线程同时打

⚠ **它要花掉真实的 embedding 调用**（conv-26 全量 ≈ 216 块），所以不进 `make test`——
`tests/test_apply.py::test_arrival_order_does_not_change_the_truth_source` 已经在单元层
盖住了同一条性质（乱序与顺序的位置 / `id` / 邻接逐字一致）。
本探针盖的是**单元测试盖不到的那一段**：真 HTTP、真并发、真 Qdrant、真 embedding。

判据（逐字，不是"大概一致"）：
  1. `(request_id, local_index)` 的集合相同
  2. 每个位置的 `(question, answer, event_time)` 逐字相同
  3. **`id` 集合相同**（位置派生 ⇒ 位置一致就该 id 一致）
  4. **`prev` / `next` 指针逐字相同**（邻接只由请求决定，与到达顺序无关——D28）

跑法：`make order-probe`（= `uv run python tools/order_probe.py`）
      `make order-probe PROCESSES=4` —— B 臂起 **4 个独立进程**共享同一套存储，
      回答的是「`uvicorn --workers N` 到底安不安全」那个问题
前置：`.env` 里的 `AML_EMB_*` 可用；`127.0.0.1:8131` / `:8132` 两个端口空着。
产物：`var/order-a/` 与 `var/order-b/`（各自的库、集合名、服务日志；都在 gitignore 里）。
⚠ **每条臂各自一个 Qdrant 集合**（`memories_ab_order_a` / `_b`）——不复用开发集合，
  否则上一轮的 point 会**静默留在里面**（[`../docs/decisions.md`](../docs/decisions.md) V9）。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

# ⚠ **本仓唯一一个 import `eval` 的 `tools/` 脚本**，所以需要下面这一行：
#   `eval` 不是已安装的包（`pyproject.toml` 只装 `src/tianximem`），它靠 cwd 或 pytest 的
#   `pythonpath` 才可见。而直接跑脚本时 `sys.path[0]` 是**脚本所在目录**（`tools/`），
#   不是仓库根 ⇒ 不插这一行就是 `ModuleNotFoundError: No module named 'eval'`。
#   `eval/harness/` 对归档 pipeline 用的是同一招（注入 `PYTHONPATH`）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets import benchmark_dir, load_locomo  # noqa: E402
from eval.harness import batches, request_id_for  # noqa: E402

# ⚠ **`parents[1]` 而不是 `parent`**：本文件在 `tools/` 下，而 `var/` 在仓库根。
#   （写成 `parent` 的话产物会静默落进 `tools/var/`。）
ROOT = Path(__file__).resolve().parents[1]
PORT_A = 8131
PORT_B = 8140  # B 臂的 N 个前端占 PORT_B .. PORT_B+N-1


def _config_dir(name: str, collection: str) -> Path:
    d = ROOT / "var" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "default.yaml").write_text(
        (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (d / "local.yaml").write_text(
        f"storage:\n  qdrant:\n    collection: {collection}\n", encoding="utf-8"
    )
    return d


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _serve(name: str, port: int, collection: str, db: Path, tag: str = "") -> subprocess.Popen:
    """起一个**单进程**前端（`--workers 1`）。

    ⚠ `db` 是**显式传进来的**：`--processes N` 时 N 个前端必须指向**同一个 SQLite 文件**
    ——那正是 `uvicorn --workers N` 的形状（N 个进程共享一套存储）。
    每个前端各有各的 Qdrant 客户端与 SQLite 连接，共享的只有库文件与集合。
    """
    # ⚠ **起之前必须先确认端口空着**——否则会撞上"上一轮没杀干净的服务"。
    #   后果不是报错，而是**静默测了别人的库**：`_serve` 的探活会立刻拿到那个残留服务的
    #   `/health` 200、返回一个马上死掉的进程，随后所有请求都打到残留服务上
    #   （它的 `TIANXIMEM_SQLITE_PATH` 指向别的目录）⇒ 断言失败在一个与被测对象无关的地方。
    #   这正是本项目反复对付的那一类失败：**它看起来像"代码坏了"，其实是"测错了对象"**。
    if not _port_is_free(port):
        raise RuntimeError(
            f"{name}:{port} 已被占用——本探针**不复用**任何已在跑的服务（那会测到别人的库）。\n"
            f"  先清掉：`ss -ltnp | grep :{port}` 找到 pid，再 kill 它。"
        )
    cfg = _config_dir(name, collection)
    env = {
        "PATH": f"{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "TIANXIMEM_PROFILE": "local",
        "TIANXIMEM_CONFIG_DIR": str(cfg),
        "TIANXIMEM_SQLITE_PATH": str(db),
        # 向量缓存落在探针自己的目录里：复用别人的缓存目录会在那个实验被清掉时静默重付一遍
        "TIANXIMEM_EMBED_CACHE_DIR": str(ROOT / "var" / "order-probe-cache"),
    }
    proc = subprocess.Popen(
        [
            str(Path.home() / ".local" / "bin" / "uv"),
            "run",
            "--env-file",
            str(ROOT / ".env"),
            "uvicorn",
            "tianximem.service.app:create_app_from_env",
            "--factory",
            "--workers",
            "1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env={**env, "HOME": str(Path.home())},
        stdout=(cfg / f"serve{tag}.log").open("wb"),
        stderr=subprocess.STDOUT,
        # ⚠ **必须自己开一个进程组**：`uv run` 是包装进程，真正的 uvicorn 是它的**子进程**。
        #   只 `kill()` 包装进程的话，uvicorn 会活下来继续占着端口——
        #   上一版就是这样在 8131/8132 上留了两个孤儿，害得下一轮跑测错了对象。
        start_new_session=True,
    )
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"{name}:{port} 服务自己退了，看 {cfg}/serve{tag}.log")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc
        except Exception:  # noqa: BLE001 — 还没起来
            time.sleep(1)
    _kill(proc)
    raise RuntimeError(f"{name}:{port} 服务起不来")


def _kill(proc: subprocess.Popen) -> None:
    """杀掉**整个进程组**（包装进程 + 它拉起来的 uvicorn）。"""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _sqlite_rows(db: Path) -> list[tuple]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT id, request_id, local_index, question, answer, event_time,"
            " prev_memory_id, next_memory_id"
            " FROM qa_pairs ORDER BY session_id, request_id, local_index"
        ).fetchall()
    finally:
        conn.close()


def main(processes: int = 1) -> int:
    sample = load_locomo(benchmark_dir())[0]
    jobs: list[tuple[str, str, tuple]] = []
    for session in sample.sessions:
        for index, batch in enumerate(batches(session.messages)):
            jobs.append(
                (
                    request_id_for(sample.user_id, session.session_id, index),
                    session.session_id,
                    batch,
                )
            )
    print(f"conv-26：{len(sample.sessions)} session · {len(jobs)} 批 Add")
    if processes > 1:
        print(f"B 臂：**{processes} 个独立进程**共享同一个 SQLite + 同一个集合")

    dir_a, dir_b = ROOT / "var" / "order-a", ROOT / "var" / "order-b"
    procs = [
        _serve("order-a", PORT_A, "memories_ab_order_a", dir_a / "tianxi.db"),
        *[
            _serve("order-b", PORT_B + i, "memories_ab_order_b", dir_b / "tianxi.db", tag=f"-{i}")
            for i in range(processes)
        ],
    ]
    b_ports = [PORT_B + i for i in range(processes)]
    try:
        # ── A 臂：顺序 ──
        with httpx.Client(timeout=120) as client:
            for rid, sid, batch in jobs:
                resp = client.post(
                    f"http://127.0.0.1:{PORT_A}/add",
                    json={
                        "request_id": rid,
                        "user_id": sample.user_id,
                        "session_id": sid,
                        "messages": [
                            {"role": m.role, "content": m.content, "timestamp": m.timestamp_ms}
                            for m in batch
                        ],
                    },
                )
                assert resp.status_code == 200, resp.text
        print("A 臂（顺序）完成")

        # ── B 臂：打乱 + 并发 ──
        # ⚠ 请求**轮流**分给 N 个前端（而不是"每个前端各拿一段连续的"）：
        #   要让同一 session 的相邻批次**落到不同进程**上，否则测不到共享存储的争用。
        shuffled = list(jobs)
        random.Random(20260927).shuffle(shuffled)
        errors: list[BaseException] = []

        def push(job, port: int) -> None:
            rid, sid, batch = job
            try:
                with httpx.Client(timeout=180) as client:
                    resp = client.post(
                        f"http://127.0.0.1:{port}/add",
                        json={
                            "request_id": rid,
                            "user_id": sample.user_id,
                            "session_id": sid,
                            "messages": [
                                {"role": m.role, "content": m.content, "timestamp": m.timestamp_ms}
                                for m in batch
                            ],
                        },
                    )
                    if resp.status_code != 200:
                        errors.append(
                            RuntimeError(f"{rid}@{port}: {resp.status_code} {resp.text[:200]}")
                        )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=push, args=(job, b_ports[i % processes]))
            for i, job in enumerate(shuffled)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"B 臂（乱序 + {len(threads)} 线程并发 → {processes} 个进程）完成")
        if errors:
            for e in errors[:5]:
                print("  ⚠", e)
            print(f"  （共 {len(errors)} 个失败）")
            return 1

        # ── 逐字比对 ──
        a = _sqlite_rows(dir_a / "tianxi.db")
        b = _sqlite_rows(dir_b / "tianxi.db")
        print(f"\n行数：A={len(a)}  B={len(b)}")
        if len(a) != len(b):
            print("❌ 行数不同")
            return 1

        pos_a = {(r[1], r[2]): r for r in a}  # 键 = (request_id, local_index)
        pos_b = {(r[1], r[2]): r for r in b}
        print(f"位置集合相同：{set(pos_a) == set(pos_b)}")
        print(f"id 集合相同  ：{ {r[0] for r in a} == {r[0] for r in b} }")

        bad = 0
        for key in sorted(set(pos_a) | set(pos_b)):
            ra, rb = pos_a.get(key), pos_b.get(key)
            if ra != rb:
                bad += 1
                if bad <= 3:
                    print(f"  ❌ {key}\n     A={ra}\n     B={rb}")
        print(f"\n逐行（含 id / 正文 / event_time / prev / next）不同的行数：{bad}")

        links_a = [(r[6], r[7]) for r in a]
        links_b = [(r[6], r[7]) for r in b]
        print(f"邻接指针逐字相同：{links_a == links_b}（长度 {len(links_a)}）")
        ok = bad == 0 and set(pos_a) == set(pos_b) and links_a == links_b
        head = "✅ 乱序+并发与顺序**逐字一致**"
        if processes > 1:
            head = f"✅ {processes} 个进程乱序并发，真源与顺序投喂**逐字一致**"
        print("\n" + (head if ok else "❌ 两者不一致"))
        return 0 if ok else 1
    finally:
        for p in procs:
            _kill(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="位置模型探针：乱序/并发投喂 vs 顺序投喂")
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        metavar="N",
        help="B 臂起 N 个**独立进程**共享同一套存储（默认 1）。"
        "N>1 回答的是'uvicorn --workers N 到底安不安全'那个问题。",
    )
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes 必须 >= 1")
    sys.exit(main(args.processes))
