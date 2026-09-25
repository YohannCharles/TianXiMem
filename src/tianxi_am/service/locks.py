"""按 `(user_id, session_id)` 串行化 Add（§15）。

## 为什么需要它

`pair_idx` 的分配是**读-改-写**：第 2 步读 `MAX(pair_idx)`、第 4 步写位置。
两个并发的同 session 批次会**读到同一个 `next_idx`**，于是生产出**重复的位置**。

> ⚠ **SQLite 的写事务不足以单独解决它**（§15 / `pairing/CLAUDE.md`）：两次事务读到的
> `MAX(pair_idx)` **会相同**——`BEGIN IMMEDIATE` 只保证事务内串行，而"A 的读-改-写"与
> "B 的读-改-写"仍可交错成 A读 B读 A写 B写。**锁必须在这之上**（按 session，不按库）。

## 三条边界

| | |
| --- | --- |
| **同 `(user_id, session_id)`** | **必须串行** |
| **不同 session** | 不互相阻塞（哪怕同一个 user） |
| **不同 user** | 不互相阻塞 |

## ⚠ 这是**进程内**锁

`service/CLAUDE.md` 与 §15 都要求**必须 `--workers 1`**：多 worker 会**静默失效**——
每个 worker 各有各的锁，两个批次照旧并发。**不做分布式锁**（v1 的既定约束）。

⚠ 每个见过的 session 会留下一个 `Lock` 对象（几十字节）；v1 不回收。
`--workers 1` + 单次 run 的 session 数下这不成问题，但**别在长期运行的服务里当缓存用**。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["SessionLockTimeout", "SessionLocks"]


class SessionLockTimeout(TimeoutError):
    """在超时内没拿到 session 锁——**这是可重试的**（同 session 正忙）。"""


class SessionLocks:
    """按 `(user_id, session_id)` 分发的进程内锁表。"""

    def __init__(self) -> None:
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._guard = threading.Lock()

    def __len__(self) -> int:
        """已登记过的 key 数（测试用）。"""
        with self._guard:
            return len(self._locks)

    def _lock_for(self, key: tuple[str, str]) -> threading.Lock:
        # 建表本身也要串行——否则两个线程会各建一把锁、各自持有 ⇒ **锁形同虚设**
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def hold(
        self, user_id: str, session_id: str, *, timeout: float | None = None
    ) -> Iterator[None]:
        """拿到锁再进入；**退出（含异常）时一定释放**。

        `timeout=None` = 一直等。给了超时且等不到 ⇒ 抛 `SessionLockTimeout`，
        让调用方返回一个**可重试**的错误，而不是无限挂住一个 30 分钟预算的请求。
        """
        lock = self._lock_for((user_id, session_id))
        acquired = lock.acquire(timeout=-1 if timeout is None else timeout)
        if not acquired:
            raise SessionLockTimeout(
                f"未能在 {timeout}s 内拿到 session 锁（user_id={user_id!r}, "
                f"session_id={session_id!r}）——同 session 的另一个 Add 正在进行"
            )
        try:
            yield
        finally:
            lock.release()
