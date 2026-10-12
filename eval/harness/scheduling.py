"""Bounded preparation/judging pipeline for independent users, using HTTP only."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypeVar

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")
DEFAULT_INGEST_WORKERS = 4


def worker_counts(ingest: int, judge: int, *, users: int, endpoints: int) -> tuple[int, int]:
    """Zero selects defaults; explicit one keeps the corresponding stage serial."""
    if ingest < 0 or judge < 0:
        raise ValueError("ingest_workers and judge_workers must be nonnegative")
    count = max(1, users)
    return min(count, ingest or DEFAULT_INGEST_WORKERS), min(count, judge or max(1, endpoints))


def run_pipeline(  # noqa: UP047 - the package also supports Python 3.11
    items: Sequence[T],
    *,
    user_id: Callable[[T], str],
    prepare: Callable[[int, T], P],
    judge: Callable[[int, T, P], R],
    ingest_workers: int,
    judge_workers: int,
) -> Iterator[tuple[int, R]]:
    """Yield completed results with their source index, without FIFO blocking.

    One preparation task owns all Add/Search events for a user. Judgment starts
    only after that task succeeds. At most ingest_workers + judge_workers users
    are in flight, including prepared contexts waiting for a judge slot. Callers
    restore source order using the returned index. Duplicate users fail before
    any HTTP call; failures cancel queued work and join running tasks before the
    caller can close the HTTP client. No task changes request IDs or batch shape.
    """
    identities = [user_id(item) for item in items]
    if len(set(identities)) != len(identities):
        raise ValueError("Concurrent evaluation requires one task per unique user_id")
    if ingest_workers < 1 or judge_workers < 1:
        raise ValueError("Pipeline worker counts must be positive")
    if ingest_workers == judge_workers == 1:
        for index, item in enumerate(items):
            yield index, judge(index, item, prepare(index, item))
        return
    preparing: dict[Future[P], int] = {}
    judging: dict[Future[R], int] = {}
    ready: deque[tuple[int, P]] = deque()
    next_index = 0
    capacity = ingest_workers + judge_workers
    preparation_pool = ThreadPoolExecutor(max_workers=ingest_workers, thread_name_prefix="ingest")
    judgment_pool = ThreadPoolExecutor(max_workers=judge_workers, thread_name_prefix="judge")
    try:
        while next_index < len(items) or preparing or ready or judging:
            while ready and len(judging) < judge_workers:
                index, prepared = ready.popleft()
                future = judgment_pool.submit(judge, index, items[index], prepared)
                judging[future] = index
            while (
                next_index < len(items)
                and len(preparing) < ingest_workers
                and len(preparing) + len(ready) + len(judging) < capacity
            ):
                index = next_index
                preparing[preparation_pool.submit(prepare, index, items[index])] = index
                next_index += 1
            completed, _ = wait((*preparing, *judging), return_when=FIRST_COMPLETED)
            for future in completed:
                if future in preparing:
                    index = preparing.pop(future)
                    ready.append((index, future.result()))
                else:
                    index = judging.pop(future)
                    yield index, future.result()
    finally:
        for future in (*preparing, *judging):
            future.cancel()
        preparation_pool.shutdown(wait=True, cancel_futures=True)
        judgment_pool.shutdown(wait=True, cancel_futures=True)
