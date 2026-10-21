"""官方重放的逐题后台作答：有界 worker、磁盘待办、用户聚合文件单写者。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.datasets.official_groups import write_json
from eval.harness.api_config import ENDPOINTS
from eval.harness.judge import run_judge
from eval.harness.judge_results import _sanitize_jsonl
from eval.jsonl_io import jsonl_line, read_jsonl, write_line
from eval.reports.official_comparison import write_summary


class StreamingOfficialJudge:
    """只消费已经冻结的单题输入，网络调用数不超过 workers。

    executor 队列只保存路径；上下文存入每题目录，只有在工作的线程才读入内存。
    每题独立运行现有 answer/evaluate，聚合写入在同一锁内，避免覆盖或重复行。
    """

    def __init__(
        self,
        output: Path,
        pipeline: Path,
        *,
        workers: int,
        max_tokens: int,
        on_error: Callable[[str, str, Exception], None],
        deferred: bool = False,
    ) -> None:
        if workers < 1:
            raise ValueError("streaming judge workers must be positive")
        self.output = output
        self.pipeline = pipeline
        self.max_tokens = max_tokens
        self.on_error = on_error
        self.workers = workers
        self.deferred = deferred
        self.waiting_jobs: list[tuple[str, str, Path]] = []
        limit = int(os.environ.get("OFFICIAL_LLM_INFLIGHT_PER_SLOT", "0"))
        if limit < 0:
            raise ValueError("OFFICIAL_LLM_INFLIGHT_PER_SLOT must be nonnegative")
        # Repeated endpoint slots encode capacity (e.g. 1:2:1).
        self.endpoint_limits = [threading.BoundedSemaphore(limit) for _ in ENDPOINTS] if limit else []
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="official-judge")
        self.users: dict[str, dict] = {}
        self.submitted: set[tuple[str, str]] = set()
        self.completed = 0
        self.active = 0
        self.failures: list[Exception] = []
        self.last_summary = time.monotonic()
        self.context_audited = {
            (row["user"], row["id"])
            for row in read_jsonl(output / "answer-context-truncations.jsonl")
        }

    def _user(self, user: str) -> dict:
        if user not in self.users:
            directory = self.output / user
            files = {}
            for name in ("input", "answers", "labels"):
                path = directory / f"{name}.jsonl"
                _sanitize_jsonl(path)
                rows = read_jsonl(path)
                indexed = {str(row["id"]): row for row in rows}
                if len(indexed) != len(rows):
                    raise ValueError(f"duplicate {name} IDs in {directory}")
                # 不把整份检索上下文留在内存；答案供已有批量作答复用。
                files[name] = indexed if name == "answers" else set(indexed)
            self.users[user] = files
        return self.users[user]

    def _progress(self) -> None:
        write_json(
            self.output / "judging-progress.json",
            {
                "workers": self.workers,
                "phase": "waiting-for-replay" if self.deferred else "answer-judge",
                "submitted": len(self.submitted),
                "active": self.active,
                "completed_this_pass": self.completed,
                "failed_this_pass": len(self.failures),
                "pending": len(self.submitted) - self.completed - len(self.failures) - self.active,
            },
        )

    def submit(self, user: str, item: dict) -> None:
        qid = str(item["id"])
        if not qid or Path(qid).name != qid or qid in (".", ".."):
            raise ValueError("question ID must be a single path component")
        with self.lock:
            files = self._user(user)
            if qid in files["labels"] or (user, qid) in self.submitted:
                return
            user_dir = self.output / user
            job = user_dir / "questions" / qid
            job.mkdir(parents=True, exist_ok=True)
            (job / "input.jsonl").write_text(jsonl_line(item), encoding="utf-8")
            if qid not in files["input"]:
                with (user_dir / "input.jsonl").open("a", encoding="utf-8") as handle:
                    write_line(handle, item)
                files["input"].add(qid)
            # 复用已经生成的答案，不为同一道题再次调用模型。
            if qid in files["answers"]:
                (job / "answers.jsonl").write_text(
                    jsonl_line(files["answers"][qid]), encoding="utf-8"
                )
            _sanitize_jsonl(job / "labels.jsonl")
            labels = read_jsonl(job / "labels.jsonl")
            # --retry-errors 会清掉聚合文件中的技术错误；同步清除对应的任务缓存。
            if any(str(row.get("label", "")).endswith("_ERROR") for row in labels):
                (job / "labels.jsonl").write_text("", encoding="utf-8")
            self.submitted.add((user, qid))
            if self.deferred:
                self.waiting_jobs.append((user, qid, job))
            else:
                self.pool.submit(self._work, user, qid, job)
            self._progress()

    def start(self) -> None:
        """Release disk-backed jobs only after every replay worker has stopped."""
        with self.lock:
            self.deferred = False
            for job in self.waiting_jobs:
                self.pool.submit(self._work, *job)
            self.waiting_jobs.clear()
            self._progress()

    def _work(self, user: str, qid: str, job: Path) -> None:
        endpoint_index = int(
            hashlib.sha256(f"{user}\0{qid}".encode()).hexdigest(), 16
        ) % len(ENDPOINTS)
        limiter = self.endpoint_limits[endpoint_index] if self.endpoint_limits else None
        if limiter is not None:
            limiter.acquire()
        with self.lock:
            self.active += 1
        try:
            # Keep endpoint selection deterministic across resumes and both arms.
            # Answer and judge for one question must use the same deployment.
            run_judge(
                self.pipeline,
                read_jsonl(job / "input.jsonl"),
                job,
                dataset="official-capture",
                max_tokens=self.max_tokens,
                endpoint_index=endpoint_index,
            )
            answers = read_jsonl(job / "answers.jsonl")
            labels = read_jsonl(job / "labels.jsonl")
            if len(answers) != 1 or len(labels) != 1:
                raise ValueError(f"incomplete single-question outputs: {user}/{qid}")
            if str(answers[0]["id"]) != qid or str(labels[0]["id"]) != qid:
                raise ValueError(f"cross-question output: {user}/{qid}")
            with self.lock:
                files = self._user(user)
                # 只有这一段会写用户级聚合文件；中断后由任务目录恢复未聚合的行。
                if qid not in files["answers"]:
                    with (self.output / user / "answers.jsonl").open("a", encoding="utf-8") as h:
                        write_line(h, answers[0])
                    files["answers"][qid] = answers[0]
                if qid not in files["labels"]:
                    with (self.output / user / "labels.jsonl").open("a", encoding="utf-8") as h:
                        write_line(h, labels[0])
                    files["labels"].add(qid)
                self.completed += 1
                if self.completed % 10 == 0 or self.completed == 1:
                    print(f"  后台作答/判分本轮完成 {self.completed} 题", flush=True)
                if time.monotonic() - self.last_summary >= 30:
                    try:
                        self.refresh_summary()
                    except (ValueError, OSError) as error:
                        print(f"  ⚠ 刷新后台 summary 失败：{error}", flush=True)
        except Exception as error:
            with self.lock:
                self.failures.append(error)
                self.on_error(user, qid, error)
        finally:
            with self.lock:
                audit_path = job / "answer-context-audit.json"
                if audit_path.is_file() and (user, qid) not in self.context_audited:
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    if audit["truncated"]:
                        with (self.output / "answer-context-truncations.jsonl").open("a", encoding="utf-8") as h:
                            write_line(h, {**audit, "user": user})
                        self.context_audited.add((user, qid))
                self.active -= 1
                self._progress()
            if limiter is not None:
                limiter.release()

    def refresh_summary(self) -> dict:
        with self.lock:
            summary = write_summary(self.output)
            self.last_summary = time.monotonic()
            return summary

    def raise_if_failed(self) -> None:
        with self.lock:
            if self.failures:
                raise self.failures[0]

    def close(self, *, drain: bool = True) -> None:
        self.pool.shutdown(wait=True, cancel_futures=not drain)
