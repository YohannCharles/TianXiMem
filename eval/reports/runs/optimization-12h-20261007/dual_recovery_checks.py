"""Offline checks for queue shutdown and preserving work after sample failures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import dual_endpoint as dual


def check():
    from eval.experiments import run

    body = (
        'import multiprocessing\nq=multiprocessing.Queue()\n'
        'q.put(b"x"*8_000_000)\n{cancel}\nq.close()\n'
    )
    try:
        subprocess.run([sys.executable, "-c", body.format(cancel="")], timeout=2, check=True)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("Unread large queue did not reproduce shutdown hang")
    subprocess.run(
        [sys.executable, "-c", body.format(cancel="q.cancel_join_thread()")],
        timeout=2, check=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        dual.OUT = Path(directory)
        dual.time.sleep = lambda _: None
        os.environ.update(AML_BASE_URL="http://primary.invalid/v1", AML_EMB_API_KEY="test")
        samples = [
            SimpleNamespace(user_id=f"user-{i}", questions=[SimpleNamespace(qid=str(i))],
                            corpus="x" * 8_000_000)
            for i in range(5)
        ]

        def native_round(**kwargs):
            sample = run._load()[0]
            (dual.OUT / sample.user_id).write_text(os.environ["AML_BASE_URL"])
            if kwargs.get("fail") and sample.user_id == "user-1":
                raise RuntimeError("HTTP 400 context limit test")
            return [sample], [SimpleNamespace(qid=sample.questions[0].qid)]

        options = dict(dataset="test", bench_dir=dual.OUT, skip_ingest=True)
        run._load = lambda *args, **kwargs: samples
        actual_samples, actual_results = dual.parallel_round(native_round, **options)
        assert actual_samples == samples
        assert [result.qid for result in actual_results] == [str(i) for i in range(5)]
        started = time.monotonic()
        try:
            dual.parallel_round(native_round, **options, fail=True)
        except RuntimeError as error:
            assert "1 native samples failed" in str(error)
        else:
            raise AssertionError("Native failure incorrectly accepted as complete")
        assert time.monotonic() - started < 5
        failures = json.loads((dual.OUT / "execution-failures-a13-full-test.json").read_text())
        assert len(failures) == 1 and failures[0]["user_id"] == "user-1"
        assert all((dual.OUT / sample.user_id).exists() for sample in samples)
    print("PASS: queue shutdown; ordered coverage; failures retained; later samples run")


if __name__ == "__main__":
    check()
