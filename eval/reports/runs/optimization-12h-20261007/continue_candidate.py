"""Resume a frozen candidate; gate associated scores before a fresh full run.

Run with ``uv run --env-file .env python continue_candidate.py a13``. Existing
answers resume only within the same arm and phase. Full has distinct run IDs,
so associated answers cannot enter its score. This script never commits code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from compare import compare
from experiment import OUT, ROOT, check_arm, now, sha, write


def continue_candidate(label: str) -> None:
    manifest = json.loads((OUT / f"arm-{label}.json").read_text())
    check_arm(manifest)
    snapshot = Path(manifest["snapshot"])
    checks = json.loads((OUT / f"local-checks-{label}.json").read_text())
    if not checks.get("complete") or any(c["exit_code"] for c in checks["checks"]):
        raise RuntimeError("Candidate local checks have not passed")
    for relative, digest in checks["product_hashes"].items():
        if sha(ROOT / relative) != digest or sha(snapshot / relative) != digest:
            raise RuntimeError(f"Tested candidate source changed: {relative}")

    # WSL proxy settings may omit loopback after a host restart. Keep external
    # gateway routing intact while reaching the private services directly.
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(
            filter(None, (os.environ.get(name), "127.0.0.1", "localhost", "::1"))
        )

    status_path = OUT / f"status-{label}-p2.json"
    write(
        OUT / f"recovery-{label}.json",
        dict(
            resumed_at=now(),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            prior_status=json.loads(status_path.read_text()) if status_path.exists() else None,
            note="Original process missing after host restart; resume same frozen arm and phase",
        ),
    )

    def run(*args: str) -> None:
        subprocess.run([sys.executable, str(OUT / "experiment.py"), *args], cwd=ROOT, check=True)

    small = compare("baseline", label, "employment-small", ["mquake-remastered"])
    for result in small["datasets"]:
        if result["after_correct"] <= result["before_correct"] or result["after_errors"]:
            raise RuntimeError("Fixed small candidate gate failed")
    manual = json.loads((OUT / "prototype-unknown-time.json").read_text())["manual"]
    if len(manual) != 4 or not all(case["passed"] for case in manual):
        raise RuntimeError("Manual unknown-time snippet validation failed")

    associated = ["mquake-remastered", "musique", "hybridqa"]
    run("--run", label, "p2", "--datasets", *associated)
    paired = compare("baseline", label, "p2", associated)
    for result in paired["datasets"]:
        if result["after_correct"] < result["before_correct"]:
            raise RuntimeError(f"Associated score gate failed: {result['dataset']}")
        if len(result["after_errors"]) > len(result["before_errors"]):
            raise RuntimeError(f"Associated technical-error gate failed: {result['dataset']}")
    mq = next(result for result in paired["datasets"] if result["dataset"] == "mquake-remastered")
    if mq["after_correct"] <= mq["before_correct"]:
        raise RuntimeError("Associated employment evidence candidate has no measured gain")

    check_arm(manifest)
    write(
        OUT / f"gate-{label}.json",
        dict(
            passed_at=now(),
            manual_unknown_time_cases=4,
            local_checks_complete=True,
            fixed_small=small,
            associated=paired,
            decision="Start fresh full native regression; product acceptance remains pending",
        ),
    )
    print(f"GATES PASSED: {label}; starting all 11 frozen native subsets", flush=True)
    run("--run", label, "full")
    subprocess.run(
        [sys.executable, str(OUT / "final_audit.py"), label], cwd=ROOT, check=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=["a13"])
    continue_candidate(parser.parse_args().label)
