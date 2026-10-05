"""归档一致性 SQLite 快照、配置指纹及冻结协议；不修改源库或向量。"""

from __future__ import annotations

import hashlib
import json
import runpy
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.targeted_eval import freeze  # noqa: E402


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def main() -> None:
    heldout = runpy.run_path(str(OUT / "heldout.py"))
    assert heldout["code_fingerprint"]() == json.loads((OUT / "heldout-code.json").read_text())
    for prefix in ("", "heldout-", "regression-"):
        assert freeze(json.loads((OUT / f"{prefix}manifest.json").read_text())) == json.loads(
            (OUT / f"{prefix}frozen.json").read_text()
        )
    record = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "product_fingerprint": heldout["code_fingerprint"](),
        "arms": {},
    }
    previous = json.loads((OUT / "regression-source-before.json").read_text())
    for arm in ("baseline", "candidate", "regression", "regression-off"):
        source = ROOT / "var/generic-evidence-20261005" / arm / "tianxi.db"
        snapshot = source.with_name("tianxi-snapshot.db")
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if arm.startswith("regression"):
                for table in ("qa_pairs", "applied_batches"):
                    rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
                    assert ordered(rows) == ordered(previous[table]), (arm, table)
            with sqlite3.connect(snapshot) as destination:
                conn.backup(destination)
        config = ROOT / "configs/runs/generic-evidence-20261005" / arm
        record["arms"][arm] = {
            "sqlite_snapshot": str(snapshot.relative_to(ROOT)),
            "sqlite_sha256": fingerprint(snapshot),
            "config_sha256": {
                name: fingerprint(config / name) for name in ("default.yaml", "local.yaml")
            },
        }
    (OUT / "archive.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("Archived all 4 arms; legacy 6237 sources and 757 batches unchanged", flush=True)


if __name__ == "__main__":
    main()
