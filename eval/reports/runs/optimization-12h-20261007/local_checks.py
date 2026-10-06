"""Run local checks only when the live product matches its frozen candidate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment import OUT, ROOT, VAR, check_arm, now, sha, write


def validate(label: str) -> None:
    import json

    manifest = json.loads((OUT / f"arm-{label}.json").read_text())
    check_arm(manifest)
    snapshot = Path(manifest["snapshot"])
    product = {
        str(Path(path).relative_to(snapshot)): digest
        for path, digest in manifest["hashes"].items()
        if Path(path).is_relative_to(snapshot / "src")
    }

    def verify() -> None:
        for relative, digest in product.items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError(f"Live product differs from {label}: {relative}")

    verify()
    record = dict(label=label, started_at=now(), product_hashes=product, checks=[])
    commands = (
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("mypy", [sys.executable, "-m", "mypy", "src/tianximem"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests"]),
    )
    for name, command in commands:
        verify()
        log = VAR / label / f"local-{name}.log"
        print(f"RUN {label}/{name}", flush=True)
        with log.open("w") as handle:
            result = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        record["checks"].append(
            dict(name=name, command=command, exit_code=result.returncode, log=str(log))
        )
        write(OUT / f"local-checks-{label}.json", record)
        print(f"DONE {name}: {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Local check failed: {name}; see {log}")
    verify()
    record.update(complete=True, finished_at=now())
    write(OUT / f"local-checks-{label}.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    validate(parser.parse_args().label)
