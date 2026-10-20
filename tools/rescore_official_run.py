#!/usr/bin/env python3
"""Correct an existing official-capture run without rerunning Search or Answer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.experiments.official_rescore import rescore_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <run-dir>/offline-rescore",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not (run_dir / "selection-manifest.json").is_file():
        print(f"not an official-capture run: {run_dir}", file=sys.stderr)
        return 2
    summary = rescore_run(run_dir, output_dir=args.output_dir)
    original = summary["original"]
    corrected = summary["corrected"]
    print(
        f"original {original['correct']}/{original['total']} "
        f"({original['accuracy']:.2%}) -> corrected "
        f"{corrected['correct']}/{corrected['total']} "
        f"({corrected['accuracy']:.2%}); changed {summary['changed_rows']}"
    )
    print(f"output: {args.output_dir or run_dir / 'offline-rescore'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
