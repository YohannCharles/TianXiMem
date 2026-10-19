"""Run the default offline LLM fact pipeline.

The pipeline is deliberately evaluation-only:
candidate extraction (Qwen v11) -> high-risk verification (Qwen v17).
It never runs in Add/Search and never writes production facts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from fact_llm_verify import DEFAULT_VERIFY_RELATIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.out_dir / "candidates-v11.jsonl"
    final = args.out_dir / "verified-v17.jsonl"
    common = [
        "--gold", str(args.gold),
        "--env-file", str(args.env_file),
        "--model", args.model,
        "--limit", str(args.limit),
        "--offset", str(args.offset),
        "--timeout", str(args.timeout),
        "--max-tokens", str(args.max_tokens),
    ]
    if args.base_url:
        common += ["--base-url", args.base_url]
    if args.api_key:
        common += ["--api-key", args.api_key]
    if args.ids:
        common += ["--ids", args.ids]
    if args.no_resume:
        common += ["--no-resume"]

    extract_cmd = [sys.executable, "tools/fact_llm_extract.py", *common,
                   "--out", str(candidates),
                   "--prompt-version", "common-facts-qwen-v11-demonstrated"]
    verify_cmd = [sys.executable, "tools/fact_llm_verify.py", *common,
                  "--candidates", str(candidates), "--out", str(final),
                  "--prompt-version", "common-facts-qwen-v17-abstain",
                  "--verify-relations", DEFAULT_VERIFY_RELATIONS]
    env = os.environ.copy()
    for command in (extract_cmd, verify_cmd):
        completed = subprocess.run(command, env=env)
        if completed.returncode != 0:
            return completed.returncode
    print(f"candidates={candidates}")
    print(f"verified={final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

