"""Resume the frozen subset, retaining terminal answer HTTP 400 as explicit errors.

The product, input plans, Search responses, prompts and scoring stay frozen. The
original answer subprocess still performs its bounded retries. A terminal HTTP
400 gets an empty answer with transport-error metadata; the unchanged scorer
classifies it as ANSWER_ERROR. Other failures still stop the run. Existing
successful answers are retained. Run via `uv run --env-file .env python <file>`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_suite.py"


def runner():
    spec = importlib.util.spec_from_file_location("new_dataset_suite", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def redacted(error: str) -> str:
    for name in ("AML_API_KEY", "AML_JUDGE_API_KEY", "TIANXIMEM_API_KEY"):
        value = os.environ.get(name)
        if value:
            error = error.replace(value, "<redacted>")
    return error


def record_terminal_error(suite, item: dict, out_dir: Path, detail: str, **extra) -> None:
    from eval.harness.corpusqa_pipeline import _fingerprint, render_answer_prompt

    prompt = render_answer_prompt(item)
    fingerprint = _fingerprint(item)
    incident = {
        "at": datetime.now(UTC).isoformat(),
        "id": item["id"],
        "status_code": 400,
        "label": "ANSWER_ERROR",
        "input_fingerprint": fingerprint,
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "requested_output_tokens": extra.pop("requested_output_tokens", 1024),
        "terminal_subprocess_error": redacted(detail),
        "policy": "Retain full input; count failure, then continue remaining questions",
        **extra,
    }
    suite.emit(HERE / "terminal-answer-errors.jsonl", incident)
    suite.emit(
        out_dir / "answers.jsonl",
        {
            "id": item["id"],
            "generated_answer": "",
            "answer_attempts": [],
            "transport_error": incident,
            "input_fingerprint": fingerprint,
        },
    )
    print(
        f"  {item['id']}: terminal answer HTTP 400 recorded as ANSWER_ERROR; "
        "full input retained, continuing",
        flush=True,
    )


def install_error_recording(suite) -> None:
    suite.frozen_imports()
    from eval import harness
    from eval.harness import judge

    original = judge.run_judge

    def run_judge(pipeline, items, out_dir, **kwargs):
        while True:
            try:
                return original(pipeline, items, out_dir, **kwargs)
            except RuntimeError as error:
                detail = redacted(str(error))
                if not (
                    Path(pipeline).name == "corpusqa_pipeline.py"
                    and kwargs.get("dataset") == "feverous"
                    and "corpusqa_pipeline.py answer 失败" in detail
                    and "Client error '400 Bad Request'" in detail
                ):
                    raise
                answers = Path(out_dir) / "answers.jsonl"
                done = {row["id"] for row in suite.rows(answers)}
                pending = [item for item in items if item["id"] not in done]
                if not pending:
                    raise
                item = pending[0]
                record_terminal_error(
                    suite,
                    item,
                    Path(out_dir),
                    detail,
                    requested_output_tokens=kwargs.get("max_tokens", 1024),
                )
                # The unchanged answer CLI resumes from this explicit error row;
                # the unchanged evaluate CLI then scores every original input ID.

    judge.run_judge = run_judge
    # run.py imports the public package export, rather than the module member.
    harness.run_judge = run_judge
    if "eval.experiments.run" in sys.modules:
        sys.modules["eval.experiments.run"].run_judge = run_judge


def seed_confirmed_failure(suite, manifest: dict) -> None:
    """Reuse an already confirmed terminal response only for the same frozen prompt."""
    suite.frozen_imports()
    from eval.harness.corpusqa_pipeline import render_answer_prompt

    directory = suite.ROOT / "eval/reports/runs" / f"{suite.NAME}-subset-feverous"
    for path in HERE.glob("gateway-error-feverous-*.json"):
        diagnostic = json.loads(path.read_text())
        if (
            diagnostic.get("status_code") != 400
            or diagnostic.get("model") != manifest["answer_model"]
        ):
            continue
        for source in directory.glob("*/input.jsonl"):
            items = suite.rows(source)
            if len(items) != 1 or items[0]["id"] != diagnostic["question_id"]:
                continue
            item = items[0]
            done = {row["id"] for row in suite.rows(source.with_name("answers.jsonl"))}
            if item["id"] in done:
                continue
            prompt = render_answer_prompt(item)
            if hashlib.sha256(prompt.encode()).hexdigest() != diagnostic["prompt_sha256"]:
                raise ValueError("Confirmed transport failure has another prompt fingerprint")
            record_terminal_error(
                suite,
                item,
                source.parent,
                "Already exhausted answer subprocess retries; confirmed response: "
                + diagnostic["error_body"],
                requested_output_tokens=diagnostic["max_tokens"],
                confirmed_response_source=path.name,
            )


class ChildProxy:
    """Change only the original suite's child entry point, preserving its lifecycle."""

    def Popen(self, args, *positional, **kwargs):
        if len(args) > 2 and Path(args[1]).resolve() == RUNNER and args[2] == "--child":
            args = list(args)
            args[1] = str(Path(__file__).resolve())
        return subprocess.Popen(args, *positional, **kwargs)

    def __getattr__(self, name):
        return getattr(subprocess, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2)
    args = parser.parse_args()
    suite = runner()
    if args.child:
        install_error_recording(suite)
        return suite.child(*args.child)
    manifest = json.loads((HERE / "manifest.json").read_text())
    suite.check_frozen(manifest)
    seed_confirmed_failure(suite, manifest)
    continuation = {
        "at": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": str(Path(__file__).relative_to(suite.ROOT)),
        "policy": (
            "Terminal answer HTTP 400 becomes an explicit ANSWER_ERROR with original stderr; "
            "full inputs, model, prompts, scorer, prior answers and qids remain unchanged"
        ),
        "diagnostic": "gateway-error-feverous-4053.json",
    }
    suite.write(HERE / "continuation-manifest.json", continuation)
    state = json.loads((HERE / "status.json").read_text())
    state["continuation"] = continuation
    suite.write(HERE / "status.json", state)
    suite.subprocess = ChildProxy()
    return suite.execute(manifest, "subset")


if __name__ == "__main__":
    raise SystemExit(main())
