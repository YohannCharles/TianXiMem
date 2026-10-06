"""Fast literal-evidence checks, without HTTP, model calls or product mutation.

Run against the chosen product/prototype import path before larger validation.
Each case checks evidence content and safety rather than predicting a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import compile_query
from tianximem.retrieve.evidence import select_evidence


def fact(value, timestamp, position):
    return extract_evidence(
        parent_memory_id=position,
        user_id="manual",
        question=f"Corpus: Book was created by {value} (this replaces the earlier value)",
        answer=None,
        event_time=timestamp,
    )[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = compile_query("Who is the creator of Book?")
    cases = [
        (
            "dated_reverse_arrival",
            [fact("Latest", 300, "a"), fact("Earlier", 200, "b")],
            {"Latest"},
        ),
        (
            "unknown_dates",
            [fact("Writer A", None, "a"), fact("Writer B", None, "b")],
            {"Writer A", "Writer B"},
        ),
        (
            "same_time_conflict",
            [fact("Writer A", 300, "a"), fact("Writer B", 300, "b")],
            {"Writer A", "Writer B"},
        ),
        (
            "same_value_returns",
            [fact("Writer A", 100, "a"), fact("Writer B", 200, "b"), fact("Writer A", 300, "c")],
            {"Writer A"},
        ),
    ]
    results = []
    for name, facts, expected in cases:
        selected = select_evidence(facts, plan, source_limit=12, hop_limit=4)
        values = {f.object for f in selected.facts}
        passed = values == expected
        results.append(
            {
                "case": name,
                "expected": sorted(expected),
                "actual": sorted(values),
                "passed": passed,
                "literal_sources": [f.source_quote for f in facts],
            }
        )
        print(name, sorted(values), "PASS" if passed else "FAIL")
    historical = compile_query("Who was the creator of Book before 1970?")
    historical_selection = (
        None
        if historical is None
        else select_evidence(
            [f for _, facts, _ in cases for f in facts], historical, source_limit=12, hop_limit=4
        )
    )
    # A valid explicit-interval plan may compile, but absent intervals must fallback.
    history_passed = (
        historical_selection is None
        and compile_query("Who was the previous creator of Book?") is None
    )
    results.append({"case": "historical_query_keeps_original_search", "passed": history_passed})
    print("historical_query_keeps_original_search", "PASS" if history_passed else "FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"cases": results, "passed": all(r["passed"] for r in results)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return int(not all(r["passed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
