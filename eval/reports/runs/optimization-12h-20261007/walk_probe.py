"""Manually verify a complete-relation guard using an in-memory code prototype.

The product files remain untouched until these source-based checks pass.
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from collections import Counter
from pathlib import Path

from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import FactPattern, compile_query
from tianximem.retrieve import evidence
from tianximem.store.sqlite_store import SqliteStore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def prototype():
    source = Path(evidence.__file__).read_text()
    old = (
        "    frontier = set(roots)\n"
        "    visited: set[str] = set()\n"
        "    selected: dict[str, EvidenceFact] = {}"
    )
    new = old + "\n    covered_relations: set[str] = set()"
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "        current = _current(scoped)\n        for fact in current:"
    new = (
        "        current = _current(scoped)\n"
        "        covered_relations.update(relation_key(f.relation) for f in current)\n"
        "        for fact in current:"
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "        frontier = {key(f.object) for f in current}\n    return list(selected.values())"
    new = (
        "        frontier = {key(f.object) for f in current}\n"
        "    if not relations.issubset(covered_relations):\n"
        "        return None\n"
        "    return list(selected.values())"
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    module = types.ModuleType("walk_guard_prototype")
    sys.modules[module.__name__] = module
    exec(compile(source, "<walk_guard_prototype>", "exec"), module.__dict__)
    return module


def snippets():
    proposed = prototype()
    query = "Who is the head of government of the country of citizenship of Alex?"
    plan = compile_query(query)
    sources = {
        "citizenship": "Corpus: Alex — country of citizenship — State",
        "leader": "Corpus: State — head of government — Leader",
        "distractor": "Corpus: Other State — head of government — Other Leader",
        "cycle": "Corpus: State — country of citizenship — Alex",
        "edit": "Corpus: Alex is a citizen of New State (this replaces the earlier value)",
        "new_leader": "Corpus: New State — head of government — New Leader",
    }
    facts = {
        name: extract_evidence(
            parent_memory_id=name,
            user_id="manual",
            question=text,
            answer=None,
            event_time=None,
        )[0]
        for name, text in sources.items()
    }
    cases = (
        ("complete_chain", ("citizenship", "leader", "distractor"), 4, {"citizenship", "leader"}),
        ("missing_last_relation", ("citizenship",), 4, None),
        ("disconnected_last_relation", ("citizenship", "distractor"), 4, None),
        ("hop_limit_is_not_a_complete_chain", ("citizenship", "leader"), 1, None),
        ("cycle_without_required_relation", ("citizenship", "cycle"), 4, None),
        ("changed_edge_breaks_old_chain", ("citizenship", "leader", "edit"), 4, None),
        (
            "changed_edge_has_literal_support",
            ("citizenship", "leader", "edit", "new_leader"),
            4,
            {"edit", "new_leader"},
        ),
    )
    results = []
    for name, names, hops, expected in cases:
        selected = proposed.select_evidence(
            [facts[n] for n in names], plan, source_limit=12, hop_limit=hops
        )
        actual = None if selected is None else {f.parent_memory_id for f in selected.facts}
        passed = actual == expected
        print(name, "PASS" if passed else "FAIL", flush=True)
        results.append(
            {
                "case": name,
                "sources": [sources[n] for n in names],
                "expected_sources": None if expected is None else sorted(expected),
                "actual_sources": None if actual is None else sorted(actual),
                "passed": passed,
            }
        )
    if not all(case["passed"] for case in results):
        raise RuntimeError("Manual snippet validation failed")
    return proposed, results


def native_scan(proposed):
    with (ROOT / "var/optimization-12h-20261007/plan-mquake-remastered.pickle").open("rb") as f:
        samples = pickle.load(f)["full"]
    store = SqliteStore(ROOT / "var/optimization-12h-20261007/a03/seven/tianxi.db")
    counts = Counter()
    changes = []
    for sample in samples:
        for question in sample.questions:
            plan = compile_query(question.question)
            if plan is None or plan.operator != "walk":
                continue
            patterns = (*plan.patterns, FactPattern(("explicit replacement",)))
            with store.read() as conn:
                facts = store.fetch_evidence(conn, sample.user_id, patterns, limit=97)
            if len(facts) > 96:
                counts["fetch_limit"] += 1
                continue
            before = evidence.evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
            after = proposed.evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
            counts[f"{before.reason}->{after.reason}"] += 1
            if (
                before.reason != after.reason
                or (before.selection is None) != (after.selection is None)
                or (
                    before.selection is not None
                    and after.selection is not None
                    and before.selection.facts != after.selection.facts
                )
            ):
                changes.append(
                    {
                        "qid": question.qid,
                        "question": question.question,
                        "before_reason": before.reason,
                        "after_reason": after.reason,
                        "before_quotes": []
                        if before.selection is None
                        else [f.source_quote for f in before.selection.facts],
                    }
                )
    print(dict(counts), flush=True)
    return {"counts": dict(counts), "changed": changes}


def main():
    proposed, cases = snippets()
    result = {"manual": cases, "native_scan": native_scan(proposed)}
    (HERE / "prototype-walk.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
