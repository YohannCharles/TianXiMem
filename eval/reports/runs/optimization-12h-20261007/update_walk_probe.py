"""Check literal leadership questions and opaque edits before product changes."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tianximem.facts import query
from tianximem.facts.grammar import extract_evidence
from tianximem.retrieve import evidence
from tianximem.store.sqlite_store import SqliteStore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VAR = ROOT / "var/optimization-12h-20261007"
SETS = (
    "corporatebench",
    "mquake-remastered",
    "tempreason",
    "memtrapbench",
    "locomo-refined",
    "longmemeval-s",
    "medmemorybench",
    "halumem",
    "musique",
    "hybridqa",
    "feverous",
)


def shadow(module, name, old, new, extra=()):
    source = Path(module.__file__).read_text()
    assert source.count(old) == 1
    source = source.replace(old, new)
    for old, new in extra:
        assert source.count(old) == 1
        source = source.replace(old, new)
    proposed = types.ModuleType(name)
    sys.modules[name] = proposed
    exec(compile(source, f"<{name}>", "exec"), proposed.__dict__)
    return proposed


def prototypes(variant="a08b"):
    proposed_query = shadow(
        query,
        "literal_ceo_query_prototype",
        '    ("author", r"\\bauthor\\b|\\bwrote\\b"),',
        '    ("chief executive officer", r"\\bchief executive officer\\b"),\n'
        '    ("author", r"\\bauthor\\b|\\bwrote\\b"),',
        extra=((r"|\bnationality\b", r"|\bnationalit(?:y|ies)\b"),) if variant == "a08b" else (),
    )
    proposed_walk = shadow(
        evidence,
        "opaque_update_walk_prototype",
        "                        if key(entity) not in key(fact.source_quote):\n"
        "                            return None\n"
        "                        selected.setdefault(fact.parent_memory_id, fact)",
        "                        return None",
        extra=(
            (
                "        scoped = [f for f in edges if key(f.subject) in frontier]\n",
                "        scoped = [f for f in edges if key(f.subject) in frontier]\n"
                "        if relations - covered_relations and not frontier.issubset(\n"
                "            {key(f.subject) for f in scoped}\n"
                "        ):\n"
                "            return None\n",
            ),
        )
        if variant == "a08b"
        else (),
    )
    return proposed_query, proposed_walk


def selected_ids(module, facts, plan, hops=4):
    if plan is None:
        return None
    decision = module.evaluate_evidence(facts, plan, source_limit=12, hop_limit=hops)
    if decision.selection is None:
        return None
    return sorted({f.parent_memory_id for f in decision.selection.facts})


def snippets(proposed_query, proposed_walk):
    sources = {
        "executive": "Corpus: Example Labs — chief executive officer — Alex",
        "second_executive": "Corpus: Example Labs — chief executive officer — Bea",
        "old_nationality": "Corpus: Alex — country of citizenship — Sweden",
        "edit": "Corpus: Example Labs is now led by Bea (this replaces the earlier value)",
        "new_nationality": "Corpus: Bea — country of citizenship — Norway",
        "other_edit": "Corpus: Other Labs is now led by Carol (this replaces the earlier value)",
        "spouse": "Corpus: Pat — spouse — Alex",
        "spouse_edit": (
            "Corpus: Pat now shares their life with Bea (this replaces the earlier value)"
        ),
        "president": "Corpus: Example Labs — president — Alex",
    }
    facts = {
        name: extract_evidence(
            parent_memory_id=name, user_id="manual", question=text, answer=None, event_time=None
        )[0]
        for name, text in sources.items()
    }
    leadership = "What is the nationality of the chief executive officer of Example Labs?"
    spouse = "What is the nationality of the spouse of Pat?"
    cases = (
        (
            "every_alternative_has_literal_support",
            leadership,
            ("executive", "second_executive", "old_nationality", "new_nationality"),
            ["executive", "new_nationality", "old_nationality", "second_executive"],
        ),
        (
            "missing_alternative_cannot_be_hidden_by_relation_union",
            leadership,
            ("executive", "second_executive", "old_nationality"),
            None,
        ),
        (
            "plural_nationalities_bind_the_same_relation",
            leadership.replace("nationality", "nationalities"),
            ("executive", "old_nationality"),
            ["executive", "old_nationality"],
        ),
        (
            "literal_leadership_chain",
            leadership,
            ("executive", "old_nationality"),
            ["executive", "old_nationality"],
        ),
        ("missing_nationality", leadership, ("executive",), None),
        (
            "unrecognized_edit_blocks_old_path",
            leadership,
            ("executive", "old_nationality", "edit", "new_nationality"),
            None,
        ),
        (
            "unrelated_edit_does_not_block",
            leadership,
            ("executive", "old_nationality", "other_edit"),
            ["executive", "old_nationality"],
        ),
        ("no_president_alias", leadership, ("president", "old_nationality"), None),
        (
            "no_ceo_abbreviation_alias",
            leadership.replace("chief executive officer", "CEO"),
            ("executive", "old_nationality"),
            None,
        ),
        (
            "history_requires_other_plan",
            leadership.replace("the chief", "the previous chief"),
            ("executive", "old_nationality"),
            None,
        ),
        (
            "negated_scope",
            leadership.replace("What is", "What is not"),
            ("executive", "old_nationality"),
            None,
        ),
        (
            "existing_spouse_chain",
            spouse,
            ("spouse", "old_nationality"),
            ["old_nationality", "spouse"],
        ),
        (
            "opaque_update_also_blocks_existing_relation",
            spouse,
            ("spouse", "old_nationality", "spouse_edit", "new_nationality"),
            None,
        ),
    )
    results = []
    for name, text, names, expected in cases:
        chosen = [facts[n] for n in names]
        plan = proposed_query.compile_query(text)
        before = selected_ids(evidence, chosen, query.compile_query(text))
        query_only = selected_ids(evidence, chosen, plan)
        actual = selected_ids(proposed_walk, chosen, plan)
        passed = actual == expected
        print(name, "PASS" if passed else "FAIL", flush=True)
        results.append(
            dict(
                case=name,
                query=text,
                sources=[sources[n] for n in names],
                expected=expected,
                before=before,
                query_only=query_only,
                actual=actual,
                passed=passed,
            )
        )
    assert results[5]["query_only"] == ["edit", "executive", "old_nationality"]
    assert results[1]["query_only"] == ["executive", "old_nationality", "second_executive"]
    assert results[-1]["before"] == ["old_nationality", "spouse", "spouse_edit"]
    return results


def native_scan(proposed_query, proposed_walk):
    counts, changes = Counter(), []
    for dataset in SETS:
        group = "seven" if dataset in SETS[:7] else "four"
        store = SqliteStore(VAR / f"a06b/{group}/tianxi.db")
        with (VAR / f"plan-{dataset}.pickle").open("rb") as handle:
            samples = pickle.load(handle)["full"]
        for sample in samples:
            for question in sample.questions:
                before_plan = query.compile_query(question.question)
                after_plan = proposed_query.compile_query(question.question)
                changed_plan = (None if before_plan is None else asdict(before_plan)) != (
                    None if after_plan is None else asdict(after_plan)
                )
                decisions = []
                for module, plan in ((evidence, before_plan), (proposed_walk, after_plan)):
                    if plan is None or plan.operator != "walk":
                        decisions.append(("no_walk", None))
                        continue
                    patterns = (*plan.patterns, query.FactPattern(("explicit replacement",)))
                    with store.read() as conn:
                        facts = store.fetch_evidence(conn, sample.user_id, patterns, limit=97)
                    if len(facts) > 96:
                        decisions.append(("fetch_limit", None))
                        continue
                    result = module.evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
                    decisions.append(
                        (
                            result.reason,
                            None
                            if result.selection is None
                            else tuple(f.id for f in result.selection.facts),
                        )
                    )
                before, after = decisions
                counts[f"{dataset}:{before[0]}->{after[0]}"] += 1
                if changed_plan or before != after:
                    changes.append(
                        dict(
                            dataset=dataset,
                            qid=question.qid,
                            question=question.question,
                            changed_plan=changed_plan,
                            before_reason=before[0],
                            after_reason=after[0],
                            changed_selection=before != after,
                        )
                    )
    print("changed queries", len(changes), flush=True)
    print(dict(Counter(r["dataset"] for r in changes)), flush=True)
    return dict(counts=dict(counts), changed=changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("a08", "a08b"), default="a08b")
    args = parser.parse_args()
    proposed_query, proposed_walk = prototypes(args.variant)
    manual = snippets(proposed_query, proposed_walk)
    passed = all(r["passed"] for r in manual)
    scan = native_scan(proposed_query, proposed_walk) if passed else None
    (HERE / f"prototype-update-walk-{args.variant}.json").write_text(
        json.dumps(dict(manual=manual, native_scan=scan), ensure_ascii=False, indent=2) + "\n"
    )
    if not passed:
        raise RuntimeError("Manual fragment gate failed; recorded without model evaluation")


if __name__ == "__main__":
    main()
