"""Test complete literal paths for new predicates without changing legacy retrieval."""

from __future__ import annotations

import pickle
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path

from experiment import SETS, VAR, write
from walk_source_probe import audit

from tianximem.facts.query import FactPattern, compile_query
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve.evidence import evaluate_evidence
from tianximem.store.sqlite_store import SqliteStore

HERE = Path(__file__).resolve().parent


def module(name, source):
    shadow = types.ModuleType(name)
    sys.modules[name] = shadow
    exec(compile(source, f"<{name}>", "exec"), shadow.__dict__)
    shadow.source_text = source
    return shadow


def prototype():
    query = (VAR / "a06b/source/src/tianximem/facts/query.py").read_text()
    old = '    bind_side: Literal["subject", "object"] = "subject"'
    assert query.count(old) == 1
    query = query.replace(old, old + "\n    require_complete_walk: bool = False")
    old = '        return QueryPlan("walk", (FactPattern(relations),), "source", query=text)'
    assert query.count(old) == 1
    query = query.replace(
        old,
        """        return QueryPlan(
            "walk", (FactPattern(relations),), "source", query=text,
            require_complete_walk=bool(re.search(
                r"\\bwork(?:s)? at\\b|^where\\s+(?:do|does)\\s+.+?\\s+work$", text, re.I
            )),
        )""",
    )
    evidence = (VAR / "a06d/source/src/tianximem/retrieve/evidence.py").read_text()
    old = "    if not relations.issubset(covered_relations):"
    assert evidence.count(old) == 1
    evidence = evidence.replace(
        old, "    if plan.require_complete_walk and not relations.issubset(covered_relations):"
    )
    old = "        current = _current(scoped)"
    assert evidence.count(old) == 1
    evidence = evidence.replace(
        old,
        """        if plan.require_complete_walk and relations - covered_relations:
            remaining = relations - covered_relations
            if any(not any(
                key(f.subject) == entity and relation_key(f.relation) in remaining
                for f in scoped
            ) for entity in frontier):
                return None
        current = _current(scoped)""",
    )
    old = '                    if f" {words_key(entity)} " in f" {words_key(fact.source_quote)} ":'
    assert evidence.count(old) == 1
    evidence = evidence.replace(
        old,
        old + "\n                        if plan.require_complete_walk:\n"
        "                            return None",
    )
    return module("closed_query_shadow", query), module("closed_evidence_shadow", evidence)


def selection(store, user, question, compiler, executor):
    plan = compiler(question)
    if plan is None or plan.operator != "walk":
        return "no_walk", None
    with store.read_snapshot() as conn:
        facts = store.fetch_evidence(
            conn, user, (*plan.patterns, FactPattern(("explicit replacement",))), limit=97
        )
        if len(facts) > 96:
            return "fetch_limit", None
        decision = executor(facts, plan, source_limit=12, hop_limit=4)
        if decision.selection is not None and getattr(plan, "require_complete_walk", False):
            reason = audit(store, conn, user, facts, plan, decision.selection, 96)
            if reason != "selected":
                return reason, None
        return decision.reason, decision.selection


def manual(query, evidence):
    spouse = "Corpus: Ann Druyan — spouse — Carl Sagan"
    employer = "Corpus: Carl Sagan — employer — Acme"
    other_spouse = "Corpus: Ann Druyan — spouse — Ada Lovelace"
    cases = (
        ("closed_path", "Where does the spouse of Ann Druyan work?", [spouse, employer], True),
        ("missing_employer", "Where does the spouse of Ann Druyan work?", [spouse], False),
        (
            "disconnected_employer",
            "Where does the spouse of Ann Druyan work?",
            [spouse, "Corpus: Someone Else — employer — Acme"],
            False,
        ),
        (
            "missing_second_branch",
            "Where does the spouse of Ann Druyan work?",
            [spouse, other_spouse, employer],
            False,
        ),
        (
            "both_branches_supported",
            "Where does the spouse of Ann Druyan work?",
            [spouse, other_spouse, employer, "Corpus: Ada Lovelace — employer — Other Labs"],
            True,
        ),
        (
            "unparsed_spouse_update",
            "Where does the spouse of Ann Druyan work?",
            [
                spouse,
                employer,
                "Corpus: Ann Druyan now has a different spouse (this replaces the earlier value)",
            ],
            False,
        ),
        ("self_literal", "Where do I work?", ["User: I work for Acme."], True),
        (
            "self_negative_update",
            "Where do I work?",
            ["User: I work for Acme.", "User: I no longer work for Acme."],
            False,
        ),
        (
            "self_unknown_job_change",
            "Where do I work?",
            ["User: I work for Acme.", "User: I left Acme and took a role at NewCo."],
            False,
        ),
        (
            "other_person_is_not_self",
            "Where do I work?",
            ["User: I work for Acme.", "Bob: I no longer work for OtherCo."],
            True,
        ),
        (
            "assistant_is_not_self",
            "Where do I work?",
            ["User: I work for Acme.", "Assistant: I no longer work for OtherCo."],
            True,
        ),
        (
            "legacy_evidence_recall_preserved",
            "Who is the head of government of the country of citizenship of Alex?",
            ["Corpus: Alex — country of citizenship — State"],
            True,
        ),
        ("workout_is_not_employment", "How often does Alex work out?", [employer], False),
        ("historical_is_not_current", "Where did Alex work before 2020?", [employer], False),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="closed-walk-") as directory:
        for index, (name, question, sources, expected) in enumerate(cases):
            store = SqliteStore.open(Path(directory) / str(index))
            apply_batch(
                store,
                AddBatch("opaque", "manual", "s", tuple(Message("user", s) for s in sources)),
                grounded_evidence=True,
            )
            reason, selected = selection(
                store, "manual", question, query.compile_query, evidence.evaluate_evidence
            )
            passed = (selected is not None) == expected
            if name == "legacy_evidence_recall_preserved":
                before = selection(store, "manual", question, compile_query, evaluate_evidence)
                passed = passed and (reason, selected.facts) == (before[0], before[1].facts)
            print(name, "PASS" if passed else "FAIL", flush=True)
            rows.append(
                dict(
                    case=name,
                    question=question,
                    sources=sources,
                    expected=expected,
                    actual=reason,
                    passed=passed,
                )
            )
    return rows


def native_scan(query, evidence):
    rows, flagged = [], 0
    for dataset in SETS:
        group = "seven" if dataset in SETS[:7] else "four"
        store = SqliteStore(VAR / f"a03/{group}/tianxi.db")
        with (VAR / f"plan-{dataset}.pickle").open("rb") as handle:
            samples = pickle.load(handle)["full"]
        for sample in samples:
            for question in sample.questions:
                before, after = (
                    compile_query(question.question),
                    query.compile_query(question.question),
                )
                if after is None or not after.require_complete_walk:
                    old, new = asdict(before) if before else None, asdict(after) if after else None
                    if new is not None:
                        new.pop("require_complete_walk")
                    assert old == new, f"Legacy plan changed: {question.qid}"
                    continue
                flagged += 1
                old = selection(
                    store, sample.user_id, question.question, compile_query, evaluate_evidence
                )
                new = selection(
                    store,
                    sample.user_id,
                    question.question,
                    query.compile_query,
                    evidence.evaluate_evidence,
                )
                if (old[1] is None) != (new[1] is None) or (
                    old[1] is not None and new[1] is not None and old[1].facts != new[1].facts
                ):
                    rows.append(
                        dict(
                            dataset=dataset,
                            qid=question.qid,
                            question=question.question,
                            before=old[0],
                            after=new[0],
                        )
                    )
    print("native flagged", flagged, "changed selections", len(rows), flush=True)
    return dict(flagged=flagged, changes=rows)


if __name__ == "__main__":
    query, evidence = prototype()
    cases = manual(query, evidence)
    passed = all(case["passed"] for case in cases)
    write(
        HERE / "prototype-closed-walk.json",
        dict(manual=cases, native_scan=native_scan(query, evidence) if passed else None),
    )
    if not passed:
        raise RuntimeError("Manual closed-walk gate failed before product edit")
