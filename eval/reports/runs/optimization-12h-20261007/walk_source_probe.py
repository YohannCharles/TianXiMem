"""Prototype a source audit for walks without modifying product files."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path

from tianximem.facts.conversation import named_utterances
from tianximem.facts.evidence import key, words_key
from tianximem.facts.query import FactPattern, compile_query
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve.evidence import SPEAKER, evaluate_evidence
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


def scope_subjects(facts, selected, patterns):
    parents = {f.parent_memory_id for f in selected}
    return sorted(
        {
            key(f.subject)
            for f in facts
            if f.parent_memory_id in parents
            and not f.get("unparsed")
            and any(p.matches(f) for p in patterns)
        }
    )


def literals(subjects):
    return sorted(
        {
            literal
            for s in subjects
            for literal in (("i", "user", "用户", "我", "my") if s == "i" else (s,))
        }
    )


def source_audit_supported(sources, selected, facts, subjects):
    selected_parents = {f.parent_memory_id for f in selected}
    for source in sources:
        # A source projection already returns this complete original QA pair.
        if source.id in selected_parents:
            continue
        for side in ("question", "answer"):
            residue = getattr(source, side) or ""
            for fact in facts:
                if (
                    fact.parent_memory_id == source.id
                    and fact.source_side == side
                    and not fact.get("unparsed")
                ):
                    residue = residue.replace(fact.source_quote, "")
            for _, utterance in named_utterances(residue):
                text = utterance.strip()
                if not text or re.fullmatch(r"[^:\n]{1,64}:\s*", text):
                    continue
                speaker = SPEAKER.fullmatch(text)
                actor, body = (key(speaker[1]), speaker[2]) if speaker else ("", text)
                if actor in {"user", "用户"}:
                    actor = "i"
                if actor in subjects:
                    return False
                if "i" in subjects and not actor and side == "question":
                    if " i " in f" {words_key(body)} " or "我" in body:
                        return False
                    if " my " in f" {words_key(body)} ":
                        return False
                for subject in set(subjects) - {"i"}:
                    if f" {words_key(subject)} " in f" {words_key(body)} ":
                        return False
    return True


def audit(store, conn, user, facts, plan, selection, scan_limit):
    subjects = scope_subjects(facts, selection.facts, plan.patterns)
    sources = {}
    for literal in literals(subjects):
        for source in store.fetch_evidence_sources(conn, user, literal, limit=scan_limit + 1):
            sources[source.id] = source
        if len(sources) > scan_limit:
            return "source_scan_limit"
    known = (
        store.fetch_evidence(
            conn, user, (FactPattern(source_ids=tuple(sources)),), limit=97, deduplicate=False
        )
        if sources
        else []
    )
    if len(known) > 96:
        return "audit_limit"
    return (
        "selected"
        if source_audit_supported(list(sources.values()), selection.facts, known, subjects)
        else "unsupported_source"
    )


def snippets(scan_limit):
    cases = (
        ("complete_self_source", "Where do I work?", ["User: I work for Acme."], "selected"),
        (
            "negative_self_update",
            "Where do I work?",
            ["User: I work for Acme.", "User: I no longer work for Acme."],
            "unsupported_source",
        ),
        (
            "unknown_self_job_change",
            "Where do I work?",
            ["User: I work for Acme.", "User: I left Acme and took a role at NewCo."],
            "unsupported_source",
        ),
        (
            "named_negative_update",
            "Where does Alex work?",
            ["Alex: I work for Acme.", "Alex: I no longer work for Acme."],
            "unsupported_source",
        ),
        (
            "other_humans_first_person_is_not_me",
            "Where do I work?",
            ["User: I work for Acme.", "Bob: I no longer work for OtherCo."],
            "selected",
        ),
        (
            "assistant_first_person_is_not_me",
            "Where do I work?",
            ["User: I work for Acme.", "Assistant: I no longer work for OtherCo."],
            "selected",
        ),
        (
            "unlabelled_question_can_update_self",
            "Where do I work?",
            ["User: I work for Acme.", "My previous employer was Acme. I have left."],
            "unsupported_source",
        ),
        (
            "known_counterfactual_chain",
            "Where does the spouse of Ann Druyan work?",
            [
                "Corpus: Ann Druyan — spouse — Carl Sagan",
                "Corpus: Carl Sagan — employer — Cornell University",
                "Corpus: Carl Sagan works for BBC (this replaces the earlier value)",
            ],
            "selected",
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="walk-source-audit-") as directory:
        for index, (name, question, sources, expected) in enumerate(cases):
            store = SqliteStore.open(Path(directory) / str(index))
            apply_batch(
                store,
                AddBatch("opaque", "manual", "s", tuple(Message("user", text) for text in sources)),
                grounded_evidence=True,
            )
            plan = compile_query(question)
            assert plan is not None and plan.operator == "walk"
            with store.read() as conn:
                facts = store.fetch_evidence(
                    conn,
                    "manual",
                    (*plan.patterns, FactPattern(("explicit replacement",))),
                    limit=97,
                )
                decision = evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
                assert decision.selection is not None
                actual = audit(store, conn, "manual", facts, plan, decision.selection, scan_limit)
            passed = expected == actual
            print(name, "PASS" if passed else "FAIL", flush=True)
            results.append(
                dict(
                    case=name,
                    question=question,
                    sources=sources,
                    before=decision.reason,
                    expected=expected,
                    actual=actual,
                    passed=passed,
                )
            )
    return results


def native_scan(scan_limit):
    counts, changes = Counter(), []
    for dataset in SETS:
        group = "seven" if dataset in SETS[:7] else "four"
        store = SqliteStore(VAR / f"a06b/{group}/tianxi.db")
        with (VAR / f"plan-{dataset}.pickle").open("rb") as handle:
            samples = pickle.load(handle)["full"]
        for sample in samples:
            for question in sample.questions:
                plan = compile_query(question.question)
                if plan is None or plan.operator != "walk":
                    continue
                with store.read() as conn:
                    facts = store.fetch_evidence(
                        conn,
                        sample.user_id,
                        (*plan.patterns, FactPattern(("explicit replacement",))),
                        limit=97,
                    )
                    if len(facts) > 96:
                        continue
                    before = evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
                    if before.selection is None:
                        continue
                    after = audit(
                        store, conn, sample.user_id, facts, plan, before.selection, scan_limit
                    )
                counts[f"{dataset}:selected->{after}"] += 1
                if after != "selected":
                    changes.append(
                        dict(
                            dataset=dataset,
                            qid=question.qid,
                            question=question.question,
                            before="selected",
                            after=after,
                        )
                    )
    print(dict(counts), flush=True)
    return dict(counts=dict(counts), changed=changes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-limit", type=int, choices=(12, 96), default=12)
    scan_limit = parser.parse_args().scan_limit
    manual = snippets(scan_limit)
    passed = all(r["passed"] for r in manual)
    result = dict(
        manual=manual,
        native_scan=native_scan(scan_limit) if passed else None,
        scan_limit=scan_limit,
    )
    (HERE / f"prototype-walk-sources-{scan_limit}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    if not passed:
        raise RuntimeError("Manual source-audit gate failed")


if __name__ == "__main__":
    main()
