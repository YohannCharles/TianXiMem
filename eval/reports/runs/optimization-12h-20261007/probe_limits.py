"""Prototype larger complete-evidence limits on frozen native source records.

Read-only and model-free: inspect actual evidence changes before changing any
product defaults. Run with PYTHONPATH pointing to the baseline src directory.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from tianximem.facts.query import FactPattern, compile_query
from tianximem.retrieve.evidence import select_evidence
from tianximem.store.sqlite_store import SqliteStore

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
VAR = ROOT / "var/optimization-12h-20261007"


def inspect(store, sample, question, limit):
    plan = compile_query(question.question)
    if plan is None:
        return {"reason": "no_plan"}
    if plan.operator == "current":
        return {"reason": "current_input"}
    patterns = plan.patterns
    if plan.resolve_references:
        patterns = tuple(FactPattern(subject=p.subject) for p in patterns)
    if plan.operator == "walk":
        patterns = (*patterns, FactPattern(("explicit replacement",)))
    cap = limit * 8
    with store.read_snapshot() as conn:
        evidence = store.fetch_evidence(conn, sample.user_id, patterns, limit=cap + 1)
        sources = []
        audit = []
        if plan.guard_literal:
            sources = store.fetch_evidence_sources(
                conn, sample.user_id, plan.guard_literal, limit=limit + 1
            )
            if sources:
                audit = store.fetch_evidence(
                    conn,
                    sample.user_id,
                    (FactPattern(source_ids=tuple(p.id for p in sources)),),
                    limit=cap + 1,
                    deduplicate=False,
                )
    if len(evidence) > cap:
        return {"reason": "fetch_limit"}
    if len(sources) > limit:
        return {"reason": "source_scan_limit"}
    if len(audit) > cap:
        return {"reason": "audit_limit"}
    selection = select_evidence(
        evidence, plan, source_limit=limit, hop_limit=4, sources=sources, audit_facts=audit
    )
    if selection is None:
        return {"reason": "no_selection"}
    return {
        "reason": "selected",
        "projection": selection.projection,
        "evidence": [
            {
                "subject": f.subject,
                "relation": f.relation,
                "object": f.object,
                "source_id": f.parent_memory_id,
                "quote": f.source_quote,
            }
            for f in selection.facts
        ],
    }


def main():
    result = {}
    for dataset in ("corporatebench", "longmemeval-s", "medmemorybench"):
        with (VAR / f"plan-{dataset}.pickle").open("rb") as handle:
            samples = pickle.load(handle)["full"]
        store = SqliteStore(ROOT / "var/facts-benchmark-20261005/tianxi.db")
        counts = {str(limit): Counter() for limit in (12, 24, 64, 100)}
        changes = []
        for sample in samples:
            for question in sample.questions:
                attempts = {
                    str(limit): inspect(store, sample, question, limit)
                    for limit in (12, 24, 64, 100)
                }
                for limit, attempt in attempts.items():
                    counts[limit][attempt["reason"]] += 1
                if attempts["12"] != attempts["64"]:
                    changes.append(
                        {
                            "qid": question.qid,
                            "question": question.question,
                            "user_id": sample.user_id,
                            "attempts": attempts,
                        }
                    )
        result[dataset] = {"counts": counts, "changed": changes}
        print(dataset, {limit: dict(c) for limit, c in counts.items()}, flush=True)
    (OUT / "prototype-limits.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
