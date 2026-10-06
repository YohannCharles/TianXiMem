"""Check preservation of undated claims against a frozen pre-change selector."""

from __future__ import annotations

import sys
import types

from experiment import OUT, VAR, write

from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import compile_query


def shadow(name, source):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


def main():
    source = (VAR / "a12/source/src/tianximem/retrieve/evidence.py").read_text()
    old = "                and f.event_time is not None\n                and f.event_time >= latest"
    assert source.count(old) == 1
    before = shadow("dated_current_before", source)
    after = shadow(
        "dated_current_after",
        source.replace(old, "                and (f.event_time is None or f.event_time >= latest)"),
    )
    rows = []
    for name, moment, expected in (
        ("unknown", None, {"edit", "ordinary"}),
        ("older", 1735689600000, {"edit"}),
        ("same", 1767225600000, {"edit", "ordinary"}),
        ("later", 1798761600000, {"edit", "ordinary"}),
    ):
        sources = (
            (
                "edit",
                "Corpus: Alex is employed by Acme (this replaces the earlier value)",
                1767225600000,
            ),
            ("ordinary", "Alex: I currently work for NewCo.", moment),
        )
        facts = [
            fact
            for parent, text, when in sources
            for fact in extract_evidence(
                parent_memory_id=parent,
                user_id="manual",
                question=text,
                answer=None,
                event_time=when,
            )
        ]
        outcomes = {}
        for arm, module in (("before", before), ("after", after)):
            result = module.evaluate_evidence(
                facts, compile_query("Where does Alex work?"), source_limit=12, hop_limit=4
            )
            outcomes[arm] = (
                sorted(f.parent_memory_id for f in result.selection.facts)
                if result.selection
                else []
            )
        passed = set(outcomes["after"]) == expected
        print(name, "PASS" if passed else "FAIL", flush=True)
        rows.append(
            dict(
                case=name,
                sources=sources,
                expected=sorted(expected),
                outcomes=outcomes,
                passed=passed,
            )
        )
    write(OUT / "prototype-unknown-time.json", dict(manual=rows))
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Unknown source-time manual gate failed")


if __name__ == "__main__":
    main()
