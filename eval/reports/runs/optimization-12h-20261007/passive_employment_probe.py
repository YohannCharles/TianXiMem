"""Validate literal passive employment before changing the source grammar."""

from __future__ import annotations

import argparse
import json
import types
from collections import Counter
from pathlib import Path

from experiment import OUT, VAR, write

from tianximem.facts import grammar
from tianximem.facts.evidence import relation_key
from tianximem.store.sqlite_store import SqliteStore

HERE = Path(__file__).resolve().parent


def prototype():
    source = Path(grammar.__file__).read_text()
    if '("employee", "is employed by"),' in source:
        raise RuntimeError("Use the pre-change a03 source snapshot to reproduce this prototype")
    old = '    ("employee", "works for"),'
    assert source.count(old) == 1
    source = source.replace(old, old + '\n    ("employee", "is employed by"),')
    old = "                for rel, phrase in INFIX_FORMS:\n"
    assert source.count(old) == 1
    source = source.replace(
        old,
        old
        + """                    if phrase == "is employed by" and (
                        _UNSAFE.search(body) or _PLANNED.search(body)
                    ):
                        continue
""",
    )
    shadow = types.ModuleType("passive_employment_shadow")
    exec(compile(source, "<passive_employment_shadow>", "exec"), shadow.__dict__)
    shadow.source_text = source
    return shadow


def native_corpus(shadow):
    counts = Counter()
    for group in ("seven", "four"):
        store = SqliteStore(VAR / f"a03/{group}/tianxi.db")
        with store.read() as conn:
            for pair in store.iter_pairs(conn):
                if not pair.question or "is employed by" not in pair.question.casefold():
                    continue
                params = dict(
                    parent_memory_id=pair.id,
                    user_id=pair.user_id,
                    question=pair.question,
                    answer=pair.answer,
                    event_time=pair.event_time,
                )

                def signature(facts):
                    return [
                        (f.subject, f.relation, f.object, f.source_quote, dict(f.qualifiers))
                        for f in facts
                    ]

                if signature(grammar.extract_evidence(**params)) != signature(
                    shadow.extract_evidence(**params)
                ):
                    counts[pair.user_id] += 1
    write(
        OUT / "prototype-passive-employment-corpus.json", dict(changed_sources_by_user=dict(counts))
    )
    print(dict(counts), flush=True)


def main(scan):
    shadow = prototype()
    cases = (
        (
            "explicit_edit",
            "Corpus: Alex is employed by Acme (this replaces the earlier value)",
            True,
        ),
        ("explicit_interval", "Corpus: Alex is employed by Acme from Jan, 2020 to Dec, 2021", True),
        (
            "negative",
            "Corpus: Alex is not employed by Acme (this replaces the earlier value)",
            False,
        ),
        (
            "conditional",
            "Corpus: If Alex is employed by Acme (this replaces the earlier value)",
            False,
        ),
        (
            "conditional_object",
            "Corpus: Alex is employed by Acme if approved (this replaces the earlier value)",
            False,
        ),
        (
            "planned",
            "Corpus: Alex will be employed by Acme (this replaces the earlier value)",
            False,
        ),
        (
            "quotation",
            'Corpus: Example reads "Alex is employed by Acme" (this replaces the earlier value)',
            False,
        ),
        (
            "unknown_object",
            "Corpus: Alex is employed by unknown (this replaces the earlier value)",
            False,
        ),
        ("plain_text_scope_preserved", "Alex is employed by Acme.", False),
    )
    rows = []
    for name, text, expected in cases:
        facts = shadow.extract_evidence(
            parent_memory_id=name, user_id="manual", question=text, answer=None, event_time=None
        )
        positive = [f for f in facts if relation_key(f.relation) == "employee"]
        passed = bool(positive) == expected and all(f.source_quote in text for f in facts)
        if positive:
            passed = passed and all(f.subject == "Alex" and f.object == "Acme" for f in positive)
        print(name, "PASS" if passed else "FAIL", flush=True)
        rows.append(
            dict(
                case=name,
                source=text,
                expected=expected,
                passed=passed,
                facts=[
                    dict(
                        subject=f.subject,
                        relation=f.relation,
                        object=f.object,
                        qualifiers=dict(f.qualifiers),
                    )
                    for f in facts
                ],
            )
        )
    (HERE / "prototype-passive-employment.json").write_text(
        json.dumps(dict(manual=rows), ensure_ascii=False, indent=2) + "\n"
    )
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Passive employment manual gate failed")
    if scan:
        native_corpus(shadow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-corpus", action="store_true")
    main(parser.parse_args().native_corpus)
