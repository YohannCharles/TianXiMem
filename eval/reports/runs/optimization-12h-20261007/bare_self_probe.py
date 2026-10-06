"""Manually test anonymous English self-claims before broadening the extractor."""

from __future__ import annotations

import json
import types
from pathlib import Path

from tianximem.facts import grammar
from tianximem.facts.evidence import relation_key

HERE = Path(__file__).resolve().parent


def main() -> None:
    source = Path(grammar.__file__).read_text()
    old = "if speaker and safe_attributes"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected extractor source; refusing a loose replacement")
    shadow = types.ModuleType("bare_self_shadow")
    exec(
        compile(source.replace(old, "if safe_attributes"), "bare_self_shadow.py", "exec"),
        shadow.__dict__,
    )
    cases = (
        ("formal_self_role", "I am an employee of Acme.", None, "employee"),
        ("literal_employment", "I work for Acme.", None, "employee"),
        ("assistant_is_not_self", None, "I work for Acme.", None),
        ("assistant_label_is_not_self", "Assistant: I work for Acme.", None, None),
        ("named_answer", None, "Nora: I am an employee of Acme.", "employee"),
        ("negative", "I no longer work for Acme.", None, None),
        ("planned", "I plan to work for Acme.", None, None),
        ("quotation", 'The example reads: "I work for Acme."', None, None),
        ("work_for_a_living_is_not_an_employer", "I work for a living.", None, None),
        ("live_in_the_moment_is_not_a_location", "I live in the moment.", None, None),
    )
    rows = []
    for case, question, answer, expected in cases:
        params = dict(
            parent_memory_id=case,
            user_id="manual",
            question=question,
            answer=answer,
            event_time=None,
        )
        before = grammar.extract_evidence(**params)
        after = shadow.extract_evidence(**params)
        relations = {relation_key(f.relation) for f in after}
        valid = all(f.source_quote in (params[f.source_side] or "") for f in after)
        passed = valid and (relations == {expected} if expected else not relations)
        print(case, "PASS" if passed else "FAIL", flush=True)
        rows.append(
            dict(
                case=case,
                question=question,
                answer=answer,
                expected=expected,
                before=[(f.subject, f.relation, f.object) for f in before],
                after=[(f.subject, f.relation, f.object) for f in after],
                passed=passed,
            )
        )
    decision = "continue manual review" if all(r["passed"] for r in rows) else "reject before code"
    (HERE / "prototype-bare-self.json").write_text(
        json.dumps(dict(cases=rows, decision=decision), ensure_ascii=False, indent=2) + "\n"
    )
    print(decision)


if __name__ == "__main__":
    main()
