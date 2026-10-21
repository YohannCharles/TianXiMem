"""LLM verification stage for offline fact-extraction candidates.

The verifier is an evaluation probe only. It never runs in Add/Search and does
not persist facts. It accepts a Qwen candidate cache, asks Qwen to keep/drop or
normalize each candidate against the original source, and writes a new cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:  # Direct script execution resolves sibling modules without the package prefix.
    from fact_entity_smoke import _score_triples, load_cases
    from fact_llm_extract import _load_env_file, _relation_vocabulary
    from fact_llm_gate import _scopes, _trigger_supported
except ModuleNotFoundError:  # Package import from pytest.
    from tools.fact_entity_smoke import _score_triples, load_cases
    from tools.fact_llm_extract import _load_env_file, _relation_vocabulary
    from tools.fact_llm_gate import _scopes, _trigger_supported

from tianximem.facts.evidence import key, relation_key
from tianximem.llm.base import LlmUnavailable
from tianximem.llm.openai_chat import OpenAiCompatChat

PROMPT_VERSION = "common-facts-qwen-v17-abstain"
DEFAULT_VERIFY_RELATIONS = (
    "occupation,employee,employer,author,plans to open,plans to expand,"
    "profession,location,place of birth,head of government,capital,founded by,"
    "country of citizenship"
)
PROMPT_VERSIONS = (
    "common-facts-qwen-v14-verify-strict",
    "common-facts-qwen-v15-verify-normalize",
    "common-facts-qwen-v16-verify-context",
    "common-facts-qwen-v17-abstain",
)


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            result[row["id"]] = row
        else:
            raise ValueError(f"cache 第 {line_number} 行缺少 id")
    return result


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_json(raw: str) -> Any:
    cleaned = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _candidate_prompt(
    source: str,
    candidates: list[tuple[str, str, str]],
    vocabulary: tuple[str, ...],
    prompt_version: str,
) -> str:
    candidate_lines = "\n".join(
        f"C{index}: subject={subject!r}; relation={relation!r}; object={object_!r}"
        for index, (subject, relation, object_) in enumerate(candidates)
    )
    if prompt_version == "common-facts-qwen-v17-abstain":
        return f"""Review untrusted fact candidates against SOURCE. Do not extract new facts.
SOURCE is authoritative, including fictional or counterfactual Corpus rows.
Do not use world knowledge. Independently check each candidate's local clause,
speaker turn, or labeled record; the presence of both names somewhere in a
document does not establish their relation.

SOURCE:
{source}

CANDIDATES:
{candidate_lines or '(none)'}

Return JSON only:
{{"decisions":[{{"candidate_index":0,"decision":"keep","evidence":"exact local quote","reason":"supported"}}]}}

Decisions:
- keep: directly supported, including independent biography facts and every
  explicit Corpus row even when a later row replaces it.
- drop: a clear semantic error. Quote the exact source clause that demonstrates
  the mistake and choose a reason below. Do not rewrite endpoints or relations.
- uncertain: ambiguous, missing context, fragmented wording, or normalization
  differences. Do not drop merely because the document is long or a label differs.

Permitted drop reasons:
- wrong-speaker: a first-person claim belongs to another named speaker.
- wrong-role: a profile Profession became employer/employee, or a To recipient
  became attendee without an explicit attendee assertion.
- unsupported-authorship: title/possession/mention alone became authorship.
- wrong-modality: a proposed/completed action is confused with an intention;
  preserve the source's commitment and tense, and abstain if ambiguous.
- wrong-relation: the local clause expresses a clearly different relation or
  direction, such as integrating districts into a city becoming founded-by-city.

Judge meaning rather than exact string equality. Output one decision per
candidate. If unsure, choose uncertain. Only drop with a decisive local quote.

Mandatory calibration examples:
- SOURCE `Jaroslav Pelikan works for Valparaiso University from 1946 to 1949.`
  plus candidate `(Jaroslav Pelikan, employee, Valparaiso University)` is KEEP.
  A source sentence whose predicate literally says `works for`, `was born in`,
  `is a citizen of`, or `was founded by` is decisive support for that relation;
  never drop it because another paragraph is long or because a gold file may
  omit the row.
- SOURCE `Scot's book is titled The Discoverie of Witchcraft` plus candidate
  `(The Discoverie of Witchcraft, author, Scot)` is DROP with reason
  `unsupported-authorship`: possession and `is titled` do not assert authorship.
- SOURCE `Jon: Lost my job as a banker` plus candidate `(Gina, occupation,
  banker)` is DROP with reason `wrong-speaker`; the banker fact belongs to Jon.
- SOURCE `I'm starting a dance studio` plus candidate `(Jon, plans to open,
  dance studio)` is UNCERTAIN unless the source explicitly says want/plan/open;
  preserve it rather than converting a completed/current action.
- SOURCE `Profession: Colour Technologist` plus candidate `(Craig Baker,
  employee, Colour Technologist)` is DROP with reason `wrong-role`; the profile
  profession is not employment.
"""
    base = f"""You are a fact-extraction verifier. SOURCE is authoritative. The C rows
are untrusted proposals from another extractor, not facts.

SOURCE:
{source}

CANDIDATES:
{candidate_lines or '(none)'}

Allowed normalized relations:
{', '.join(vocabulary)}

For every candidate, inspect the exact source wording. Return JSON only:
{{"decisions":[{{"candidate_index":0,"decision":"keep","subject":"...","relation":"...","object":"..."}}]}}

Decision meanings:
- `keep`: the candidate's subject, relation meaning, and object are explicitly
  asserted by the same source clause or structured record.
- `correct`: the candidate is about an explicit source fact but its relation
  label is a bad normalization. Return the corrected lowercase relation from
  the allowed vocabulary. Do not invent a new fact.
- `drop`: the candidate is inferred, misattributed, reversed, a question/plan
  mismatch, a title or apposition error, or otherwise unsupported.

Rules:
1. Never create a decision for a candidate index that was not supplied. Never
   add facts that are not represented by a candidate.
2. Keep grammatical direction. A named dialogue speaker owns that line's I/me/my;
   never transfer a first-person fact to another speaker.
3. A candidate fails when it turns a title into authorship, a mention into a
   relation, a date into a place, or a person mentioned in a memorial into a
   speaker/organizer. Reject second-hop and world-knowledge inferences.
4. A relation is explicit only when the source predicate or a recognized field
   label licenses it. For structured profile/email/calendar records, respect
   `Name`, `Profession`, `Location`, `Organizer`, and `Attendees` boundaries.
5. Output at most one decision per candidate and no explanation.
"""
    if prompt_version == "common-facts-qwen-v14-verify-strict":
        return base + """
Strict mode: keep the candidate's subject and object exactly. Use `correct` only
to normalize the relation label; do not change endpoints.
"""
    if prompt_version == "common-facts-qwen-v15-verify-normalize":
        return base + """
Normalization mode: preserve candidate endpoints when they are source spans, but
correct relation semantics when the source clearly supports another allowed
label. For example, “I'm starting a dance studio” supports `started`, not
`plans to open`; “I want to open a studio” supports `plans to open`.
"""
    if prompt_version == "common-facts-qwen-v16-verify-context":
        return base + """
Context mode: additionally resolve record ownership. In `Speaker: ...`, bind I
to that Speaker. A profile field binds to the nearest Name in that profile. An
explicit Attendees list licenses attendee facts; a To header alone does not.
Drop candidates that use an entity from another speaker/record, even if both
endpoints appear somewhere in SOURCE. Preserve independent biography and
employment rows.
"""
    raise ValueError(f"unknown verifier prompt version: {prompt_version}")


def _parse_decisions(
    raw: str,
    candidates: list[tuple[str, str, str]],
    source: str,
    vocabulary: set[str],
    *,
    abstain: bool = False,
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    payload = _parse_json(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        raise ValueError("verifier response must contain decisions")
    accepted: list[tuple[str, str, str]] = []
    decisions: list[dict[str, Any]] = []
    source_folded = source.casefold()
    seen: set[int] = set()
    for item in payload["decisions"]:
        if not isinstance(item, dict):
            continue
        index = item.get("candidate_index")
        if not isinstance(index, int) or index < 0 or index >= len(candidates) or index in seen:
            continue
        seen.add(index)
        decision = item.get("decision")
        if decision not in {"keep", "correct", "drop", "uncertain"}:
            continue
        original = candidates[index]
        if abstain:
            evidence = item.get("evidence", "")
            reason = item.get("reason", "")
            decisive = (
                decision == "drop"
                and isinstance(evidence, str)
                and len(evidence.strip()) >= 3
                and evidence.strip() in source
                and reason in {
                    "wrong-speaker", "wrong-role", "unsupported-authorship",
                    "wrong-modality", "wrong-relation",
                }
            )
            scope = _candidate_scope(source, original[0], original[2])
            protected = scope is not None and _trigger_supported(scope, original[1], original[2])
            if decisive and protected:
                decisive = False
                reason = "literal-support-protected"
            if _modality_conflict(source, original):
                decisive = True
                reason = "local-modality-conflict"
                evidence = scope or evidence
            if not decisive:
                accepted.append(original)
            decisions.append({
                "candidate_index": index,
                "decision": "drop" if decisive else "keep" if decision == "keep" else "uncertain",
                "input": dict(zip(("subject", "relation", "object"), original, strict=True)),
                "evidence": evidence,
                "reason": reason,
                "accepted": not decisive,
            })
            continue
        out_subject = item.get("subject", original[0])
        out_relation = item.get("relation", original[1])
        out_object = item.get("object", original[2])
        if not all(isinstance(value, str) and value.strip() for value in (out_subject, out_relation, out_object)):
            continue
        triple = (out_subject.strip(), out_relation.strip(), out_object.strip())
        accepted_row = False
        if decision in {"keep", "correct"}:
            if (
                relation_key(triple[1]) not in vocabulary
                or triple[0].casefold() not in source_folded
                or triple[2].casefold() not in source_folded
            ):
                decision = "drop"
            else:
                accepted.append(triple)
                accepted_row = True
        decisions.append(
            {
                "candidate_index": index,
                "decision": decision,
                "input": {
                    "subject": original[0],
                    "relation": original[1],
                    "object": original[2],
                },
                "output": {
                    "subject": triple[0],
                    "relation": triple[1],
                    "object": triple[2],
                },
                "accepted": accepted_row,
            }
        )
    if abstain:
        accepted.extend(candidate for index, candidate in enumerate(candidates) if index not in seen)
    return list(dict.fromkeys(accepted)), decisions


def _select_cases(cases: tuple[dict[str, Any], ...], args: argparse.Namespace):
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        return tuple(case for case in cases if case["id"] in wanted)
    selected = cases[args.offset :]
    return selected[: args.limit] if args.limit is not None else selected


def _candidate_scope(source: str, subject: str, object_: str) -> str | None:
    """Find the smallest record that owns both endpoints.

    Named-speaker subjects are resolved to their own turn before generic
    same-scope matching, preventing a name mentioned in another turn from
    protecting a cross-speaker hallucination.
    """
    lines = source.splitlines()
    has_speaker = False
    for index, line in enumerate(lines):
        match = re.match(r"^\s*([^:]{1,80}):\s*", line)
        if match and match.group(1).strip().casefold() == subject.casefold():
            has_speaker = True
            end = index + 1
            while end < len(lines) and not re.match(r"^\s*([^:]{1,80}):\s*", lines[end]):
                end += 1
            turn = "\n".join(lines[index:end])
            if object_.casefold() in turn.casefold():
                return turn
            continue
    if has_speaker:
        return None
    for scope in _scopes(source):
        folded = scope.casefold()
        if subject.casefold() in folded and object_.casefold() in folded:
            return scope
    return None


def _modality_conflict(source: str, triple: tuple[str, str, str]) -> bool:
    """Identify a plan label attached to a current/completed action clause."""
    subject, relation, object_ = triple
    if relation_key(relation) != "plans to open":
        return False
    scope = _candidate_scope(source, subject, object_)
    if scope is None:
        return False
    folded = scope.casefold()
    object_folded = object_.casefold()
    plan_words = (
        "want to open", "wanna open", "plan to open",
        "search for a place to open", "looking for a place to open",
    )
    action_words = ("i'm starting", "im starting", "started", "start my own")
    return (
        object_folded in folded
        and any(word in folded for word in action_words)
        and not any(word in folded for word in plan_words)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    parser.add_argument(
        "--verify-relations",
        default=DEFAULT_VERIFY_RELATIONS,
        help="comma-separated high-risk relation labels to verify",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    _load_env_file(args.env_file)
    base_url = args.base_url or os.environ.get("AML_BASE_URL", "")
    api_key = args.api_key or os.environ.get("AML_API_KEY", "")
    if not base_url or not api_key:
        print("LLM endpoint is not configured: set AML_BASE_URL and AML_API_KEY", file=sys.stderr)
        return 2

    all_cases = load_cases(args.gold)
    cases = _select_cases(all_cases, args)
    vocabulary = set(_relation_vocabulary(all_cases))
    verify_relations = {
        relation_key(value)
        for value in args.verify_relations.split(",")
        if value.strip()
    }
    input_rows = _read_cache(args.candidates)
    output_rows = {} if args.no_resume else _read_cache(args.out)
    client = OpenAiCompatChat(base_url=base_url, api_key=api_key, model=args.model, timeout=args.timeout)
    try:
        for index, case in enumerate(cases, 1):
            if (
                case["id"] in output_rows
                and output_rows[case["id"]].get("prompt_version") == args.prompt_version
                and not args.no_resume
            ):
                continue
            input_row = input_rows.get(case["id"], {})
            candidates = []
            for item in input_row.get("relations", []):
                if isinstance(item, dict):
                    values = tuple(item.get(name) for name in ("subject", "relation", "object"))
                    if all(isinstance(value, str) and value.strip() for value in values):
                        candidates.append(values)  # type: ignore[arg-type]
            risky_indices = [
                index
                for index, (_subject, relation, _object_) in enumerate(candidates)
                if not verify_relations or relation_key(relation) in verify_relations
            ]
            risky_candidates = [candidates[index] for index in risky_indices]
            row: dict[str, Any] = {
                "id": case["id"],
                "text": case["text"],
                "model": args.model,
                "prompt_version": args.prompt_version,
                "input_cache": str(args.candidates),
                "input_relations": [
                    {"subject": s, "relation": r, "object": o} for s, r, o in candidates
                ],
                "verified_relation_labels": sorted(verify_relations),
            }
            try:
                raw = ""
                relations = list(candidates)
                decisions = []
                if risky_candidates:
                    raw = client.complete(
                        _candidate_prompt(
                            case["text"],
                            risky_candidates,
                            tuple(sorted(vocabulary)),
                            args.prompt_version,
                        ),
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                    )
                    verified, local_decisions = _parse_decisions(
                        raw, risky_candidates, case["text"], vocabulary,
                        abstain=args.prompt_version == "common-facts-qwen-v17-abstain",
                    )
                    # Keep every non-risky candidate unchanged, and replace the
                    # risky subset with only verifier-approved rows.
                    relations = [
                        candidates[index]
                        for index in range(len(candidates))
                        if index not in risky_indices
                    ] + verified
                    for decision in local_decisions:
                        decision["candidate_index"] = risky_indices[decision["candidate_index"]]
                    decisions = local_decisions
                row["relations"] = [
                    {"subject": s, "relation": r, "object": o} for s, r, o in relations
                ]
                row["decisions"] = decisions
                row["raw"] = raw
            except (LlmUnavailable, ValueError, json.JSONDecodeError) as exc:
                row["relations"] = (
                    [{"subject": s, "relation": r, "object": o} for s, r, o in candidates]
                    if args.prompt_version == "common-facts-qwen-v17-abstain" else []
                )
                row["decisions"] = []
                row["error"] = str(exc)
            _append_row(args.out, row)
            print(f"[{index}/{len(cases)}] {case['id']}", file=sys.stderr)
    finally:
        client.close()

    rows = _read_cache(args.out)
    predicted = []
    selected_cases = []
    for case in cases:
        selected_cases.append(case)
        row = rows.get(case["id"], {})
        predicted.append(
            {
                (key(item["subject"]), relation_key(item["relation"]), key(item["object"]))
                for item in row.get("relations", [])
                if isinstance(item, dict)
                and all(isinstance(item.get(name), str) for name in ("subject", "relation", "object"))
            }
        )
    report = {
        "model": args.model,
        "prompt_version": args.prompt_version,
        "gold": str(args.gold),
        "candidates": str(args.candidates),
        "out": str(args.out),
        "cases": len(selected_cases),
        "score": _score_triples(tuple(selected_cases), predicted),
    }
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
