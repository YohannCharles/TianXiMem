"""Semantic fact evaluation for the manual diverse gold set.

The evaluator keeps exact triple scores as the primary regression metric.  For
triples that do not match exactly, it can ask the configured LLM whether a
predicted fact and a gold fact express the same asserted relation.  Semantic
matches are one-to-one and are reported separately; they never replace the
strict score.

This is an offline evaluation tool.  It does not change Add/Search or persist
LLM-generated facts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from fact_entity_smoke import load_cases
from fact_llm_extract import _load_env_file

from tianximem.facts.entities import EntityExtractor, load_spacy_extractor
from tianximem.facts.evidence import key, relation_key
from tianximem.facts.grammar import _model_fact_candidates
from tianximem.llm.base import LlmUnavailable
from tianximem.llm.openai_chat import OpenAiCompatChat


def _norm(triple: tuple[str, str, str]) -> tuple[str, str, str]:
    return key(triple[0]), relation_key(triple[1]), key(triple[2])


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _read_qwen_cache(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    result: dict[str, list[tuple[str, str, str]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"Qwen cache 第 {line_number} 行缺少 id")
        triples = []
        for item in row.get("relations", []):
            if not isinstance(item, dict):
                continue
            values = tuple(item.get(name) for name in ("subject", "relation", "object"))
            if all(isinstance(value, str) and value.strip() for value in values):
                triples.append(values)  # type: ignore[arg-type]
        result[row["id"]] = list(dict.fromkeys(triples))
    return result


def _spaCy_predictions(cases: tuple[dict[str, Any], ...], extractor: EntityExtractor):
    result: dict[str, list[tuple[str, str, str]]] = {}
    for case in cases:
        triples = []
        for subject, relation, object_, _quote in _model_fact_candidates(
            source=case["text"], extractor=extractor
        ):
            triples.append((subject, relation, object_))
        result[case["id"]] = list(dict.fromkeys(triples))
    return result


def _gold(case: dict[str, Any]) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys(case.get("relations", ())))


def _strict_matches(
    gold: list[tuple[str, str, str]], predicted: list[tuple[str, str, str]]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    by_triple: defaultdict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, triple in enumerate(predicted):
        by_triple[_norm(triple)].append(index)
    pairs: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    unmatched_gold: list[int] = []
    for gold_index, triple in enumerate(gold):
        choices = by_triple.get(_norm(triple), [])
        choice = next((index for index in choices if index not in used_pred), None)
        if choice is None:
            unmatched_gold.append(gold_index)
        else:
            used_pred.add(choice)
            pairs.append((gold_index, choice))
    unmatched_pred = [index for index in range(len(predicted)) if index not in used_pred]
    return pairs, unmatched_gold, unmatched_pred


def _semantic_prompt(
    source: str,
    gold: list[tuple[str, str, str]],
    predicted: list[tuple[str, str, str]],
    gold_indices: list[int],
    predicted_indices: list[int],
) -> str:
    gold_lines = "\n".join(
        f"G{index}: subject={gold[index][0]!r}; relation={gold[index][1]!r}; object={gold[index][2]!r}"
        for index in gold_indices
    )
    predicted_lines = "\n".join(
        f"P{index}: subject={predicted[index][0]!r}; relation={predicted[index][1]!r}; object={predicted[index][2]!r}"
        for index in predicted_indices
    )
    return f"""You are an exact semantic equivalence judge for information extraction.

SOURCE:
{source}

GOLD RELATIONS:
{gold_lines or '(none)'}

PREDICTED RELATIONS:
{predicted_lines or '(none)'}

For each pair, decide whether the predicted relation expresses the same fact
explicitly asserted by SOURCE as the gold relation. Accept harmless casing,
inflection, relation synonyms, and shorter/longer entity spans when they refer
to the same entity or event. Reject inferred facts, different entities,
different relation direction, contradictions, plans/questions, and relations
that only share a word. A predicted relation can match at most one gold
relation, and a gold relation can match at most one prediction.

Return JSON only:
{{"matches":[{{"gold_index":0,"predicted_index":0,"equivalent":true}}]}}
Include only equivalent=true pairs. Use the original numeric indices after G/P.
"""


def _semantic_pairs(
    client: OpenAiCompatChat,
    source: str,
    gold: list[tuple[str, str, str]],
    predicted: list[tuple[str, str, str]],
    gold_indices: list[int],
    predicted_indices: list[int],
) -> tuple[list[tuple[int, int]], str | None]:
    if not gold_indices or not predicted_indices:
        return [], None
    try:
        raw = client.complete(
            _semantic_prompt(source, gold, predicted, gold_indices, predicted_indices),
            max_tokens=512,
            temperature=0.0,
        )
        payload = _parse_json(raw)
    except (LlmUnavailable, ValueError, json.JSONDecodeError) as exc:
        return [], str(exc)
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        return [], "judge response has no matches array"
    allowed_gold, allowed_pred = set(gold_indices), set(predicted_indices)
    candidates: list[tuple[int, int]] = []
    for item in payload["matches"]:
        if not isinstance(item, dict) or item.get("equivalent") is not True:
            continue
        gi, pi = item.get("gold_index"), item.get("predicted_index")
        if isinstance(gi, int) and isinstance(pi, int) and gi in allowed_gold and pi in allowed_pred:
            candidates.append((gi, pi))
    # The response is advisory; enforce one-to-one matching deterministically.
    selected: list[tuple[int, int]] = []
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        selected.append((gi, pi))
    return selected, None


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    cases: tuple[dict[str, Any], ...],
    predictions: dict[str, list[tuple[str, str, str]]],
    client: OpenAiCompatChat | None,
) -> dict[str, Any]:
    strict_tp = strict_fp = strict_fn = 0
    semantic_tp = semantic_fp = semantic_fn = 0
    judge_calls = judge_errors = semantic_added = 0
    rows = []
    for case in cases:
        gold, predicted = _gold(case), predictions.get(case["id"], [])
        strict, missing_gold, extra_pred = _strict_matches(gold, predicted)
        strict_tp += len(strict)
        strict_fn += len(missing_gold)
        strict_fp += len(extra_pred)
        semantic = list(strict)
        error = None
        if client is not None and missing_gold and extra_pred:
            judge_calls += 1
            judged, error = _semantic_pairs(
                client, case["text"], gold, predicted, missing_gold, extra_pred
            )
            if error:
                judge_errors += 1
            semantic.extend(judged)
            semantic_added += len(judged)
        semantic_tp += len(semantic)
        semantic_fn += len(gold) - len(semantic)
        semantic_fp += len(predicted) - len(semantic)
        rows.append(
            {
                "id": case["id"],
                "strict_matches": len(strict),
                "semantic_matches": len(semantic),
                "gold_count": len(gold),
                "predicted_count": len(predicted),
                "semantic_added": len(semantic) - len(strict),
                "judge_error": error,
            }
        )
    return {
        "strict": _metrics(strict_tp, strict_fp, strict_fn),
        "semantic": _metrics(semantic_tp, semantic_fp, semantic_fn),
        "judge_calls": judge_calls,
        "judge_errors": judge_errors,
        "semantic_added_true_positives": semantic_added,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args(argv)

    _load_env_file(args.env_file)
    base_url = args.base_url or os.environ.get("AML_BASE_URL", "")
    api_key = args.api_key or os.environ.get("AML_API_KEY", "")
    cases = load_cases(args.gold)
    qwen_predictions = _read_qwen_cache(args.qwen_cache)
    extractor = load_spacy_extractor(args.spacy_model)
    spacy_predictions = _spaCy_predictions(cases, extractor)

    client = None
    if not args.no_judge:
        if not base_url or not api_key:
            print("semantic judge requires AML_BASE_URL and AML_API_KEY", file=sys.stderr)
            return 2
        client = OpenAiCompatChat(
            base_url=base_url, api_key=api_key, model=args.model, timeout=120.0
        )
    try:
        report = {
            "gold": str(args.gold),
            "qwen_cache": str(args.qwen_cache),
            "judge_model": args.model if client else None,
            "spacy_model": args.spacy_model,
            "qwen": evaluate(cases, qwen_predictions, client),
            "spacy": evaluate(cases, spacy_predictions, client),
        }
    finally:
        if client is not None:
            client.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
