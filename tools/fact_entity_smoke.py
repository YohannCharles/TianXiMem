"""Small, model-independent smoke for the fact entity sidecar.

This intentionally measures only sidecar candidates.  It does not call
Add/Search.  A JSONL gold file can additionally provide exact fact triples for
precision/recall/F1 without treating the existing database as gold.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from tianximem.facts.entities import (
    EntityExtractor,
    SpacyEntityExtractor,
    load_spacy_extractor,
)
from tianximem.facts.evidence import key, relation_key
from tianximem.facts.grammar import (
    _model_fact_candidates,
    _model_sources,
    extract_evidence,
)

CASES = (
    {
        "id": "role",
        "text": "Alice works for Acme Labs.",
        "entities": ("Alice", "Acme Labs"),
    },
    {
        "id": "quantity",
        "text": "I sold 12 cups for $3.50 each.",
        "entities": ("12", "$3.50"),
    },
    {
        "id": "date-place",
        "text": "The team met in Paris on 2025-01-03.",
        "entities": ("Paris", "2025-01-03"),
    },
)


def load_cases(path: Path) -> tuple[dict[str, Any], ...]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"gold JSONL 第 {line_number} 行非法: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"gold JSONL 第 {line_number} 行需要 text 字段")
        entities = row.get("entities", [])
        if not isinstance(entities, list) or not all(isinstance(item, str) for item in entities):
            raise ValueError(f"gold JSONL 第 {line_number} 行 entities 需要字符串数组")
        relations = row.get("relations", [])
        if not isinstance(relations, list):
            raise ValueError(f"gold JSONL 第 {line_number} 行 relations 需要数组")
        normalized_relations = []
        for relation in relations:
            if isinstance(relation, dict):
                subject = relation.get("subject")
                predicate = relation.get("relation")
                object_ = relation.get("object")
            elif isinstance(relation, (list, tuple)) and len(relation) == 3:
                subject, predicate, object_ = relation
            else:
                raise ValueError(
                    f"gold JSONL 第 {line_number} 行 relation 需要 subject/relation/object"
                )
            if not all(isinstance(item, str) for item in (subject, predicate, object_)):
                raise ValueError(f"gold JSONL 第 {line_number} 行 relation 字段必须是字符串")
            normalized_relations.append((subject, predicate, object_))
        cases.append(
            {
                "id": str(row.get("id", line_number)),
                "text": row["text"],
                "entities": tuple(entities),
                "relations": tuple(normalized_relations),
            }
        )
    return tuple(cases)


def evaluate(extractor: EntityExtractor, cases: tuple[dict, ...] = CASES) -> dict:
    rows = []
    expected_total = found_total = 0
    for case in cases:
        result = extractor.extract(case["text"])
        found = {entity.text for entity in result.entities}
        expected = set(case["entities"])
        hits = sorted(expected & found)
        expected_total += len(expected)
        found_total += len(hits)
        rows.append(
            {
                "id": case["id"],
                "expected": sorted(expected),
                "found": sorted(found),
                "hits": hits,
                "recall": len(hits) / len(expected) if expected else 1.0,
                "backend": result.backend,
            }
        )
    predicted_total = sum(len(row["found"]) for row in rows)
    gold_total = sum(len(row["expected"]) for row in rows)
    hit_total = sum(len(row["hits"]) for row in rows)
    precision = hit_total / predicted_total if predicted_total else 1.0
    recall = hit_total / gold_total if gold_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "cases": rows,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
    }
    relation_rows = []
    expected_relations = set()
    found_relations = set()
    for case in cases:
        gold = {
            (key(subject), relation_key(predicate), key(object_))
            for subject, predicate, object_ in case.get("relations", ())
        }
        predicted = {
            (
                relation.subject.normalized,
                relation_key(relation.relation),
                relation.object.normalized,
            )
            for relation in extractor.extract(case["text"]).relations
            if relation.object is not None
        }
        expected_relations.update(gold)
        found_relations.update(predicted)
        relation_rows.append(
            {
                "id": case["id"],
                "expected": sorted(gold),
                "found": sorted(predicted),
                "hits": sorted(gold & predicted),
            }
        )
    if expected_relations or any(case.get("relations") for case in cases):
        relation_hits = expected_relations & found_relations
        relation_precision = (
            len(relation_hits) / len(found_relations) if found_relations else 1.0
        )
        relation_recall = (
            len(relation_hits) / len(expected_relations) if expected_relations else 1.0
        )
        report.update(
            {
                "relation_cases": relation_rows,
                "relation_precision": relation_precision,
                "relation_recall": relation_recall,
                "relation_f1": (
                    2 * relation_precision * relation_recall
                    / (relation_precision + relation_recall)
                    if relation_precision + relation_recall
                    else 0.0
                ),
            }
        )
    return report


def _gold_triples(case: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (key(subject), relation_key(predicate), key(object_))
        for subject, predicate, object_ in case.get("relations", ())
    }


def _fact_triples(facts: Any) -> set[tuple[str, str, str]]:
    return {
        (key(fact.subject), relation_key(fact.relation), key(fact.object))
        for fact in facts
        if fact.relation != "explicit replacement"
    }


def _model_triples(extractor: EntityExtractor, text: str) -> set[tuple[str, str, str]]:
    return {
        (key(subject), relation_key(relation), key(object_))
        for subject, relation, object_, _quote in _model_fact_candidates(
            source=text, extractor=extractor
        )
    }


def _score_triples(
    cases: tuple[dict[str, Any], ...], predicted: list[set[tuple[str, str, str]]]
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    errors: list[dict[str, Any]] = []
    for case, found in zip(cases, predicted, strict=True):
        gold = _gold_triples(case)
        hits = gold & found
        false_positives = sorted(found - gold)
        false_negatives = sorted(gold - found)
        true_positive += len(hits)
        false_positive += len(false_positives)
        false_negative += len(false_negatives)
        if false_positives or false_negatives:
            errors.append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                }
            )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "error_count": len(errors),
        "errors": errors[:50],
    }


def evaluate_gold_comparison(
    extractor: EntityExtractor, cases: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    """Compare deterministic rules, model sidecar, and their combination."""
    rules: list[set[tuple[str, str, str]]] = []
    model: list[set[tuple[str, str, str]]] = []
    combined: list[set[tuple[str, str, str]]] = []
    for index, case in enumerate(cases):
        text = case["text"]
        rules.append(
            _fact_triples(
                extract_evidence(
                    parent_memory_id=f"gold-{index}",
                    user_id="gold",
                    question=text,
                    answer=None,
                    event_time=None,
                )
            )
        )
        model.append(_model_triples(extractor, text))
        combined.append(
            _fact_triples(
                extract_evidence(
                    parent_memory_id=f"gold-{index}",
                    user_id="gold",
                    question=text,
                    answer=None,
                    event_time=None,
                    entity_extractor=extractor,
                )
            )
        )
    return {
        "cases": len(cases),
        "rules": _score_triples(cases, rules),
        "model_sidecar": _score_triples(cases, model),
        "combined": _score_triples(cases, combined),
    }


def evaluate_sqlite(extractor: EntityExtractor, database: Path, limit: int) -> dict:
    """Run the sidecar over a few real QA rows without touching the database."""
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT id, question, answer FROM qa_pairs ORDER BY rowid LIMIT ?", (limit,)
        ).fetchall()
    finally:
        connection.close()
    reports = []
    entity_count = 0
    for memory_id, question, answer in rows:
        text = "\n".join(part for part in (question, answer) if part)
        result = extractor.extract(text)
        entity_count += len(result.entities)
        reports.append(
            {
                "id": memory_id,
                "chars": len(text),
                "entities": len(result.entities),
                "relations": len(result.relations),
                "backend": result.backend,
            }
        )
    return {"rows": reports, "entity_count": entity_count}


def evaluate_fact_endpoints(extractor: EntityExtractor, database: Path, limit: int) -> dict:
    """Compare entity candidates with endpoints already recorded as facts.

    This is a coverage probe, not a gold-standard accuracy score: facts whose
    endpoint is not written in their quote (for example a synthetic parent id)
    are excluded from the denominator and reported separately.
    """
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT subject, relation, object, source_quote FROM memory_facts "
            "ORDER BY rowid LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    endpoint_total = endpoint_hits = quote_total = 0
    subject_total = object_total = relation_total = 0
    subject_hits = object_hits = relation_hits = 0
    triple_hits = triple_total = 0
    reports = []
    for subject, relation, object_, quote in rows:
        result = extractor.extract(quote)
        found = {entity.normalized for entity in result.entities}
        subject_key = key(str(subject))
        object_key = key(str(object_))
        quote_key = key(quote)
        observable_subject = bool(subject_key and subject_key in quote_key)
        observable_object = bool(object_key and object_key in quote_key)
        expected = [
            endpoint
            for endpoint, observable in (
                (subject_key, observable_subject),
                (object_key, observable_object),
            )
            if observable
        ]
        hits = [endpoint for endpoint in expected if endpoint in found]
        endpoint_total += len(expected)
        endpoint_hits += len(hits)
        quote_total += 1
        subject_candidates = {
            relation.subject.normalized for relation in result.relations
        }
        object_candidates = {
            relation.object.normalized
            for relation in result.relations
            if relation.object is not None
        }
        relation_candidates = {relation_key(relation.relation) for relation in result.relations}
        expected_relation = relation_key(str(relation))
        subject_total += int(observable_subject)
        object_total += int(observable_object)
        subject_hits += int(observable_subject and subject_key in subject_candidates)
        object_hits += int(observable_object and object_key in object_candidates)
        relation_observable = observable_subject and observable_object
        relation_total += int(relation_observable)
        relation_hits += int(relation_observable and expected_relation in relation_candidates)
        triple_total += int(relation_observable)
        triple_hits += int(
            relation_observable
            and any(
                candidate.subject.normalized == subject_key
                and candidate.object is not None
                and candidate.object.normalized == object_key
                and relation_key(candidate.relation) == expected_relation
                for candidate in result.relations
            )
        )
        reports.append(
            {
                "quote": quote,
                "expected_endpoints": expected,
                "hit_endpoints": hits,
                "relation": expected_relation,
                "candidate_relations": sorted(relation_candidates),
                "backend": result.backend,
            }
        )
    return {
        "quotes": quote_total,
        "endpoint_hits": endpoint_hits,
        "endpoint_total": endpoint_total,
        "endpoint_recall": endpoint_hits / endpoint_total if endpoint_total else None,
        "subject_recall": subject_hits / subject_total if subject_total else None,
        "object_recall": object_hits / object_total if object_total else None,
        "relation_recall": relation_hits / relation_total if relation_total else None,
        "triple_recall": triple_hits / triple_total if triple_total else None,
        "rows": reports,
    }


def evaluate_model_fact_precision(
    extractor: EntityExtractor, database: Path, limit: int
) -> dict:
    """Compare model-side triples with same-source indexed facts.

    This is a regression reference, not an independent gold score.
    """
    connection = sqlite3.connect(database)
    try:
        qa_rows = connection.execute(
            "SELECT id, question, answer FROM qa_pairs ORDER BY rowid LIMIT ?", (limit,)
        ).fetchall()
        fact_rows = connection.execute(
            "SELECT parent_memory_id, subject, relation, object, source_side FROM memory_facts"
        ).fetchall()
    finally:
        connection.close()
    reference: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for parent_id, subject, relation, object_, side in fact_rows:
        reference.setdefault((parent_id, side), set()).add(
            (key(str(subject)), relation_key(str(relation)), key(str(object_)))
        )
    predicted = hit = 0
    misses: list[dict[str, str]] = []
    for parent_id, question, answer in qa_rows:
        for side, source in (("question", question), ("answer", answer)):
            for fragment in _model_sources(side, source):
                for subject, relation, object_, quote in _model_fact_candidates(
                    source=fragment, extractor=extractor
                ):
                    predicted += 1
                    triple = (key(subject), relation_key(relation), key(object_))
                    if triple in reference.get((parent_id, side), set()):
                        hit += 1
                    elif len(misses) < 20:
                        misses.append(
                            {
                                "parent_memory_id": parent_id,
                                "side": side,
                                "subject": subject,
                                "relation": relation,
                                "object": object_,
                                "quote": quote,
                            }
                        )
    return {
        "rows": len(qa_rows),
        "predicted": predicted,
        "hits": hit,
        "precision": hit / predicted if predicted else None,
        "misses": misses,
        "reference_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="",
        help="可选 spaCy 模型名；缺省使用 tokenizer + 规则补缺",
    )
    parser.add_argument("--database", type=Path, help="可选 SQLite truth.db，只读真实 QA smoke")
    parser.add_argument("--limit", type=int, default=5, help="真实 QA smoke 行数")
    parser.add_argument(
        "--fact-endpoints",
        action="store_true",
        help="从 truth.db 的 memory_facts 做只读端点覆盖 smoke",
    )
    parser.add_argument(
        "--model-fact-precision",
        action="store_true",
        help="按同一来源已有事实评估模型侧路精确率（仅回归参考）",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        help=(
            "JSONL: {id,text,entities:[...],relations:[{subject,relation,object}]}，"
            "评估实体和事实"
        ),
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args(argv)

    if args.model:
        extractor = load_spacy_extractor(args.model)
    else:
        try:
            import spacy
        except ImportError as exc:
            print(f"spaCy 未安装：{exc}", file=sys.stderr)
            return 2
        extractor = SpacyEntityExtractor(spacy.blank("en"), model_name="blank-en")
    if args.gold:
        cases = load_cases(args.gold)
        report = {
            "entity_relation_smoke": evaluate(extractor, cases),
            "gold_comparison": evaluate_gold_comparison(extractor, cases),
        }
    elif args.database and args.model_fact_precision:
        report = evaluate_model_fact_precision(extractor, args.database, args.limit)
    elif args.database and args.fact_endpoints:
        report = evaluate_fact_endpoints(extractor, args.database, args.limit)
    elif args.database:
        report = evaluate_sqlite(extractor, args.database, args.limit)
    else:
        report = evaluate(extractor)
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
