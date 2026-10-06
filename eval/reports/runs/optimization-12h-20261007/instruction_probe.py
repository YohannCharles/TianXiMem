"""Compare query instructions on literal snippets before any config experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

from tianximem.common.config import load_config
from tianximem.common.render import render
from tianximem.embed.query_instruction import DEFAULT_TASK, apply_query_instruction
from tianximem.service.app import build_embedder

MEMORY_TASK = (
    "Given a question about a user's memories, retrieve original statements "
    "and linked context that directly support the answer"
)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b)
    )


def main():
    cases = (
        (
            "current_preference",
            "Which sport does Maya enjoy now?",
            (
                render("Maya: I no longer enjoy badminton. I enjoy basketball now.", None),
                render("Maya: I enjoyed badminton when I was younger.", None),
                render(None, "Assistant: Many people enjoy sports such as basketball."),
            ),
            {0},
        ),
        (
            "recorded_result_and_source",
            "What was my recorded TSH result on September 12?",
            (
                render("User: My September 12 report lists a TSH of 6.8.", None),
                render(
                    None,
                    "Assistant: A doctor may recommend repeating your TSH on September 12.",
                ),
                render("User: I read a general article about TSH target ranges.", None),
            ),
            {0},
        ),
        (
            "both_supporting_relations",
            "What is the birthplace of the author of Book Alpha?",
            (
                render("Corpus: Book Alpha — author — Evan Stone", None),
                render("Corpus: Evan Stone — place of birth — Hilltown", None),
                render("Corpus: Book Beta — author — Linda Wood", None),
                render("Corpus: Book Alpha — publisher — Hilltown Press", None),
            ),
            {0, 1},
        ),
    )
    config = load_config()
    embedder = build_embedder(config)
    results = []
    try:
        for name, question, documents, expected in cases:
            document_vectors = embedder.encode(documents)
            variants = []
            for label, task in (("off", ""), ("web", DEFAULT_TASK), ("memory", MEMORY_TASK)):
                vector = embedder.encode([apply_query_instruction(question, task)])[0]
                scores = [cosine(vector, doc) for doc in document_vectors]
                ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                passed = set(ranking[: len(expected)]) == expected
                variants.append(
                    {
                        "variant": label,
                        "instruction": task,
                        "scores": scores,
                        "ranking": ranking,
                        "passed": passed,
                    }
                )
                print(name, label, ranking, "PASS" if passed else "FAIL", flush=True)
            results.append(
                {
                    "case": name,
                    "question": question,
                    "literal_sources": documents,
                    "expected_top": sorted(expected),
                    "variants": variants,
                }
            )
    finally:
        embedder.close()
    Path(__file__).with_name("prototype-instruction.json").write_text(
        json.dumps({"model": config.models.embedder, "cases": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
