"""Manual relevance checks before enabling rerank in any candidate profile."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import monotonic

from tianximem.common.config import load_config
from tianximem.service.app import build_reranker

OUT = Path(__file__).resolve().parent


def main():
    config = load_config()
    config = replace(
        config,
        rerank=replace(config.rerank, enabled=True, envelope="queries"),
        reranker_model="qwen3-reranker-4b",
    )
    reranker = build_reranker(config)
    if reranker is None:
        raise RuntimeError("Reranker is not configured")
    cases = [
        {
            "query": "What time does the night train leave?",
            "documents": [
                "Q: My favourite colour is teal. A: Noted.",
                "Q: The night train leaves at 23:40 from platform 3.",
            ],
            "expected_first": 1,
        },
        {
            "query": "2024-01-08的检查中患者的促甲状腺激素（TSH）检查结果是多少？",
            "documents": [
                "Q: 医生您好，我希望用药尽量稳定，不需要太频繁复诊。\n"
                "A: 甲状腺这边之前的检查里 TSH 是明显压低的。",
                "Q: 我2024-01-08的检查结果是 TSH 0.01 mIU/L。",
            ],
            "expected_first": 1,
        },
        {
            "query": "Which westerner first sighted Angaur island?",
            "documents": [
                "Q: Title: Malakal Island\nMalakal is an island in Koror, Palau.",
                "Q: Title: Angaur\nThe first sighting of Angaur by westerners was by "
                "the Spanish expedition of Ruy López de Villalobos.",
            ],
            "expected_first": 1,
        },
    ]
    results = []
    for case in cases:
        start = monotonic()
        scores = reranker.score(query=case["query"], documents=case["documents"])
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        passed = order[0] == case["expected_first"] and sorted(order) == list(range(len(scores)))
        result = case | {
            "scores": scores,
            "order": order,
            "passed": passed,
            "seconds": monotonic() - start,
        }
        results.append(result)
        print("PASS" if passed else "FAIL", case["query"], scores, flush=True)
    (OUT / "prototype-rerank.json").write_text(
        json.dumps({"model": reranker.name, "cases": results}, ensure_ascii=False, indent=2)
    )
    return int(not all(case["passed"] for case in results))


if __name__ == "__main__":
    raise SystemExit(main())
