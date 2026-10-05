"""两个失败问法的原文诊断：仅减噪或固定顺序，不改答案链。"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import pipeline_for, run_judge  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert freeze(manifest) == json.loads((OUT / "frozen.json").read_text())
    items = {
        row["id"]: row for row in map(json.loads, (OUT / "input.jsonl").read_text().splitlines())
    }
    folder = OUT / "http/generalization-20261005-backpack"
    reference = json.loads(
        (folder / "generalization-backpack-count-canonical/search.json").read_text()
    )["hits"]
    source = json.loads((OUT / "sources.json").read_text())["backpack"][0]
    clean_hit = next(hit for hit in reference if source in hit["content"])
    contexts = {
        "single-original-source": clean_hit["content"],
        "canonical-raw-order": "\n".join(hit["content"] for hit in reference),
    }
    protocol = {
        "scope": "post-failure diagnostic only; not an improved service score",
        "original_question_gold_model_prompt_scorer_unchanged": True,
        "contexts": {
            key: {"content": value, "sha256": hashlib.sha256(value.encode()).hexdigest()}
            for key, value in contexts.items()
        },
    }
    protocol_path = OUT / "probe-protocol.json"
    encoded = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    if protocol_path.exists():
        assert protocol_path.read_text() == encoded
    protocol_path.write_text(encoded)
    rows = []
    for condition, context in contexts.items():
        for variant in ("list-paraphrase", "count-paraphrase"):
            qid = f"generalization-backpack-{variant}"
            item = {**items[qid], "retrieved_context": context}
            result = run_judge(
                pipeline_for(benchmark_dir(), "corporatebench"),
                [item],
                OUT / "probes" / condition / qid,
                dataset="corporatebench",
            )[0]
            rows.append({"condition": condition, **asdict(result)})
            (OUT / "probe-results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
            )
            print(condition, qid, result.is_correct, result.generated_answer, flush=True)


if __name__ == "__main__":
    main()
