"""离线复核各 AML 冻结口径：不调用服务、模型或下载器。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from eval.datasets.aml import load_plans
from eval.datasets.aml.feverous import visible_json_evidence
from eval.datasets.aml.plan import AddEvent, SearchEvent
from eval.datasets.aml.registry import SUPPORTED
from eval.datasets.registry import benchmark_dir
from eval.experiments.recipes import recipe_for
from eval.harness.plan_driver import compile_plan, plans_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=benchmark_dir({}))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("plan-checks.json"))
    parser.add_argument("--aml-pool", type=Path)
    args = parser.parse_args()
    rows = []
    for dataset in sorted(SUPPORTED):
        started = time.monotonic()
        recipe = recipe_for(dataset, input_contract="aml-v1")
        plans = load_plans(
            args.dir,
            dataset,
            limit=recipe.limit,
            spread=recipe.spread,
            pool=args.aml_pool if dataset == "feverous" else None,
        )
        coverage = None
        if dataset == "feverous":
            # 评分侧诊断；不回过头来改变页面池、选题或 HTTP 事件。
            available = set()
            for plan in plans:
                for event in plan.events:
                    if isinstance(event, AddEvent):
                        for message in event.messages:
                            available.update(visible_json_evidence(message["content"]))
            verifiable = [
                q
                for plan in plans
                for q in plan.sample.questions
                if q.gold["label"] != "NOT ENOUGH INFO"
            ]
            covered = [
                q
                for q in verifiable
                if any(
                    group and all("_".join(element) in available for element in group)
                    for group in q.gold["evidence"]
                )
            ]
            coverage = {
                "supports_refutes": len(verifiable),
                "complete_gold_evidence_in_declared_add_pool": len(covered),
            }
        plans = [compile_plan(p, namespace="offline-plan-check-20261006") for p in plans]
        fingerprint = plans_fingerprint(plans)
        assert fingerprint["n_questions"] == recipe.n_questions, (dataset, fingerprint)
        assert len({p.sample.user_id for p in plans}) == len(plans)
        requests = []
        for plan in plans:
            plan.validate()
            assert not plan.sample.sessions  # 原始语料不进入评分文件。
            for event in plan.events:
                if isinstance(event, SearchEvent):
                    assert event.query and event.qid
                    continue
                assert isinstance(event, AddEvent) and event.batch_ready and event.request_id
                assert 0 < len(event.messages) <= 20
                requests.append(event.request_id)
                for message in event.messages:
                    assert set(message) <= {"role", "content", "timestamp"}
                    assert len(message["content"]) <= 8000
        assert len(set(requests)) == len(requests)
        row = {
            "dataset": dataset,
            "elapsed_s": round(time.monotonic() - started, 3),
            "recipe": {"limit": recipe.limit, "spread": recipe.spread},
            "fingerprint": fingerprint,
        }
        if coverage is not None:
            row["corpus_coverage_diagnostic"] = coverage
        rows.append(row)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "elapsed_s": row["elapsed_s"],
                    **{
                        k: fingerprint[k]
                        for k in ("n_samples", "n_questions", "n_adds", "n_messages")
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
