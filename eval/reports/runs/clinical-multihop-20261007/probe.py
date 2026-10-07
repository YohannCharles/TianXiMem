"""Frozen clinical QA diagnosis; stored Search text, unchanged upstream judge.

prepare audits released source visits against the stored Search text. Oracle
contexts contain only original dialogue turns from annotated source sessions;
gold chooses sources only in that explicitly diagnostic arm, never in Search.
run is serial and resumable, with prompt/model fingerprints checked on resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/harness"))

from eval.datasets.prepare import sha256_file  # noqa: E402
from eval.datasets.registry import benchmark_dir  # noqa: E402
from eval.harness import extra_pipeline as ep  # noqa: E402
from eval.jsonl_io import read_jsonl, write_line  # noqa: E402
from tools.run_products import rows_by_sample  # noqa: E402

SOURCE = "optimization-12h-20261007-a13-full-medmemorybench"
BASELINE_COMMIT = "6dbba175937a4ae0014808b38e8d25add6fbc0cc"
MAX_TOKENS = 2000
EXPLAIN_PROMPT = """You are an assistant answering from the patient's retrieved visit history.

Rules:
1. Use only information in the memories. Do not invent medical history,
   measurements, diagnoses, dates, medications, or causal mechanisms.
2. When asked why symptoms or events are related, explain the supported links
   across visits. Include the relevant patient-specific facts, dates, medication
   details and measurements that support those links. A generic conclusion alone
   is insufficient. Keep the explanation focused on the question.
3. Distinguish current facts from earlier facts and statements of uncertainty.
   If a necessary link is missing, state that limitation instead of inventing it.
4. If the question provides answer options, return only the letters of all
   selected options. For a direct name, value, date or status question, return
   only the requested information.
5. If the history provides no answer, reply exactly: Cannot determine from the memories.

Memories:
{memories}

Question: {question}

Answer:"""


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staged.replace(path)


def inputs() -> dict[str, dict]:
    result = {}
    for (_, qid), row in rows_by_sample(ROOT / "eval/reports/runs" / SOURCE, "input.jsonl").items():
        if qid in result:
            raise ValueError(f"duplicate input: {qid}")
        result[qid] = row
    return result


def prepare() -> dict:
    import pyarrow.parquet as pq

    items = inputs()
    clinical = sorted(
        qid for qid, row in items.items() if row["category"] == "multi_hop_clinical_deduction"
    )
    assert len(items) == 388 and len(clinical) == 39
    source = benchmark_dir() / "medmemorybench/data/zh/dialogues.parquet"
    visits = defaultdict(list)
    for batch in pq.ParquetFile(source).iter_batches(
        columns=["persona_id", "session_id", "turn", "role", "content", "event_info"]
    ):
        for row in batch.to_pylist():
            visits[int(row["persona_id"]), int(row["session_id"])].append(row)
    audits, oracle = [], []
    for qid in clinical:
        item = items[qid]
        persona = int(qid.split("-")[1])
        checkpoint = int(re.search(r"session_(\d+)", qid)[1])
        sessions = sorted(set(item["gold_answer"]["metadata"].get("sessions_involved", [])))
        assert all(0 < session <= checkpoint for session in sessions)
        normalized = "".join(item["retrieved_context"].split())
        parts, checks = [], []
        for session in sessions:
            turns = sorted(visits[persona, session], key=lambda row: int(row["turn"]))
            if not turns:
                raise ValueError(f"missing source visit: {qid}/{session}")
            eligible, full, prefix = 0, 0, 0
            for turn in turns:
                body = str(turn["content"] or "").strip()
                norm = "".join(body.split())
                if len(norm) >= 80:
                    eligible += 1
                    full += norm in normalized
                    prefix += norm[:120] in normalized
                info = turn["event_info"]
                if isinstance(info, str):
                    info = json.loads(info)
                day = str((info or {}).get("date") or "")[:10]
                parts.append(f"[{day}] {turn['role']}: {body}")
            checks.append(
                dict(
                    session=session,
                    source_date=day,
                    source_turns=len(turns),
                    eligible_turns=eligible,
                    full_turn_matches=full,
                    prefix_matches=prefix,
                    dated_header_matches=normalized.count(f"[{day}]") if day else None,
                )
            )
        audits.append(
            dict(
                id=qid,
                context_characters=len(item["retrieved_context"]),
                source_visits=checks,
                note="Exact source-text presence only; not a semantic node-coverage score.",
            )
        )
        oracle.append(dict(item, retrieved_context="\n\n".join(parts)))
    clinical_by_hash = sorted(clinical, key=digest)
    # Frozen before new model calls; at most one case per persona in the quick gate.
    selected, personas = [], set()
    for qid in clinical_by_hash:
        persona = qid.split("-")[1]
        if persona not in personas:
            selected.append(qid)
            personas.add(persona)
        if len(selected) == 6:
            break
    regression = []
    for category in sorted(
        {row["category"] for row in items.values()} - {"multi_hop_clinical_deduction"}
    ):
        group = sorted(
            (qid for qid, row in items.items() if row["category"] == category), key=digest
        )
        regression.extend(group[:6])
    manifest = dict(
        source_run=SOURCE,
        input_hash=digest(items),
        source_sha256=sha256_file(source),
        original_prompt=ep.ANSWER_PROMPT,
        explain_prompt=EXPLAIN_PROMPT,
        quick_qids=selected,
        clinical_qids=clinical,
        regression_qids=regression,
        oracle_scope=(
            "Gold selects source visits only; original dialogue content, "
            "no answers or reasoning-chain annotations."
        ),
        max_tokens=MAX_TOKENS,
        judge="unchanged upstream MedMemoryBench",
    )
    existing = OUT / "manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError("frozen manifest changed; use a new experiment directory")
    save(existing, manifest)
    save(OUT / "source-audit.json", audits)
    with (OUT / "oracle-input.jsonl").open("w") as stream:
        for row in oracle:
            write_line(stream, row)
    print(
        json.dumps(
            dict(
                prepared=True,
                clinical=len(clinical),
                quick=selected,
                regression=len(regression),
                visits_without_full_match=sum(
                    visit["full_turn_matches"] == 0
                    for row in audits
                    for visit in row["source_visits"]
                ),
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return manifest


def run(phase: str, arms: list[str], limit: int | None) -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    items = inputs()
    if digest(items) != manifest["input_hash"]:
        raise ValueError("source Search inputs changed")
    oracle = {row["id"]: row for row in read_jsonl(OUT / "oracle-input.jsonl")}
    selected = manifest[phase + "_qids"]
    if limit:
        selected = selected[:limit]
    answer, judge = ep._config("answer"), ep._config("judge")
    backend = {
        kind: {"base": config[0], "model": config[2]}
        for kind, config in [("answer", answer), ("judge", judge)]
    }
    path = OUT / "results.jsonl"
    completed = {(row["arm"], row["id"]): row for row in read_jsonl(path)}
    with path.open("a") as stream:
        for qid in selected:
            for arm in arms:
                answer_budget = 1024 if arm == "explain_1024" else MAX_TOKENS
                item = oracle[qid] if arm.startswith("oracle_") else items[qid]
                template = (
                    manifest["explain_prompt"]
                    if arm.endswith("explain") or arm == "explain_1024"
                    else manifest["original_prompt"]
                )
                prompt = template.format(
                    memories=item["retrieved_context"] or "(no memories)", question=item["question"]
                )
                if arm == "explain_1024":
                    actual = ep.render_answer_prompt(item)
                    if actual != prompt:
                        raise ValueError(
                            "implemented clinical prompt differs from frozen candidate"
                        )
                    prompt = actual
                fingerprint = digest(
                    dict(
                        prompt=prompt,
                        gold=item["gold_answer"],
                        backend=backend,
                        max_tokens=answer_budget,
                        judge=manifest["judge"],
                    )
                )
                if (arm, qid) in completed:
                    if completed[arm, qid]["fingerprint"] != fingerprint:
                        raise ValueError("completed answer/judge fingerprint changed")
                    continue
                began = time.monotonic()
                output = ep._chat(*answer, prompt, max_tokens=answer_budget, timeout=180)
                verdict = ep._judge_medmemorybench(qid, item, output, judge=judge, max_tokens=1024)
                row = dict(
                    verdict,
                    arm=arm,
                    generated_answer=output,
                    answer_max_tokens=answer_budget,
                    fingerprint=fingerprint,
                    elapsed_seconds=round(time.monotonic() - began, 2),
                    backend=backend,
                    at=datetime.now(UTC).isoformat(),
                )
                write_line(stream, row)
                completed[arm, qid] = row
                print(
                    json.dumps(
                        dict(
                            arm=arm,
                            id=qid,
                            correct=verdict["is_correct"],
                            label=verdict["label"],
                            answer_characters=len(output),
                            elapsed_seconds=row["elapsed_seconds"],
                        )
                    ),
                    flush=True,
                )
    report()


def report() -> None:
    rows = read_jsonl(OUT / "results.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"]].append(row)
    summary = {
        arm: dict(
            n=len(group),
            correct=sum(row["is_correct"] for row in group),
            labels=dict(Counter(row["label"] for row in group)),
        )
        for arm, group in groups.items()
    }
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def isolation() -> None:
    baseline = types.ModuleType("frozen_extra_pipeline")
    baseline.__file__ = ep.__file__
    source = subprocess.run(
        ["git", "show", BASELINE_COMMIT + ":eval/harness/extra_pipeline.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exec(source, baseline.__dict__)
    counts, changed = {}, []
    for run_dir in sorted(
        (ROOT / "eval/reports/runs").glob("optimization-12h-20261007-a13-full-*")
    ):
        if not run_dir.is_dir():
            continue
        items = rows_by_sample(run_dir, "input.jsonl")
        delta = 0
        for (_, qid), item in items.items():
            if baseline.render_answer_prompt(item) != ep.render_answer_prompt(item):
                assert item["dataset"] == "medmemorybench"
                assert item["category"] == "multi_hop_clinical_deduction"
                changed.append(qid)
                delta += 1
        counts[run_dir.name] = dict(items=len(items), changed_prompts=delta)
    assert len(counts) == 11 and len(changed) == 39
    save(
        OUT / "prompt-isolation.json",
        dict(
            baseline_commit=BASELINE_COMMIT,
            baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            counts=counts,
            changed_qids=changed,
        ),
    )
    print(
        json.dumps(
            dict(
                changed=len(changed),
                unchanged=sum(group["items"] for group in counts.values()) - len(changed),
            )
        ),
        flush=True,
    )


def reparse() -> None:
    """Apply the current parser to saved malformed responses; retain originals."""
    items = inputs()
    rows = read_jsonl(OUT / "results.jsonl")
    recovered = []
    with (OUT / "reparsed-results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            if row["label"] == "JUDGE_ERROR":
                _, separator, raw = row["judge_response"].partition("｜")
                if separator:
                    ok, why = ep.judge_mmb_llm_verdict(raw)
                    if ok is not None:
                        category = items[row["id"]]["category"]
                        label = category + (":CORRECT" if ok else ":WRONG")
                        recovered.append(
                            dict(
                                id=row["id"],
                                arm=row["arm"],
                                original_label=row["label"],
                                reparsed_label=label,
                                reparsed_is_correct=ok,
                                correctness_changed=ok != row["is_correct"],
                            )
                        )
                        row = dict(
                            row,
                            original_label=row["label"],
                            original_judge_response=row["judge_response"],
                            label=label,
                            is_correct=ok,
                            judge_response=why,
                        )
            write_line(stream, row)
    groups = defaultdict(list)
    for row in read_jsonl(OUT / "reparsed-results.jsonl"):
        groups[row["arm"]].append(row)
    summary = {
        arm: dict(
            n=len(group),
            correct=sum(row["is_correct"] for row in group),
            labels=dict(Counter(row["label"] for row in group)),
        )
        for arm, group in groups.items()
    }
    audit = dict(
        original_results_sha256=sha256_file(OUT / "results.jsonl"),
        parser_source_sha256=sha256_file(Path(ep.__file__)),
        recovered=recovered,
        summary=summary,
    )
    save(OUT / "quote-recovery.json", audit)
    print(json.dumps(audit["summary"], ensure_ascii=False), flush=True)
    print(json.dumps(recovered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report", "isolation", "reparse"])
    parser.add_argument("--phase", choices=["quick", "clinical", "regression"], default="quick")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["original", "explain", "explain_1024", "oracle_original", "oracle_explain"],
        default=["original", "explain"],
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run(args.phase, args.arms, args.limit)
    elif args.command == "report":
        report()
    elif args.command == "isolation":
        isolation()
    else:
        reparse()
