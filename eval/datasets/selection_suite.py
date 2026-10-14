"""Portable ID-only selection executed through existing native adapters only.

The pool is Memory600 plus a stratified tenth of scorable captured Search records.
Source identity includes answerability/knowledge world. Replacing a captured or
curated scope with the native scope is explicit; it is not an AML reproduction.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from tools.build_official_kit import PROMPT_REFS

from eval.datasets.aml.plan import digest
from eval.datasets.frozen_bundle import file_sha256 as sha256
from eval.datasets.frozen_bundle import read_jsonl, validate
from eval.datasets.preprocess import Question, Sample

VERSION = "selection-suite-v1"
PRIORITY = {"native": 0}
LOADERS = {
    "beam": ("beam", "load_beam"),
    "clbench": ("clbench", "load_clbench"),
    "corporatebench": ("corporatebench", "load_corporatebench"),
    "halumem": ("halumem", "load_halumem"),
    "hybridqa": ("hybridqa", "load_hybridqa"),
    "locomo-refined": ("locomo", "load_locomo"),
    "longmemeval-s": ("longmemeval", "load_longmemeval"),
    "medmemorybench": ("medmemorybench", "load_medmemorybench"),
    "memtrapbench": ("memtrapbench", "load_memtrapbench"),
    "mquake-remastered": ("mquake", "load_mquake"),
    "musique": ("musique", "load_musique"),
    "personamem-v2": ("personamem", "load_personamem"),
}
#: 只给这些数据集传 `question_ids`（按题筛语料而非整份加载）。
#: ⚠ **feverous 已于 2026-10-14 移出本套件**：它的 25 题要带 56G 语料（占全套 98%）。
#: loader（`eval/datasets/feverous.py`）与 `recipes.py` 的配方**都保留**——
#: `run.py --dataset feverous` 那条独立评测路径不受影响，这里只是不再选它。
TARGETED = {"longmemeval-s", "hybridqa", "medmemorybench"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def classified_rows(capture: Path) -> list[dict]:
    searches = read_jsonl(capture / "official-searches.jsonl")
    kit = read_jsonl(capture / "official-eval-kit.jsonl")
    by_seq = {int(r["seq"]): r for r in kit}
    if len(by_seq) != len(kit) or len({int(r["seq"]) for r in searches}) != len(searches):
        raise ValueError("duplicate captured seq")
    if set(by_seq) != {int(r["seq"]) for r in searches}:
        raise ValueError("captured Search/kit IDs differ")
    output = []
    reliable = {"tag", "official_feedback", "public_gold", "gold_extra"}
    for search in searches:
        row = by_seq[int(search["seq"])]
        if any(row.get(k) != search.get(k) for k in ("user_id", "query", "ts")):
            raise ValueError("captured Search/kit binding differs")
        has_gold = any(v is not None and str(v).strip() for v in row.get("gold_answers") or [])
        scorable = (
            row.get("dataset") in PROMPT_REFS
            and row.get("dataset") != "unknown"
            and row.get("attribution") in reliable
            and row.get("judge_kind") not in (None, "none")
            and (has_gold or bool(row.get("gold_rubric")))
        )
        output.append({**row, **search, "scorable": scorable})
    return output


def native_id(dataset: str, source_id: str) -> str:
    if dataset == "musique":
        return f"musique-{source_id}-answerable"
    if dataset == "hybridqa":
        return f"hybridqa-{source_id}"
    return source_id


def load_native(root: Path, wanted: dict[str, set[str]]) -> list[Sample]:
    """Reuse existing loaders; select questions without trimming their memory scopes."""
    output = []
    for dataset, ids in sorted(wanted.items()):
        if not ids:
            continue
        module, function = LOADERS[dataset]
        loader = getattr(importlib.import_module(f"eval.datasets.{module}"), function)
        options = {"question_ids": ids} if dataset in TARGETED else {}
        for sample in loader(root, **options):
            questions = tuple(q for q in sample.questions if q.qid in ids)
            if questions:
                output.append(replace(sample, questions=questions))
    return output


def stratified_tenth(rows: list[dict], seed: str) -> list[dict]:
    """Largest remainders, total ceil(N/10); hash ranking independent of file order."""
    if len({r["seq"] for r in rows}) != len(rows):
        raise ValueError("duplicate captured seq")
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    quotas = {d: len(rs) // 10 for d, rs in groups.items()}
    remainder = math.ceil(len(rows) / 10) - sum(quotas.values())
    for d in sorted(groups, key=lambda d: (-(len(groups[d]) % 10), d))[:remainder]:
        quotas[d] += 1
    selected = []
    for d, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda r: (digest([seed, d, int(r["seq"])]), int(r["seq"])))
        selected.extend(ranked[: quotas[d]])
    return sorted(selected, key=lambda r: int(r["seq"]))


def capture_links(capture: Path) -> tuple[dict, dict]:
    extras = {}
    for path in sorted(capture.glob("official-gold-extra*.jsonl")):
        for row in read_jsonl(path):
            extras.setdefault(int(row["seq"]), row)
    public = {
        int(r["seq"]): r.get("public_gold") or {}
        for r in read_jsonl(capture / "official-eval-questions.jsonl")
    }
    return extras, public


def official_native_id(row: dict, extra: dict, public: dict, root: Path) -> str | None:
    """Resolve addresses, not arbitrary text matches across users/variants."""
    d = row["dataset"]
    loc = extra.get("source_locator") or {}
    gold = row.get("gold_native") or {}
    if d == "locomo-refined":
        return public.get("gold_locomo_qa_id")
    if d == "musique" and (loc.get("id") or gold.get("id")):
        variant = "answerable" if loc.get("answerable", gold.get("answerable")) else "unanswerable"
        return f"musique-{loc.get('id') or gold['id']}-{variant}"
    if d == "hybridqa" and (loc.get("question_id") or gold.get("question_id")):
        return f"hybridqa-{loc.get('question_id') or gold['question_id']}"
    if d == "clbench":
        return gold.get("task_id")
    if d == "halumem" and loc.get("uuid"):
        return (
            f"halumem-{loc['uuid']}-s{int(loc['session_index']) + 1:03d}"
            f"-q{int(loc['question_index']):03d}"
        )
    if d == "mquake-remastered" and loc.get("file"):
        return f"{loc['file'].split('-')[0]}-{loc['case_id']}-{loc['variant']}"
    if d == "corporatebench" and loc.get("file"):
        return f"{loc['file'].split('/')[0]}-{loc['id']}"
    if d == "memtrapbench" and loc.get("file"):
        return f"{Path(loc['file']).stem}-{loc['id']}"
    if d == "personamem-v2" and loc.get("persona_id") is not None:
        from .personamem import BENCHMARK_CSV, _query_text

        matches = []
        index = 0
        with (root / d / BENCHMARK_CSV).open(encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                if str(source["persona_id"]) != str(loc["persona_id"]):
                    continue
                if _query_text(source["user_query"]).strip() == row["question"].strip():
                    matches.append(f"pm-{source['persona_id']}-{index}")
                index += 1
        return matches[0] if len(matches) == 1 else None
    if d == "beam" and loc.get("conversation_id") is not None:
        from .beam import PARQUET, _questions, _rows

        for source in _rows(root / PARQUET):
            if str(source["conversation_id"]) != str(loc["conversation_id"]):
                continue
            matches = [
                q.qid
                for q in _questions(source, str(source["conversation_id"]))
                if q.category == loc.get("group") and q.question.strip() == row["question"].strip()
            ]
            return matches[0] if len(matches) == 1 else None
    if d == "medmemorybench" and loc.get("query_id") is not None:
        from .medmemorybench import _rows

        matches = [
            f"mmb-{int(s['persona_id'])}-{s['query_id']}"
            for s in _rows(root / d / "data/zh/queries.parquet")
            if s["query_id"] == loc["query_id"] and s["question"].strip() == row["question"].strip()
        ]
        return matches[0] if len(matches) == 1 else None
    return None


def sample_scope(sample: Sample) -> str:
    return digest(
        {"sessions": [asdict(s) for s in sample.sessions], "speaker_names": sample.speaker_names}
    )


def question_identity(question: Question) -> str:
    return digest(asdict(question))


def deduplicate(candidates: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in candidates:
        grouped[tuple(row["source_key"])].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        ranked = sorted(
            rows, key=lambda r: (PRIORITY[r["route"]], r["origin"], str(r["origin_id"]))
        )
        winner = dict(ranked[0])
        winner["members"] = [
            {
                k: r[k]
                for k in (
                    "origin",
                    "origin_id",
                    "route",
                    "audit",
                    "original_scope",
                    "original_evaluation",
                )
            }
            for r in ranked
        ]
        winner["id"] = digest(key)[:24]
        output.append(winner)
    return output


def prepare(root: Path, capture: Path, bundle: Path, *, seed: str = "tianximem-20261014") -> dict:
    validate(bundle)
    all_rows = classified_rows(capture)
    eligible = [r for r in all_rows if r["scorable"]]
    official = stratified_tenth(eligible, seed)
    extras, public = capture_links(capture)
    findings = json.loads((bundle / "audit-status.json").read_text())[
        "questions_with_open_findings"
    ]
    by_id = defaultdict(list)
    for finding in findings:
        by_id[finding["id"]].append({k: finding[k] for k in ("aspect", "status")})
    candidates, excluded, wanted = [], [], defaultdict(set)
    failed_sources = {}
    bundle_manifest = json.loads((bundle / "manifest.json").read_text())
    for q in read_jsonl(bundle / "questions.jsonl"):
        d, qid = q["source_dataset"], native_id(q["source_dataset"], q["source_question_id"])
        world = "updated" if d == "mquake-remastered" else "default"
        audit = by_id[q["id"]] or (
            [{"status": "independent_audit_pending"}]
            if not q["id"].startswith("memory500-")
            else []
        )
        if any(f["status"] == "failed" for f in audit):
            failed_sources[(d, qid, world)] = audit
            excluded.append({"origin": "memory600", "origin_id": q["id"], "audit": audit})
            continue
        candidates.append(
            {
                "origin": "memory600",
                "origin_id": q["id"],
                "dataset": d,
                "native_qid": qid,
                "source_key": [d, qid, world],
                "audit": audit,
                "original_scope": bundle_manifest["space_scope_fingerprints"][q["space_id"]][
                    "sha256"
                ],
                "original_evaluation": digest(q),
                "route": "memory600",
            }
        )
        if d in LOADERS:
            wanted[d].add(qid)
    for row in official:
        d = row["dataset"]
        qid = official_native_id(row, extras.get(row["seq"], {}), public.get(row["seq"], {}), root)
        world = row.get("gold_variant") or "default"
        if d == "mquake-remastered" and world.startswith("updated"):
            world = "updated"
        key = [d, qid, world] if qid else ["capture", int(row["seq"])]
        if tuple(key) in failed_sources:
            excluded.append(
                {
                    "origin": "official",
                    "origin_id": int(row["seq"]),
                    "audit": failed_sources[tuple(key)],
                }
            )
            continue
        candidates.append(
            {
                "origin": "official",
                "origin_id": int(row["seq"]),
                "dataset": d,
                "native_qid": qid,
                "source_key": key,
                "audit": [],
                "route": "official",
                "original_scope": digest([row["user_id"], row["prior_adds_for_user"], row["ts"]]),
                "original_evaluation": digest(row),
            }
        )
        if qid and d in LOADERS and (d != "mquake-remastered" or world == "updated"):
            wanted[d].add(qid)
    native = {(s.dataset, q.qid): (s, q) for s in load_native(root, wanted) for q in s.questions}
    scopes = {}
    supported = []
    for row in candidates:
        found = native.get((row["dataset"], row["native_qid"]))
        if found and (row["dataset"] != "mquake-remastered" or row["source_key"][-1] == "updated"):
            s, q = found
            scope_key = (s.dataset, s.user_id)
            if scope_key not in scopes:
                scopes[scope_key] = sample_scope(s)
            row.update(
                route="native",
                scope_sha256=scopes[scope_key],
                evaluation_sha256=question_identity(q),
            )
            supported.append(row)
        else:
            reason = (
                "native_adapter_missing"
                if row["dataset"] not in LOADERS
                else "native_original_world_unsupported"
                if row["dataset"] == "mquake-remastered" and row["source_key"][-1] != "updated"
                else "native_source_id_unresolved_or_missing"
            )
            excluded.append(
                {
                    "origin": row["origin"],
                    "origin_id": row["origin_id"],
                    "dataset": row["dataset"],
                    "source_key": row["source_key"],
                    "reason": reason,
                    "audit": row["audit"],
                }
            )
    questions = deduplicate(supported)
    files = {
        name: sha256(capture / name)
        for name in ("official-searches.jsonl", "official-eval-kit.jsonl")
    }
    files["official-eval-questions.jsonl"] = sha256(capture / "official-eval-questions.jsonl")
    files.update({p.name: sha256(p) for p in sorted(capture.glob("official-gold-extra*.jsonl"))})
    return {
        "version": VERSION,
        "policy": {
            "execution": "existing native loaders and pipelines only; unsupported tasks skipped",
            "seed": seed,
            "sample": "scorable Search records; dataset stratified; largest remainders; ceil(N/10)",
            "dedup": (
                "source dataset + source ID + answerability/knowledge world; "
                "unknown mapping retains seq"
            ),
            "replacement": (
                "native scopes/queries/judges may replace captured or curated scopes; "
                "not AML reproduction"
            ),
            "failed_audit": "exclude; pending audits retained and reported",
        },
        "sources": {"capture": files, "bundle_manifest_sha256": sha256(bundle / "manifest.json")},
        "official_selection": [{"seq": int(r["seq"]), "dataset": r["dataset"]} for r in official],
        "excluded": excluded,
        "questions": questions,
        "counts": {
            "captured_searches": len(all_rows),
            "scorable_capture": len(eligible),
            "official_sample": len(official),
            "memory600": 600,
            "candidates_after_exclusions": len(supported),
            "dropped_unsupported": len(candidates) - len(supported),
            "unique_tasks": len(questions),
            "routes": dict(Counter(r["route"] for r in questions)),
            "official_strata": dict(Counter(r["dataset"] for r in official)),
        },
    }


def load_selection(path: Path) -> dict:
    if path.suffix == ".tsv":
        return load_ids(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported selection version")
    rows = manifest["questions"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty selection or duplicate task ID")
    if len({digest(r["source_key"]) for r in rows}) != len(rows):
        raise ValueError("duplicate source identity")
    if any(r["route"] not in PRIORITY for r in rows):
        raise ValueError("unknown execution route")
    return manifest


def row_audits(row: dict) -> list[dict]:
    facts = row.get("audit", []) + [
        finding for member in row.get("members", []) for finding in member.get("audit", [])
    ]
    return [
        dict(aspect=aspect, status=status)
        for aspect, status in sorted({(f.get("aspect", ""), f["status"]) for f in facts})
    ]


def write_ids(path: Path, manifest: dict) -> None:
    """Two columns plus sparse audit comments; no per-question hashes/provenance."""
    lines = ["# dataset\tnative_qid"]
    rows = sorted(manifest["questions"], key=lambda r: (r["dataset"], r["native_qid"]))
    for row in rows:
        for finding in row_audits(row):
            lines.append(
                "\t".join(
                    [
                        "# audit",
                        row["dataset"],
                        row["native_qid"],
                        finding["aspect"],
                        finding["status"],
                    ]
                )
            )
    lines.extend(f"{r['dataset']}\t{r['native_qid']}" for r in rows)
    text = "\n".join(lines) + "\n"
    # Validate before writing, including malformed IDs and audit/source binding.
    parse_ids(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_ids(text: str) -> dict:
    rows, seen, audits = [], set(), defaultdict(list)
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("# audit\t"):
            fields = line.split("\t")
            if len(fields) != 5 or not fields[-1] or fields[-1] == "failed":
                raise ValueError(f"invalid audit comment at line {number}")
            audits[(fields[1], fields[2])].append({"aspect": fields[3], "status": fields[4]})
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not all(f and f == f.strip() for f in fields):
            raise ValueError(f"expected dataset and native_qid at line {number}")
        dataset, qid = fields
        if dataset not in LOADERS:
            raise ValueError(f"unknown native dataset at line {number}: {dataset}")
        if (dataset, qid) in seen:
            raise ValueError(f"duplicate native question at line {number}")
        seen.add((dataset, qid))
        key = [dataset, qid, "updated" if dataset == "mquake-remastered" else "default"]
        rows.append(
            {
                "id": digest(key)[:24],
                "dataset": dataset,
                "native_qid": qid,
                "source_key": key,
                "route": "native",
                "members": [],
            }
        )
    if not rows:
        raise ValueError("empty selection")
    if set(audits) - seen:
        raise ValueError("audit comment references an unselected question")
    rows.sort(key=lambda r: (r["dataset"], r["native_qid"]))
    for row in rows:
        row["audit"] = row_audits({"audit": audits[(row["dataset"], row["native_qid"])]})
    return {
        "version": VERSION,
        "policy": {"input": "compact native question IDs"},
        "questions": rows,
        "excluded": [],
        "counts": {"unique_tasks": len(rows), "routes": {"native": len(rows)}},
    }


def load_ids(path: Path) -> dict:
    return parse_ids(path.read_text(encoding="utf-8"))


def verify_sources(manifest: dict, capture: Path, bundle: Path) -> None:
    for name, expected in manifest["sources"]["capture"].items():
        if Path(name).name != name or sha256(capture / name) != expected:
            raise ValueError(f"captured source changed: {name}")
    if sha256(bundle / "manifest.json") != manifest["sources"]["bundle_manifest_sha256"]:
        raise ValueError("Memory600 bundle manifest changed")
    validate(bundle)
