"""Read-only comparison of public QA inputs with the captured AML Full requests.

Run from the repository root with `.venv/bin/python <this file>`. No preparation,
network/model calls, service imports, or changes to the active benchmark run.
Raw query text and gold answers are not copied into the JSON output.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CAPTURE = ROOT / "official-dataset-2026-09-29"
DATA = ROOT / "dataset"
FAMILIES = (
    "corporatebench",
    "mquake-remastered",
    "tempreason",
    "memtrapbench",
    "locomo-refined",
    "longmemeval-s",
    "medmemorybench",
    "halumem",
    "musique",
    "hybridqa",
    "feverous",
)
TEMPORAL_INTERVAL = re.compile(
    r" from (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec), \d+ to "
)
PARAGRAPH = re.compile(r"^Corpus: \[Paragraph (\d+)\]\nTitle: ([^\n]*)\n\n(.*)$", re.S)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parquet(path: Path, columns: list[str]):
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        yield from batch.to_pylist()


def norm(text: str) -> str:
    return " ".join(text.split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def native_questions():
    known: dict[str, set[str]] = {name: set() for name in FAMILIES}
    for sub in ("kb_qa", "topic_qa", "integrated_qa"):
        path = DATA / "corporatebench/data" / sub / "zenith_questions.json"
        known["corporatebench"].update(
            row["question"].strip() for row in json.loads(path.read_text())["questions"]
        )
    for path in sorted((DATA / "memtrapbench/memtrapbench").glob("*/*.json")):
        known["memtrapbench"].update(
            row["final_trigger"].strip()
            for row in json.loads(path.read_text())
            if row.get("context_history")
        )
    known["medmemorybench"].update(
        row["question"].strip()
        for row in parquet(DATA / "medmemorybench/data/zh/queries.parquet", ["question"])
    )
    for path in sorted((DATA / "mquake-remastered/data").glob("*.parquet")):
        for row in parquet(path, ["questions"]):
            known["mquake-remastered"].update(text.strip() for text in row["questions"])
    known["locomo-refined"].update(
        row["question"].strip() for row in jsonl(DATA / "locomo-refined/questions.jsonl")
    )
    lme_bodies = set()
    for row in json.loads((DATA / "longmemeval-s/lme_s_cleaned.json").read_text()):
        known["longmemeval-s"].add(row["question"].strip())
        for session in row["haystack_sessions"]:
            lme_bodies.update(
                turn["content"].strip() for turn in session if turn["content"].strip()
            )
    for row in jsonl(DATA / "halumem/HaluMem-Medium.jsonl"):
        for session in row["sessions"]:
            known["halumem"].update(q["question"].strip() for q in session.get("questions", []))
    for row in jsonl(DATA / "musique/musique_full_v1.0_dev.jsonl"):
        known["musique"].add(row["question"].strip())
    known["hybridqa"].update(
        row["question"].strip() for row in json.loads((DATA / "hybridqa/dev.json").read_text())
    )
    known["feverous"].update(
        row["claim"].strip()
        for row in jsonl(DATA / "feverous/feverous_dev_challenges.jsonl")
        if row["claim"]
    )
    for name in ("test_l2.json", "test_l3.json"):
        known["tempreason"].update(
            row["question"].strip() for row in jsonl(DATA / "tempreason" / name)
        )
    return known, lme_bodies


def concatenated_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    position = 0
    objects = []
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        obj, position = decoder.raw_decode(text, position)
        objects.append(obj)
    return objects


def main() -> None:
    kit = list(jsonl(CAPTURE / "official-eval-kit.jsonl"))
    attribution = {row["user_id"]: row for row in jsonl(CAPTURE / "official-attribution.jsonl")}
    raw_searches = {row["seq"]: row for row in jsonl(CAPTURE / "official-searches.jsonl")}
    known, lme_bodies = native_questions()
    by_user = defaultdict(list)
    for question in kit:
        by_user[question["user_id"]].append(question)
    stats = {
        name: {
            "adds": 0,
            "messages": 0,
            "roles": Counter(),
            "timestamp": Counter(),
            "add_message_counts": Counter(),
        }
        for name in FAMILIES
    }
    add_times = defaultdict(list)
    sessions = defaultdict(set)
    first_contents = {}
    retained = defaultdict(list)
    generic_locomo = 0
    generic_locomo_lme_hits = 0
    for add in jsonl(CAPTURE / "official-adds.jsonl"):
        uid = add["user_id"]
        add_times[uid].append(add["ts"])
        sessions[uid].add(add["session_id"])
        first_contents.setdefault(uid, add["messages"][0]["content"])
        family = attribution.get(uid, {}).get("dataset")
        if family not in stats:
            continue
        s = stats[family]
        s["adds"] += 1
        s["messages"] += len(add["messages"])
        s["add_message_counts"][len(add["messages"])] += 1
        for message in add["messages"]:
            s["roles"][message["role"]] += 1
            s["timestamp"]["present" if "timestamp" in message else "missing"] += 1
            if family == "locomo-refined" and message["content"].startswith(
                ("User: ", "Assistant: ")
            ):
                generic_locomo += 1
                generic_locomo_lme_hits += (
                    message["content"].split(": ", 1)[1].strip() in lme_bodies
                )
        if family in {"corporatebench", "musique", "hybridqa", "feverous"}:
            retained[uid].append(add)
    for family in FAMILIES:
        cohort = [q for q in kit if q["dataset"] == family]
        users = {q["user_id"] for q in cohort}
        s = stats[family]
        s.update(
            labelled_queries=len(cohort),
            query_users=len(users),
            native_unique_questions=len(known[family]),
            exact_stripped_query_matches=sum(q["query"].strip() in known[family] for q in cohort),
            exact_stripped_matches_across_all_queries=sum(
                q["query"].strip() in known[family] for q in kit
            ),
            top_k_counts=Counter(raw_searches[q["seq"]]["top_k"] for q in cohort),
            interleaved_users=sum(
                any(ts > min(q["ts"] for q in by_user[uid]) for ts in add_times[uid])
                for uid in users
            ),
            unique_sessions=sum(len(sessions[uid]) for uid in users),
        )
    temporal_users = {
        uid
        for uid, text in first_contents.items()
        if attribution[uid]["dataset"] == "tempreason"
        or (text.startswith("source: ") and TEMPORAL_INTERVAL.search(text))
    }
    temporal_queries = [q for q in kit if q["user_id"] in temporal_users]
    extra = {
        "tempreason_unlabelled_interval_cohort": {
            "basis": (
                "tag OR source-prefixed first message with explicit month/year intervals; not gold"
            ),
            "users": len(temporal_users),
            "queries": len(temporal_queries),
            "native_exact_question_matches": sum(
                q["query"].strip() in known["tempreason"] for q in temporal_queries
            ),
        },
        "locomo_user_attribution_generic_messages": {
            "messages": generic_locomo,
            "exact_lme_message_body_matches": generic_locomo_lme_hits,
            "basis": "User/Assistant-prefixed bodies, stripped, against all public LME-S turns",
        },
        "mquake_gold_variants": Counter(
            row.get("gold_variant")
            for row in jsonl(CAPTURE / "official-gold-extra.jsonl")
            if row["source_dataset"] == "mquake-remastered"
        ),
    }
    musique = {
        (row["id"], row["answerable"]): row
        for row in jsonl(DATA / "musique/musique_full_v1.0_dev.jsonl")
    }
    mu_matches = Counter()
    for question in (q for q in kit if q["dataset"] == "musique"):
        metadata = question["gold_native"]
        public = musique[(metadata["id"], metadata["answerable"])]
        original = [
            (int(p["idx"]), p["title"], norm(p["paragraph_text"])) for p in public["paragraphs"]
        ]
        actual = []
        for add in sorted(retained[question["user_id"]], key=lambda row: row["ts"]):
            for message in add["messages"]:
                match = PARAGRAPH.fullmatch(message["content"])
                if match is None:
                    raise ValueError("Unparsed captured MuSiQue message")
                actual.append((int(match[1]), match[2], norm(match[3])))
        mu_matches["ordered_idx_title_body_matches"] += original == actual
        mu_matches["suffix_stripped_query_matches"] += (
            question["query"].split("\n\nAnswer using only")[0].strip() in known["musique"]
        )
    extra["musique"] = mu_matches
    hybrid_matches = Counter()
    for uid, table_id in {
        q["user_id"]: q["gold_native"]["table_id"] for q in kit if q["dataset"] == "hybridqa"
    }.items():
        text = "".join(
            m["content"].removeprefix("Corpus: ")
            for add in sorted(retained[uid], key=lambda row: row["ts"])
            for m in add["messages"]
        )
        head = "Complete table document in JSON. Each header/data cell is [text, link IDs].\n"
        if not text.startswith(head):
            raise ValueError("Unexpected HybridQA preamble")
        table, end = json.JSONDecoder().raw_decode(text[len(head) :])
        public = json.loads((DATA / "hybridqa/tables_tok" / f"{table_id}.json").read_text())
        hybrid_matches["structured_table_matches"] += table == public
        passages = json.loads((DATA / "hybridqa/request_tok" / f"{table_id}.json").read_text())
        markers = list(re.compile(r"Passage ID: (/wiki/[^\n]+)\n").finditer(text, end + len(head)))
        actual = {
            m[1]: text[
                m.end() : markers[i + 1].start() if i + 1 < len(markers) else len(text)
            ].strip()
            for i, m in enumerate(markers)
        }
        hybrid_matches["all_linked_passage_matches"] += set(actual) == set(passages) and all(
            norm(actual[k]) == norm(v) for k, v in passages.items()
        )
    extra["hybridqa"] = hybrid_matches
    fever_uid = next(q["user_id"] for q in kit if q["dataset"] == "feverous")
    fever_text = "".join(
        m["content"].removeprefix("Corpus: ")
        for add in sorted(retained[fever_uid], key=lambda row: row["ts"])
        for m in add["messages"]
    )
    pages = concatenated_objects(fever_text)
    extra["feverous"] = {
        "complete_json_pages": len(pages),
        "unique_page_titles": len({p["title"] for p in pages}),
        "joined_json_chars": len(fever_text),
        "claim_extracted_query_matches": sum(
            q["query"].split("\n\nClaim: ")[-1].strip() in known["feverous"]
            for q in kit
            if q["dataset"] == "feverous"
        ),
    }
    # Use the actual renderer for this payload equality check, without an HTTP client.
    from eval.datasets.corporatebench import load_corporatebench
    from eval.harness.add_shape import shape_batch

    sample = load_corporatebench(DATA, limit=1)[0]
    original = shape_batch(
        sample.sessions[0].messages,
        dataset=sample.dataset,
        speaker_names=sample.speaker_names,
        shape="official",
    )
    corp_uid = next(q["user_id"] for q in kit if q["dataset"] == "corporatebench")
    actual = [m for add in retained[corp_uid] for m in add["messages"]]
    extra["corporatebench"] = {
        "native_messages": len(original),
        "ordered_message_payloads_identical": original == actual,
    }
    result = {
        "capture_sha256": {
            name: sha256(CAPTURE / name)
            for name in (
                "official-adds.jsonl",
                "official-searches.jsonl",
                "official-eval-kit.jsonl",
            )
        },
        "query_comparison": (
            "str.strip() exact equality, without lowercasing or paraphrase matching"
        ),
        "cohort_note": (
            "dataset uses existing kit attribution; "
            "temporal interval cohort is explicitly heuristic"
        ),
        "families": stats,
        "extra_checks": extra,
    }
    destination = Path(__file__).with_name("audit.json")
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
