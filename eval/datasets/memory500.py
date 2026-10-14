"""Offline curation/export utilities; source histories never depend on gold snippets.

This module does not call Add, Search, embeddings, or a language model. Selection
and semantic review are external, versioned records, separate from source memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from . import halumem, hybridqa, locomo, longmemeval, mquake, musique, personamem
from .manifest import MANIFEST
from .preprocess import Message, Session, normalize_content
from .registry import benchmark_dir


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def contains(text: str, answer: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text, re.I))


def raw_sources(root: Path) -> dict:
    return {
        "longmemeval": json.loads(
            (root / "longmemeval-s/lme_s_cleaned.json").read_text(encoding="utf-8")
        ),
        "locomo": jsonl(root / "locomo-refined/questions.jsonl"),
        "halumem": jsonl(root / "halumem/HaluMem-Medium.jsonl"),
        "musique": [r for r in musique.read_rows(root) if r["answerable"]],
        "hybridqa": json.loads((root / "hybridqa/dev.json").read_text(encoding="utf-8")),
        "mquake": list(mquake._rows(root / "mquake-remastered/data/CF3k-00000-of-00001.parquet")),
    }


def inventory(root: Path, work: Path) -> None:
    raw = raw_sources(root)
    write_json(
        work / "lme-candidates.json",
        [
            {
                "id": r["question_id"],
                "type": r["question_type"],
                "question": r["question"],
                "answer": r["answer"],
                "question_date": r["question_date"],
                "support": [
                    {"session_id": sid, "date": date, "messages": msgs}
                    for sid, date, msgs in zip(
                        r["haystack_session_ids"],
                        r["haystack_dates"],
                        r["haystack_sessions"],
                        strict=True,
                    )
                    if sid in r["answer_session_ids"]
                ],
            }
            for r in raw["longmemeval"]
            if r["question_type"] != "multi-session" or r["question_id"].endswith("_abs")
        ],
    )
    write_json(
        work / "locomo-candidates.json",
        [r for r in raw["locomo"] if str(r["category"]) in {"2", "4"}],
    )
    hc = []
    for user in raw["halumem"]:
        for si, session in enumerate(user["sessions"]):
            for qi, q in enumerate(session.get("questions") or []):
                hc.append(
                    {
                        "id": f"halumem-{user['uuid']}-s{si:03d}-q{qi:03d}",
                        "user_uuid": user["uuid"],
                        "session_index": si,
                        **q,
                        "current_dialogue": session["dialogue"],
                        "current_memory_points": session.get("memory_points", []),
                    }
                )
    write_json(work / "halu-candidates.json", hc)
    mc = []
    for r in raw["musique"]:
        hops = r["question_decomposition"]
        if not musique._is_strict_chain(hops):
            continue
        ps = {p["idx"]: p for p in r["paragraphs"]}
        if not all(
            contains(ps[h["paragraph_support_idx"]]["paragraph_text"], h["answer"]) for h in hops
        ):
            continue
        mc.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "hops": [{**h, "paragraph": ps[h["paragraph_support_idx"]]} for h in hops],
            }
        )
    write_json(work / "musique-candidates.json", mc)
    mq = []
    for start in range(0, len(raw["mquake"]), mquake.CASES_PER_USER):
        group = raw["mquake"][start : start + mquake.CASES_PER_USER]
        facts = defaultdict(set)
        for r in group:
            for s, p, o in r["orig_triples"]:
                facts[(s, p)].add(o)
        for r in group:
            for s, p, o in r["edit_triples"]:
                facts[(s, p)] = {o}
        for r in group:
            chain = r["new_triples"]
            if not all(chain[i - 1][2] == chain[i][0] for i in range(1, len(chain))):
                continue
            if not all(facts.get(tuple(t[:2])) == {t[2]} for t in chain):
                continue
            mq.append(
                {
                    "id": f"CF3k:{r['case_id']}",
                    "group_start": start,
                    "question": r["questions"][0],
                    "answer": r["new_answer"],
                    "hops": r["new_triples_labeled"],
                    "rewrites": r["requested_rewrite"],
                }
            )
    write_json(work / "mquake-candidates.json", mq)
    with (root / "personamem-v2/benchmark/text/benchmark.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        pm, counts = [], Counter()
        for i, r in enumerate(csv.DictReader(f)):
            persona = r["persona_id"]
            qid = f"pm-{persona}-{counts[persona]}"
            counts[persona] += 1
            pm.append(
                {
                    "id": qid,
                    "row_index": i,
                    **r,
                    "question": personamem._query_text(r["user_query"]),
                }
            )
    write_json(work / "persona-candidates.json", pm)
    write_json(work / "hybrid-candidates.json", raw["hybridqa"])
    print("candidate inventories ready", flush=True)


def portable_hybrid(root: Path, work: Path, table_ids: list[str]) -> dict:
    """Extract whole selected tables/pages to hashed paths (Windows colon safe).

    Verify the fixed original archive first; never extract an archive path to an
    OS path. No gold-based paragraph filtering. The original archive is unchanged.
    """
    entry = next(e for e in MANIFEST if e["name"] == "hybridqa/wiki_tables_with_links.tar.gz")
    archive = root / "hybridqa/wiki_tables_with_links.tar.gz"
    if digest(archive) != entry["sha256"]:
        raise ValueError("HybridQA archive fingerprint mismatch")
    wanted = {
        f"{kind}/{tid}.json": (tid, kind)
        for tid in table_ids
        for kind in ("tables_tok", "request_tok")
    }
    receipt = {}
    prefix = entry["archive_root"] + "/"
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name.removeprefix(prefix)
            if not member.isfile() or name not in wanted:
                continue
            tid, kind = wanted[name]
            file = tar.extractfile(member)
            if file is None:
                raise ValueError(name)
            body = file.read()
            dest = (
                work
                / "hybrid-full-source"
                / (hashlib.sha256(tid.encode()).hexdigest()[:20] + "-" + kind + ".json")
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            receipt.setdefault(tid, {})[kind] = {
                "path": str(dest.resolve()),
                "archive_member": member.name,
                "sha256": hashlib.sha256(body).hexdigest(),
            }
    if set(receipt) != set(table_ids) or any(
        set(v) != {"tables_tok", "request_tok"} for v in receipt.values()
    ):
        raise ValueError("HybridQA whole selected table/page extraction incomplete")
    write_json(
        work / "hybrid-source-receipt.json", {"archive_sha256": entry["sha256"], "tables": receipt}
    )
    print(f"whole tables and page maps extracted: {len(receipt)}", flush=True)
    return receipt


def propose(root: Path, work: Path) -> None:
    """Produce candidate selections for subsequent source review, not approval."""
    selections = []

    def take(dataset, task, row, qid=None, **extra):
        selections.append(
            {"dataset": dataset, "task": task, "source_qid": qid or row["id"], **extra}
        )

    rows = json.loads((work / "musique-candidates.json").read_text(encoding="utf-8"))
    indexes = {
        2: [
            1,
            2,
            10,
            15,
            17,
            18,
            20,
            22,
            30,
            33,
            45,
            47,
            48,
            51,
            57,
            61,
            62,
            65,
            85,
            91,
            95,
            97,
            98,
            100,
            103,
            104,
            105,
            107,
            109,
            112,
            113,
            114,
            116,
            117,
            123,
            130,
            131,
            132,
            138,
            141,
            144,
            147,
            150,
        ],
        3: [2, 10, 46, 47, 50, 64, 86],
    }
    for hops, ii in indexes.items():
        candidates = [r for r in rows if len(r["hops"]) == hops]
        for i in ii:
            take("musique", "multi_hop", candidates[i])
    rows = json.loads((work / "mquake-candidates.json").read_text(encoding="utf-8"))
    # Distinct cases, no paraphrases repeated. Whole native groups retained.
    answers = Counter()
    for hops in [2, 3, 4]:
        first_relations = Counter()
        count = 0
        for r in rows:
            if (
                len(r["hops"]) != hops
                or first_relations[r["hops"][0][1]] >= 2
                or answers[r["answer"]] >= 2
            ):
                continue
            take("mquake", "multi_hop", r, group_start=r["group_start"])
            first_relations[r["hops"][0][1]] += 1
            answers[r["answer"]] += 1
            count += 1
            if count == 10:
                break
        if count != 10:
            raise ValueError("MQuAKE diversity/complete-chain quota not met")
    rows = json.loads((work / "hybrid-candidates.json").read_text(encoding="utf-8"))
    for i in [0, 1, 5, 8, 12, 16, 19, 36, 37, 47, 56, 74, 76, 77, 79, 86, 87, 88, 89, 90]:
        r = rows[i]
        take("hybridqa", "multi_hop", r, qid=r["question_id"], table_id=r["table_id"])
    rows = json.loads((work / "lme-candidates.json").read_text(encoding="utf-8"))
    for typ in (
        "single-session-user",
        "single-session-assistant",
        "knowledge-update",
        "single-session-preference",
    ):
        candidates = [r for r in rows if r["type"] == typ and not r["id"].endswith("_abs")]
        for i, r in enumerate(candidates):
            if (
                typ == "single-session-user"
                and i not in [8, 19, 28, 30, 38, 40, 44, 46, 57]
                or typ == "single-session-assistant"
                and i
                in [
                    0,
                    3,
                    4,
                    6,
                    7,
                    11,
                    12,
                    13,
                    14,
                    20,
                    25,
                    26,
                    36,
                    47,
                    55,
                ]
            ):
                take("longmemeval", "single_hop_fact", r)
            elif typ == "knowledge-update" and i not in [
                1,
                5,
                13,
                16,
                20,
                21,
                23,
                25,
                27,
                28,
                30,
                32,
                33,
                34,
                51,
                55,
                64,
                65,
                66,
                67,
                68,
                70,
            ]:
                take("longmemeval", "knowledge_update", r)
            elif typ == "single-session-preference" and i != 26:
                take("longmemeval", "preference_rule", r)
    for r in rows:
        if r["id"].endswith("_abs") and r["type"] in {"single-session-user", "knowledge-update"}:
            take("longmemeval", "abstention", r)
    rows = json.loads((work / "locomo-candidates.json").read_text(encoding="utf-8"))
    users = sorted({r["sample_id"] for r in rows})
    extra_temporal = []
    extra_facts = []
    for user in users:
        for task, typ, quota in [("single_hop_fact", "4", 6), ("temporal", "2", 7)]:
            candidates = [
                r
                for r in rows
                if r["sample_id"] == user
                and str(r["category"]) == typ
                and not r["is_multi_modality"]
                and len(r.get("evidence_messages", [])) == 1
                and not r["evidence_messages"][0].get("has_multimodal_context")
                and not re.search(r"\b(photo|photograph|picture|pic|image)\b", r["question"], re.I)
                and (
                    task != "temporal"
                    or (
                        r["question"].lower().startswith("when")
                        and re.search(
                            r"\b(yesterday|tomorrow|last|next|ago|today|this month|"
                            r"this weekend|\d{4})\b",
                            r["evidence_messages"][0]["text"],
                            re.I,
                        )
                    )
                )
                and (
                    task != "single_hop_fact"
                    or (
                        r["question"].lower().startswith(("what", "who", "where", "which"))
                        and any(
                            contains(r["evidence_messages"][0]["text"], str(a)) for a in r["answer"]
                        )
                    )
                )
            ]
            # Spread through the entire conversation, rather than its first session.
            count = min(quota, len(candidates))
            chosen = (
                [candidates[(len(candidates) - 1) * i // (count - 1)] for i in range(count)]
                if count > 1
                else candidates
            )
            for r in chosen:
                take("locomo", task, r, qid=r["qa_id"])
            if task == "temporal":
                extra_temporal.extend(r for r in candidates if r not in chosen)
            else:
                extra_facts.extend(r for r in candidates if r not in chosen)
    missing = 70 - sum(s["task"] == "temporal" for s in selections)
    for r in extra_temporal[:missing]:
        take("locomo", "temporal", r, qid=r["qa_id"])
    missing = 60 - sum(
        s["dataset"] == "locomo" and s["task"] == "single_hop_fact" for s in selections
    )
    for r in extra_facts[:missing]:
        take("locomo", "single_hop_fact", r, qid=r["qa_id"])
    rows = json.loads((work / "halu-candidates.json").read_text(encoding="utf-8"))
    raw = jsonl(root / "halumem/HaluMem-Medium.jsonl")
    usermap = {r["uuid"]: r for r in raw}
    users = list(usermap)
    for ui, user in enumerate(users):
        history = usermap[user]["sessions"]
        for task, typ in [
            ("single_hop_fact", "Basic Fact Recall"),
            ("knowledge_update", "Dynamic Update"),
        ]:
            candidates = []
            for r in rows:
                if r["user_uuid"] != user or r["question_type"] != typ:
                    continue
                if task == "knowledge_update" and (
                    r["session_index"] < 4 or not r["question"].startswith("What")
                ):
                    continue
                if task == "single_hop_fact" and (
                    r["session_index"] not in {1, 2}
                    or re.search(r"\b(when|date|change|why|how|and)\b", r["question"], re.I)
                ):
                    continue
                # Explicit short fact, in actual dialogue, not just memory_points.
                text = "\n".join(
                    t["content"]
                    for s in history[: r["session_index"] + 1]
                    for t in s["dialogue"]
                    if t["role"] == "user"
                )
                if len(r["answer"]) < 100 and contains(text, r["answer"].removeprefix("$")):
                    candidates.append(r)
            if not candidates:
                print(f"skip ungrounded source candidate: {ui}/{user}/{task}", flush=True)
                continue
            r = candidates[ui % min(len(candidates), 5)]
            take(
                "halumem",
                task,
                r,
                session_index=r["session_index"],
                user_uuid=user,
                source_qid_override=(
                    f"halumem-{user}-s{r['session_index'] + 1:03d}"
                    f"-q{int(r['id'].rsplit('q', 1)[1]):03d}"
                ),
            )
        candidates = [
            r
            for r in rows
            if r["user_uuid"] == user
            and r["question_type"] == "Memory Boundary"
            and r["session_index"] >= 4
            and re.search(
                r"unknown|not (provided|specified|mentioned)|no (information|data)",
                r["answer"],
                re.I,
            )
            and not re.search(r"\b(why|how|and|before|after)\b", r["question"], re.I)
        ]
        # Prefer varied missing attributes over twenty middle-name paraphrases.
        candidates.sort(key=lambda r: ("middle name" in r["question"].lower(), r["session_index"]))
        count = 2 if ui < 8 else 1
        used_text = set()
        for r in candidates:
            signature = re.sub(r"as of .+|on .+", "", r["question"], flags=re.I)
            if signature in used_text:
                continue
            take(
                "halumem",
                "abstention",
                r,
                session_index=r["session_index"],
                user_uuid=user,
                source_qid_override=(
                    f"halumem-{user}-s{r['session_index'] + 1:03d}"
                    f"-q{int(r['id'].rsplit('q', 1)[1]):03d}"
                ),
            )
            used_text.add(signature)
            if len(used_text) == count:
                break
    for task, typ, quota in [
        ("knowledge_update", "Dynamic Update", 20),
        ("single_hop_fact", "Basic Fact Recall", 20),
    ]:
        for r in rows:
            if sum(s["dataset"] == "halumem" and s["task"] == task for s in selections) >= quota:
                break
            if (
                r["question_type"] != typ
                or r["session_index"] < 4
                or not r["question"].startswith("What")
            ):
                continue
            if any(s["dataset"] == "halumem" and s["source_qid"] == r["id"] for s in selections):
                continue
            history = usermap[r["user_uuid"]]["sessions"]
            text = "\n".join(
                t["content"]
                for x in history[: r["session_index"] + 1]
                for t in x["dialogue"]
                if t["role"] == "user"
            )
            if len(r["answer"]) < 100 and contains(text, r["answer"].removeprefix("$")):
                take(
                    "halumem",
                    task,
                    r,
                    session_index=r["session_index"],
                    user_uuid=r["user_uuid"],
                    source_qid_override=(
                        f"halumem-{r['user_uuid']}-s{r['session_index'] + 1:03d}"
                        f"-q{int(r['id'].rsplit('q', 1)[1]):03d}"
                    ),
                )
    pref = next(
        r
        for r in rows
        if r["question_type"] == "Basic Fact Recall" and "beverage" in r["question"].lower()
    )
    take(
        "halumem",
        "preference_rule",
        pref,
        session_index=pref["session_index"],
        user_uuid=pref["user_uuid"],
        source_qid_override=(
            f"halumem-{pref['user_uuid']}-s{pref['session_index'] + 1:03d}"
            f"-q{int(pref['id'].rsplit('q', 1)[1]):03d}"
        ),
    )
    rows = json.loads((work / "persona-candidates.json").read_text(encoding="utf-8"))
    allowed = {"neutral_preferences", "anti_stereotypical_pref", "stereotypical_pref"}
    topics = Counter()
    persons = Counter()
    for r in rows:
        if r["who"] != "self" or r["updated"] != "False" or r["pref_type"] not in allowed:
            continue
        if persons[r["persona_id"]] >= 2 or (r["persona_id"] not in persons and len(persons) >= 20):
            continue
        if topics[r["topic_query"]] >= 10:
            continue
        path = root / "personamem-v2/data/chat_history_32k" / Path(r["chat_history_32k_link"]).name
        history = personamem._history(path)
        snippet = json.loads(r["related_conversation_snippet"])
        snippets = [m["content"].replace("\\n", "\n") for m in snippet if m["content"].strip()]
        native = [m.content.replace("\\n", "\n") for m in history]
        if not snippets or not all(
            any(s == text or s in text for text in native) for s in snippets
        ):
            continue
        take("personamem", "preference_rule", r, persona_id=r["persona_id"], csv_row=r["row_index"])
        topics[r["topic_query"]] += 1
        persons[r["persona_id"]] += 1
        if sum(persons.values()) == 40:
            break
    write_json(
        work / "curation-proposed.json",
        {
            "version": "memory500-v1",
            "selections": selections,
            "review_status": "candidate-source-review-required",
        },
    )
    print(json.dumps(Counter(r["task"] for r in selections)), len(selections), flush=True)


QUOTAS = {
    "multi_hop": 100,
    "single_hop_fact": 150,
    "temporal": 70,
    "knowledge_update": 70,
    "preference_rule": 70,
    "abstention": 40,
}

NATIVE_DATASETS = {
    "musique": "musique",
    "mquake": "mquake-remastered",
    "hybridqa": "hybridqa",
    "longmemeval": "longmemeval-s",
    "locomo": "locomo-refined",
    "halumem": "halumem",
    "personamem": "personamem-v2",
}


def build(root: Path, work: Path, output: Path) -> None:
    """Export full normalized source spaces, gold and independently addressed audits."""
    curation = json.loads((work / "curation.json").read_text(encoding="utf-8"))
    if curation["review_status"] != "source-reviewed":
        raise ValueError("Semantic source review must precede export")
    selected = curation["selections"]
    if Counter(r["task"] for r in selected) != QUOTAS:
        raise ValueError("Question/task quotas differ")
    raw = raw_sources(root)
    maps = {
        "longmemeval": {r["question_id"]: r for r in raw["longmemeval"]},
        "locomo": {r["qa_id"]: r for r in raw["locomo"]},
        "halumem": {r["uuid"]: r for r in raw["halumem"]},
        "musique": {r["id"]: r for r in raw["musique"]},
        "hybridqa": {r["question_id"]: r for r in raw["hybridqa"]},
        "mquake": {r["case_id"]: r for r in raw["mquake"]},
        "personamem": {
            r["id"]: r
            for r in json.loads((work / "persona-candidates.json").read_text(encoding="utf-8"))
        },
    }
    convs = {s.user_id: s for s in locomo.load_locomo(root)}
    hybrid_receipt = json.loads((work / "hybrid-source-receipt.json").read_text(encoding="utf-8"))[
        "tables"
    ]
    spaces, questions, audits, source_files = {}, [], [], {}

    def source(path: Path) -> str:
        key = path.relative_to(root).as_posix()
        if key not in source_files:
            source_files[key] = {"path": str(path.resolve()), "sha256": digest(path)}
        return key

    files = {
        "longmemeval": source(root / "longmemeval-s/lme_s_cleaned.json"),
        "locomo": source(root / "locomo-refined/questions.jsonl"),
        "halumem": source(root / "halumem/HaluMem-Medium.jsonl"),
        "musique": source(root / "musique/musique_full_v1.0_dev.jsonl"),
        "hybridqa": source(root / "hybridqa/dev.json"),
        "mquake": source(root / "mquake-remastered/data/CF3k-00000-of-00001.parquet"),
        "personamem": source(root / "personamem-v2/benchmark/text/benchmark.csv"),
    }
    source(root / "locomo-refined/locomo_refined.json")
    source(root / "hybridqa/wiki_tables_with_links.tar.gz")
    source(root / "hybridqa/dev_reference.json")

    def space(dataset, uid, sessions, names, contract, scope, expected_sessions, expected_messages):
        sid = f"memory500-v1/{dataset}/{uid}"
        if sid not in spaces:
            native = [
                {"session_id": s.session_id, "messages": [m.to_add_payload() for m in s.messages]}
                for s in sessions
            ]
            sp = {
                "space_id": sid,
                "user_id": sid,
                "dataset": dataset,
                "source_dataset": NATIVE_DATASETS[dataset],
                "source_user_id": uid,
                "speaker_names": list(names),
                "loading_contract": contract,
                "source_scope": scope,
                "sessions": native,
            }
            if (
                len(native) != expected_sessions
                or sum(len(s["messages"]) for s in native) != expected_messages
            ):
                raise ValueError(f"incomplete source scope: {sid}")
            sp["normalized_source_sha256"] = hashlib.sha256(
                json.dumps(
                    native, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            for s in native:
                for i, m in enumerate(s["messages"]):
                    m["source_message_id"] = hashlib.sha256(
                        f"{sid}\0{s['session_id']}\0{i}".encode()
                    ).hexdigest()[:24]
                    m["source_message_index"] = i
            spaces[sid] = sp
        return spaces[sid]

    def all_messages(sp):
        return [m for s in sp["sessions"] for m in s["messages"]]

    def refs_for(sp, text):
        ids = [m["source_message_id"] for m in all_messages(sp) if text.strip() in m["content"]]
        if not ids:
            raise ValueError(
                f"source quote missing from exported memory: {sp['space_id']} / {text[:90]}"
            )
        return ids

    def proof(sp, text, **extra):
        return {"source_message_ids": refs_for(sp, text), "quote": text.strip(), **extra}

    # Corpus construction uses every candidate paragraph; gold never selects memory.
    musi_ids = [r["source_qid"] for r in selected if r["dataset"] == "musique"]
    musi = musique.load_musique_haystack(root, musi_ids)
    all_paragraphs = {
        normalize_content(f"Title: {p['title']}\n{p['paragraph_text'].strip()}", where="memory500")
        for qid in musi_ids
        for p in maps["musique"][qid]["paragraphs"]
    }
    musi_sp = space(
        "musique",
        "selected-candidate-haystack",
        musi.sessions,
        ("Corpus", "assistant"),
        "selected-complete-candidates-title-body-dedup-v1",
        {
            "source_question_ids": musi_ids,
            "dedup_key": ["title", "paragraph_text"],
            "scope_note": (
                "Local shared candidate pool, not full Wikipedia or the native per-question pool."
            ),
        },
        1,
        len(all_paragraphs),
    )
    if {m["content"] for m in all_messages(musi_sp)} != all_paragraphs:
        raise ValueError("MuSiQue full candidate pool differs, including distractors")

    for i, pick in enumerate(selected, 1):
        dataset, task = pick["dataset"], pick["task"]
        qid = pick.get("source_qid_override", pick["source_qid"])
        record = {
            "id": f"memory500-{i:04d}",
            "dataset": dataset,
            "source_dataset": NATIVE_DATASETS[dataset],
            "source_question_id": qid,
            "task_type": task,
            "is_multi_hop": task == "multi_hop",
            "is_abstention": task == "abstention",
            "answerable": task != "abstention",
            "source_file": files[dataset],
            "evaluation_format": "free_response",
        }
        audit = {
            "id": record["id"],
            "dataset": dataset,
            "source_question_id": qid,
            "task_type": task,
            "evidence": [],
            "steps": [],
        }
        if dataset == "musique":
            r = maps[dataset][qid]
            sp = musi_sp
            record.update(
                question=r["question"],
                gold={
                    "answer": r["answer"],
                    "answer_aliases": r["answer_aliases"],
                    "answerable": True,
                },
                source_category=f"{len(r['question_decomposition'])}-hop/answerable",
            )
            ps = {p["idx"]: p for p in r["paragraphs"]}
            for j, h in enumerate(r["question_decomposition"]):
                paragraph = ps[h["paragraph_support_idx"]]
                audit["steps"].append(
                    {
                        "id": f"s{j + 1}",
                        "query_template": h["question"],
                        "answer": h["answer"],
                        "depends_on": [f"s{x}" for x in re.findall(r"#(\d+)", h["question"])],
                        "source_paragraph_idx": h["paragraph_support_idx"],
                        "source_title": paragraph["title"],
                        "evidence": [proof(sp, paragraph["paragraph_text"])],
                    }
                )
            audit["reason"] = "依赖前步实体逐步取属性；已核对关系语义，支持正文完整进入共享候选池。"
        elif dataset == "mquake":
            case = int(pick["source_qid"].split(":")[1])
            r = maps[dataset][case]
            start = pick["group_start"]
            group = raw["mquake"][start : start + mquake.CASES_PER_USER]
            uid = f"CF3k-group-{start:05d}"
            originals = [
                Message("user", mquake._triple_text(*t))
                for x in group
                for t in x["orig_triples_labeled"]
            ]
            updates = [
                Message("user", mquake._update_text(x))
                for c in group
                for x in c["requested_rewrite"]
            ]
            sp = space(
                dataset,
                uid,
                (Session(uid, tuple(originals + updates)),),
                ("user", "assistant"),
                "native-group64-originals-then-updates-v1",
                {
                    "case_ids": [x["case_id"] for x in group],
                    "ordering_note": (
                        "Native group size and full facts; all original statements "
                        "precede updates, matching the published edit-before-query task. "
                        "No new_triples are added."
                    ),
                },
                1,
                sum(len(x["orig_triples_labeled"]) + len(x["requested_rewrite"]) for x in group),
            )
            record.update(
                source_question_id=f"CF3k-{case}-0",
                question=r["questions"][0],
                gold=sorted(set([r["new_answer"], *(r.get("new_answer_alias") or [])])),
                source_category="CF3k",
                source_case_id=case,
                knowledge_world="published-counterfactual-after-updates",
            )
            facts, origins, edited = defaultdict(set), {}, {}
            for c in group:
                for ids, labels in zip(c["orig_triples"], c["orig_triples_labeled"], strict=True):
                    facts[tuple(ids[:2])].add(ids[2])
                    origins[tuple(ids)] = mquake._triple_text(*labels)
            for c in group:
                subject_labels = {
                    ids[0]: labels[0]
                    for ids, labels in zip(
                        c["orig_triples"] + c["new_triples"],
                        c["orig_triples_labeled"] + c["new_triples_labeled"],
                        strict=True,
                    )
                }
                for ids, rewrite in zip(c["edit_triples"], c["requested_rewrite"], strict=True):
                    # Published arrays are parallel, including edits outside the QA chain.
                    if (
                        subject_labels.get(ids[0], rewrite["subject"]) != rewrite["subject"]
                        or ids[1] != rewrite["relation_id"]
                        or ids[2] != rewrite["target_new_id"]
                    ):
                        raise ValueError(f"MQuAKE edit/rewrite arrays not aligned: {c['case_id']}")
                    facts[tuple(ids[:2])] = {ids[2]}
                    edited[tuple(ids[:2])] = (ids[2], mquake._update_text(rewrite))
            for j, (ids, labels) in enumerate(
                zip(r["new_triples"], r["new_triples_labeled"], strict=True)
            ):
                if facts.get(tuple(ids[:2])) != {ids[2]}:
                    raise ValueError(f"MQuAKE missing/conflicting updated hop: {case}/{j}")
                if j and r["new_triples"][j - 1][2] != ids[0]:
                    raise ValueError("MQuAKE entity chain disconnected")
                key = tuple(ids[:2])
                text = edited[key][1] if key in edited else origins[tuple(ids)]
                audit["steps"].append(
                    {
                        "id": f"s{j + 1}",
                        "subject": labels[0],
                        "relation": labels[1],
                        "answer": labels[2],
                        "triple_ids": ids,
                        "depends_on": [] if not j else [f"s{j}"],
                        "evidence": [proof(sp, text)],
                    }
                )
            if r["new_triples_labeled"][-1][2] not in record["gold"]:
                raise ValueError("MQuAKE final chain node does not match gold")
            audit["reason"] = (
                "完整原始事实组及公开更新提供链条；逐关系按实体ID核对最新值，无缺边或冲突。"
            )
        elif dataset == "hybridqa":
            r = maps[dataset][qid]
            tid = r["table_id"]
            receipt = hybrid_receipt[tid]
            docs = []
            for kind in ("tables_tok", "request_tok"):
                path = Path(receipt[kind]["path"])
                if digest(path) != receipt[kind]["sha256"]:
                    raise ValueError(f"changed portable HybridQA source: {tid}")
                docs.append(json.loads(path.read_text(encoding="utf-8")))
            table, pages = docs
            heads = [hybridqa._cell(c) for c in table["header"]]
            context = [f"Table: {tid}", f"Title: {table['title'].strip()}"]
            context += [
                f"{k}: {table[k].strip()}"
                for k in ("intro", "section_title", "section_text")
                if table.get(k)
            ]
            context.append("Columns: " + " | ".join(heads))
            texts = ["\n".join(context)]
            for n, row in enumerate(table["data"]):
                texts.append(
                    f"Table: {tid}\nRow {n + 1}: "
                    + " | ".join(
                        f"{h}: {hybridqa._cell(c)}" for h, c in zip(heads, row, strict=True)
                    )
                )
            for link, body in pages.items():
                title = unquote(link.removeprefix("/wiki/")).replace("_", " ")
                texts.append(f"Page: {title}\nLink: {link}\n{body.strip()}")
            uid = hashlib.sha256(tid.encode()).hexdigest()[:20]
            sp = space(
                dataset,
                uid,
                (Session(uid, tuple(Message("user", t.strip()) for t in texts)),),
                ("Corpus", "assistant"),
                "hybridqa-table-passages-v1-portable-paths",
                {
                    "table_id": tid,
                    "table_rows": len(table["data"]),
                    "linked_passages": len(pages),
                    "source_members": receipt,
                },
                1,
                1 + len(table["data"]) + len(pages),
            )
            record.update(
                question=r["question"],
                gold={"answer": r["answer-text"]},
                source_category="table-passage-join",
                table_id=tid,
            )
            for j, step in enumerate(pick["reviewed_steps"]):
                ps = []
                for ref in step["refs"]:
                    text = texts[1 + ref["index"]] if ref["kind"] == "row" else pages[ref["link"]]
                    ps.append(proof(sp, text, source_locator=ref))
                audit["steps"].append(
                    {
                        "id": f"s{j + 1}",
                        "answer": step["answer"],
                        "relation": step["relation"],
                        "depends_on": [] if not j else [f"s{j}"],
                        "evidence": ps,
                    }
                )
            if audit["steps"][-1]["answer"] != r["answer-text"]:
                raise ValueError(f"HybridQA reviewed final node != gold: {qid}")
            audit["reason"] = "通过未知中间实体连接表格筛选与链接正文；整张表与全部链接页面保留。"
        elif dataset == "longmemeval":
            r = maps[dataset][qid]
            sessions = longmemeval._sessions(r, source=root / files[dataset])
            total = sum(
                bool(str(m["content"]).strip()) for ss in r["haystack_sessions"] for m in ss
            )
            sp = space(
                dataset,
                "lme-" + qid,
                sessions,
                ("user", "assistant"),
                "longmemeval-s-cleaned-full-haystack-v1",
                {
                    "question_id": qid,
                    "haystack_session_ids": r["haystack_session_ids"],
                    "empty_messages_skipped": sum(len(x) for x in r["haystack_sessions"]) - total,
                },
                len(r["haystack_sessions"]),
                total,
            )
            record.update(
                question=r["question"],
                gold=r["answer"],
                source_category=r["question_type"],
                question_date=r["question_date"],
            )
            for session in sp["sessions"]:
                if session["session_id"] in r["answer_session_ids"]:
                    audit["evidence"].append(
                        {
                            "session_id": session["session_id"],
                            "source_message_ids": [
                                m["source_message_id"] for m in session["messages"]
                            ],
                        }
                    )
            if set(r["answer_session_ids"]) - {s["session_id"] for s in sp["sessions"]}:
                raise ValueError(f"LongMemEval gold support session not in full haystack: {qid}")
            audit["reason"] = {
                "single_hop_fact": "已知人物/物件的直接属性或既有回答记录，无未知桥接实体。",
                "knowledge_update": "同一属性的历史/当前版本选择；旧值干扰会话完整保留。",
                "preference_rule": "使用已知用户偏好或约束进行个性化应用，不需依次解析未知实体。",
                "abstention": "上游原生_abs题：询问未提供的属性，完整haystack保留。",
            }[task]
        elif dataset == "locomo":
            r = maps[dataset][qid]
            sample = convs[r["sample_id"]]
            sp = space(
                dataset,
                sample.user_id,
                sample.sessions,
                sample.speaker_names,
                "locomo-refined-complete-text-conversation-v1",
                {
                    "sample_id": sample.user_id,
                    "scope_note": (
                        "All source text turns and session dates. "
                        "Image-dependent questions excluded; "
                        "no gold evidence_messages used for memory."
                    ),
                },
                len(sample.sessions),
                sample.message_count,
            )
            record.update(
                question=r["question"], gold=r["answer"], source_category=str(r["category"])
            )
            for m in r["evidence_messages"]:
                audit["evidence"].append(proof(sp, m["text"], dia_id=m["dia_id"]))
            audit["reason"] = (
                "单个已知事件的日期/相对时间加原始会话日期锚点；无实体依赖链。"
                if task == "temporal"
                else "已知实体的直接事实属性，原文明确提供。"
            )
        elif dataset == "halumem":
            user, si = pick["user_uuid"], pick["session_index"]
            qs = maps[dataset][user]["sessions"][si]["questions"]
            qi = int(qid.rsplit("q", 1)[1])
            r = qs[qi]
            prefix = maps[dataset][user]["sessions"][: si + 1]
            sessions = tuple(halumem._session(x, user, j) for j, x in enumerate(prefix))
            sp = space(
                dataset,
                f"halumem-{user}-s{si + 1:03d}",
                sessions,
                ("user", "assistant"),
                "halumem-medium-native-checkpoint-prefix-v1",
                {
                    "uuid": user,
                    "session_index_zero_based": si,
                    "cutoff_end_time": prefix[-1]["end_time"],
                    "future_sessions_excluded": True,
                    "annotations_excluded": [
                        "persona_info",
                        "memory_points",
                        "questions",
                        "evidence",
                    ],
                },
                si + 1,
                sum(len(s["dialogue"]) for s in prefix),
            )
            record.update(question=r["question"], gold=dict(r), source_category=r["question_type"])
            if task != "abstention":
                target = pick.get("grounding_quote_hint", r["answer"].removeprefix("$"))
                ms = [
                    m
                    for m in all_messages(sp)
                    if m["role"] == "user" and contains(m["content"], target)
                ]
                if not ms:
                    raise ValueError(f"HaluMem answer has no explicit dialogue proof: {qid}")
                # Keep whole source turn, not an extracted/synthesized Add fact.
                audit["evidence"] = [proof(sp, ms[-1]["content"])]
            else:
                if r["question_type"] != "Memory Boundary":
                    raise ValueError("HaluMem abstention must be native Memory Boundary")
                audit["native_unknown_annotation"] = r["answer"]
            audit["reason"] = (
                "按原生检查点保留全部历史正文，不加入persona或memory_points标注；类别依据原题意。"
            )
        else:
            r = maps["personamem"][qid]
            path = (
                root / "personamem-v2/data/chat_history_32k" / Path(r["chat_history_32k_link"]).name
            )
            source(path)
            history = personamem._history(path)
            sp = space(
                dataset,
                "pm-" + r["persona_id"],
                (Session("pm-" + r["persona_id"], history),),
                ("user", "assistant"),
                "personamem-v2-full-native-32k-history-v1",
                {
                    "persona_id": r["persona_id"],
                    "history_file": path.relative_to(root).as_posix(),
                    "system_turn_preserved": True,
                    "annotations_not_added": [
                        "expanded_persona",
                        "preference",
                        "related_conversation_snippet",
                    ],
                },
                1,
                len(json.loads(path.read_text(encoding="utf-8"))["chat_history"]),
            )
            record.update(
                question=r["question"],
                gold={
                    "correct_answer": r["correct_answer"],
                    "incorrect_answers": json.loads(r["incorrect_answers"]),
                    "persona_id": r["persona_id"],
                },
                source_category=r["pref_type"],
                topic=r["topic_query"],
                evaluation_format="native_mcq",
                source_csv_row=r["row_index"],
            )
            for m in json.loads(r["related_conversation_snippet"]):
                text = m["content"].replace("\\n", "\n")
                ids = [
                    x["source_message_id"]
                    for x in all_messages(sp)
                    if text in x["content"].replace("\\n", "\n")
                ]
                if not ids:
                    raise ValueError(
                        "PersonaMem related snippet absent from actual full 32k history"
                    )
                audit["evidence"].append(
                    {
                        "source_message_ids": ids,
                        "quote": text,
                        "match_normalization": "literal-backslash-n-to-newline-for-audit-only",
                    }
                )
            audit["reason"] = (
                "原生MCQ偏好应用；相关文本在完整历史中逐字可定位，标注未另行加入语料。"
            )
        record["space_id"] = sp["space_id"]
        record["user_id"] = sp["user_id"]
        audit["space_id"] = sp["space_id"]
        audit["source_question_id"] = record["source_question_id"]
        audit["answerable"] = record["answerable"]
        if task == "multi_hop" and len(audit["steps"]) < 2:
            raise ValueError("Multi-hop requires an explicitly reviewed dependency chain")
        questions.append(record)
        audits.append(audit)
    if len({q["question"].strip().lower() for q in questions}) != len(questions):
        raise ValueError("Duplicate question wording across selected sources")
    if len({(q["dataset"], q["source_question_id"]) for q in questions}) != len(questions):
        raise ValueError("Repeated original question/case")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with gzip.open(output / "spaces.jsonl.gz", "wt", encoding="utf-8", newline="\n") as f:
        for sp in spaces.values():
            f.write(json.dumps(sp, ensure_ascii=False) + "\n")
    with (output / "question-audits.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for a in audits:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    # Full source table JSON and page maps remain available as portable snapshots.
    snapshots = {
        tid: {
            kind: json.loads(Path(meta["path"]).read_text(encoding="utf-8"))
            for kind, meta in hybrid_receipt[tid].items()
        }
        for tid in {p["table_id"] for p in selected if p["dataset"] == "hybridqa"}
    }
    with gzip.open(output / "hybrid-full-source.json.gz", "wt", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False)
    write_json(output / "curation.json", curation)
    write_json(
        output / "manifest.json",
        {
            "version": curation["version"],
            "source_files": source_files,
            "multi_hop_definition": curation["multi_hop_definition"],
            "seed": None,
            "selection_method": (
                "Source-grounded explicit curation; no model answer or retrieval result used"
            ),
            "questions_file": "questions.jsonl",
            "spaces_file": "spaces.jsonl.gz",
            "audits_file": "question-audits.jsonl",
            "space_scope_fingerprints": {
                k: {
                    "sha256": v["normalized_source_sha256"],
                    "loading_contract": v["loading_contract"],
                }
                for k, v in spaces.items()
            },
            "output_sha256": {
                name: digest(output / name)
                for name in [
                    "questions.jsonl",
                    "spaces.jsonl.gz",
                    "question-audits.jsonl",
                    "hybrid-full-source.json.gz",
                    "curation.json",
                ]
            },
            "source_urls": {
                "musique": "https://github.com/stonybrooknlp/musique",
                "hybridqa": "https://github.com/wenhuchen/HybridQA",
                "mquake": "https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered",
                "longmemeval": "https://github.com/xiaowu0162/LongMemEval",
                "locomo": "https://github.com/mem-eval-suite/LoCoMo_refined",
                "halumem": "https://github.com/MemTensor/HaluMem",
                "personamem": "https://github.com/bowen-upenn/PersonaMem-v2",
            },
        },
    )
    validate(output, work)
    print("full source bundle exported", output.resolve(), flush=True)


def validate(output: Path, work: Path) -> dict:
    """Read saved artifacts back; fail closed on missing memory, changed scopes or quotas."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["output_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"export fingerprint changed: {name}")
    questions = jsonl(output / "questions.jsonl")
    audits = jsonl(output / "question-audits.jsonl")
    with gzip.open(output / "spaces.jsonl.gz", "rt", encoding="utf-8") as f:
        spaces = [json.loads(line) for line in f]
    indexed = {s["space_id"]: s for s in spaces}
    if len(indexed) != len(spaces) or len({s["user_id"] for s in spaces}) != len(spaces):
        raise ValueError("source isolation identifiers collide")
    if Counter(q["task_type"] for q in questions) != QUOTAS or len(questions) != 500:
        raise ValueError("exported quotas incomplete")
    if len(questions) != len(audits) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("question/audit IDs incomplete")
    messages = {}
    for sp in spaces:
        clean = [
            {
                "session_id": s["session_id"],
                "messages": [
                    {
                        k: v
                        for k, v in m.items()
                        if k not in {"source_message_id", "source_message_index"}
                    }
                    for m in s["messages"]
                ],
            }
            for s in sp["sessions"]
        ]
        fingerprint = hashlib.sha256(
            json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if fingerprint != manifest["space_scope_fingerprints"][sp["space_id"]]["sha256"]:
            raise ValueError("complete source memory scope changed on serialization")
        for s in sp["sessions"]:
            for m in s["messages"]:
                if not m["content"].strip() or set(m) - {
                    "role",
                    "content",
                    "timestamp",
                    "source_message_id",
                    "source_message_index",
                }:
                    raise ValueError("blank memory or annotation field in Add source")
                mid = m["source_message_id"]
                if mid in messages:
                    raise ValueError("memory source identifiers collide")
                messages[mid] = (sp["space_id"], m)
    for q, a in zip(questions, audits, strict=True):
        if q["id"] != a["id"] or q["space_id"] not in indexed:
            raise ValueError("question missing isolated source memory")
        if q["is_multi_hop"] and (q["is_abstention"] or not q["answerable"] or len(a["steps"]) < 2):
            raise ValueError("multi-hop is unanswerable or has no chain")
        proofs = a["evidence"] + [p for step in a["steps"] for p in step["evidence"]]
        for p in proofs:
            for mid in p["source_message_ids"]:
                if mid not in messages or messages[mid][0] != q["space_id"]:
                    raise ValueError("evidence absent or crosses user isolation")
                if p.get("quote"):
                    text = messages[mid][1]["content"]
                    if p.get("match_normalization"):
                        text = text.replace("\\n", "\n")
                    if p["quote"] not in text:
                        raise ValueError("audit quote not contained in its source message")
    result = {
        "status": "passed",
        "questions": len(questions),
        "task_counts": dict(Counter(q["task_type"] for q in questions)),
        "dataset_counts": dict(Counter(q["dataset"] for q in questions)),
        "isolated_spaces": len(spaces),
        "memory_source_messages": len(messages),
        "multi_hop_answerable": sum(q["is_multi_hop"] and q["answerable"] for q in questions),
        "multi_hop_hops": dict(
            Counter(len(a["steps"]) for a in audits if a["task_type"] == "multi_hop")
        ),
        "unanswerable_multi_hop": sum(q["is_multi_hop"] and not q["answerable"] for q in questions),
        "missing_evidence_references": 0,
        "cross_space_evidence_references": 0,
        "full_normalized_scope_fingerprints_verified": len(spaces),
        "retrieval_or_answer_model_calls": 0,
        "scope_stats": [
            {
                "space_id": s["space_id"],
                "dataset": s["dataset"],
                "sessions": len(s["sessions"]),
                "messages": sum(len(ss["messages"]) for ss in s["sessions"]),
                "loading_contract": s["loading_contract"],
            }
            for s in spaces
        ],
    }
    write_json(work / "validation.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "scope_stats"}, ensure_ascii=False),
        flush=True,
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inventory", "hybrid", "propose", "build", "validate"))
    p.add_argument("--root", type=Path, default=Path(benchmark_dir()))
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--tables", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "inventory":
        inventory(args.root, args.work)
    elif args.action == "propose":
        propose(args.root, args.work)
    elif args.action == "build" and args.output:
        build(args.root, args.work, args.output)
    elif args.action == "validate" and args.output:
        validate(args.output, args.work)
    elif args.tables:
        portable_hybrid(args.root, args.work, json.loads(args.tables.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
