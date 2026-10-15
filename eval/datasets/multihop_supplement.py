"""Offline, independent 100-question supplement; never modifies the audited base.

Selections must have source reviews before export. Full source scopes, distractors,
original questions and original gold are preserved. No Add, Search or model calls.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from . import hybridqa, mquake, musique
from .manifest import MANIFEST
from .preprocess import normalize_content
from .registry import benchmark_dir

VERSION = "memory100-multihop-v1"
COUNTS = {"musique": 40, "mquake": 40, "hybridqa": 20}
NATIVE = {"musique": "musique", "mquake": "mquake-remastered", "hybridqa": "hybridqa"}
FILES = {
    "musique": "musique/musique_full_v1.0_dev.jsonl",
    "mquake": "mquake-remastered/data/CF3k-00000-of-00001.parquet",
    "hybridqa": "hybridqa/dev.json",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def norm(text):
    return normalize_content(text, where="multihop supplement")


def query_key(text):
    return re.sub(r"\W+", "", text).casefold()


def corpus_texts(tid, table, pages):
    headers = [hybridqa._cell(c) for c in table["header"]]
    context = [f"Table: {tid}", f"Title: {table['title'].strip()}"]
    for key in ("intro", "section_title", "section_text"):
        if table.get(key):
            context.append(f"{key}: {table[key].strip()}")
    context.append("Columns: " + " | ".join(headers))
    texts = ["\n".join(context)]
    for i, row in enumerate(table["data"]):
        texts.append(
            f"Table: {tid}\nRow {i + 1}: "
            + " | ".join(f"{h}: {hybridqa._cell(c)}" for h, c in zip(headers, row, strict=True))
        )
    for link, body in pages.items():
        title = unquote(link.removeprefix("/wiki/")).replace("_", " ")
        texts.append(f"Page: {title}\nLink: {link}\n{body.strip()}")
    return [norm(t) for t in texts]


def mq_group(group):
    """Full local group, originals then updates; gold never constructs memories.

    Group64 and ordering are declared local adaptations, not an official protocol.
    Native loader interleaves each case's originals and edits instead.
    """
    return [norm(mquake._triple_text(*t)) for r in group for t in r["orig_triples_labeled"]] + [
        norm(mquake._update_text(e)) for r in group for e in r["requested_rewrite"]
    ]


def mq_effective(group):
    facts, origins, edited = defaultdict(set), {}, {}
    text_facts, labels_for_relation = defaultdict(set), {}
    for r in group:
        for ids, labels in zip(
            r["orig_triples"] + r["new_triples"],
            r["orig_triples_labeled"] + r["new_triples_labeled"],
            strict=True,
        ):
            prior = labels_for_relation.setdefault(ids[1], labels[1])
            if prior != labels[1]:
                raise ValueError("relation ID has inconsistent labels")
        for ids, labels in zip(r["orig_triples"], r["orig_triples_labeled"], strict=True):
            facts[tuple(ids[:2])].add(ids[2])
            origins[tuple(ids)] = norm(mquake._triple_text(*labels))
            text_facts[tuple(labels[:2])].add(labels[2])
    for r in group:
        subjects = {
            ids[0]: labels[0]
            for ids, labels in zip(
                r["orig_triples"] + r["new_triples"],
                r["orig_triples_labeled"] + r["new_triples_labeled"],
                strict=True,
            )
        }
        for ids, edit in zip(r["edit_triples"], r["requested_rewrite"], strict=True):
            if (
                subjects.get(ids[0], edit["subject"]) != edit["subject"]
                or ids[1] != edit["relation_id"]
                or ids[2] != edit["target_new_id"]
            ):
                raise ValueError("edit and rewrite arrays are misaligned")
            facts[tuple(ids[:2])] = {ids[2]}
            edited[tuple(ids[:2])] = norm(mquake._update_text(edit))
            relation = labels_for_relation[ids[1]]
            text_facts[(edit["subject"], relation)] = {edit["target_new_str"]}
    return facts, origins, edited, text_facts


def build(root, base, work, output):
    c = read(work / "curation.json")
    picks = c["selections"]
    if c["review_status"] != "source-reviewed" or Counter(p["dataset"] for p in picks) != COUNTS:
        raise ValueError("source reviews or quotas missing")
    originals = rows(base / "questions.jsonl")
    if len(originals) != 500:
        raise ValueError("audited base is not the expected 500-question bundle")
    source_rows = {
        "musique": {r["id"]: r for r in musique.read_rows(root) if r["answerable"]},
        "mquake": list(mquake._rows(root / FILES["mquake"])),
        "hybridqa": {r["question_id"]: r for r in read(root / FILES["hybridqa"])},
    }
    mq = {r["case_id"]: r for r in source_rows["mquake"]}
    receipt = read(work / "hybrid-source-receipt.json")
    reference = read(root / "hybridqa/dev_reference.json")["reference"]
    spaces, questions, audits, raw_hybrid = {}, [], [], {}

    def space(dataset, uid, texts, contract, scope):
        sid = f"{VERSION}/{dataset}/{uid}"
        if sid not in spaces:
            sessions = [
                {"session_id": uid, "messages": [{"role": "user", "content": t} for t in texts]}
            ]
            source_hash = fingerprint(sessions)
            for i, m in enumerate(sessions[0]["messages"]):
                m["source_message_id"] = hashlib.sha256(f"{sid}\0{uid}\0{i}".encode()).hexdigest()[
                    :24
                ]
                m["source_message_index"] = i
            spaces[sid] = {
                "space_id": sid,
                "user_id": sid,
                "dataset": dataset,
                "source_dataset": NATIVE[dataset],
                "source_user_id": uid,
                "speaker_names": ["Corpus" if dataset != "mquake" else "user", "assistant"],
                "loading_contract": contract,
                "source_scope": scope,
                "normalized_source_sha256": source_hash,
                "sessions": sessions,
            }
        return spaces[sid]

    def proof(sp, quote, **extra):
        quote = norm(quote)
        ids = [
            m["source_message_id"] for m in sp["sessions"][0]["messages"] if quote in m["content"]
        ]
        if not ids:
            raise ValueError(f"quote missing: {quote[:100]}")
        return {"quote": quote, "source_message_ids": ids, **extra}

    wanted = [p["source_qid"] for p in picks if p["dataset"] == "musique"]
    pool = dict.fromkeys(
        norm(f"Title: {p['title']}\n{norm(p['paragraph_text'])}")
        for qid in wanted
        for p in source_rows["musique"][qid]["paragraphs"]
    )
    msp = space(
        "musique",
        "selected-candidate-haystack",
        list(pool),
        "selected-complete-candidates-title-body-dedup-v1",
        {
            "source_question_ids": wanted,
            "dedup_key": ["title", "paragraph_text"],
            "scope_note": (
                "Local shared candidate pool; all per-question candidates, not full Wikipedia."
            ),
        },
    )
    for i, p in enumerate(picks, 1):
        d, qid = p["dataset"], p["source_qid"]
        q = {
            "id": f"memory100-mh-{i:04d}",
            "dataset": d,
            "source_dataset": NATIVE[d],
            "source_question_id": qid,
            "source_file": FILES[d],
            "task_type": "multi_hop",
            "is_multi_hop": True,
            "is_abstention": False,
            "answerable": True,
            "evaluation_format": "free_response",
            "knowledge_world": "frozen-source-snapshot",
        }
        a = {
            "id": q["id"],
            "dataset": d,
            "source_question_id": qid,
            "task_type": "multi_hop",
            "evidence": [],
            "steps": [],
            "review_status": p["review_status"],
            "review_basis": p["review_basis"],
            "review_note": p.get("review_note", ""),
        }
        if d == "musique":
            r, sp = source_rows[d][qid], msp
            if not musique._is_strict_chain(r["question_decomposition"]):
                raise ValueError("MuSiQue dependency chain is not strict")
            q.update(
                question=r["question"],
                gold={
                    "answer": r["answer"],
                    "answer_aliases": r["answer_aliases"],
                    "answerable": True,
                },
            )
            paras = {x["idx"]: x for x in r["paragraphs"]}
            for j, h in enumerate(r["question_decomposition"], 1):
                para = paras[h["paragraph_support_idx"]]
                a["steps"].append(
                    {
                        "id": f"s{j}",
                        "query_template": h["question"],
                        "answer": h["answer"],
                        "depends_on": [f"s{x}" for x in re.findall(r"#(\d+)", h["question"])],
                        "source_paragraph_idx": para["idx"],
                        "source_title": para["title"],
                        "evidence": [proof(sp, para["paragraph_text"])],
                    }
                )
            if a["steps"][-1]["answer"] != r["answer"]:
                raise ValueError("MuSiQue final node differs from source gold")
        elif d == "mquake":
            r = mq[int(qid.split(":")[1])]
            start = p["group_start"]
            group = source_rows[d][start : start + 64]
            if len(group) != 64 or r not in group:
                raise ValueError("invalid full MQuAKE local group")
            sp = space(
                d,
                f"CF3k-group-{start:05d}",
                mq_group(group),
                "local-group64-originals-then-updates-v1",
                {
                    "group_start": start,
                    "case_ids": [x["case_id"] for x in group],
                    "ordering_note": (
                        "Local group64; all original facts then all rewrites. "
                        "Native loader interleaves per-case facts and updates. "
                        "No gold/new_triples injected."
                    ),
                },
            )
            q.update(
                source_question_id=f"CF3k-{r['case_id']}-0",
                source_case_id=r["case_id"],
                question=r["questions"][0],
                gold=sorted(set([r["new_answer"], *(r.get("new_answer_alias") or [])])),
                knowledge_world="published-counterfactual-after-updates",
            )
            facts, origins, edited, text_facts = mq_effective(group)
            for j, (ids, labels) in enumerate(
                zip(r["new_triples"], r["new_triples_labeled"], strict=True), 1
            ):
                if (
                    facts.get(tuple(ids[:2])) != {ids[2]}
                    or text_facts.get(tuple(labels[:2])) != {labels[2]}
                    or (j > 1 and r["new_triples"][j - 2][2] != ids[0])
                ):
                    raise ValueError(f"missing/conflicting MQuAKE hop: {qid}/{j}")
                quote = edited.get(tuple(ids[:2]), origins.get(tuple(ids)))
                a["steps"].append(
                    {
                        "id": f"s{j}",
                        "subject": labels[0],
                        "relation": labels[1],
                        "answer": labels[2],
                        "triple_ids": ids,
                        "depends_on": [] if j == 1 else [f"s{j - 1}"],
                        "evidence": [proof(sp, quote)],
                    }
                )
            if a["steps"][-1]["answer"] not in q["gold"]:
                raise ValueError("MQuAKE final node differs from source gold")
        else:
            r = source_rows[d][qid]
            tid = r["table_id"]
            docs = []
            for kind in ("tables_tok", "request_tok"):
                loc = receipt["tables"][tid][kind]
                if sha(loc["path"]) != loc["sha256"]:
                    raise ValueError("portable raw HybridQA snapshot changed")
                docs.append(read(loc["path"]))
            table, pages = docs
            texts = corpus_texts(tid, table, pages)
            sp = space(
                d,
                hashlib.sha256(tid.encode()).hexdigest()[:20],
                texts,
                "hybridqa-table-passages-v1-portable-paths",
                {
                    "table_id": tid,
                    "table_rows": len(table["data"]),
                    "linked_passages": len(pages),
                    "source_members": receipt["tables"][tid],
                },
            )
            raw_hybrid[tid] = {
                "table": table,
                "pages": pages,
                "source_members": receipt["tables"][tid],
            }
            if r["answer-text"] != reference[qid]:
                raise ValueError("HybridQA gold and reference disagree")
            q.update(question=r["question"], gold={"answer": r["answer-text"]}, table_id=tid)
            for j, s in enumerate(p["reviewed_steps"], 1):
                proofs = []
                for ref in s["refs"]:
                    quote = texts[1 + ref["index"]] if ref["kind"] == "row" else pages[ref["link"]]
                    proofs.append(proof(sp, quote, source_locator=ref))
                a["steps"].append(
                    {
                        "id": f"s{j}",
                        "relation": s["relation"],
                        "answer": s["answer"],
                        "depends_on": [] if j == 1 else [f"s{j - 1}"],
                        "evidence": proofs,
                    }
                )
            if a["steps"][-1]["answer"] != r["answer-text"]:
                raise ValueError("HybridQA reviewed final node differs from source gold")
        q.update(space_id=sp["space_id"], user_id=sp["user_id"], hops=len(a["steps"]))
        a["space_id"] = sp["space_id"]
        questions.append(q)
        audits.append(a)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("questions.jsonl", questions), ("question-audits.jsonl", audits)):
        (output / name).write_text(
            "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8"
        )
    with gzip.open(output / "spaces.jsonl.gz", "wt", encoding="utf-8") as handle:
        for sp in spaces.values():
            handle.write(json.dumps(sp, ensure_ascii=False) + "\n")
    with gzip.open(output / "hybrid-full-source.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(raw_hybrid, handle, ensure_ascii=False)
    write(output / "curation.json", c)
    source_files = {
        name: {"path": str((root / name).resolve()), "sha256": sha(root / name)}
        for name in [
            *FILES.values(),
            "hybridqa/dev_reference.json",
            "hybridqa/wiki_tables_with_links.tar.gz",
        ]
    }
    urls = {k: v for k, v in read(base / "manifest.json")["source_urls"].items() if k in COUNTS}
    manifest = {
        "version": VERSION,
        "questions": 100,
        "base_bundle": str(base.resolve()),
        "base_snapshot": read(work / "base-snapshot.json"),
        "source_files": source_files,
        "source_urls": urls,
        "dataset_counts": dict(Counter(q["dataset"] for q in questions)),
        "space_scope_fingerprints": {
            sid: sp["normalized_source_sha256"] for sid, sp in spaces.items()
        },
        "output_sha256": {
            name: sha(output / name)
            for name in (
                "questions.jsonl",
                "question-audits.jsonl",
                "spaces.jsonl.gz",
                "hybrid-full-source.json.gz",
                "curation.json",
            )
        },
        "reviewer": c["reviewer"],
        "no_add_or_model_calls": True,
        "combination_note": (
            "Separate supplement only. No 600-question merge is written; "
            "original 500 remains under independent audit."
        ),
    }
    write(output / "manifest.json", manifest)
    validate(root, base, work, output)


def validate(root, base, work, output):
    manifest = read(output / "manifest.json")
    for name, expected in manifest["output_sha256"].items():
        if sha(output / name) != expected:
            raise ValueError(f"artifact changed: {name}")
    for name, loc in manifest["source_files"].items():
        if sha(root / name) != loc["sha256"]:
            raise ValueError(f"source file changed: {name}")
    snapshot = manifest["base_snapshot"]
    for name, expected in snapshot["sha256"].items():
        if sha(base / name) != expected:
            raise ValueError(f"audited base changed since snapshot: {name}; do not overwrite it")
    original_qs = rows(base / "questions.jsonl")
    qs, audits = rows(output / "questions.jsonl"), rows(output / "question-audits.jsonl")
    with gzip.open(output / "spaces.jsonl.gz", "rt", encoding="utf-8") as handle:
        spaces = [json.loads(s) for s in handle]
    with gzip.open(output / "hybrid-full-source.json.gz", "rt", encoding="utf-8") as handle:
        hybrid = json.load(handle)
    if len(qs) != 100 or Counter(q["dataset"] for q in qs) != COUNTS or len(audits) != 100:
        raise ValueError("supplement quotas differ")
    if len({q["id"] for q in qs + original_qs}) != 600:
        raise ValueError("question IDs overlap the base or each other")
    if sum(q["is_multi_hop"] for q in original_qs) != 100:
        raise ValueError("base multi-hop count changed during its separate audit")
    old_keys = {(q["dataset"], q["source_question_id"]) for q in original_qs}
    old_texts = {query_key(q["question"]) for q in original_qs}
    if (
        len({(q["dataset"], q["source_question_id"]) for q in qs}) != 100
        or any(
            (q["dataset"], q["source_question_id"]) in old_keys
            or query_key(q["question"]) in old_texts
            for q in qs
        )
        or len({query_key(q["question"]) for q in qs}) != 100
    ):
        raise ValueError("source question or normalized text duplicate")
    prior = read(base / "curation.json")["selections"]
    old_cases = {p["source_qid"] for p in prior if p["dataset"] == "mquake"}
    c = read(output / "curation.json")
    if any(p["source_qid"] in old_cases for p in c["selections"] if p["dataset"] == "mquake"):
        raise ValueError("MQuAKE case paraphrase already selected")
    old_audits = rows(base / "question-audits.jsonl")

    def chain_key(a):
        return tuple(
            (query_key(s.get("query_template", "")), query_key(s["answer"])) for s in a["steps"]
        )

    old_chains = {chain_key(a) for a in old_audits if a["dataset"] == "musique"}
    new_chains = [chain_key(a) for a in audits if a["dataset"] == "musique"]
    if len(set(new_chains)) != len(new_chains) or set(new_chains) & old_chains:
        raise ValueError("same MuSiQue dependency questions and outputs already selected")
    mr = {r["id"]: r for r in musique.read_rows(root) if r["answerable"]}
    mqrows = list(mquake._rows(root / FILES["mquake"]))
    mqmap = {r["case_id"]: r for r in mqrows}
    hr = {r["question_id"]: r for r in read(root / FILES["hybridqa"])}
    proof_map, clean_spaces = {}, {}
    old_spaces = set(read(base / "manifest.json")["space_scope_fingerprints"])
    for sp in spaces:
        if sp["space_id"] in old_spaces or sp["user_id"] != sp["space_id"]:
            raise ValueError("space isolation collision")
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
        if fingerprint(clean) != manifest["space_scope_fingerprints"][sp["space_id"]]:
            raise ValueError("full source scope fingerprint differs")
        actual = [m["content"] for s in clean for m in s["messages"]]
        scope = sp["source_scope"]
        if sp["dataset"] == "musique":
            expected = list(
                dict.fromkeys(
                    norm(f"Title: {p['title']}\n{norm(p['paragraph_text'])}")
                    for qid in scope["source_question_ids"]
                    for p in mr[qid]["paragraphs"]
                )
            )
        elif sp["dataset"] == "mquake":
            start = scope["group_start"]
            group = mqrows[start : start + 64]
            if scope["case_ids"] != [r["case_id"] for r in group]:
                raise ValueError("full original group scope differs")
            expected = mq_group(group)
        else:
            h = hybrid[scope["table_id"]]
            expected = corpus_texts(scope["table_id"], h["table"], h["pages"])
            if len(actual) != 1 + len(h["table"]["data"]) + len(h["pages"]):
                raise ValueError("whole table rows or linked pages missing")
        if actual != expected:
            raise ValueError("serialized scope differs from whole raw scope")
        clean_spaces[sp["space_id"]] = sp
        for s in sp["sessions"]:
            for m in s["messages"]:
                if (
                    set(m) != {"role", "content", "source_message_id", "source_message_index"}
                    or m["role"] != "user"
                ):
                    raise ValueError("Add payload includes audit annotations")
                if m["source_message_id"] in proof_map:
                    raise ValueError("source message IDs collide")
                proof_map[m["source_message_id"]] = (sp["space_id"], m["content"])
    for q, a in zip(qs, audits, strict=True):
        if q["id"] != a["id"] or not q["answerable"] or q["is_abstention"] or len(a["steps"]) < 2:
            raise ValueError("invalid answerable multi-hop review")
        if q["dataset"] == "musique":
            r = mr[q["source_question_id"]]
            expected_gold = {
                "answer": r["answer"],
                "answer_aliases": r["answer_aliases"],
                "answerable": True,
            }
        elif q["dataset"] == "mquake":
            r = mqmap[q["source_case_id"]]
            expected_gold = sorted(set([r["new_answer"], *(r.get("new_answer_alias") or [])]))
            r = {**r, "question": r["questions"][0]}
            sp = clean_spaces[q["space_id"]]
            start = sp["source_scope"]["group_start"]
            facts, _, _, labels = mq_effective(mqrows[start : start + 64])
            for ids, label in zip(r["new_triples"], r["new_triples_labeled"], strict=True):
                if facts.get(tuple(ids[:2])) != {ids[2]} or labels.get(tuple(label[:2])) != {
                    label[2]
                }:
                    raise ValueError("effective updated chain unsupported")
        else:
            r = hr[q["source_question_id"]]
            expected_gold = {"answer": r["answer-text"]}
        if q["question"] != r["question"] or q["gold"] != expected_gold:
            raise ValueError("original question or gold altered")
        for j, step in enumerate(a["steps"], 1):
            if step["id"] != f"s{j}" or (j > 1 and not step["depends_on"]):
                raise ValueError("dependent step missing")
            if any(dep not in {f"s{k}" for k in range(1, j)} for dep in step["depends_on"]):
                raise ValueError("dependency points forward or outside plan")
            for proof in step["evidence"]:
                for mid in proof["source_message_ids"]:
                    sid, body = proof_map[mid]
                    if sid != q["space_id"] or proof["quote"] not in body:
                        raise ValueError("proof missing or crosses source isolation")
    # Independently reconcile the portable raw snapshots to the fixed upstream tar.
    archive = root / "hybridqa/wiki_tables_with_links.tar.gz"
    entry = next(e for e in MANIFEST if e["name"] == "hybridqa/wiki_tables_with_links.tar.gz")
    if sha(archive) != entry["sha256"]:
        raise ValueError("official fixed archive hash mismatch")
    wanted = {
        loc["archive_member"]: loc["sha256"]
        for h in hybrid.values()
        for loc in h["source_members"].values()
    }
    expected_members = {
        loc["archive_member"]: h["table" if kind == "tables_tok" else "pages"]
        for h in hybrid.values()
        for kind, loc in h["source_members"].items()
    }
    seen = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name in wanted:
                body = tar.extractfile(member).read()
                if hashlib.sha256(body).hexdigest() != wanted[member.name]:
                    raise ValueError("portable snapshot differs from original archive member")
                if fingerprint(json.loads(body)) != fingerprint(expected_members[member.name]):
                    raise ValueError("embedded whole source differs from original archive JSON")
                seen.add(member.name)
    if seen != set(wanted):
        raise ValueError("whole table or whole page map absent in original archive")
    result = {
        "status": "passed",
        "questions": 100,
        "answerable_multi_hop": 100,
        "abstention": 0,
        "dataset_counts": dict(Counter(q["dataset"] for q in qs)),
        "hop_counts": dict(Counter(q["hops"] for q in qs)),
        "isolated_spaces": len(spaces),
        "memory_source_messages": len(proof_map),
        "base_existing_files_unchanged": True,
        "base_question_overlap": 0,
        "normalized_question_overlap": 0,
        "mquake_case_paraphrase_overlap": 0,
        "musique_decomposition_signature_overlap": 0,
        "space_id_overlap": 0,
        "source_question_gold_exact_matches": 100,
        "whole_scopes_verified": len(spaces),
        "original_hybrid_archive_members_verified": len(seen),
        "cross_space_or_missing_proofs": 0,
        "add_search_model_calls": 0,
        "combined_manifest_counts_only": {
            "questions": 600,
            "multi_hop": 200,
            "other": 400,
            "note": "Base semantic qualification remains subject to its ongoing independent audit.",
        },
    }
    write(work / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "validate"))
    parser.add_argument("--root", type=Path, default=Path(benchmark_dir()))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (build if args.action == "build" else validate)(args.root, args.base, args.work, args.output)


if __name__ == "__main__":
    main()
