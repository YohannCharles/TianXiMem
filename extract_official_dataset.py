#!/usr/bin/env python3
"""从 requests*.jsonl 采集原文里抽出**AML 官方**的 /add 与 /search，生成可复放的数据集。

官方流量的判据（见 README 报告「口径」一节）：
  client ∈ {221.194.152.144, 221.194.152.241, 221.194.152.171, 111.205.43.252}
  且 user_id 形如 u_<64 hex>
本机自测（127.0.0.1 的 pf-*、172.28.0.1 的 verify-*）**全部剔除**。

用法：python3 extract.py [采集文件所在目录] [输出目录]
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

OFFICIAL_CLIENTS = {
    "221.194.152.144",
    "221.194.152.241",
    "221.194.152.171",
    "111.205.43.252",
}
USER_ID_RE = re.compile(r"^u_[0-9a-f]{64}$")

# 官方本轮的**开场 11 分钟**只在这份旧导出里有：09:56:39.969Z–10:07:52.318Z，24 个 add 被反复投递
# 328 次、**全部失败**（当时部署版本的 ingest.chunk_ordinal_pattern 匹配不上 r_<64hex> 形式的
# request_id，`pairing.parse_chunk_ordinal` 响亮失败）。采集分片 p1–p7 最早只到 13:13:00Z，
# 所以**必须**把这份并进来，否则数据集的时间窗与重试史都是错的。
# ⚠ 它与采集分片的行格式不同：一行 = 一个 rid，把该窗口内的投递次数压进了 `retry_count`。
EARLY_EXPORT = "official-add-bodies-2026-09-29.jsonl"

OPTION_RE = re.compile(r"^([A-Z])\.\s+(.*)$")

# 官方 /search 的 query 末尾是一项**作答指令**，它同时也是题型标签。
# 判据用 `enclosed in parentheses`（实测 1894 条，与题型分类的并集完全吻合），
# 不要用「题干里出现 A. B. C.」——PDF 正文里的 "G. Levine et al." 会被误判成选项
# （实测有 291 条这类假阳性）。
INSTR_RE = re.compile(
    r"\n\s*Please [^\n]*enclosed in parentheses[^\n]*$",
    re.S,
)
TASK_KINDS = (
    ("only correct answer", "single_choice"),
    ("all correct options", "multi_select"),
    ("options in the correct order", "ordering"),
    ("correct order of options", "ordering"),
)


def parse_query(query: str) -> dict:
    """切成 题干 / 选项 / 题型。非选择题 options 为空、task='open'。"""
    text = (query or "").rstrip()
    m = INSTR_RE.search(text)
    if not m:
        return {"question": text, "options": [], "task": "open", "instruction": None}
    instruction = m.group(0).strip()
    body = text[: m.start()].rstrip()
    task = next((k for needle, k in TASK_KINDS if needle in instruction), "other_mcq")

    # 选项从 "A. " 开始，且字母必须**从 A 起连续**——这条约束就是上面那 291 条
    # 假阳性的过滤器（PDF 正文里的 G./W./D. 不连续）。
    lines = body.split("\n")
    start = next(
        (i for i, ln in enumerate(lines) if OPTION_RE.match(ln.strip()) and ln.strip()[0] == "A"),
        None,
    )
    if start is None:
        return {"question": body, "options": [], "task": task, "instruction": instruction}

    stem = "\n".join(lines[:start]).strip()
    options: list[dict] = []
    next_letter = "A"
    for ln in lines[start:]:
        s = ln.strip()
        om = OPTION_RE.match(s)
        if om and om.group(1) == next_letter:
            options.append({"letter": om.group(1), "text": om.group(2).strip()})
            next_letter = chr(ord(next_letter) + 1)
        elif options and s:
            options[-1]["text"] += " " + s  # 同一条选项的续行
    if len(options) < 2:
        return {"question": body, "options": [], "task": task, "instruction": instruction}
    return {"question": stem, "options": options, "task": task, "instruction": instruction}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 只认采集原文分片（requests.jsonl / requests.partN.jsonl 及其导出副本 p1..pN），
    # 外加记录官方开场那 11 分钟的旧导出 EARLY_EXPORT（缺了它时间窗就错，见该常量的注释）。
    # ⚠ 不要用 official-add-bodies-*.jsonl 这种宽通配：根上还有一份 ...-full.jsonl，
    # 它是 p1 的前 742 条子集，并进来会把那 742 条重复计数。
    src_files = sorted(
        set(SRC_DIR.glob("requests*.jsonl"))
        | set(SRC_DIR.glob("official-add-bodies-*-p[0-9].jsonl"))
        | {p for p in [SRC_DIR / EARLY_EXPORT] if p.exists()}
    )
    if not src_files:
        print(f"✗ {SRC_DIR} 下没有采集原文分片（requests*.jsonl）", file=sys.stderr)
        return 1

    adds_by_rid: dict[str, dict] = {}
    add_order: list[str] = []
    searches: list[dict] = []
    dropped = Counter()
    seen_files = []

    for path in src_files:
        seen_files.append(path.name)
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") != "req":
                continue
            client = rec.get("client")
            body = rec.get("body")
            if client not in OFFICIAL_CLIENTS or not isinstance(body, dict):
                dropped["非官方 client"] += 1
                continue
            uid = body.get("user_id")
            if not USER_ID_RE.match(str(uid)):
                dropped["user_id 不合官方形状"] += 1
                continue

            common = {
                "ts": rec["ts"],
                "client": client,
                "status": rec.get("status"),
                "error": rec.get("error"),
                "latency_ms": rec.get("latency_ms"),
            }

            if rec["path"] == "/add":
                rid = body.get("request_id")
                # 旧导出（官方开场那 11 分钟）是一行一个 rid，把该窗口内的投递次数压缩进了
                # `retry_count`；采集分片则是一行一次投递。两者都要计入总投递数。
                n_deliveries = 1
                if path.name == EARLY_EXPORT:
                    n_deliveries = int(rec.get("retry_count") or 1)

                prev = adds_by_rid.get(rid)
                if prev is not None:
                    prev["deliveries"] += n_deliveries
                    prev["last_ts"] = max(prev["last_ts"], rec["ts"])
                    prev["ts"] = min(prev["ts"], rec["ts"])
                    prev["retry_count"] = prev["deliveries"] - 1
                    # 只信「成功那一次」的 status/latency——首投失败、重试成功时，
                    # 拿失败那次的 latency 会得到荒谬的数字（实测差 35 倍）。
                    if rec.get("status") == 200 or not prev["succeeded"]:
                        prev["succeeded"] = prev["succeeded"] or rec.get("status") == 200
                        prev["status"] = rec.get("status")
                        prev["error"] = rec.get("error")
                        prev["latency_ms"] = rec.get("latency_ms")
                    continue
                msgs = body.get("messages") or []
                stamps = [
                    m.get("timestamp")
                    for m in msgs
                    if isinstance(m, dict) and isinstance(m.get("timestamp"), (int, float))
                ]
                adds_by_rid[rid] = {
                    **common,
                    "request_id": rid,
                    "user_id": uid,
                    "session_id": body.get("session_id"),
                    "messages": msgs,
                    "n_messages": len(msgs),
                    "deliveries": n_deliveries,
                    "retry_count": n_deliveries - 1,
                    "last_ts": rec["ts"],
                    "succeeded": rec.get("status") == 200,
                    "msg_ts_min": min(stamps) if stamps else None,
                    "msg_ts_max": max(stamps) if stamps else None,
                }
                add_order.append(rid)
            elif rec["path"] == "/search":
                q = parse_query(body.get("query", ""))
                searches.append(
                    {
                        **common,
                        "user_id": uid,
                        "query": body.get("query", ""),
                        "top_k": body.get("top_k"),
                        "options_field": body.get("options"),
                        "question": q["question"],
                        "options": q["options"],
                        "task": q["task"],
                        "instruction": q["instruction"],
                    }
                )
            else:
                dropped["非 add/search"] += 1

    adds = [adds_by_rid[r] for r in add_order]

    # ── 时间线：adds + searches 按 ts 归并，标注每条 search 之前该用户已有多少条 add ──
    stream = [(a["ts"], 0, a) for a in adds] + [(s["ts"], 1, s) for s in searches]
    stream.sort(key=lambda x: (x[0], x[1]))
    prior = defaultdict(int)
    timeline = []
    for seq, (ts, kind, rec) in enumerate(stream):
        if kind == 0:
            prior[rec["user_id"]] += 1
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "kind": "add",
                    "user_id": rec["user_id"],
                    "session_id": rec["session_id"],
                    "request_id": rec["request_id"],
                    "n_messages": rec["n_messages"],
                    "status": rec["status"],
                    "prior_adds_for_user": prior[rec["user_id"]] - 1,
                }
            )
        else:
            rec["prior_adds_for_user"] = prior[rec["user_id"]]
            rec["seq"] = seq
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "kind": "search",
                    "user_id": rec["user_id"],
                    "top_k": rec["top_k"],
                    "task": rec["task"],
                    "status": rec["status"],
                    "prior_adds_for_user": rec["prior_adds_for_user"],
                    "query": rec["query"],
                }
            )

    # ── 写文件 ──
    def dump(name, rows):
        with (OUT_DIR / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    n_adds = dump("official-adds.jsonl", adds)
    n_searches = dump("official-searches.jsonl", searches)
    n_tl = dump("official-timeline.jsonl", timeline)
    mcq = [s for s in searches if s["task"] != "open"]
    n_mcq = dump("official-qa-mcq.jsonl", mcq)

    users = {a["user_id"] for a in adds} | {s["user_id"] for s in searches}
    users_with_adds = {a["user_id"] for a in adds}
    users_with_search = {s["user_id"] for s in searches}

    source_fingerprints = [
        {
            "name": p.name,
            "bytes": p.stat().st_size,
            "lines": sum(1 for _ in p.open(encoding="utf-8")),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in src_files
    ]

    manifest = {
        "generated_from": seen_files,
        # 采集是**边跑边写**的：这份数据集是下面这个时刻的快照，源文件之后还会继续长。
        # 用 (bytes, sha256) 钉住出处，下次重跑就能判断有没有漂移。
        "snapshot": {
            "last_official_request_utc": timeline[-1]["ts"],
            "source_files": source_fingerprints,
        },
        "official_clients": sorted(OFFICIAL_CLIENTS),
        "rule": "client ∈ 官方 4 个出口 IP，且 user_id 匹配 u_<64hex>；自测流量已剔除",
        "dropped_non_official": dict(dropped),
        "time_window": {"first": timeline[0]["ts"], "last": timeline[-1]["ts"]},
        "counts": {
            "adds_unique": n_adds,
            "adds_with_retry": sum(1 for a in adds if a["retry_count"] > 0),
            "add_retry_redeliveries": sum(a["retry_count"] for a in adds),
            "searches": n_searches,
            "searches_mcq": n_mcq,
            "searches_open": n_searches - n_mcq,
            "timeline": n_tl,
        },
        "search_tasks": dict(Counter(s["task"] for s in searches)),
        "users": {
            "total": len(users),
            "with_adds": len(users_with_adds),
            "with_search": len(users_with_search),
            "search_without_any_add": len(users_with_search - users_with_adds),
        },
        "status": {
            "adds": dict(Counter(str(a["status"]) for a in adds)),
            "searches": dict(Counter(str(s["status"]) for s in searches)),
        },
        "top_k": dict(Counter(str(s["top_k"]) for s in searches)),
        "files": {
            "official-adds.jsonl": n_adds,
            "official-searches.jsonl": n_searches,
            "official-timeline.jsonl": n_tl,
            "official-qa-mcq.jsonl": n_mcq,
        },
    }
    (OUT_DIR / "official-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
