#!/usr/bin/env python3
"""把官方采集数据集接上**可判分的标签**，产出评测用文件。

两类标签来源：
  A. 官方内嵌判分（唯一来自 AML 自己的信号）——`/add` 的 message 里混着平台回写的
     "Streaming online QA feedback" / "Streaming intra-transfer train trajectory"，
     带 predicted_answer / score / qa_id，question 与该 user 的某条 search 对应。
  B. 公开金标回捞——语料本身来自公开数据集（ScriptMem / LoCoMo-Refined / PersonaMem-v2），
     用归一化后的题面到公开版本里逐字匹配，取回标准答案。

用法：python3 build_eval_set.py [数据集目录] [公开金标目录]
"""

import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DS = Path(sys.argv[1] if len(sys.argv) > 1 else "official-dataset-2026-09-29")
GOLD = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gold")

QA_FEEDBACK_PREFIX = "Streaming online QA feedback:"
TRAJ_FEEDBACK_PREFIX = "Streaming intra-transfer train trajectory:"


def norm(text: str) -> str:
    """题面归一化：NFKC + 转小写 + 空白折叠。两侧都用它，否则匹配率会假低。"""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


# 公开金标的 `question` 字段**自带选项**（"…stem…\nA. …\nB. …"），而我们从 query 里
# 解析出来的 question 只有题干。不切齐就会全部匹配失败（实测 ScriptMem 0 命中）。
_OPT_LINE_RE = re.compile(r"(?:^|\s)([A-Za-z])[.、]\s")


def ts_key(t: str) -> str:
    return t  # ISO8601 同格式，字符串序即时间序


def _secs(iso: str) -> float:
    """ISO8601 → epoch 秒（只用于取最近邻，精度够用）。"""
    from datetime import datetime

    return datetime.fromisoformat(iso).timestamp()


def stem_of(text: str) -> str:
    """切掉选项块，只留题干。大小写都认。"""
    lines = (text or "").split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if len(s) > 2 and s[0] in "Aa" and s[1] == "." and s[2] == " ":
            return "\n".join(lines[:i]).strip()
    return (text or "").strip()


def extract_feedback(adds_path: Path):
    """从 /add 的消息里抽出平台回写的判分记录。"""
    qa, traj, broken = [], [], Counter()
    for line in adds_path.open(encoding="utf-8"):
        rec = json.loads(line)
        for msg in rec.get("messages") or []:
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            for prefix, bucket in (
                (QA_FEEDBACK_PREFIX, qa),
                (TRAJ_FEEDBACK_PREFIX, traj),
            ):
                if not content.startswith(prefix):
                    continue
                payload = content[len(prefix) :].strip()
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    # 实测有 8 条轨迹记录在 8000 字符处被截断
                    broken[prefix] += 1
                    continue
                bucket.append(
                    {
                        "add_ts": rec["ts"],
                        "user_id": rec["user_id"],
                        "request_id": rec["request_id"],
                        **parsed,
                    }
                )
                break
    return qa, traj, dict(broken)


def load_public_gold(gold_dir: Path):
    """公开金标 → {归一化题面: 标签}。三家格式各不相同，逐个适配。"""
    out: dict[str, dict] = {}
    counts = Counter()

    sm = gold_dir / "scriptmem_questions.jsonl"
    if sm.exists():
        for line in sm.open(encoding="utf-8"):
            r = json.loads(line)
            out.setdefault(norm(stem_of(r["question"])), {}).update(
                {
                    "gold_scriptmem_answer_letters": r.get("answer_letters"),
                    "gold_scriptmem_answer": r.get("answer"),
                    "gold_scriptmem_qa_id": r.get("qa_id"),
                    "gold_scriptmem_source": r.get("source"),
                    "gold_scriptmem_qa_type": r.get("qa_type"),
                }
            )
            counts["scriptmem"] += 1

    lc = gold_dir / "locomo_refined_questions.jsonl"
    if lc.exists():
        for line in lc.open(encoding="utf-8"):
            r = json.loads(line)
            out.setdefault(norm(stem_of(r["question"])), {}).update(
                {
                    "gold_locomo_answer": r.get("answer"),
                    "gold_locomo_qa_id": r.get("qa_id"),
                    "gold_locomo_category": r.get("category"),
                    "gold_locomo_is_multi_modality": r.get("is_multi_modality"),
                    "gold_locomo_evidence": r.get("evidence"),
                }
            )
            counts["locomo"] += 1

    pm = gold_dir / "personamem_benchmark.csv"
    if pm.exists():
        with pm.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                query = row.get("user_query") or ""
                # 这一列是 Python dict 的 repr 串，取里面的 content
                content = query
                try:
                    content = eval(query, {"__builtins__": {}}).get("content", query)  # noqa: S307
                except Exception:  # noqa: BLE001 — repr 千奇百怪，抽不出就退回原文
                    m = re.search(r"'content':\s*'(.*?)',?\s*\}", query, re.S)
                    if m:
                        content = m.group(1)
                out.setdefault(norm(content), {}).update(
                    {
                        "gold_personamem_answer": row.get("correct_answer"),
                        "gold_personamem_persona_id": row.get("persona_id"),
                    }
                )
                counts["personamem"] += 1
    return out, dict(counts)


def main() -> int:
    adds_path = DS / "official-adds.jsonl"
    searches_path = DS / "official-searches.jsonl"
    if not adds_path.exists() or not searches_path.exists():
        print(f"✗ {DS} 下缺 official-adds.jsonl / official-searches.jsonl", file=sys.stderr)
        return 1

    qa, traj, broken = extract_feedback(adds_path)
    gold, gold_counts = load_public_gold(GOLD)
    searches = [json.loads(line) for line in searches_path.open(encoding="utf-8")]

    # ── 官方判分：按 (user, 题面) 取候选，再用**时间**定序 ──
    # 关键：同一道题会在不同记忆阶段被反复问，每次的 score 都可能不同。只按题面匹配会
    # 一对多（实测 930 条歧义），取哪条都是错的。判分是在对应 /search 之后约 3 秒被
    # 回写进 /add 的，所以正解是「同 user、同题面、且 add_ts 不早于 search.ts 的最近一条」。
    qa_idx: dict[tuple[str, str], list[dict]] = {}
    for q in qa:
        # ⚠ 两侧都要 stem_of：选择题的 feedback.question 与我们解析出的 question
        # 一个带选项、一个不带，不切齐就会整族漏配（实测两个 ScriptMem 大用户全丢）。
        qa_idx.setdefault((q["user_id"], norm(stem_of(q.get("question", "")))), []).append(q)
    for v in qa_idx.values():
        v.sort(key=lambda x: ts_key(x["add_ts"]))

    matched = 0
    ambiguous = 0
    no_time_match = 0
    for s in searches:
        key = (s["user_id"], norm(s["question"]))
        hits = qa_idx.get(key)
        if not hits:
            continue
        after = [h for h in hits if ts_key(h["add_ts"]) >= ts_key(s["ts"])]
        if len(hits) > 1:
            ambiguous += 1
        if not after:
            # 判分早于提问（同题重问、回写落在上一次）：退回时间上最近的一条
            no_time_match += 1
            h = min(hits, key=lambda x: abs(_secs(ts_key(x["add_ts"])) - _secs(ts_key(s["ts"]))))
        else:
            h = after[0]
        matched += 1
        s["official_score"] = h.get("score")
        s["official_qa_id"] = h.get("qa_id")
        s["official_method"] = h.get("method")
        s["official_predicted_answer"] = h.get("predicted_answer")
        s["official_feedback_ts"] = h["add_ts"]
        s["official_n_candidates"] = len(hits)
        # score==1 时平台的答案被判为正确——但**两类判分含义不同**（对抗验证实测）：
        #   *_exact  —— 答案必须是选项集的规范形式，pred 就是官方答案本身
        #   llm_judge —— 自由文本被裁判接受；同一题实测有 18 组出现 ≥2 个不同文本都判 1
        #                ⇒ pred 是「一个被接受的正确答案」，不是唯一标准答案
        judge_kind = "exact" if str(h.get("method", "")).endswith("exact") else "llm_judge"
        s["official_judge_kind"] = judge_kind
        s["official_accepted_answer"] = h.get("predicted_answer") if h.get("score") == 1 else None
        s["gold_official_answer"] = (
            h.get("predicted_answer") if (h.get("score") == 1 and judge_kind == "exact") else None
        )
        # 多阶段：qa_id 形如 <dataset>:qNNNN:stepMM
        m = re.search(r":step(\d+)$", str(h.get("qa_id") or ""))
        s["official_step"] = int(m.group(1)) if m else None
    unmatched = len(searches) - matched

    # ── 公开金标：按归一化题面匹配 ──
    pub = 0
    for s in searches:
        g = gold.get(norm(s["question"]))
        if g:
            s["public_gold"] = g
            pub += 1

    for s in searches:
        s.setdefault("official_score", None)
        s.setdefault("gold_official_answer", None)
        s.setdefault("public_gold", None)

    def dump(name, rows):
        with (DS / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    dump("official-feedback-labels.jsonl", qa + traj)
    n_eval = dump("official-eval-questions.jsonl", searches)
    assert n_eval == len(searches)

    # 推荐评测子集：去掉失败行与"查询时还没有记忆"的行，留下带标签的
    subset = [
        s
        for s in searches
        if s["status"] == 200
        and s["prior_adds_for_user"] > 0
        and (s["official_score"] is not None or s["public_gold"] is not None)
    ]
    dump("official-eval-subset.jsonl", subset)
    scores = Counter(str(q.get("score")) for q in qa)
    rewards = Counter(str((q.get("eval_feedback") or {}).get("reward")) for q in traj)
    summary = {
        "official_feedback": {
            "qa_records": len(qa),
            "trajectory_records": len(traj),
            "unparseable_truncated": broken,
            "qa_score_distribution": dict(scores),
            "qa_accuracy": (sum(1 for q in qa if q.get("score") == 1) / len(qa) if qa else None),
            "trajectory_reward_distribution": dict(rewards),
            "qa_methods": dict(Counter(q.get("method") for q in qa)),
            "distinct_users": len({q["user_id"] for q in qa}),
        },
        "join": {
            "searches_total": len(searches),
            "searches_with_official_score": matched,
            "searches_with_official_accepted_answer": sum(
                1 for s in searches if s.get("official_accepted_answer")
            ),
            "searches_with_official_gold_answer_exact_only": sum(
                1 for s in searches if s.get("gold_official_answer")
            ),
            "official_judge_kind": dict(
                Counter(
                    s.get("official_judge_kind") for s in searches if s.get("official_judge_kind")
                )
            ),
            "ambiguous_multiple_feedback": ambiguous,
            "feedback_matched_by_fallback_not_after": no_time_match,
            "searches_without_official_feedback": unmatched,
            "searches_with_public_gold": pub,
            "recommended_subset_rows": len(subset),
            "public_gold_source_rows": gold_counts,
        },
        "note": (
            "official_score / gold_official_answer 来自 AML 自己回写进 /add 的判分记录；"
            "public_gold 来自公开数据集逐字匹配（见 README）。两者都不是本服务产生的。"
        ),
    }
    (DS / "official-eval-manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
