#!/usr/bin/env python3
"""给官方采集的 persona 访谈段接上 HaluMem-Medium 金标。

## 为什么要在意这一段

[`../official-dataset-2026-09-29/README.md`] §4.2 的普查里，最大一族是
`user:` / `assistant:` 的 persona 分享体（32,444 条 add），当时只能记成
"没有足够特征把它归属到某一个具体数据集"。本脚本把其中**可归属的子集**钉死：
add 开头是 `Hello! I'd like to share my basic personal information ...`
（同一段文字也出现为 `user: Hello! ...` 与 `[Time: ...] Hello! ...` 两种包装）
的 186 个采集 user，逐字命中 **HaluMem-Medium**（MemTensor）。

## 三个互相独立的检测器，集合完全一致（"逐字"的证据）

| 检测器 | 判据 | user 数 |
| --- | --- | --- |
| a 开头语 | 该 user 的 add 含 `share my basic personal information` | 186 |
| b 正文 | 该 user **全部** add message 去包装后命中 HaluMem 的 session dialogue | 186 |
| c 题面 | 该 user 的 search question 逐字命中 HaluMem 的 per-session questions | 186 |

实测：a=b=c=186，差集全空；命中 search 共 **1,481** 条。user → persona：
Sarah Garcia 50 / Taylor David 48 / Williams Donna 44 / Joseph Garcia 44
（HaluMem-Medium 共 20 个 persona，采集只回放了这 4 个）。
b 的阈值远在两个分布中间：命中 user 全部 100% 命中，非命中 user 实测最高 0%。

## 上游判分口径（逐字引用，不发布精确匹配打分器）

HaluMem 仓库对 QA **不发布**字符串匹配打分器，只发一份 **LLM judge** 提示词：
`eval/eval_tools.py` 的 `EVALUATION_PROMPT_FOR_QUESTION`
（MemTensor/HaluMem，main @ `718f16ff0c83413b1c86fa83fc13cc1a639871f9`，
该文件 sha256 `0c08e5ecb8c93945bafc4bd0336bd6c9756b40d175f442ce44aca4a43169ee3b`）。
`eval/evaluation.py` 用 (question, answer, evidence 的 memory_content 换行拼接, 系统作答)
调用它，回 `Correct | Hallucination | Omission`；**语义等价即 Correct**。
`eval/prompts.py` 里的 `PROMPT_*` 是**作答侧**提示词，不是裁判。
逐字规则抄进每行的 `gold_judging`（引用前必读）。

## 数据来源与指纹（缺数据时按 `eval/datasets/manifest.py` 的清单校验后落 `dataset/halumem/`）

* 数据集：<https://huggingface.co/datasets/IAAR-Shanghai/HaluMem>
* 文件：`HaluMem-Medium.jsonl`，33,511,525 B（≈33.5 MB），20 个 user / 3,467 题
* 直链：<https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl>
* sha256：`486fbc130a5c8781a2af27ffa508a1d7855245137aa449c193ac4d29c45634e7`
  （与 HF LFS oid 一致，2026-10-06 实测）
* 许可：**CC-BY-NC-ND-4.0**（HF README front-matter）——仅本仓内部评测用，
  不得再分发；同库另有 `HaluMem-Long.jsonl`（106,535,674 B），本脚本不用。
* 采集侧：`official-eval-questions.jsonl`（题面 join 用）与 `official-adds.jsonl`
  （三个检测器用），都在采集导出目录里。

## 题目重名的一条口径

HaluMem 有 6 个题面在多 session 重复（如 `What is Sarah Garcia's middle name?`
在 18 个 session 出现），措辞不同、语义全同（都是"未提供"一类）。
本脚本优先取 user **锚定 session**（其唯一命中题的 session）里的实例，
取不到再取最小 session；并把全部变体写进 `gold_answers`（主答案在前）——
判分按语义等价，任一变体都是上游对某个实例的标准答案。

## 怎么跑（可重复）

    uv run python tools/recover_halumem.py    # 缺数据会按清单下载并校验 sha256
    uv run ruff check tools/recover_halumem.py

产物：`official-dataset-2026-09-29/official-gold-extra-halumem.jsonl`，
一行一条命中的 search（seq/ts/user_id/question 与采集逐字相同）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ⚠ 直接跑脚本时 `sys.path[0]` 是 `tools/`，仓库根不在上面；而 `eval` 不是已安装的包
#   （只有 `src/` 打了包）⇒ 补一条路径（与 `tools/recover_official_gold.py` 同一处置）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets.registry import benchmark_dir, capture_dir  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

MEDIUM_BYTES = 33_511_525
OUT_NAME = "official-gold-extra-halumem.jsonl"

# add 正文命中阈值：实测 186 个命中 user 全在 100%、其余 user 最高 0%，这里是分布中间值。
_ADD_HIT_FRAC = 0.95
_ADD_HIT_MIN = 12

_OPENING = "share my basic personal information"
_NAME_RE = re.compile(r"Name: ([^;]+);")
_TIME_PREFIX = re.compile(r"^\[Time: [^\]]+\]\s*")
_ROLE_PREFIX = re.compile(r"^(?:user|assistant|human|ai):\s*", re.IGNORECASE)

#: 上游 judge 提示词里与本数据集判分直接相关的逐字规则（出处见模块 docstring）。
#: 引号与 `**` 强调都是原文照抄，别顺手改成中文标点。
_JUDGING_RULES = (
    "上游对 QA 不发布精确匹配打分器，只发布一份 LLM judge 提示词："
    "MemTensor/HaluMem 的 `eval/eval_tools.py` 里 `EVALUATION_PROMPT_FOR_QUESTION`"
    "（main @718f16ff…，文件 sha256 0c08e5ec…），"
    "由 `eval/evaluation.py` 以（question, answer, "
    '`"\\n".join([i["memory_content"] for i in qa["evidence"]])`, 系统作答）调用，'
    "回 `Correct | Hallucination | Omission`。逐字规则："
    '"Classify it as one of **“Correct”**, **“Hallucination”**, or **“Omission.”** '
    'Do **not** use any external knowledge or subjective inference."；'
    "Correct = "
    '"its content is **semantically equivalent** to the “Reference Answer.”… '
    'Synonyms, paraphrasing, and reasonable summarization are acceptable."；'
    "Hallucination 之一 = "
    '"When the “Reference Answer” is labeled as *unknown/uncertain*, '
    'yet the response provides a specific verifiable fact or conclusion."；'
    "Omission 之一 = "
    '"It explicitly states “don’t know,” “can’t remember,” or “no related memory,” '
    'even though relevant information exists in the “Key Memory Points.”" 且 '
    '"For multi-element questions, **all elements must be correct and present**; '
    'omission of **any** element is considered an **Omission**."；'
    "冲突时 = "
    '"If the response contains **both missing necessary information** and '
    '**fabricated/contradictory information**, classify it as **Hallucination**."；'
    "数值容差 = "
    '"Equivalent expressions of numbers, times, and units are acceptable, '
    'but the **numerical values themselves must not differ**."；'
    "未知答案 = "
    '"If the reference answer is *“unknown / cannot be determined”* and the system '
    "provides a definite fact, that is a **Hallucination**. If the system also answers "
    '*“unknown”* (without guessing), it may be **Correct**."'
)


def ensure_medium(bench_dir: Path, *, offline: bool = False) -> Path:
    """复用固定版本清单与临时下载，防止未校验文件进入评测目录。"""
    from eval.datasets.prepare import ensure_dataset

    ensure_dataset("halumem", bench_dir, offline=offline)
    return bench_dir / "halumem" / "HaluMem-Medium.jsonl"


def strip_wrapper(content: str) -> str:
    """去掉采集加的两种包装：`[Time: …] ` 前缀与 `user:` / `assistant:` 字面前缀。

    两层可能叠加出现（[Time:] 那批只有时间前缀；`user:` 那批只有角色前缀），
    各去一次即可；正文本身一个字不动——匹配是逐字的，**不做大小写或空白归一**。
    """
    text = _TIME_PREFIX.sub("", content or "")
    return _ROLE_PREFIX.sub("", text).strip()


@dataclass
class _AddScan:
    """一个采集 user 的 add 扫描汇总（只留计数，不留正文，以便单遍扫完 300+ MB）。"""

    phrase: bool = False
    messages: int = 0
    hit: int = 0
    persona_score: Counter[str] = field(default_factory=Counter)


@dataclass
class _MediumIndex:
    """HaluMem-Medium 的索引：persona 元数据 + 正文索引 + 题目索引。"""

    #: uuid -> (persona 名, 文件内的 0 基行号)——行号供人 `sed -n 'Np'` 回查。
    info: dict[str, tuple[str, int]]
    #: 去包装后的正文 -> 含这句话的 uuid 集合（同一句可能出现在多个 persona 里）。
    dialogue: dict[str, set[str]]
    #: 题面 -> [(uuid, session_index, question_index, qa 对象), …]（去空白后逐字相同）。
    questions: dict[str, list[tuple[str, int, int, dict[str, Any]]]]


def build_medium_index(path: Path) -> _MediumIndex:
    info: dict[str, tuple[str, int]] = {}
    dialogue: dict[str, set[str]] = defaultdict(set)
    questions: dict[str, list[tuple[str, int, int, dict[str, Any]]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for rec_index, line in enumerate(handle):
            record = json.loads(line)
            uuid = record["uuid"]
            name = _NAME_RE.search(record["persona_info"]).group(1)
            info[uuid] = (name, rec_index)
            for session_index, session in enumerate(record["sessions"]):
                for turn in session["dialogue"]:
                    dialogue[turn["content"].strip()].add(uuid)
                for q_index, qa in enumerate(session.get("questions", [])):
                    questions[qa["question"].strip()].append((uuid, session_index, q_index, qa))
    return _MediumIndex(info=info, dialogue=dict(dialogue), questions=dict(questions))


def scan_capture_adds(path: Path, dialogue: dict[str, set[str]]) -> dict[str, _AddScan]:
    """单遍扫采集 adds：开头语、去包装正文命中数、persona 归属计分一把算完。"""
    scans: dict[str, _AddScan] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            add = json.loads(line)
            scan = scans.setdefault(add["user_id"], _AddScan())
            for message in add.get("messages", []):
                raw = str(message.get("content") or "")
                if _OPENING in raw:
                    scan.phrase = True
                scan.messages += 1
                uuids = dialogue.get(strip_wrapper(raw))
                if uuids:
                    scan.hit += 1
                    for uuid in uuids:
                        scan.persona_score[uuid] += 1
    return scans


def load_searches(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def judging_text(
    rec_index: int, uuid: str, name: str, session_index: int, q_index: int, dup: list[int]
) -> str:
    """每行一段：金标在哪 + 上游怎么判 + （重名题）取的是哪个实例。"""
    head = (
        f"HaluMem 金标：HaluMem-Medium.jsonl 第 {rec_index} 条 user"
        f"（uuid {uuid}，persona {name}）session {session_index} 第 {q_index} 题；"
        "answer / evidence / difficulty / question_type 逐字取自该 QA 对象，"
        "kind=exact 说的是金标形状（短答串），判分方式见下。"
    )
    if dup:
        head += (
            f"⚠ 该题面在本 persona 的 {len(dup) + 1} 个 session 重复出现"
            f"（本行取 session {session_index} 的实例，另有 {dup}）；"
            "各实例的标准答案措辞不同、语义等价，"
            "gold_answers 已列全部变体，判分按语义等价，任一变体可接受。"
        )
    return head + _JUDGING_RULES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="采集导出目录（默认 TIANXIMEM_CAPTURE_DIR 或 official-dataset-2026-09-29）",
    )
    parser.add_argument(
        "--benchmark-dir", default=None, help="题库目录（默认 TIANXIMEM_BENCHMARK_DIR）"
    )
    parser.add_argument("--out", default=OUT_NAME, help="输出文件名（写在采集目录下）")
    parser.add_argument("--offline", action="store_true", help="禁止下载缺失的题库材料")
    args = parser.parse_args()

    capture = Path(args.dataset_dir) if args.dataset_dir else capture_dir()
    bench = Path(args.benchmark_dir) if args.benchmark_dir else benchmark_dir()
    if not bench.is_absolute():
        # ⚠ benchmark_dir() 的缺省是相对路径（相对 cwd）——这里锚到仓库根，跑法就与 cwd 无关。
        bench = _REPO_ROOT / bench
    adds_path = capture / "official-adds.jsonl"
    search_path = capture / "official-eval-questions.jsonl"
    for required in (adds_path, search_path):
        if not required.exists():
            print(f"✗ 找不到 {required}", file=sys.stderr)
            return 1

    medium_path = ensure_medium(bench, offline=args.offline)
    print(f"✓ {medium_path}  sha256 与预期一致（{MEDIUM_BYTES} B）")
    medium = build_medium_index(medium_path)
    print(
        f"  HaluMem-Medium：{len(medium.info)} 个 persona，"
        f"{len(medium.dialogue)} 句正文，{len(medium.questions)} 个不同题面"
    )

    scans = scan_capture_adds(adds_path, medium.dialogue)
    searches = load_searches(search_path)
    print(f"  采集：{len(scans)} 个 user（adds），{len(searches)} 条 search")

    # ---- 三个独立检测器 -------------------------------------------------
    set_a = {user for user, scan in scans.items() if scan.phrase}
    set_b = {
        user
        for user, scan in scans.items()
        if scan.messages >= _ADD_HIT_MIN and scan.hit / scan.messages >= _ADD_HIT_FRAC
    }
    set_c = {
        query["user_id"]
        for query in searches
        if str(query.get("question") or "").strip() in medium.questions
    }
    print(f"检测器 a（开头语）        ：{len(set_a)} users")
    print(f"检测器 b（add 正文逐字）  ：{len(set_b)} users")
    print(f"检测器 c（search 题面逐字）：{len(set_c)} users")
    print(f"两两相等：a==b {set_a == set_b} / a==c {set_a == set_c} / b==c {set_b == set_c}")
    outside = sorted(
        (scan.hit / scan.messages, user)
        for user, scan in scans.items()
        if user not in set_b and scan.messages >= _ADD_HIT_MIN
    )
    ceiling = f"{outside[-1][0]:.3f}" if outside else "（非命中用户无一过线）"
    print(f"非命中 user 的正文命中率上限：{ceiling}")

    users = set_a & set_b & set_c
    if set_a != set_b or set_a != set_c or set_b != set_c:
        print("⚠ 三个检测器不完全一致：只导出交集（逐条可回查）")

    persona_of = {user: scans[user].persona_score.most_common(1)[0][0] for user in users}
    persona_names = Counter(medium.info[uuid][0] for uuid in persona_of.values())
    print(f"persona 分布：{dict(persona_names)}")

    # ---- 组行 ------------------------------------------------------------
    matched = sorted((q for q in searches if q["user_id"] in users), key=lambda q: q["seq"])
    # 锚定 session：该 user 唯一命中题所在 session；重名题优先取锚定 session 里的实例。
    anchors: dict[str, set[int]] = defaultdict(set)
    hits_cache: dict[int, list[tuple[str, int, int, dict[str, Any]]]] = {}
    for query in matched:
        uuid = persona_of[query["user_id"]]
        hits = [
            h for h in medium.questions[str(query.get("question") or "").strip()] if h[0] == uuid
        ]
        hits_cache[query["seq"]] = hits
        if len({h[1] for h in hits}) == 1:
            anchors[query["user_id"]] |= {h[1] for h in hits}

    rows: list[dict[str, Any]] = []
    dup_rows = 0
    missing = 0
    for query in matched:
        uuid = persona_of[query["user_id"]]
        hits = hits_cache[query["seq"]]
        if not hits:
            missing += 1
            continue
        sessions = {h[1] for h in hits}
        if len(sessions) == 1:
            chosen = hits[0]
        else:
            cands = [h for h in hits if h[1] in anchors[query["user_id"]]] or hits
            chosen = min(cands, key=lambda h: (h[1], h[2]))
            dup_rows += 1
        answers = [chosen[3]["answer"]]
        seen = {chosen[3]["answer"].strip().lower()}
        for hit in sorted(hits, key=lambda h: (h[1], h[2])):
            key = hit[3]["answer"].strip().lower()
            if key not in seen:
                seen.add(key)
                answers.append(hit[3]["answer"])
        name, rec_index = medium.info[uuid]
        locator: dict[str, Any] = {
            "file": "HaluMem-Medium.jsonl",
            "record_index": rec_index,
            "uuid": uuid,
            "persona": name,
            "session_index": chosen[1],
            "question_index": chosen[2],
        }
        other_sessions = sorted(sessions - {chosen[1]})
        if other_sessions:
            locator["also_asked_in_sessions"] = other_sessions
        rows.append(
            {
                "seq": query["seq"],
                "ts": query["ts"],
                "user_id": query["user_id"],
                "question": query["question"],
                "source_dataset": "halumem",
                "source_locator": locator,
                "gold_kind": "exact",
                "gold_answers": answers,
                "gold_rubric": None,
                "gold_native": {
                    "answer": chosen[3]["answer"],
                    "evidence": chosen[3].get("evidence", []),
                    "question_type": chosen[3].get("question_type"),
                    "difficulty": chosen[3].get("difficulty"),
                },
                "gold_judging": judging_text(
                    rec_index, uuid, name, chosen[1], chosen[2], other_sessions
                ),
            }
        )

    out_path = capture / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n命中 {len(rows)} 行 → {out_path}")
    if missing:
        print(f"  ⚠ 有 {missing} 条 search 在其 persona 下找不到题面实例（未导出）")
    print(f"  其中题面跨 session 重复、取了锚定实例的：{dup_rows} 行（gold_answers 带变体）")
    print(f"  gold_kind: {dict(Counter(row['gold_kind'] for row in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
