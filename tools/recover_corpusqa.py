#!/usr/bin/env python3
"""「Corpus:」语料族里 **HybridQA / MuSiQue / FEVEROUS** 三家的公开金标回收。

## 为什么需要它

官方 2026-09-29 采集（`official-eval-questions.jsonl` 10,144 条 search）里有一片带
「`Corpus: `」前缀的语料，README §4.2 只把其中 `Corpus: {…}`（JSON 三元组）那一支
认成 MQuAKE-Remastered——**那个归属只覆盖了这一族的一支**。逐 user 核过（2026-10-06）：
按 user 语料的形态，这一族其实有 **591 个 user**（592 减去 1 个只是碰巧出现子串的）：

| 族 | user 数 | search 数 | 语料指纹 |
| --- | ---: | ---: | --- |
| MQuAKE-Remastered | 249 | 1,125 | `Corpus: {"subject"…}` / `UPDATE:`——**本脚本不管** |
| MuSiQue | 240 | 240 | `Corpus: [Paragraph 0]` + `Title: …` |
| HybridQA | 100 | 113 | `Complete table document…` ＋ `Passage ID: /wiki/…` |
| FEVEROUS | 1 | 249 | `Corpus: {"order": ["table_0"…]}`（wiki 页面序列化） |
| books/films（无公开金标） | 1 | 84 | `Corpus: Document title: …` |

MQuAKE 那一支由 `recover_official_gold.py` 负责（它已补了 1,124 条），不在这里重复。

**上一轮调查说「240 users = MuSiQue，240 questions、金标就是答案」——count 对，金标口径只对一半**：
MuSiQue 那一支的语料逐段**多重集**分别等于 `musique_full_v1.0_dev.jsonl` 里同一个 `id` 的
`answerable=True` / `answerable=False` 实例（实测 120 / 120，一一对上；同一题面被喂给两个 user，
一个拿可答版、一个拿不可答版）。⇒ **120 条的金标不是答案串，而是「证据不足」**——否则会把
不可答题的答案判成对的。判据不是只看题面（两版题面相同），而是语料段落逐条比对。

另外：有 1 个 user 的对话里恰好出现 "Corpus:" 子串（Foucault 问答 + City Council 报告），
**不是这一族**，别把它算进来——所以「带 `Corpus:` 的 user」是 592，本族是 591。

## 输出与 join

`<capture 目录>/official-gold-extra-corpusqa.jsonl`，**一行一条被匹配上的 search**，
按 `seq` 与 `official-eval-questions.jsonl` join；`question` 逐字取自采集。
未被匹配的行**不输出**（books/films 那一族 84 条就是这样被略过的），只在 stdout 里报数。
`official-gold-extra*.jsonl` 会被 `build_official_kit.py` 按文件名 glob 收走（一族一份），
所以文件名就按那个约定取。

## 数据出处（URL + sha256）——全部落在 `dataset/<族>/`（目录 gitignore）

必需三份（缺一份脚本会带着 curl 命令退出）：每份记「本地路径 → URL、sha256」。

* `hybridqa/dev.json`
  ← https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev.json
  sha256 `424272b233735a70ed8ef5af4a615373d114f472168c686c4370d54c92d58ac1`
* `musique/musique_full_v1.0_dev.jsonl`
  ← https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_full_v1.0_dev.jsonl
  sha256 `8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a`
* `feverous/feverous_dev_challenges.jsonl`
  ← https://fever.ai/download/feverous/feverous_dev_challenges.jsonl
  sha256 `1ac8cfd964d4dcedc5de3375850fe3f93d39b89a73a475734bde864da1701f8f`

旁证两份（在就校验、做交叉核对；缺了只提示）：

* `hybridqa/dev_reference.json`
  ← https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev_reference.json
  sha256 `617cd141a09550e85a0634b68b2e16727a87f9a6005002d31f40d839dfa389ed`
* `musique/musique_ans_v1.0_dev.jsonl`
  ← https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl
  sha256 `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`

MuSiQue 官方发布是 Google Drive 的 `musique_v1.0.zip`
（`gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`，仓库 `StonyBrookNLP/musique/download_data.sh`）；
上表用的 HF 镜像文件名逐字相同，两份镜像（`dgslibisey/MuSiQue` 与 `voidful/MuSiQue`）
对 `musique_ans_v1.0_dev.jsonl` 的 sha256 一致，已核。
FEVEROUS 的 URL 取自官方仓库 `Raldir/FEVEROUS/download_data.sh`。

## 已核实的身份锚点（2026-10-06）

* HybridQA：113/113 条题面与 `dev.json` **逐字**相同（100 个 user），`answer-text` 与
  `dev_reference.json` 的 `reference[question_id]` 归一化后逐条一致；表语料抽一条与
  `WikiTables-WithLinks` 的 `tables_tok/Primera_B_Nacional_0.json` 比对——`json.dumps(…,
  ensure_ascii=False, indent=2)` 后 8,000 字符捕获上限内**逐字节**吻合（repo 文件是 `\\uXXXX`
  转义写法，所以是「同一次序列化、不同转义」）。
* MuSiQue：240/240 条语料多重集精确等于 `musique_full_v1.0_dev.jsonl` 里某个实例的
  `paragraphs`（120 可答 + 120 不可答）；`musique_ans` 那份的段集与 `musique_full` **并不相同**，
  所以段集正确的对齐源是 full 文件（金标也按它取）。
* FEVEROUS：249/249 条 claim 在 `feverous_dev_challenges.jsonl` 里**唯一**命中
  （SUPPORTS 89 / REFUTES 80 / NOT ENOUGH INFO 80；证据组 1/2/3 组的行数为 226/22/1）。

## 怎么重跑

    mkdir -p dataset/hybridqa dataset/musique dataset/feverous
    curl -L -o dataset/hybridqa/dev.json <上表 URL>
    …（三份必需文件，哈希对不上脚本会拒绝跑）
    uv run python tools/recover_corpusqa.py
    # 采集目录默认 eval.datasets.registry.capture_dir()，可用 --capture-dir 改

目录都会先按 sha256 校验，对不上就停——**哈希是「到手的是不是同一份」的唯一判据**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ⚠ 直接跑脚本时 `sys.path[0]` 是 `tools/`，仓库根不在上面（与 `recover_official_gold.py`
#   同一条处置）；也可以 `python -m tools.recover_corpusqa` 跑。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets.prepare import sha256_file  # noqa: E402
from eval.datasets.registry import benchmark_dir, capture_dir  # noqa: E402

__all__ = ["main", "norm"]

OUT_NAME = "official-gold-extra-corpusqa.jsonl"

#: `(相对 benchmark_data 的路径, URL, sha256)`——缺一份就照抄 URL 用 curl 取回。
REQUIRED: tuple[tuple[str, str, str], ...] = (
    (
        "hybridqa/dev.json",
        "https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev.json",
        "424272b233735a70ed8ef5af4a615373d114f472168c686c4370d54c92d58ac1",
    ),
    (
        "musique/musique_full_v1.0_dev.jsonl",
        "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_full_v1.0_dev.jsonl",
        "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a",
    ),
    (
        "feverous/feverous_dev_challenges.jsonl",
        "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
        "1ac8cfd964d4dcedc5de3375850fe3f93d39b89a73a475734bde864da1701f8f",
    ),
)

#: 交叉核对用；缺了只打印一行提示，不挡跑。
OPTIONAL: tuple[tuple[str, str, str], ...] = (
    (
        "hybridqa/dev_reference.json",
        "https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev_reference.json",
        "617cd141a09550e85a0634b68b2e16727a87f9a6005002d31f40d839dfa389ed",
    ),
    (
        "musique/musique_ans_v1.0_dev.jsonl",
        "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",
        "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    ),
)

#: MuSiQue 题面的尾指令——**逐字**（240/240 实测都以它结尾），判族靠它。
MUSIQUE_SUFFIX = (
    "\n\nAnswer using only the supplied paragraphs. "
    "If the evidence is insufficient, respond with INSUFFICIENT_EVIDENCE."
)

#: FEVEROUS 题面的开头（claim 正文在 `\n\nClaim: ` 之后）。
FEVEROUS_HEAD = "Verify the following claim using only the supplied corpus."
FEVEROUS_SPLIT = "\n\nClaim: "

#: 语料形态标记（判定 user 属于哪一族）。
HYBRIDQA_MARKS = ("Complete table document in JSON", "Passage ID: /wiki/")
MUSIQUE_MARK = "Corpus: [Paragraph"

#: 无法补公开金标的那一族（books/films）的题面标记——只用来报数。
BOOKSFILM_MARK = "Use only the provided document corpus. Return all document titles"

#: `Corpus: [Paragraph N]\nTitle: <title>\n\n<text>`——text 取到该 message 末尾。
PARA_RE = re.compile(r"^Corpus: \[Paragraph \d+\]\nTitle: [^\n]*\n\n(?P<text>.*)$", re.S)

MUSIQUE_JUDGING_ANSWERABLE = (
    "MuSiQue（官方仓库 StonyBrookNLP/musique，CC BY 4.0）：该 user 的语料段"
    "（`Corpus: [Paragraph N]` 逐条，归一化后多重集）与 `musique_full_v1.0_dev.jsonl` 中该 `id` 的 "
    "**answerable=True** 实例的 `paragraphs` 完全相等——本行就是这么钉住的（题面在两版之间相同，"
    "单靠题面分不出版本）。官方 `evaluate_v1.0.py`："
    '`ground_truth_answers = [ground_truth_instance["answer"]]'
    ' + ground_truth_instance["answer_aliases"]`，'
    "用 metrics/answer.py 的 SQuAD 式 `normalize_answer`（Lower text and remove punctuation, "
    "articles and extra whitespace，去冠词正则 `\\b(a|an|the)\\b`）算 "
    "`compute_exact` / `compute_f1`，并以 `metric_max_over_ground_truths` 取最大 "
    "⇒ 本行按 answer_em / answer_f1 口径判，`gold_answers` 已含全部官方别名。"
)

MUSIQUE_JUDGING_UNANSWERABLE = (
    "MuSiQue（官方仓库 StonyBrookNLP/musique，CC BY 4.0）：该 user 的语料段"
    "（`Corpus: [Paragraph N]` 逐条，归一化后多重集）与 `musique_full_v1.0_dev.jsonl` 中该 `id` 的 "
    "**answerable=False**（不可答）实例的 `paragraphs` 完全相等——当局把某条支撑段换成了干扰段，"
    "官方对这类实例只算组内 sufficiency（`evaluate_v1.0.py` + metrics/group_answer_sufficiency.py："
    "`sufficiency_score = group.predicted_sufficiencies == group.gold_sufficiencies`，"
    "即同一 id 的两条要分别判「可答 / 不可答」），答案文本**不参与** answer_f1。"
    "题面注入的指令逐字写着「Answer using only the supplied paragraphs. If the evidence is "
    "insufficient, respond with INSUFFICIENT_EVIDENCE.」"
    "⇒ 本行 `gold_answers=[INSUFFICIENT_EVIDENCE]`；"
    "`gold_native.answer` 是**可答孪生**的官方答案，拿它当正确答案判就是错的。"
)

HYBRIDQA_JUDGING = (
    "HybridQA（EMNLP2020；官方仓库 wenhuchen/HybridQA，MIT）：题面与 `released_data/dev.json` "
    "逐字相同，金标即该行 `answer-text`"
    "（已核：与 `dev_reference.json` 的 `reference[question_id]` 归一化后一致）。"
    "官方评测入口 `python evaluate_script.py predictions.json released_data/dev_reference.json`；"
    "判分规则（evaluate_script.py 原文）：`normalize_answer`＝「Lower text and remove punctuation, "
    "articles and extra whitespace」（去冠词正则 `\\b(a|an|the)\\b`）；逐题取 "
    "`max(compute_exact(a, pred))` 与 `max(compute_f1(a, pred))`，"
    "官方汇总打印 `total exact` / `total f1`。"
    "⇒ 本行按同一口径判：预测串与 `gold_answers[0]` 归一化后相等记 exact 命中。"
)

FEVEROUS_JUDGING = (
    "FEVEROUS（官方仓库 Raldir/FEVEROUS）：金标取自 dev challenge 同一 `id` 行的 `label`"
    "（SUPPORTS / REFUTES / NOT ENOUGH INFO）与 `evidence`（证据组；`content` 是原生 id，形如 "
    "`page_sentence_0` / `page_cell_0_9_1` / `page_table_caption_0`，`context` 给出所属页面标题）。"
    "官方 feverous_scorer.py 判分：label 正确＝"
    '`instance["label"].upper() == instance["predicted_label"].upper()`；'
    "`is_strictly_correct` 要求预测证据（`(page,type,position)` 三元组列表，如 "
    "`['Wolfgang Niedecken', 'sentence', '1']` 或 `['Korean Air', 'cell', '1_19_0']`）完整包含 "
    "`evidence` 里的**任一整组**（原文：Only return true if an entire group of actual sentences is "
    "in the predicted sentences），另有 evidence macro precision/recall/F1（预测证据截断上限 "
    "max_evidence=5 / max_evidence_cell=25）。⇒ 本 kit 用顶层 `gold_evidence_ids`（全部证据组 "
    "content id 的扁平表）与 `gold_native.evidence`（原始分组＋context）复核证据；label 与 "
    "`gold_answers[0]` 判等。"
)


def norm(text: str) -> str:
    """题面/段落归一化：NFKC + 转小写 + 空白折叠（与上一轮调查同一套，否则匹配率会假低）。"""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def check_file(path: Path, url: str, want: str, *, required: bool) -> bool:
    """存在性 + 哈希校验。必需文件对不上就退出；旁证文件只提示。"""
    if not path.exists():
        level = "缺失" if required else "旁证缺失"
        print(f"[{level}] {path}\n    curl -L -o {path} {url}")
        if required:
            raise SystemExit(f"✗ 必需文件不存在：{path}")
        return False
    got = sha256_file(path)
    if got != want:
        print(f"[哈希不符] {path}\n    期望 {want}\n    实际 {got}\n    来源 {url}")
        if required:
            raise SystemExit("✗ 文件与清单不符——重新下载前不要相信任何匹配结果")
        return False
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_adds_index(
    adds_path: Path, musique_users: set[str]
) -> tuple[dict[str, list[str]], set[str]]:
    """过一遍 300MB 的 adds。

    返回 `(musique user -> 其语料段 message 列表, 带 HybridQA 语料标记的 user 集合)`——
    只需要这两样，别的 message 看一眼就丢。
    """
    paragraphs: dict[str, list[str]] = defaultdict(list)
    hybridqa_users: set[str] = set()
    with adds_path.open(encoding="utf-8") as handle:
        for line in handle:
            add = json.loads(line)
            user = add["user_id"]
            for message in add.get("messages") or []:
                content = message.get("content") or ""
                if user in musique_users and content.startswith(MUSIQUE_MARK):
                    paragraphs[user].append(content)
                elif any(mark in content for mark in HYBRIDQA_MARKS):
                    hybridqa_users.add(user)
    return paragraphs, hybridqa_users


def para_counter(corpus_adds: list[str]) -> tuple[Counter[str], int]:
    """把 `Corpus: [Paragraph …]` 段解析成『归一化正文 -> 次数』的多重集，外加无法解析的条数。"""
    counter: Counter[str] = Counter()
    bad = 0
    for content in corpus_adds:
        match = PARA_RE.match(content)
        if match:
            counter[norm(match.group("text"))] += 1
        else:
            bad += 1
    return counter, bad


def row_para_counter(row: dict[str, Any]) -> Counter[str]:
    return Counter(norm(p["paragraph_text"]) for p in row["paragraphs"])


def musique_hit(
    question: str, corpus: Counter[str], by_question: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """按语料多重集在 full-dev 的同一题面候选里找实例；返回命中的行（理论上有且仅有一条）。"""
    return [row for row in by_question.get(norm(question), []) if row_para_counter(row) == corpus]


def emit_row(
    capture_row: dict[str, Any],
    source_dataset: str,
    locator: dict[str, Any],
    answers: list[str],
    judging: str,
    native: dict[str, Any] | None,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seq": capture_row["seq"],
        "ts": capture_row["ts"],
        "user_id": capture_row["user_id"],
        "question": capture_row["question"],
        "source_dataset": source_dataset,
        "source_locator": locator,
        "gold_kind": "exact",
        "gold_answers": answers,
        "gold_rubric": None,
        "gold_native": native,
        "gold_judging": judging,
    }
    if evidence_ids is not None:
        row["gold_evidence_ids"] = evidence_ids
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="回收 Corpus: 族（HybridQA/MuSiQue/FEVEROUS）公开金标"
    )
    parser.add_argument("--capture-dir", type=Path, default=capture_dir(), help="采集目录")
    parser.add_argument("--benchmark-dir", type=Path, default=benchmark_dir(), help="题库归档目录")
    parser.add_argument(
        "--out", type=Path, default=None, help=f"输出路径（默认 <capture-dir>/{OUT_NAME}）"
    )
    parser.add_argument("--offline", action="store_true", help="禁止下载缺失的题库材料")
    args = parser.parse_args()

    bm: Path = args.benchmark_dir
    capture: Path = args.capture_dir
    out_path: Path = args.out or capture / OUT_NAME
    from eval.datasets.prepare import ensure_dataset

    for family in ("hybridqa", "musique", "feverous"):
        ensure_dataset(family, bm, offline=args.offline)

    paths: dict[str, Path] = {}
    for rel, url, want in REQUIRED:
        path = bm / rel
        check_file(path, url, want, required=True)
        paths[rel] = path
    cross: dict[str, bool] = {}
    for rel, url, want in OPTIONAL:
        path = bm / rel
        cross[rel] = check_file(path, url, want, required=False)
        if cross[rel]:
            paths[rel] = path

    dev = json.loads(paths["hybridqa/dev.json"].read_text(encoding="utf-8"))
    hybridqa_by_q = {norm(r["question"]): r for r in dev}

    full_rows = load_jsonl(paths["musique/musique_full_v1.0_dev.jsonl"])
    musique_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        musique_by_q[norm(row["question"])].append(row)

    feverous_by_claim: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(paths["feverous/feverous_dev_challenges.jsonl"]):
        if row.get("claim"):
            feverous_by_claim[norm(row["claim"])] = row

    questions = load_jsonl(capture / "official-eval-questions.jsonl")
    searches = [r for r in questions if r.get("query") and r.get("question")]

    musique_users = {r["user_id"] for r in searches if r["question"].endswith(MUSIQUE_SUFFIX)}
    paragraphs, hybridqa_users = build_adds_index(capture / "official-adds.jsonl", musique_users)

    out: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    problems: list[str] = []
    for cap in searches:
        question: str = cap["question"]
        if question.startswith(FEVEROUS_HEAD) and FEVEROUS_SPLIT in question:
            claim = question.split(FEVEROUS_SPLIT, 1)[1].strip()
            gold = feverous_by_claim.get(norm(claim))
            if gold is None:
                stats["feverous 未命中"] += 1
                problems.append(f"seq={cap['seq']} feverous claim 未命中：{claim[:80]}")
                continue
            evidence_ids = [cid for group in gold["evidence"] for cid in group["content"]]
            out.append(
                emit_row(
                    cap,
                    "feverous",
                    {
                        "dir": "feverous",
                        "file": "feverous_dev_challenges.jsonl",
                        "id": gold["id"],
                        "challenge": gold.get("challenge"),
                    },
                    [gold["label"]],
                    FEVEROUS_JUDGING,
                    {
                        "id": gold["id"],
                        "label": gold["label"],
                        "challenge": gold.get("challenge"),
                        "evidence": gold["evidence"],
                    },
                    evidence_ids,
                )
            )
            stats["feverous"] += 1
            continue
        if question.endswith(MUSIQUE_SUFFIX):
            bare = question[: -len(MUSIQUE_SUFFIX)]
            corpus, bad = para_counter(paragraphs.get(cap["user_id"], []))
            if bad:
                problems.append(f"seq={cap['seq']} 有 {bad} 条语料段解析失败")
            hits = musique_hit(bare, corpus, musique_by_q)
            if len(hits) != 1:
                stats["musique 未命中/有歧义"] += 1
                problems.append(
                    f"seq={cap['seq']} musique 语料对不上（命中 {len(hits)} 条）：{bare[:80]}"
                )
                continue
            gold = hits[0]
            answerable = bool(gold["answerable"])
            if answerable:
                answers = [gold["answer"], *gold["answer_aliases"]]
                judging = MUSIQUE_JUDGING_ANSWERABLE
                stats["musique 可答"] += 1
            else:
                answers = ["INSUFFICIENT_EVIDENCE"]
                judging = MUSIQUE_JUDGING_UNANSWERABLE
                stats["musique 不可答"] += 1
            support = [p["idx"] for p in gold["paragraphs"] if p["is_supporting"]]
            out.append(
                emit_row(
                    cap,
                    "musique",
                    {
                        "dir": "musique",
                        "file": "musique_full_v1.0_dev.jsonl",
                        "id": gold["id"],
                        "answerable": answerable,
                    },
                    answers,
                    judging,
                    {
                        "id": gold["id"],
                        "answerable": answerable,
                        "answer": gold["answer"],
                        "answer_aliases": gold["answer_aliases"],
                        "support_paragraph_idxs": support,
                        "question_decomposition": gold["question_decomposition"],
                    },
                )
            )
            continue
        gold = hybridqa_by_q.get(norm(question))
        if gold is not None:
            if cap["user_id"] not in hybridqa_users:
                stats["hybridqa 命中但语料标记不符"] += 1
                problems.append(f"seq={cap['seq']} 题面命中 HybridQA 但该 user 无表/段语料标记")
                continue
            out.append(
                emit_row(
                    cap,
                    "hybridqa",
                    {
                        "dir": "hybridqa",
                        "file": "dev.json",
                        "question_id": gold["question_id"],
                        "table_id": gold["table_id"],
                    },
                    [gold["answer-text"]],
                    HYBRIDQA_JUDGING,
                    dict(gold),
                )
            )
            stats["hybridqa"] += 1
            continue
        if BOOKSFILM_MARK in question:
            stats["books/films（无公开金标，略过）"] += 1

    # ── 旁证交叉核对（在就核，不在就跳过）────────────────────────────────
    if cross.get("hybridqa/dev_reference.json"):
        ref = json.loads(paths["hybridqa/dev_reference.json"].read_text(encoding="utf-8"))
        bad = sum(
            1
            for row in out
            if row["source_dataset"] == "hybridqa"
            and norm(ref["reference"][row["source_locator"]["question_id"]])
            != norm(row["gold_answers"][0])
        )
        stats["旁证：hybridqa answer-text≠dev_reference 的条数"] += bad
    if cross.get("musique/musique_ans_v1.0_dev.jsonl"):
        ans_dev = {r["id"]: r for r in load_jsonl(paths["musique/musique_ans_v1.0_dev.jsonl"])}
        bad = 0
        for row in out:
            if row["source_dataset"] != "musique" or not row["source_locator"]["answerable"]:
                continue
            twin = ans_dev.get(row["source_locator"]["id"])
            if twin is None:
                stats["旁证：ans 文件缺该 id"] += 1
                continue
            if [twin["answer"], *twin["answer_aliases"]] != row["gold_answers"]:
                bad += 1  # 官方两份文件的别名表有 30 个 id 不一致，仅报表用，不失败
        stats["旁证：ans 与 full 别名表不一致的条数"] += bad

    out.sort(key=lambda row: row["seq"])
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out), encoding="utf-8"
    )

    print(f"写到 {out_path}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"  总输出行数: {len(out)}")
    if problems:
        print("可留意的行：")
        for line in problems[:20]:
            print("  -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
