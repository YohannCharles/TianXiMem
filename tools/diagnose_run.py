"""跑批诊断 —— **把"低分"和"模型不行"分开**。

```bash
uv run python tools/diagnose_run.py --run-id base-mqk
```

## 它为什么存在

2026-10-03 一天里在**四个**数据集上查出同一类缺陷，**四个全是"我们接错了"**，
而不是"模型在这个数据集上很差"：

| 数据集 | 表面分数 | 实际病因 |
| --- | ---: | --- |
| tempreason | 0.066 | 加载器**丢了一个字段**（`fact_context`），gold 只有 27% 在记忆里 |
| memtrapbench | 0.144 | **用错了答案 prompt** + 裁判输出被 `max_tokens=256` 截断 |
| mquake | 0.400 | prompt **少了冲突解决规则**、又禁掉了数据集**设计上就要求**的世界知识 |
| corporatebench | 0.260 | 覆盖限制 + 聚合作答 + 评测适配（详见下面的修复报告） |

CorporateBench 的修复与重判记录见 `eval/reports/corporatebench-fix-20261004.md`。

修完前三个：**0.551 / 0.424 / 0.540**。

**这四次的诊断手法完全一样**，本脚本把它固化下来——不然下次还得从头敲一遍，
而**"接错"的表现酷似"方法很差"**（分数低、逐题看也像模型答不上来），
唯一的区分手段就是**回产物里量**。

## 判读树（脚本最后会自己给结论）

```text
① 拒答率高（模型逐字回同一句）?
     ↓ 是
② 那些拒答题里，gold 在不在**检索返回的上下文**里?
     ├─ 在  ⇒ **prompt 接错**（有规则逼它放弃）—— memtrapbench / mquake 是这一类
     └─ 不在 ⇒ 往上一层：**语料/加载器**（tempreason）
                还是**结构性**（corporatebench：语料装不进去）
③ JUDGE_ERROR 占比高 ⇒ **判分链路**（解析失败 / 输出被截断）
④ 逐类分化大（某类全对、某类全崩）⇒ 别读总分，读那一类
```

⚠ **②要先过「自校准」才准读**：它靠**子串匹配**认 gold，而 gold 是不是短实体、
判分宽不宽容，逐数据集差得很远（实测召回从 **33% 到 100%**）。
脚本拿**判对的题**当已知有证据的对照组量出这条判据的召回；**低于 80% 就拒绝下结论**
（并明说"别照它去查语料"）——否则它会把人**送去查错的那一层**。

> ⚠ **本脚本只回产物、不打网关、不改任何东西**。它回答"分数是怎么来的"，
> 不回答"该怎么改"——那是人的判断（或 [`../../../CLAUDE.md`](../../../CLAUDE.md) 说的那种决定）。

## 一条使用纪律

**拒答句是可配的**（`--refusal`，可重复）：默认认 `Cannot determine from the memories.`
这一句。换了数据集/换了 prompt 之后，模型可能换一句拒答——**它换了而你没改，
脚本会报"0 拒答"，那是假绿**。脚本会把**答案里出现最多的那些前缀**打出来供你核对。

（② 的自校准是**自动**的，不需要你记得做——但**结论里说"不可信"时不要去读上面那两行**。）
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Final

#: **答案本来就不该在记忆里**的数据集 —— 判读树对它们**不适用**。
#:
#: ⚠ 这一条是本脚本**第一次真跑就自己撞出来的**：它对 memtrapbench 判了
#: "证据不在 ⇒ 查语料"，而那份数据集的**设计正好相反**——官方 README 原文
#: "No-Memory Solvability: the final query must be answerable correctly **even without
#: the history**"，gold 是 "the correct answer that **ignores the misleading memory**"。
#: ⇒ 它上面"gold 不在记忆里"是**预期行为**，不是缺陷。
#: ⚠ `personamem-v2` 同理，而且更彻底：它**根本不读检索字段**。
_NO_MEMORY_DATASETS: Final[dict[str, str]] = {
    "memtrapbench": "答案**设计成不在记忆里**（No-Memory Solvability）",
    "personamem-v2": "**根本不读检索字段**",
}

#: 默认的拒答句——**模型逐字回同一句**是"被 prompt 逼的"最典型的指纹。
DEFAULT_REFUSALS: Final[tuple[str, ...]] = (
    "cannot determine from the memories",
    "cannot determine",
    "unable to determine",
)

#: 记忆文本可能的字段名（**逐数据集不同**，见 `eval/CLAUDE.md` 那张契约表）。
MEMORY_FIELDS: Final[tuple[str, ...]] = (
    "retrieved_context",
    "speaker_1_memories",
    "speaker_2_memories",
)

_SUMMARY_HEAD: Final[int] = 3


def _read_jsonl(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _read_batches(run_dir: Path, filename: str) -> dict[tuple[str, str], dict]:
    """读产物，键是 **`(批目录名, 题号)`**。

    ⚠ **不能只用题号**：一个 run 的产物按 `sample.user_id` 分目录，而题号只在
    **一个样本内**保证唯一。`medmemorybench` 一度就是**全局**撞的（`session_10_eem_1`
    在 20 个 persona 上各出现一次）⇒ 按题号配对的读法把 172 行压成 20 个键，
    报出来的"答案只有 20/172 题、跑批中断？"**是工具自己造的**。
    加载器已改成全局唯一，但**核对产物时不该假定它**——这里以目录为准。
    """
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(run_dir.glob(f"*/{filename}")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows[(path.parent.name, str(row.get("id") or row.get("idx")))] = row
    return rows


def _norm(text: object) -> str:
    """比字符串用：只留字母数字与空格。**重音与标点都会被抹掉**。

    ⚠ 这是**诊断**用的宽容口径，不是判分口径——它的目的是"gold 有没有被给到"，
    抹掉重音正好能**避免**把 `Markarian` / `Markarián` 那种表面差异读成"没给到"
    （那种差异是**判分**该关心的事，不是"证据在不在"该关心的事）。
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def memory_text(row: dict) -> str:
    """把这一行里**注入给答案模型的记忆文本**拼出来（逐数据集字段不同）。"""
    parts: list[str] = []
    for key in MEMORY_FIELDS:
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts += [str(item) for item in value]
    retrieval = row.get("retrieval")
    if isinstance(retrieval, dict):
        parts += [
            json.dumps(item, ensure_ascii=False) for item in (retrieval.get("selected") or [])
        ]
    return "\n".join(parts)


def gold_strings(row: dict) -> list[str]:
    """把金标摊成一组可比字符串。

    ⚠ **`gold_answer` 的形状逐数据集不同，实测四种**（`input.jsonl` 的产物扫过一遍）：

    | 形状 | 数据集 |
    | --- | --- |
    | `list` / `str` / `int` | loCoMo · LME · mquake · tempreason |
    | `{"answer", "answer_type"}` | corporatebench |
    | `{"answers": [{"content", "is_correct"}], "query_type"}` | **medmemorybench** |
    | `{"gold_standard", "expected_failure_output"}` | **memtrapbench**（判分**要点**） |

    ⚠ 只认 `"answer"` 的那一版对 **mmb 恒返回空列表** ⇒ `_evidence_present` 恒为假
    ⇒ 报出来的"97% 证据不在"**整个是判据自己造的**，而结论会指向"查语料"。
    （是 ① 的**自校准**把它抓出来的：判对的题里「证据在」= 0%。）
    """
    gold = row.get("gold_answer")
    if not isinstance(gold, dict):
        if isinstance(gold, list):
            return [str(item) for item in gold]
        return [] if gold is None else [str(gold)]
    parts: list[str] = []
    value = gold.get("answer")
    if isinstance(value, list):
        parts += [str(item) for item in value]
    elif value is not None:
        parts.append(str(value))
    for entry in gold.get("answers") or []:  # medmemorybench：正确答案住在 answers[]
        if isinstance(entry, dict) and entry.get("is_correct"):
            parts.append(str(entry.get("content", "")))
        elif isinstance(entry, str):
            parts.append(entry)
    for key in ("gold_standard", "golden_answer", "reference_answer", "correct_answer"):
        if isinstance(gold.get(key), str):
            parts.append(gold[key])
    return [part for part in parts if part]


def _evidence_present(
    row: dict, *, cache: dict[tuple[str, str], str], key: tuple[str, str]
) -> bool:
    """gold 里**有没有任何一条**出现在这一行给模型的记忆文本里。"""
    body = cache.setdefault(key, _norm(memory_text(row)))
    return any(_norm(g) and _norm(g) in body for g in gold_strings(row))


def _is_refusal(answer: str, phrases: tuple[str, ...]) -> bool:
    head = _norm(answer)[:120]
    return any(_norm(p) in head for p in phrases)


#: "换了措辞的拒答"的**粗启发式**——只用来**报警**，不用来改判。
#:
#: ⚠ 这条是为了一次**差点骗过人的假绿**加的（2026-10-04，`tr-promptfix`）：
#: 换了 prompt 之后模型不再回 `Cannot determine from the memories.`，改成回
#: `The provided memories do not contain information about …`——**12 条、全判错**，
#: 而工具当时印的是"**拒答率 0/332 = 0.0%**"。脚本确实把"答案里出现最多的前缀"
#: 打在下面了（那是纪律 1 的设计），**但"0%"那行太好信**。
#:
#: ⇒ 命中这里的条数会**单独印一行并要求人工核对**，不计进拒答率
#: （真正算数要靠 `--refusal` 显式加词——启发式会误伤"答案里本来就有的否定句"）。
_MAYBE_REFUSAL_WORDS: Final[tuple[str, ...]] = (
    "cannot",
    "can not",
    "unable",
    "not contain",
    "no information",
    "do not know",
    "don t know",
    "insufficient",
    "none of the memories",
)


def _looks_like_refusal(answer: str) -> bool:
    head = _norm(answer)[:160]
    return any(word in head for word in _MAYBE_REFUSAL_WORDS)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:5.1f}%" if whole else "  n/a"


#: 校准用的最小样本量——**低于它就别谈召回**（判对的题太少，比例本身没意义）。
_CALIB_MIN_N = 20

#: 召回低于这个数就**不许据「证据在不在」下结论**。
_CALIB_TRUSTWORTHY = 0.80


def _recall_ceiling(calib: collections.Counter[str]) -> float:
    """「gold 在不在记忆里」这条判据的**召回**——用**判对的题**当已知有证据的对照组。

    ## 为什么必须自校准

    这条判据是**子串匹配**（`gold` 归一化后出现在记忆文本里），它只在
    **gold 是短实体、且检索返回的就是原文**时才有分辨率。判对的题**必然**有过证据，
    所以"判对的题里它认出几条"就是它的**召回上限**——一旦这个数低，
    "证据不在"里混的主要是**判据自己的漏**，而不是真的没给到。

    ## 实测（2026-10-03，各 run 的产物）

    | run | 判对的题里「证据在」 | 能不能读 |
    | --- | ---: | --- |
    | `base-mqk` / `tr-fixtest` | **100%** | 能——"证据不在"就是真没给到 |
    | `base-cb`（corporatebench） | 86% | 勉强 |
    | `base-lme2` | 67% | 勉强 |
    | **`base-mmb`（medmemorybench）** | **33%** | ⛔ **不能** |

    mmb 的 gold 多是**中文整句**（判分本身就是"语义上对不对"，不要求复述），
    模型答对也是**改写**⇒ 子串匹配天然大面积漏 ⇒ 它那 97% 的"证据不在"
    **绝大部分是判据自己的漏**。不校准就会照着它去查**语料**——查错层。
    """
    hit = calib["对在"]
    total = hit + calib["对不在"]
    return hit / total if total else 0.0


def clinical_report(run_id: str, reports_dir: Path, refusals: tuple[str, ...]) -> int:
    """打一份诊断报告。**只读产物。**"""
    run_dir = reports_dir / "runs" / run_id
    if not run_dir.is_dir():
        print(f"⛔ 没有这个 run 目录：{run_dir}", file=sys.stderr)
        return 2

    record_path = reports_dir / "runs" / f"{run_id}.json"
    record: dict[str, Any] = (
        json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    )
    fingerprint = record.get("data_fingerprint") or {}

    inputs = _read_batches(run_dir, "input.jsonl")
    answers = _read_batches(run_dir, "answers.jsonl")
    labels = _read_batches(run_dir, "labels.jsonl")

    print(f"\n═══ {run_id} ═══")
    if fingerprint:
        print(
            f"  数据集 {fingerprint.get('dataset')} · n={fingerprint.get('n_questions')}"
            f" · add_shape={fingerprint.get('add_shape')}"
        )
    if record:
        scores = record.get("scores") or {}
        print(f"  **overall = {scores.get('overall')}**")
        if (record.get("breakdown") or {}).get("partial_credit", {}).get("n"):
            print(f"  partial_credit = {(record['breakdown'])['partial_credit']}")
    if not inputs or not answers:
        print("  ⚠ 产物不全（input/answers 缺）——跑批可能没跑完，下面只报能算的")
    if inputs and len(answers) < len(inputs):
        print(f"  ⚠ 答案只有 {len(answers)}/{len(inputs)} 题（跑批中断？）")

    # ── ① 拒答 ──────────────────────────────────────────────────────
    cache: dict[tuple[str, str], str] = {}
    table: collections.Counter[str] = collections.Counter()
    calib: collections.Counter[str] = collections.Counter()
    heads: collections.Counter[str] = collections.Counter()
    refused_ids: list[str] = []
    reworded = 0
    for key, row in inputs.items():
        answer = str((answers.get(key) or {}).get("generated_answer") or "")
        if key not in answers:
            continue
        heads[" ".join(_norm(answer).split()[:6])] += 1
        refused = _is_refusal(answer, refusals)
        if refused:
            refused_ids.append(key[1])
        elif _looks_like_refusal(answer):
            reworded += 1
        present = _evidence_present(row, cache=cache, key=key)
        table[("拒答" if refused else "作答") + "·证据" + ("在" if present else "不在")] += 1
        label = labels.get(key)
        if label is not None and "ERROR" not in str(label.get("label")).upper():
            calib[("对" if label.get("is_correct") else "错") + ("在" if present else "不在")] += 1

    total = sum(table.values())
    print(f"\n── ① 答案侧（{total} 题）──")
    for key in ("拒答·证据在", "拒答·证据不在", "作答·证据在", "作答·证据不在"):
        if table.get(key):
            print(f"   {key:14s} {table[key]:5d}  {_pct(table[key], total)}")
    refused_total = sum(v for k, v in table.items() if k.startswith("拒答"))
    print(f"   **拒答率 {refused_total}/{total} = {_pct(refused_total, total).strip()}**")
    if reworded:
        # ⚠ **这一行是防"假绿"的**：换 prompt 之后模型常换一句拒答，而它在 `--refusal`
        #   之外 ⇒ 上面那行会印 0.0%。**别信那个 0**，先看这里。
        print(
            f"   ⚠ 另有 **{reworded} 条看起来是换了措辞的拒答**（不在 `--refusal` 里，未计入）"
            f" ⇒ 核一下下面的前缀，要算进来就把它加进 `--refusal`"
        )

    # ⚠ **这条判据要先自校准再读** —— 见 `_recall_ceiling` 的 docstring。
    recall = _recall_ceiling(calib)
    judged = calib["对在"] + calib["对不在"]
    if judged >= _CALIB_MIN_N:
        print(
            f"\n   校准：「gold 在不在记忆里」这条判据，在**判对的**题上只认出 "
            f"**{recall * 100:.1f}%**（{calib['对在']}/{judged}）"
            f" ← 这是它的**召回上限**"
        )
        if recall < _CALIB_TRUSTWORTHY:
            print(
                "     ⛔ **低于 80% ⇒ 上面那两条「证据在/不在」的推论都不作数**"
                "——别照它去查语料/prompt"
            )

    print("\n   答案里出现最多的前缀（**核对拒答句有没有换**）：")
    for head, count in heads.most_common(_SUMMARY_HEAD):
        print(f"     {count:5d}  {head!r}")

    # ── ② 判分侧 ────────────────────────────────────────────────────
    print(f"\n── ② 判分侧（{len(labels)} 条 label）──")
    by_label: collections.Counter[str] = collections.Counter(
        str(row.get("label")) for row in labels.values()
    )
    for label, count in by_label.most_common(8):
        flag = "  ← **判分链路在丢题**" if "ERROR" in label.upper() else ""
        print(f"   {label:18s} {count:5d}  {_pct(count, len(labels))}{flag}")

    # ── ③ 逐类 ──────────────────────────────────────────────────────
    by_cat: dict[str, list[bool]] = {}
    judge_errors: dict[str, int] = {}
    for key, row in inputs.items():
        label = labels.get(key)
        if label is None:
            continue
        category = str(row.get("category") or "?")
        if "ERROR" in str(label.get("label")).upper():
            judge_errors[category] = judge_errors.get(category, 0) + 1
            continue
        by_cat.setdefault(category, []).append(bool(label.get("is_correct")))
    if len(by_cat) > 1 or judge_errors:
        print(f"\n── ③ 逐类（{len(by_cat)} 类）──")
        # ⚠ **ERROR 的那几道必须从分母里剔掉**，否则"判分链路丢题"会伪装成"这一类得 0 分"
        #   ——2026-10-03 的 base-mmb 就是这样：`multi_hop_clinical_deduction` 看起来
        #   0/17，其实是 15 道裁在输出被截断、2 道没跑到评委那一步。
        for category, flags in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            ok = sum(flags)
            tail = ""
            if judge_errors.get(category):
                tail = f"  ← 另有 {judge_errors[category]} 条 JUDGE_ERROR 未计入"
            print(f"   {category:34s} {ok:4d}/{len(flags):<4d} {_pct(ok, len(flags))}{tail}")
        for category, count in sorted(judge_errors.items()):
            if category not in by_cat:
                print(f"   {category:34s}    —/—     全部 {count} 条都是 JUDGE_ERROR")

    # ── ④ 上下文体量 ────────────────────────────────────────────────
    sizes = sorted(len(cache[k]) for k in cache)
    if sizes:
        print(
            f"\n── ④ 记忆文本体量（{len(sizes)} 题，归一化后字符数）──"
            f"\n   中位 {sizes[len(sizes) // 2]:,} · p90 {sizes[int(len(sizes) * 0.9)]:,}"
            f" · 最大 {sizes[-1]:,}"
        )

    # ── 结论 ────────────────────────────────────────────────────────
    print("\n── 结论 ──")
    verdicts: list[str] = []
    refused_with = table.get("拒答·证据在", 0)
    refused_without = table.get("拒答·证据不在", 0)
    # ⚠ **「证据在不在」这一支先过校准**：召回不到 80% 时说"证据不在 ⇒ 查语料"
    #   是**把人送去查错的那一层**（mmb 上实测召回只有 33%）。
    trustworthy = _recall_ceiling(calib) >= _CALIB_TRUSTWORTHY
    if refused_total and refused_with / refused_total >= 0.2 and trustworthy:
        verdicts.append(
            f"⚠ **{refused_with}/{refused_total} 的拒答题里证据就在它眼前**"
            " ⇒ 先查 **prompt**（有没有规则逼它放弃）"
            "——memtrapbench / mquake 当初都是这一类"
        )
    dataset = str(fingerprint.get("dataset") or "")
    if dataset in _NO_MEMORY_DATASETS:
        verdicts.append(
            f"ℹ️ **{dataset}：{_NO_MEMORY_DATASETS[dataset]}**"
            " ⇒ 判读树的「证据在不在」这一支**对它不适用**，别照着上面那两条读"
        )
    elif refused_without > refused_with and refused_total and trustworthy:
        verdicts.append(
            f"⚠ **{refused_without}/{refused_total} 的拒答题里证据不在**"
            " ⇒ 往上一层查**语料/加载器**（tempreason 缺 `fact_context`）"
            "，或**结构性**（corporatebench 装不下）"
        )
    elif not trustworthy and refused_total:
        verdicts.append(
            "⛔ **「证据在不在」这条判据在这份数据集上不可信**"
            "（gold 是长句 / 判分宽容 ⇒ 子串匹配大面积漏）"
            " ⇒ 它报的「证据不在」**不能**用来指认语料或加载器；"
            "要查就得**逐题读原文**"
        )
    errors = sum(v for k, v in by_label.items() if "ERROR" in k.upper())
    if labels and errors / len(labels) >= 0.05:
        verdicts.append(
            f"⚠ **{errors}/{len(labels)} 条 label 是 ERROR**"
            " ⇒ 查**判分链路**（解析失败 / 输出被截断）"
        )
    if len(by_cat) > 1:
        rates = [sum(v) / len(v) for v in by_cat.values() if v]
        if rates and max(rates) - min(rates) >= 0.4:
            verdicts.append("⚠ **逐类分化 ≥40pt** ⇒ 别读总分，读那一类")
    if not verdicts:
        verdicts.append("🟢 没看到上面那几类指纹——低分更像是**系统在这份数据集上的真实水平**")
    for line in verdicts:
        print(f"   {line}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑批诊断：把'低分'与'模型不行'分开（只读产物）")
    parser.add_argument("--run-id", required=True, help="eval/reports/runs/<run-id>")
    parser.add_argument(
        "--reports-dir", default="eval/reports", help="报告根目录（缺省 eval/reports）"
    )
    parser.add_argument(
        "--refusal",
        action="append",
        default=None,
        help="认定「拒答」的句子（可重复）。⚠ **换了数据集/prompt 之后要核对**",
    )
    args = parser.parse_args(argv)
    phrases = tuple(args.refusal) if args.refusal else DEFAULT_REFUSALS
    return clinical_report(args.run_id, Path(args.reports_dir), phrases)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
