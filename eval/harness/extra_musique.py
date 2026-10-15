"""MuSiQue 家族的 answer / evaluate：两套答案 prompt ＋ 规则 / 模型两套裁判。

拆自 `extra_pipeline.py`（2026-10-15，只搬位置）：那层现在是薄 façade、把本模块的名字
全部重新导出。⛔ 拒答题恒走规则那一格的处置在 `judge_musique` 的注释里；
模型裁判的调用编排留在 façade（补丁点，见那边的注释）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ⚠ 与 `extra_pipeline.py` 同一处置：subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根。
#   本目录的模块既被 `eval.harness.*` 包内引用、也被当脚本/顶层模块跑，两种跑法都要能
#   解析到 `eval.*`（包内 import 时仓库根已在路径上，这里是幂等的）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets.musique import ANSWER_CONTRACT as MUSIQUE_ANSWER_CONTRACT  # noqa: E402
from eval.datasets.musique import LOOSE_ANSWER_CONTRACT as MUSIQUE_LOOSE_CONTRACT  # noqa: E402
from eval.datasets.musique import REFUSAL as MUSIQUE_REFUSAL  # noqa: E402
from eval.harness.extra_base import _contains, normalize  # noqa: E402


def judge_musique(
    generated: str, gold: object, *, contract: str = MUSIQUE_ANSWER_CONTRACT
) -> tuple[bool, str]:
    """本地 MuSiQue：可答题**归一化后与别名表比对**，不可答题严格匹配拒答标记。

    ## 两套口径（`contract` 选，**默认严**）

    | `contract` | 判据 | 什么时候用 |
    | --- | --- | --- |
    | `musique-local-qa-v1`（默认） | 归一化后**相等**于别名 | MuSiQue 原生那一档（含不可答题） |
    | `musique-local-qa-v3-direct` | **同上，逐字相同** | **链式多跳那一档**——只改答案 prompt |
    | `musique-local-qa-v2-loose` | 别名作为一个词出现在回答里 | 只问"捞到了没有"时 |

    ⚠ **宽松口径只放宽一处**：把"答得比金标更具体"也算对
    （`five` ← `up to five games per year`）。**它同时会制造假阳性**：
    金标 `California` 会被 `Rio Linda, California` 命中，而那是**比金标更笼统**的答案，
    按 MuSiQue 自己的口径**应当算错**。⇒ **两套口径必须并排报，不许拿宽松那套替换严的那套**。

    ⚠ 实测（2026-10-11 那一批 25 题）宽松口径**只翻 1 道**（`…46196…`，gold `five`）。
    ⇒ 它**不是**"分数低的原因"。改成它之前先看那份逐题对照。
    """
    if not isinstance(gold, dict) or not isinstance(gold.get("answerable"), bool):
        return False, "MuSiQue：金标缺 bool 类型的 answerable"
    if not gold["answerable"]:
        ok = generated.strip() == MUSIQUE_REFUSAL
        return ok, f"MuSiQue：不可答题必须只输出 {MUSIQUE_REFUSAL}"
    aliases = {
        normalize(value)
        for value in [gold.get("answer", ""), *(gold.get("answer_aliases") or [])]
        if normalize(value)
    }
    if not aliases:
        return False, "MuSiQue：金标答案及别名均为空"
    answer = normalize(generated)
    if contract == MUSIQUE_LOOSE_CONTRACT:
        hit = next((alias for alias in sorted(aliases) if _contains(answer, alias)), None)
        why = f"命中 {hit!r}" if hit else "别名一个都没出现"
        return hit is not None, f"MuSiQue/{MUSIQUE_LOOSE_CONTRACT}：{why}"
    return answer in aliases, "MuSiQue：参考答案或别名归一化后精确匹配"


# ── MuSiQue：可选模型裁判────────────────────────────────────────────────────
#
# ⚠ **为什么要有这一档**：`judge_musique` 是**归一化精确匹配**，而 MuSiQue 的 gold
#   **不是规范答案、是原文里的"答案跨度"**（可能是一整个从句），别名表又常常是空的
#   （本项目那 50 题里 32 题 `answer_aliases` 为 `[]`）。于是"语义正确但措辞不同"的答案
#   被系统性判错，实测（2026-10-13，MuSiQue 50 题）：
#
#     E0 混合链 严格 15/50 → 官方 token-F1(≥0.5) 22/50
#     E1-plan   严格 24/50 → 官方 token-F1(≥0.5) 32/50
#
#   ⇒ 少报 7–10 道。人复核那 9/10/9 道"原判错、语义正确"的就是这一批
#   （`22` ← `22 times`；`rises in northern Minnesota and meanders slowly southwards`
#   ← `southwards`；`for the conclave in rome` ← `Roncalli left Venice for the conclave in Rome.`)。
#
#
# ⚠ **默认仍是历史规则口径**：`rule` 那套原样保留并作为默认，
#   台账里的既有分数可以逐位复现。需要语义等价判分时显式传 `--musique-judge llm`。
#
# ⛔ **不动拒答那一条**：`answerable=false` 的分支仍走规则（"必须只输出拒答标记"是**输出契约**，
#   不是语义判断），理由见根 `CLAUDE.md` 那条"不要改别处的 refusal 口径"。
MUSIQUE_JUDGE_LLM = "llm"
MUSIQUE_JUDGE_RULE = "rule"
MUSIQUE_JUDGE_MODES: tuple[str, ...] = (MUSIQUE_JUDGE_LLM, MUSIQUE_JUDGE_RULE)
DEFAULT_MUSIQUE_JUDGE = MUSIQUE_JUDGE_RULE


#: 模型裁判的提示词。**我们写的**（MuSiQue 官方没有发布 AML 判分 pipeline）。
#: 只给"问题 + 参考答案（+别名） + 模型答案"——**不给检索到的段落**：
#: 判的是"答得对不对"，不是"检索够不够"（后者由覆盖率那一列回答）。
MUSIQUE_JUDGE_PROMPT = """\
You grade whether a model's answer means the same thing as the reference answer.

Question: {question}
Reference answer: {gold}{aliases}

Model answer: {response}

How to judge:
1. Judge MEANING, not wording. The reference is taken from the source text and may be a whole
   span (a clause or a full sentence). A shorter answer that states the same fact is CORRECT,
   and an answer that wraps the same fact in a sentence or adds correct qualifiers is CORRECT.
2. Equivalent surface forms are fine: "22" = "22 times"; "May 1990" = "in May 1990";
   a full name with or without a middle initial.
3. Do NOT accept an answer that is LESS specific than the reference (a broader category, a
   superordinate place or person, a different year, a different entity), and do not accept one
   that contradicts it.
4. Judge only from the question and the reference. Do not use outside knowledge to decide
   whether the reference itself is right.

Reply with ONE short line of reasoning, then a final line that is exactly:
Verdict: CORRECT
or
Verdict: WRONG"""


def build_musique_judge_prompt(question: str, gold: object, generated: str) -> str:
    """把金标（answer + 别名）与模型答案填进裁判模板。"""
    answer = ""
    aliases: list[str] = []
    if isinstance(gold, dict):
        answer = str(gold.get("answer") or "")
        aliases = [str(a) for a in (gold.get("answer_aliases") or []) if str(a).strip()]
    block = f"\nAccepted alternative wordings: {' | '.join(aliases)}" if aliases else ""
    return MUSIQUE_JUDGE_PROMPT.format(
        question=question,
        gold=answer or "(none)",
        aliases=block,
        response=(generated or "").strip() or "(empty)",
    )


def judge_musique_llm_verdict(raw_response: str) -> tuple[bool | None, str]:
    """解析模型裁判的 `Verdict: CORRECT|WRONG`。**解不出来回 `None`**（调用方记 `JUDGE_ERROR`）。

    ⚠ **优先认 `Verdict:` 那一行**，认不到才退化成"全文最后一次出现的 CORRECT/WRONG"
    ——因为推理那一行里可能提到反例（"...would be WRONG if it said 1916"），
    而那**不是**结论。两处都认不到 ⇒ `None`，**绝不默认成对**。
    """
    text = (raw_response or "").strip()
    if not text:
        return None, "MuSiQue/llm：裁判响应为空"
    if text.startswith("JUDGE CALL FAILED"):
        return None, f"MuSiQue/llm：{text[:200]}"
    tagged = re.findall(r"Verdict\s*[:：]\s*(CORRECT|WRONG)\b", text, re.IGNORECASE)
    if tagged:
        ok = tagged[-1].upper() == "CORRECT"
        return ok, f"MuSiQue/llm：Verdict={tagged[-1].upper()}｜{text[:400]}"
    loose = re.findall(r"(?<![A-Z])(CORRECT|WRONG)(?![A-Z])", text.upper())
    if loose:
        ok = loose[-1] == "CORRECT"
        return ok, f"MuSiQue/llm（无 Verdict 行，取末次出现）：{loose[-1]}｜{text[:400]}"
    return None, "MuSiQue/llm：响应里既没有 Verdict 行、也没有 CORRECT/WRONG"


MUSIQUE_ANSWER_PROMPT = (
    "Answer the question using ONLY the retrieved candidate paragraphs below. "
    "Do not use outside knowledge. Return only the short answer, without explanation. "
    f"If the paragraphs are insufficient, return exactly: {MUSIQUE_REFUSAL}\n\n"
    "Retrieved candidate paragraphs:\n{memories}\n\nQuestion: {question}\n\nAnswer:"
)


#: **链式多跳那一档**（`DIRECT_ANSWER_CONTRACT`）：**不给拒答出口**。
#:
#: 与上面那份**只差那一句**——`If the paragraphs are insufficient, return exactly: …`。
#: ⛔ 那一句在这一档上不是保险，是**出口**：本仓量过两次它的后果（250 题上 160/250 逐字拒答、
#: 通过 0；TempReason 76 道拒答而证据就在眼前）——只删那句子就分别把 0.3997 拉到 0.4857、
#: 把 44/76 拉成答对。MuSiQue 链式这 25 题上同样 10/25 是模型自己拒答。
#: ⚠ 判分**不变**（仍是 v1 的归一化精确匹配）⇒ 拒答在这一档照样算错，只是不再**被提示词邀请**。
#: ⛔ **只换这一档**：MuSiQue 原生的可答/不可答变体、以及别的数据集，评测口径一个字不改。
MUSIQUE_DIRECT_ANSWER_PROMPT = (
    "Answer the question using ONLY the retrieved candidate paragraphs below. "
    "Do not use outside knowledge. Return only the short answer, without explanation.\n\n"
    "Retrieved candidate paragraphs:\n{memories}\n\nQuestion: {question}\n\nAnswer:"
)
