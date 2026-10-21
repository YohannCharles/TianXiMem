"""MQuAKE 家族的 answer / evaluate：答案 prompt ＋ 别名表纯函数判分。

`extra_pipeline.py` 是薄 façade，重新导出本模块的名字。
**"这是我们的契约、不是官方口径"**（分数只在仓内前后对比）那条边界见
`extra_pipeline.py` 的模块 docstring 第 1 条。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ⚠ 与 `extra_pipeline.py` 同一处置：subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根。
#   本目录的模块既被 `eval.harness.*` 包内引用、也被当脚本/顶层模块跑，两种跑法都要能
#   解析到 `eval.*`（包内 import 时仓库根已在路径上，这里是幂等的）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.harness.extra_base import _contains, normalize  # noqa: E402

# ── 判分：纯函数（可离线重算、可单测）──────────────────────────────────────


def judge_mquake(generated: str, gold: object) -> tuple[bool, str]:
    """MQuAKE：`gold` 是别名表，**任一命中即对**。"""
    aliases = [normalize(a) for a in (gold or []) if a]
    if not aliases:
        return False, "MQuAKE：金标别名表为空"
    answer = normalize(generated)
    for alias in sorted(aliases, key=len, reverse=True):
        if answer == alias or _contains(answer, alias):
            return True, f"MQuAKE：命中别名 {alias!r}"
    return False, f"MQuAKE：{len(aliases)} 个别名一个都没出现"


#: **MQuAKE 专用**的答案 prompt。
#:
#: ⛔ **不要对它用通用 `ANSWER_PROMPT`。** 那份有两条对 MQuAKE **有害**的规则：
#:   · 「只用记忆、不用外部知识」——而 **MQuAKE-CF 的"未编辑跳"本来就要求世界知识**
#:     （那是它的设计），禁掉就答不出；
#:   · 「记忆里没有就逐字回 `Cannot determine from the memories.`」——记忆里
#:     **旧值与 "…is X (this replaces the earlier value)" 并存**，模型一看到冲突就引用这条拒答。
#:
#: ⚠ **这是 MQuAKE 的接法问题，不是"通用 prompt 该改"**：通用那份还被
#: medmemorybench / CorporateBench 的单值查询仍沿用；CorporateBench 聚合任务已独立。
#: （tempreason **不在**名单里——它有自己那份 `TEMPR_ANSWER_PROMPT`。）
MQUAKE_ANSWER_PROMPT = (
    "You are an assistant answering a question using the memories provided below.\n\n"
    "The memories may contain **updates**: a statement marked as replacing an earlier one.\n"
    "When two statements conflict, **the replacement is the current truth** — use it.\n\n"
    "Rules:\n"
    "1. Prefer the memories. If a step of the reasoning chain is missing from them,\n"
    "   you may fill it with what you already know.\n"
    "2. Do not reply that you cannot determine the answer unless the memories are\n"
    "   genuinely unhelpful — a conflict between an old and a replacing statement is\n"
    "   **not** a reason to refuse.\n"
    "3. Answer concisely — one short sentence or the value itself.\n\n"
    "Memories:\n"
    "{memories}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
