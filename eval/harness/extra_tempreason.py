"""TempReason 家族的 answer / evaluate：A/B 定版的答案 prompt ＋ 可接受答案串纯函数判分。

拆自 `extra_pipeline.py`（2026-10-15，只搬位置）：那层现在是薄 façade、把本模块的名字
全部重新导出。
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

from eval.harness.extra_base import normalize  # noqa: E402


def judge_tempreason(generated: str, gold: object) -> tuple[bool, str]:
    """TempReason：**任一可接受答案串出现在回答里即算对**（归一化后）。

    ⚠ 口径是**我们定的**（上游没随数据发布评测脚本，我们只下了 4 个 json）。
    ⚠ **不做月份缩写/数字格式的等价**（`Mar, 1192` 与 `March 1192` 现在**不算**同一条）——
    这是已知的偏严，写在这里免得下一个人以为是 bug；真要放松就改这一处，两臂同时生效。
    """
    wants = [normalize(a) for a in (gold or []) if normalize(a)]
    if not wants:
        return False, "TempReason：金标为空"
    answer = normalize(generated)
    for want in wants:
        if want in answer:
            return True, f"TempReason：命中 {want!r}"
    return False, f"TempReason：{len(wants)} 个可接受答案都没出现"


#: **TempReason 专用**的答案 prompt（2026-10-04 加，A/B 定版）。
#:
#: ⛔ **不要对它用通用 `ANSWER_PROMPT`。** 那份第 2 条写的是「记忆里**没有**答案就逐字回
#: `Cannot determine from the memories.`」——而 **TempReason 的答案从来不在记忆里字面出现**：
#: 记忆给的是**区间**（`Raúl Servín plays for Atlas F.C. from Jan, 1991 to Jan, 1992.`），
#: 问题问的是**一个时点**（`… in May, 1991?`），要读完区间再落到点上。
#: 9B 的模型把 "contain" 读成字面匹配 ⇒ 一律拒答。
#:
#: ⚠ 第 2 条在这里**是准确的**：这份数据集的 gold 恒在上下文里（`tr-splitfix` 的 332 题，
#: 逐字拒答的 76 道**全部**"证据就在眼前"；校准口径下判对的题 212/212 都认得出 gold）。
#: ⇒ "必须作答"不是鼓励瞎猜，而是**去掉一条它误触发的弃权出口**。
#:
#: 实测（`tr-splitfix` 的 332 题，**只重答那 76 条逐字拒答、其余 256 条不动**，
#: 三个候选同条件对照；旧总分 0.6386）：
#:
#:     B1 只删拒答条款        新答对 44/76  ⇒ 0.7711
#:     B2 B1+「区间说明」     新答对 43/76  ⇒ 0.7681   ← 看起来更"对症"，实测**反而差一点**
#:     B3 本份（+必须作答）   新答对 52/76  ⇒ **0.7952**
#:
#: ⚠ **B2 那次是"想当然更对症"的反例**：单题探针上它就答错了（把 May 1991 答成
#: `Real Madrid Fc`），76 条上仍然不如 B1。**别按"哪份 prompt 更贴近任务"选，按数字选。**
TEMPR_ANSWER_PROMPT = (
    "You are an assistant that answers a question using the memories provided below.\n"
    "\n"
    "Rules:\n"
    "1. Use only the information in the memories. Do not use outside knowledge.\n"
    "2. The question is always answerable from the memories — commit to the best\n"
    "   answer they support, even if you must reason across time intervals.\n"
    "3. Answer concisely — one short sentence or the value itself.\n"
    "\n"
    "Memories:\n"
    "{memories}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Answer:"
)
