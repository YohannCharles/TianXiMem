"""TempReason 家族的 answer / evaluate：A/B 定版的答案 prompt ＋ 可接受答案串纯函数判分。

`extra_pipeline.py` 是薄 façade，重新导出本模块的名字。
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


#: TempReason 使用的专用答案 prompt。
#:
#: 通用 `ANSWER_PROMPT` 在找不到字面答案时要求拒答，但 TempReason 的答案通常需要从
#: 原文时间区间中推导。此处的规则要求模型在证据支持时给出简洁答案。
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
