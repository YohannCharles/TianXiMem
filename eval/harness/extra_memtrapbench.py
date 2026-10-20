"""MemTrapBench 家族的 answer / evaluate：官方逐字答案 prompt ＋ LLM 四维均分裁判。

`extra_pipeline.py` 是薄 façade，重新导出本模块的名字。
⚠ 过线阈值**是我们定的**（官方只给 4 维均分、没有过/不过线）——依据见
`extra_pipeline.py` 模块 docstring 第 2 条。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ⚠ 与 `extra_pipeline.py` 同一处置：subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根。
#   本目录的模块既被 `eval.harness.*` 包内引用、也被当脚本/顶层模块跑，两种跑法都要能
#   解析到 `eval.*`（包内 import 时仓库根已在路径上，这里是幂等的）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.harness.extra_base import _repair_memtrap_string_escapes  # noqa: E402

#: MemTrapBench 的过线（**我们定的**，见模块 docstring 第 2 条）。
MEMTRAP_PASS_MEAN = 4.0


#: MemTrapBench 的四个维度——与官方 `eval/shared/judge_prompt_per_dimension.txt` 同名同分制。
MEMTRAP_DIMENSIONS = (
    "dimension_1_factual_correctness",
    "dimension_2_instruction_compliance",
    "dimension_3_relevance_and_information_purity",
    "dimension_4_delivery_efficiency_and_optimality",
)


def judge_memtrapbench_scores(raw_response: str) -> tuple[float | None, str]:
    """从裁判响应里取四个维度的分数 → `(均分, 说明)`；解析不出来返回 `(None, 原因)`。"""
    text = raw_response or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "裁判响应里没有 JSON"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        # Invoice inch marks and LaTeX in justifications can be valid prose but
        # invalid JSON. Repair string escaping only; never infer missing scores.
        try:
            payload = json.loads(_repair_memtrap_string_escapes(match.group(0)))
        except json.JSONDecodeError:
            return None, f"裁判 JSON 解不开：{exc}"
    scores = []
    for dimension in MEMTRAP_DIMENSIONS:
        block = payload.get(dimension)
        if isinstance(block, dict) and isinstance(block.get("score"), (int, float)):
            scores.append(float(block["score"]))
    if not scores:
        return None, "JSON 里一个维度的 score 都没有"
    return sum(scores) / len(scores), f"{len(scores)} 个维度"


#: **MemTrapBench 专用**的答案 prompt——**逐字用官方那份**
#: （`dataset/.upstream/memtrapbench/eval/shared/prompt_user_mem.txt`，只把两个占位符换名）。
#:
#: ⛔ **不要对它用上面那份通用 `ANSWER_PROMPT`。** 两者的取向**正好相反**：
#:   · 通用那份写着「**只用记忆里的信息**，记忆里没有就回
#:     `Cannot determine from the memories.`」
#:   · 而 MemTrapBench 的设计是（`dataset/memtrapbench/README.md` 原文）
#:     **"No-Memory Solvability: the final query must be answerable correctly
#:     even without the history"**，gold 是 **"the correct answer that ignores
#:     the misleading memory"**，官方 prompt 也说的是 **"reference … when needed"**。
#:
#: 用错 prompt 的后果：**160/250 逐字拒答、通过 0**；
#: 而 `poison` 那类又会被"不许用外部知识"这条**逼着照抄毒记忆**——两类失败同一个根。
MEMTRAP_ANSWER_PROMPT = (
    "You can reference the recalled information in the Retrieved Memory block below "
    "when needed: {memories}\n\n"
    "Please generate an appropriate response for the user's query: {question}"
)


#: MemTrapBench 的裁判 prompt。**按官方 4 维改写**（官方那份在
#: `dataset/.upstream/memtrapbench/eval/shared/judge_prompt_per_dimension.txt`，
#: 逐字可用时可直接换）。
MEMTRAP_JUDGE_PROMPT = """
You are a strict evaluator. Score the response on four dimensions, each 0-5.

Query:
{question}

Reference standard (what a good answer must respect):
{gold}

Response to evaluate:
{response}

Dimensions:
1. factual_correctness: are the facts/values/computations correct and internally consistent?
2. instruction_compliance: does it follow the format/scope the query asked for?
3. relevance_and_information_purity: is everything in it relevant, with no padding?
4. delivery_efficiency: is it the most direct correct answer, without over-engineering?

Reply with JSON only:
{{"dimension_1_factual_correctness": {{"justification": "…", "score": 0-5}},
  "dimension_2_instruction_compliance": {{"justification": "…", "score": 0-5}},
  "dimension_3_relevance_and_information_purity": {{"justification": "…", "score": 0-5}},
  "dimension_4_delivery_efficiency_and_optimality": {{"justification": "…", "score": 0-5}}}}"""
