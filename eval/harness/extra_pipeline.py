#!/usr/bin/env python3
"""我们自写的五份 `official-extra` 数据集的 `answer` / `evaluate`——**官方没有发布它们的 pipeline**。

CLI 与归档 pipeline **同形**（harness 的 `run_judge` 就是按这个形状调的）：

    python extra_pipeline.py answer --input items.jsonl --output answers.jsonl [--max-tokens N]
    python extra_pipeline.py evaluate --input items.jsonl --answers answers.jsonl \
        --output labels.jsonl

输入项由 [`judge._build_extra_items`](./judge.py) 产出：
`{id, dataset, question, gold_answer, retrieved_context, category}`

输出：

    answer   → `{"id", "generated_answer"}`（**追加**模式，按 id 跳过已完成的题）
    evaluate → `{"id", "is_correct", "label", "judge_response"}`（**覆盖**模式）

## 三句必须先读的话

1. **这是"我们自己的契约"，不是官方口径。** 官方那 6 份的 answer/judge prompt 在
   `dataset/.upstream/aml/pipeline_*.py` 里逐字可查；这五份没有 ⇒ **分数只能在本仓内部前后对比**，
   别拿去和官方分数对齐（§12.4：跨数据集的裁判不同，横向比没有意义）。
2. **裁判分两种**：MQuAKE / CorporateBench / TempReason 是**纯函数**（字符串 / 集合匹配，不花钱、
   确定性强、可离线重算），MedMemoryBench 照上游发布的 `metrics/`；**MemTrapBench 必须 LLM**——
   官方只给 4 维 0–5 的**均分**、没有过/不过线，所以：
   * 阈值 `MEMTRAP_PASS_MEAN` 是**我们定的**；
   * **四个原始分原样写进 `judge_response`** ⇒ 换阈值**不用重跑裁判**。
3. **失败不静默**：判分出错时 `label` 带 `JUDGE_ERROR`、`is_correct=False`，
   原始响应留着——与 CL-Bench"API/JSON 失败一律记 0"同一处置，但**在 label 里说明白**。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import string
import sys
import unicodedata
from pathlib import Path

# ⚠ 归档目录走 D16 的唯一入口（`benchmark_dir()` 读 `TIANXIMEM_BENCHMARK_DIR`，
#   默认 `dataset`）。subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根，
#   与 `tools/*.py` 同一处置。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets.musique import ANSWER_CONTRACT as MUSIQUE_ANSWER_CONTRACT  # noqa: E402
from eval.datasets.musique import REFUSAL as MUSIQUE_REFUSAL  # noqa: E402
from eval.datasets.registry import benchmark_dir  # noqa: E402
from eval.harness.corporatebench_pipeline import (  # noqa: E402
    build_answer_prompt as build_corporatebench_answer_prompt,
)
from eval.harness.corporatebench_pipeline import (  # noqa: E402
    judge_corporatebench,
    score_corporatebench,
)
from eval.jsonl_io import read_jsonl, write_line  # noqa: E402

__all__ = [
    "MEMTRAP_PASS_MEAN",
    "build_docpp_judge_prompt",
    "judge_docpp_verdict",
    "judge_tempreason",
    "judge_corporatebench",
    "judge_memtrapbench_scores",
    "judge_mquake",
    "judge_musique",
    "normalize",
]

#: MemTrapBench 的过线（**我们定的**，见模块 docstring 第 2 条）。
MEMTRAP_PASS_MEAN = 4.0

#: MemTrapBench 的四个维度——与官方 `eval/shared/judge_prompt_per_dimension.txt` 同名同分制。
MEMTRAP_DIMENSIONS = (
    "dimension_1_factual_correctness",
    "dimension_2_instruction_compliance",
    "dimension_3_relevance_and_information_purity",
    "dimension_4_delivery_efficiency_and_optimality",
)

ANSWER_TIMEOUT = 180.0
JUDGE_TIMEOUT = 180.0


# ── 归一化：所有精确判分共用一份（改它等于改分，别在别处再写一个）───────────────
def normalize(text: object) -> str:
    """NFKC + 转小写 + 空白折叠 + 去掉常见包装标点。"""
    raw = unicodedata.normalize("NFKC", str(text if text is not None else ""))
    return re.sub(r"\s+", " ", raw).strip().lower().strip("。.,;:!?\"'`()[]{}")


def _contains(haystack: str, needle: str) -> bool:
    """`needle` 是否作为**一个词**出现在 `haystack` 里。

    ASCII 词用 `\\b` 边界——否则 MQuAKE 那种长度 2 的别名（`de` = German）会满屏假阳性。
    非 ASCII（中文等）没有词边界，退回子串判断。
    """
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9 .'/-]+", needle):
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack


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


def judge_musique(generated: str, gold: object) -> tuple[bool, str]:
    """本地 MuSiQue：可答题归一化后精确匹配别名，不可答题严格匹配拒答标记。"""
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
    return normalize(generated) in aliases, "MuSiQue：参考答案或别名归一化后精确匹配"


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


def judge_memtrapbench_scores(raw_response: str) -> tuple[float | None, str]:
    """从裁判响应里取四个维度的分数 → `(均分, 说明)`；解析不出来返回 `(None, 原因)`。"""
    text = raw_response or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "裁判响应里没有 JSON"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"裁判 JSON 解不开：{exc}"
    scores = []
    for dimension in MEMTRAP_DIMENSIONS:
        block = payload.get(dimension)
        if isinstance(block, dict) and isinstance(block.get("score"), (int, float)):
            scores.append(float(block["score"]))
    if not scores:
        return None, "JSON 里一个维度的 score 都没有"
    return sum(scores) / len(scores), f"{len(scores)} 个维度"


# ── MedMemoryBench（zh）：判分口径**照上游 `metrics/` 实现**，不是我们发明的 ──────────
#
# 上游仓（`dataset/.upstream/medmemorybench/`，另下的 GitHub 份）里：
#   `metrics/__init__.py` 的 DEFAULT_METRIC_MAPPING 把 6 类题分给三个度量：
#     entity_exact_match → string_contain · multiple_choice → option_match
#     temporal_localization / state_update / inference_generation → llm_judge
#     multi_hop_clinical_deduction → llm_judge_mcd
# 本文件下面两条是**逐行照抄它的规则**（含标点集与那 6 条选项正则）；
# LLM 那几类的 prompt **直接从归档里的 `utils/prompts_judge.py` 读**（`ast.literal_eval`，
# 不 import、不复制）——复制一份就等于开了第二个家。
MMB_CODE_DIR = "medmemorybench-code"
MMB_PROMPTS = "utils/prompts_judge.py"

#: 上游 `string_match.normalize_text` 的标点集：ASCII + 中文标点。
_MMB_PUNCT = string.punctuation + "，。！？、；：''（）【】《》·…—～－–·"

#: 上游 `extract_option_letters` 的六条模式（顺序不影响结果，∪ 起来）。
_MMB_OPTION_PATTERNS = (
    r"\b([A-F])\b",
    r"选([A-F])",
    r"答案[是为：:]*\s*([A-F])",
    r"choose\s*([A-F])",
    r"answer[:\s]*([A-F])",
    r"([A-F])选项",
)


def mmb_normalize(text: object) -> str:
    """上游 `normalize_text`：**去标点 + 去空白 + 转小写**（不折叠、不切词）。"""
    stripped = str(text if text is not None else "").translate(str.maketrans("", "", _MMB_PUNCT))
    return "".join(stripped.split()).lower()


def mmb_option_letters(text: object) -> set[str]:
    """上游 `extract_option_letters`：六条正则一起扫（在**大写化**后的文本上）。"""
    upper = str(text if text is not None else "").upper()
    found: set[str] = set()
    for pattern in _MMB_OPTION_PATTERNS:
        found.update(re.findall(pattern, upper))
    return found


def judge_mmb_string_contain(generated: str, gold: dict) -> tuple[bool, str]:
    """`entity_exact_match`：**每一条** `is_correct` 的答案都要在输出里出现，缺一即错。"""
    expected = [str(a.get("content", "")) for a in gold.get("answers") or [] if a.get("is_correct")]
    if not expected:
        return False, "MedMemoryBench/string_contain：没有 is_correct 的答案"
    out = mmb_normalize(generated)
    matched = [e for e in expected if mmb_normalize(e) and mmb_normalize(e) in out]
    ok = len(matched) == len(expected)
    return ok, f"MedMemoryBench/string_contain：命中 {len(matched)}/{len(expected)}"


def judge_mmb_option_match(generated: str, gold: dict) -> tuple[bool, str]:
    """`multiple_choice`：**选项集合完全相等**才算对（多选、少选、多选错都算错）。"""
    correct: set[str] = set()
    for answer in gold.get("answers") or []:
        if not answer.get("is_correct"):
            continue
        match = re.match(r"^([A-F])[.、:\s]", str(answer.get("content", "")).upper())
        if match:
            correct.add(match.group(1))
    selected = mmb_option_letters(generated)
    ok = bool(correct) and selected == correct
    return ok, f"MedMemoryBench/option_match：选中 {sorted(selected)}，正确 {sorted(correct)}"


def _mmb_prompt_templates() -> dict[str, str]:
    """从归档里读上游的裁判模板（**纯数据文件**，`ast.literal_eval` 即可，不必 import）。"""
    from eval.datasets.layout import archive_file

    path = archive_file(benchmark_dir(), f"{MMB_CODE_DIR}/{MMB_PROMPTS}")
    if not path.exists():
        raise SystemExit(
            f"缺 {path}——MedMemoryBench 的裁判 prompt 在上游代码仓里（见 docs/benchmark-data.md）"
        )
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "JUDGE_TEMPLATES":
            return ast.literal_eval(node.value)
    raise SystemExit(f"{path} 里找不到 JUDGE_TEMPLATES——上游改过结构？")


def _mmb_metadata_info(metadata: dict) -> str:
    """上游 `judge_inference_generation` 里那段 `metadata_info` 的拼法。"""
    info = ""
    if metadata.get("inference_type"):
        info += f"\nInference type: {metadata['inference_type']}"
    trap = metadata.get("trap_design") or {}
    if isinstance(trap, dict):
        if trap.get("trap_mechanism"):
            info += f"\nTrap mechanism: {trap['trap_mechanism']}"
        if trap.get("required_patient_info"):
            info += f"\nRequired patient info: {', '.join(trap['required_patient_info'])}"
    wrong = metadata.get("common_wrong_answer")
    if isinstance(wrong, dict):
        info += f"\nCommon wrong answer: {wrong.get('content', '')}"
        info += f"\nError reason: {wrong.get('why_wrong', '')}"
    return info


def _mmb_mcd_params(metadata: dict) -> dict:
    """上游 `judge_multi_hop_clinical_deduction` 里那两段的拼法（节点逐条列出）。"""
    nodes = ""
    for index, node in enumerate(metadata.get("reasoning_chain") or []):
        nodes += (
            f"\nNode {node.get('node_id', index + 1)}:"
            f"\n  - Source: Session {node.get('session_id', '?')} ({node.get('source_info', '')})"
            f"\n  - Role: {node.get('role', '')}"
            f"\n  - Content: {node.get('content', '')}\n"
        )
    if nodes:
        nodes = "\n[Reasoning chain nodes (to be verified one by one)]\n" + nodes
    required = ""
    for node in metadata.get("required_memory_nodes") or []:
        required += f"- {node}\n"
    if required:
        required = "\n[Information that must be recalled from memory]\n" + required
    return {
        "nodes_for_validation": nodes,
        "required_nodes_str": required,
        "hop_count": metadata.get("hop_count", 0),
        "reasoning_pattern": metadata.get("reasoning_pattern", ""),
    }


def build_mmb_judge_prompt(query_type: str, question: str, generated: str, gold: dict) -> str:
    """四类 LLM 裁判的 prompt——模板取自上游，参数填法也照上游那个 wrapper。"""
    template = _mmb_prompt_templates().get(f"medmemorybench_{query_type}_judge")
    if not template:
        raise SystemExit(f"上游模板里没有 medmemorybench_{query_type}_judge")
    answers = gold.get("answers") or []
    first_correct = next((a for a in answers if a.get("is_correct")), {})
    params = {
        "question": question,
        # ⚠ 上游只取**第一条**正确答案与它的 explanation（`expected_answers[0]`）
        "expected_answer": str(first_correct.get("content", "")),
        "explanation": str(first_correct.get("explanation", "")),
        "model_output": generated,
    }
    metadata = gold.get("metadata") or {}
    if query_type == "inference_generation":
        params["metadata_info"] = _mmb_metadata_info(metadata)
    elif query_type == "multi_hop_clinical_deduction":
        params.update(_mmb_mcd_params(metadata))
    return template.format(**params)


#: 走 LLM 裁判的四类（其余两类的口径见上面两条纯函数）。
MMB_LLM_TYPES = (
    "temporal_localization",
    "state_update",
    "inference_generation",
    "multi_hop_clinical_deduction",
)

#: 多跳那一类的**单独预算**——上游 `judge_multi_hop_clinical_deduction` 就是
#: `self._call_llm(prompt, max_tokens=2000)`（其余三类 500）。
#:
#: ⚠ **它的输出比其余三类长一个量级**：要逐节点回 `node_validations`，每个节点还带
#: `note`。2026-10-03 的 base-mmb 实测——**17 道 mcd 里 15 道**的裁判输出被砍在
#: 618–778 字符处，JSON 没收尾 ⇒ 全记 `JUDGE_ERROR`（而 `is_correct` 一律当 False
#: ⇒ 看起来像"这一类能力为 0"，其实是**判分链路在丢题**）。
#: 单独按 2000 实测 9 道：输出 1,286–1,801 字符（中位 1,483）、**全部正常收尾并解析**，
#: 其中一道判 `True` ⇒ 修复后这一类**是可分辨的**，不是结构性 0 分。
MMB_MCD_MAX_TOKENS = 2000


def judge_mmb_llm_verdict(raw_response: str) -> tuple[bool | None, str]:
    """从裁判响应里取 `is_correct`（上游 `_extract_json_from_text` 的等价物）。

    ⚠ **不能用非贪婪正则找 `{...}`**：多跳那一类（`llm_judge_mcd`）的裁决是**嵌套** JSON
    （`node_validations` 里还有一层对象），非贪婪匹配只会命中里层的节点对象，永远够不到
    外层的 `is_correct`——2026-09-30 实测就是这么把一条**判对了的**回答记成 `JUDGE_ERROR` 的。
    `raw_decode` 从每个 `{` 起尝试解析**完整的** JSON 值，嵌套天然没问题。
    """
    text = raw_response or ""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        verdict = _as_bool(payload.get("is_correct")) if isinstance(payload, dict) else None
        if verdict is not None:
            return verdict, str(payload.get("reason", ""))

    # 严格路径全灭 ⇒ 兜底：把字符串里的裸引号转义掉再来一遍（见 `_escape_stray_quotes`）。
    head, tail = text.find("{"), text.rfind("}")
    if 0 <= head < tail:
        try:
            payload = json.loads(_escape_stray_quotes(text[head : tail + 1]))
        except json.JSONDecodeError:
            payload = None
        verdict = _as_bool(payload.get("is_correct")) if isinstance(payload, dict) else None
        if verdict is not None:
            reason = str(payload.get("reason", ""))
            return verdict, f"{reason}（⚠ 判词含未转义引号，转义后解析）"
    return None, "裁判响应里没有可解析的 is_correct"


def _as_bool(value: object) -> bool | None:
    """`is_correct` 归一化。**读不出布尔就交回 `None`（⇒ `JUDGE_ERROR`），不要瞎猜。**

    ⚠ 不归一化会让 `bool("false")` **为真**——裁判偶尔把布尔写成字符串，而那样
    一条**判负**的回答会被静默记成**判对**。宁可响亮地记一条 ERROR。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _escape_stray_quotes(text: str) -> str:
    """把**字符串值里**的裸 `"` 转义掉。**只在严格解析已经失败之后才配用它。**

    ## 为什么需要它（2026-10-03 实测）

    裁判写中文理由时会**混用全角开引号与半角闭引号**：

    ```text
    …且未明确引用或体现患者“终身禁用 NSAIDs"这一关键禁忌症信息…
    ```

    那个半角 `"` 把 JSON 字符串**提前截断** ⇒ 整份 JSON 非法 ⇒ 记 `JUDGE_ERROR`
    ——而**判词本身是完整、可读、判断正确的**。上游 `_extract_json_from_text` 的
    括号计数法在这里**同样失效**（它靠 `"` 翻转 `in_string`，撞上这个引号后
    后面的 `}` 被当成了字符串内容），所以这不是"照上游实现"能解决的事。

    ## 判据：这个 `"` 后面是不是结构字符

    在字符串里，`"` 只有在**后面（跳过空白）跟着 `,` `}` `]` `:` 或已到结尾**时
    才是真的收尾引号。`…NSAIDs"这一…` 后面是 `这` ⇒ 它是内容，转义；
    `…推理。"}` 后面是 `}` ⇒ 收尾，照旧。
    """
    out: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and in_string:  # 已转义的序列整个跳过，别二次转义
            out.append(text[index : index + 2])
            index += 2
            continue
        if char != '"':
            out.append(char)
            index += 1
            continue
        if not in_string:
            in_string = True
        else:
            look = index + 1
            while look < len(text) and text[look] in " \t\r\n":
                look += 1
            if look >= len(text) or text[look] in ",}]:":
                in_string = False
                out.append(char)
            else:
                out.append('\\"')
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


# ── Doc-PP：判分口径**照上游 `prompts/judge_evaluation.py`**（`with_policy=false` 两条路）──
#
# 上游按 `type` × `with_policy` 分四种 case（见 `src/03_judge_evaluation.py` 的注释）。
# **我们走 `with_policy=false`**：记忆里只有文档正文、没有把政策条文喂进去——
# 于是 direct 走"回答里有没有那个信息"、indirect 走"逐条清单满足了吗"。
DOCPP_PROMPTS = "doc-pp/prompts/judge_evaluation.py"


def _docpp_prompts() -> dict[str, str]:
    """从归档里读上游的裁判模板（纯数据文件；赋值形如 `NAME = 三引号字符串.strip()`）。"""
    from eval.datasets.layout import archive_file

    path = archive_file(benchmark_dir(), DOCPP_PROMPTS)
    if not path.exists():
        raise SystemExit(f"缺 {path}——Doc-PP 的裁判 prompt 在上游仓里（见 docs/benchmark-data.md）")
    values: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        name = getattr(node.targets[0], "id", "")
        value = node.value
        if isinstance(value, ast.Call) and getattr(value.func, "attr", "") == "strip":
            # ⚠ `"""…""".strip()` 里字符串是**接收者**（`func.value`），而 `.strip()` **不收参数**
            #   ——取 `args[0]` 会拿到空、整条抽取静默返回 {}（2026-09-30 实测）。
            receiver = getattr(value.func, "value", None)
            if isinstance(receiver, ast.Constant):
                value = receiver
        if name and isinstance(value, ast.Constant) and isinstance(value.value, str):
            values[name] = value.value
    return values


def build_docpp_judge_prompt(question: str, generated: str, gold: dict) -> tuple[str, str]:
    """返回 `(合并后的 prompt, 这一条的口径名)`——两条都来自上游模板。"""
    prompts = _docpp_prompts()
    kind = str(gold.get("type") or "")
    policy_value = str(gold.get("policy_value") or "")
    checklist = [str(c) for c in (gold.get("checklist") or [])]
    if kind == "indirect":
        # 上游：`combined_criteria = [「回答里要有 policy_value」] + checklist`，再逐条编号
        criteria = [f"The response contains the following information: {policy_value}", *checklist]
        body = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(criteria))
        template = prompts["JUDGE_INDIRECT_WITHOUT_POLICY_USER"]
        prompt = (
            prompts["JUDGE_INDIRECT_WITHOUT_POLICY_SYSTEM"]
            + "\n\n"
            + template.replace("{combined_criteria}", body).replace("{model_response}", generated)
        )
        return prompt, "indirect/checklist"
    template = prompts["JUDGE_DIRECT_WITHOUT_POLICY_USER"]
    prompt = (
        prompts["JUDGE_DIRECT_WITHOUT_POLICY_SYSTEM"]
        + "\n\n"
        + template.replace("{policy_value}", policy_value).replace("{model_response}", generated)
    )
    return prompt, "direct/present"


def judge_docpp_verdict(raw_response: str, kind: str) -> tuple[bool | None, str]:
    """解析上游那两条的 JSON：direct 取 `present`、indirect 取 `results`（**全真才算对**）。"""
    text = raw_response or ""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if kind == "indirect/checklist":
            results = payload.get("results")
            if isinstance(results, list) and results:
                passed = sum(1 for item in results if item is True)
                reason = str(payload.get("reasoning", ""))[:300]
                return all(item is True for item in results), (
                    f"Doc-PP/checklist：{passed}/{len(results)} 条满足｜{reason}"
                )
        elif "present" in payload:
            return bool(payload["present"]), (
                f"Doc-PP/direct：present={payload['present']}｜{payload.get('reasoning', '')[:300]}"
            )
    return None, "Doc-PP：裁判响应里没有可解析的 present/results"


# ── 模型调用 ────────────────────────────────────────────────────────────────
def _config(which: str) -> tuple[str, str, str]:
    """取端点配置。**与归档 pipeline 同一个来源**：`eval/harness/api_config.py`。

    它在 subprocess 里由 `run_judge` 经 `PYTHONPATH` 注入；单元测试 import 本模块时
    不会走到这里（`answer` / `evaluate` 才需要）。
    """
    try:
        import api_config
    except ImportError as exc:  # pragma: no cover — 只在没按 harness 的方式跑时发生
        raise SystemExit(
            "找不到 api_config（应当由 eval/harness/judge.py 的 run_judge 经 PYTHONPATH 注入）"
        ) from exc
    if which == "judge":
        return api_config.JUDGE_API_BASE, api_config.JUDGE_API_KEY, api_config.JUDGE_MODEL
    return api_config.ANSWER_API_BASE, api_config.ANSWER_API_KEY, api_config.ANSWER_MODEL


def _chat(base: str, key: str, model: str, prompt: str, *, max_tokens: int, timeout: float) -> str:
    import httpx

    if not base:
        raise SystemExit("端点未配置（AML_BASE_URL）——本脚本不能离线跑模型调用")
    response = httpx.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


#: 答案 prompt。**我们写的**（官方那 6 份的模板在 `dataset/.upstream/aml/pipeline_*.py`）。
ANSWER_PROMPT = """
You are an assistant that answers a question using ONLY the memories provided below.

Rules:
1. Use only the information in the memories. Do not use outside knowledge.
2. If the memories do not contain the answer, reply exactly: Cannot determine from the memories.
3. Answer concisely — one short sentence or the value itself.

Memories:
{memories}

Question: {question}

Answer:"""

MUSIQUE_ANSWER_PROMPT = (
    "Answer the question using ONLY the retrieved candidate paragraphs below. "
    "Do not use outside knowledge. Return only the short answer, without explanation. "
    f"If the paragraphs are insufficient, return exactly: {MUSIQUE_REFUSAL}\n\n"
    "Retrieved candidate paragraphs:\n{memories}\n\nQuestion: {question}\n\nAnswer:"
)

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
#: 用错 prompt 的后果（2026-10-03 实测，250 题）：**160/250 逐字拒答、通过 0**；
#: 而 `poison` 那类又会被"不许用外部知识"这条**逼着照抄毒记忆**——两类失败同一个根。
MEMTRAP_ANSWER_PROMPT = (
    "You can reference the recalled information in the Retrieved Memory block below "
    "when needed: {memories}\n\n"
    "Please generate an appropriate response for the user's query: {question}"
)


#: **MQuAKE 专用**的答案 prompt（2026-10-03 加）。
#:
#: ⛔ **不要对它用通用 `ANSWER_PROMPT`。** 那份有两条对 MQuAKE **有害**的规则：
#:   · 「只用记忆、不用外部知识」——而 **MQuAKE-CF 的"未编辑跳"本来就要求世界知识**
#:     （那是它的设计），禁掉就答不出；
#:   · 「记忆里没有就逐字回 `Cannot determine from the memories.`」——记忆里
#:     **旧值与 "…is X (this replaces the earlier value)" 并存**，模型一看到冲突就引用这条拒答。
#:
#: 实测（768 题，**只重答那 201 条拒答、其余 567 条不动**，四个候选同条件对照）：
#:
#:     B0 现状（含拒答条款）  新答对  0  ⇒ 总分 0.3997
#:     B1 只删拒答条款        新答对 66  ⇒ 总分 0.4857
#:     B2 +改写优先           新答对 88  ⇒ 总分 0.5143
#:     B3 本份（三条都给上）   新答对 93  ⇒ 总分 0.5208
#:
#: ⚠ **这是 MQuAKE 的接法问题，不是"通用 prompt 该改"**：通用那份还被
#: medmemorybench / CorporateBench 的单值查询仍沿用；CorporateBench 聚合任务已独立。
#: （tempreason 一度也在名单里，**10-04 起它有自己那份 `TEMPR_ANSWER_PROMPT`**。）
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


# ── 两个子命令 ──────────────────────────────────────────────────────────────


def render_answer_prompt(item: dict) -> str:
    """按 `item["dataset"]` 挑答案 prompt——**`cmd_answer` 与本包的分派 pipeline 共用这一处**。

    （`official_capture_pipeline.py` 逐题调它，而不是把整个文件交给 `cmd_answer`：
    那样会把别的数据集的题也顺带跑一遍。）
    """
    template = {
        "memtrapbench": MEMTRAP_ANSWER_PROMPT,
        "mquake-remastered": MQUAKE_ANSWER_PROMPT,
        "tempreason": TEMPR_ANSWER_PROMPT,
    }.get(item.get("dataset"), ANSWER_PROMPT)
    if item.get("dataset") == "musique" and item.get("answer_contract") == MUSIQUE_ANSWER_CONTRACT:
        template = MUSIQUE_ANSWER_PROMPT
    if item.get("dataset") == "corporatebench":
        return build_corporatebench_answer_prompt(item, scalar_template=ANSWER_PROMPT)
    return template.format(
        memories=item.get("retrieved_context") or "(no memories)",
        question=item["question"],
    )


def cmd_answer(args: argparse.Namespace) -> int:
    items = read_jsonl(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in read_jsonl(output)}
    base, key, model = _config("answer")
    with output.open("a", encoding="utf-8") as handle:
        for item in items:
            if item["id"] in done:
                continue
            # **逐数据集挑 prompt**（判据是 item 里那个 `dataset` 字段，与 evaluate 同一口径）。
            # 分派理由见各自模板注释及 CorporateBench 的独立适配器。
            prompt = render_answer_prompt(item)
            generated = _chat(
                base, key, model, prompt, max_tokens=args.max_tokens, timeout=ANSWER_TIMEOUT
            )
            write_line(handle, {"id": item["id"], "generated_answer": generated})
    return 0


def _verdict(qid: str, ok: bool, why: str) -> dict:
    return {
        "id": qid,
        "is_correct": ok,
        "label": "CORRECT" if ok else "WRONG",
        "judge_response": why,
    }


def _judge_one(
    dataset: str | None,
    qid: str,
    item: dict,
    generated: str,
    *,
    judge: tuple[str, str, str],
    max_tokens: int,
) -> dict:
    """一题的判分——**按数据集四选一**。每一支都写清楚口径的出处。"""
    gold = item.get("gold_answer")
    if dataset == "mquake-remastered":
        ok, why = judge_mquake(generated, gold)
        return _verdict(qid, ok, why)
    if dataset == "corporatebench":
        score, why = score_corporatebench(generated, gold)
        return _verdict(qid, score == 1.0, why) | {"partial": score}
    if dataset == "tempreason":
        ok, why = judge_tempreason(generated, gold)
        return _verdict(qid, ok, why)
    if dataset == "musique":
        ok, why = judge_musique(generated, gold)
        return _verdict(qid, ok, why)
    if dataset == "docpp":
        return _judge_docpp(qid, item, generated, judge=judge, max_tokens=max_tokens)
    if dataset == "memtrapbench":
        gold = gold or {}
        prompt = MEMTRAP_JUDGE_PROMPT.format(
            question=item["question"],
            gold=(gold.get("gold_standard") if isinstance(gold, dict) else gold) or "(none)",
            response=generated or "(empty)",
        )
        raw = _call_judge(judge, prompt, max_tokens=max_tokens)
        mean, why = judge_memtrapbench_scores(raw)
        if mean is None:
            return {
                "id": qid,
                "is_correct": False,
                "label": "JUDGE_ERROR",
                "judge_response": f"{why}｜{raw[:2000]}",
            }
        return {
            "id": qid,
            "is_correct": mean >= MEMTRAP_PASS_MEAN,
            "label": f"MEAN={mean:.2f}",
            "judge_response": raw,
        }
    if dataset == "medmemorybench":
        return _judge_medmemorybench(qid, item, generated, judge=judge, max_tokens=max_tokens)
    return {
        "id": qid,
        "is_correct": False,
        "label": "JUDGE_ERROR",
        "judge_response": f"未知数据集 {dataset!r}",
    }


def _judge_medmemorybench(
    qid: str, item: dict, generated: str, *, judge: tuple[str, str, str], max_tokens: int
) -> dict:
    """MedMemoryBench：**两条纯函数 + 四条 LLM**（分流照上游 `DEFAULT_METRIC_MAPPING`）。"""
    gold = item.get("gold_answer") or {}
    query_type = str(gold.get("query_type") or "")
    if query_type == "entity_exact_match":
        ok, why = judge_mmb_string_contain(generated, gold)
        return _verdict(qid, ok, why)
    if query_type == "multiple_choice":
        ok, why = judge_mmb_option_match(generated, gold)
        return _verdict(qid, ok, why)
    if query_type not in MMB_LLM_TYPES:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": f"MedMemoryBench：没见过的 query_type {query_type!r}",
        }
    # ⚠ 上游对**空输出**直接判错（`_is_empty_output`），这里照办——省一次裁判调用。
    if not (generated or "").strip():
        return _verdict(qid, False, "MedMemoryBench：模型输出为空（上游判错）")
    prompt = build_mmb_judge_prompt(query_type, item["question"], generated, gold)
    budget = MMB_MCD_MAX_TOKENS if query_type == "multi_hop_clinical_deduction" else max_tokens
    raw = _call_judge(judge, prompt, max_tokens=budget)
    ok, why = judge_mmb_llm_verdict(raw)
    if ok is None:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": f"{why}｜{raw[:2000]}",
        }
    return {
        "id": qid,
        "is_correct": ok,
        "label": f"{query_type}:" + ("CORRECT" if ok else "WRONG"),
        "judge_response": why or raw[:2000],
    }


def _judge_docpp(
    qid: str, item: dict, generated: str, *, judge: tuple[str, str, str], max_tokens: int
) -> dict:
    """Doc-PP：**两条 LLM 裁判**（`with_policy=false` 那两条路，prompt 取自上游）。"""
    gold = item.get("gold_answer") or {}
    if not (generated or "").strip():
        # 上游那条 direct 裁判明写"模型说找不到/拒答 ⇒ 不算 present" ⇒ 空输出直接判负，省一次调用。
        return _verdict(qid, False, "Doc-PP：模型输出为空")
    prompt, kind = build_docpp_judge_prompt(item["question"], generated, gold)
    raw = _call_judge(judge, prompt, max_tokens=max_tokens)
    ok, why = judge_docpp_verdict(raw, kind)
    if ok is None:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": f"{why}｜{raw[:2000]}",
        }
    return {
        "id": qid,
        "is_correct": ok,
        "label": kind + (":CORRECT" if ok else ":WRONG"),
        "judge_response": why,
    }


def _call_judge(judge: tuple[str, str, str], prompt: str, *, max_tokens: int) -> str:
    """裁判调用**失败不打死整轮**（留痕在 `judge_response` 里）——与 409 那条纪律同一取向。"""
    base, key, model = judge
    try:
        return _chat(base, key, model, prompt, max_tokens=max_tokens, timeout=JUDGE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return f"JUDGE CALL FAILED: {type(exc).__name__}: {exc}"


def cmd_evaluate(args: argparse.Namespace) -> int:
    items = {item["id"]: item for item in read_jsonl(Path(args.input))}
    answers = {row["id"]: row.get("generated_answer", "") for row in read_jsonl(Path(args.answers))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    judge = _config("judge")

    with output.open("w", encoding="utf-8") as handle:
        for qid, item in items.items():
            row = _judge_one(
                item.get("dataset"),
                qid,
                item,
                answers.get(qid, ""),
                judge=judge,
                max_tokens=args.max_tokens,
            )
            write_line(handle, row)
    return 0


# ── 公开入口（同包的分派 pipeline 用）─────────────────────────────────────
# [`official_capture_pipeline.py`](./official_capture_pipeline.py) **逐题**调这几个函数，
# 而不是把整份文件交给 `cmd_answer` / `cmd_evaluate`——那会多出一份临时文件、
# 且无法与别的数据集混在同一趟里。**实现仍只有一份**，这里只是给它们公开名字。
config = _config
chat = _chat
verdict = _verdict
judge_one = _judge_one


def build_parser() -> argparse.ArgumentParser:
    """CLI 形状**必须与归档 pipeline 一致**：`--max-tokens` 挂在**子命令上**。

    ⚠ 2026-09-30 的教训：一开始把它挂在**顶层** parser 上，而 `run_judge` 是按官方那几份的
    写法**在子命令之后**传的（`… answer --input … --output … --max-tokens 256`）⇒
    argparse 子解析器不认识它，整轮以 `unrecognized arguments` 死在第一步。
    ⇒ **两个子命令都要收**（`run_judge` 对 locomo / LME / 我们这五份是两边都传的）。
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-tokens", type=int, default=512, help="answer / 裁判各自的输出上限")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    answer = sub.add_parser(
        "answer", parents=[common], help="作答（**追加**模式，按 id 跳过已完成的题）"
    )
    answer.add_argument("--input", required=True)
    answer.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", parents=[common], help="判分（**覆盖**模式）")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--answers", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
