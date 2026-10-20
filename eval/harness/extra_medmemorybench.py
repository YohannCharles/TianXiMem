"""MedMemoryBench（zh）家族的 answer / evaluate：照上游 `metrics/` 的判分 ＋ 临床多跳答案 prompt。

拆自 `extra_pipeline.py`（2026-10-15，只搬位置）：那层现在是薄 façade、把本模块的名字
全部重新导出。上游两份来源（`dataset/.upstream/medmemorybench/` 那份与归档里的
`utils/prompts_judge.py`）与"逐行照抄 / 直接从归档读"的范围见下面那段注释；
临床多跳答案的绕行与原始就诊对照见 [`../reports/optimization-20261007.md`](../reports/optimization-20261007.md)。
"""

from __future__ import annotations

import ast
import json
import re
import string
import sys
from pathlib import Path

# ⚠ 与 `extra_pipeline.py` 同一处置：subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根。
#   本目录的模块既被 `eval.harness.*` 包内引用、也被当脚本/顶层模块跑，两种跑法都要能
#   解析到 `eval.*`（包内 import 时仓库根已在路径上，这里是幂等的）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets.registry import benchmark_dir  # noqa: E402
from eval.harness.extra_base import _as_bool, _escape_stray_quotes  # noqa: E402

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
    head, tail = text.find("{"), text.rfind("}")
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            # Repair the outer object before scanning nested fragments: quoted
            # note contents must not supply a different is_correct decision.
            if start == head and head < tail:
                repaired = _repair_mmb_judge_strings(text[head : tail + 1])
                try:
                    payload = json.loads(repaired)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and "is_correct" in payload:
                    verdict = _as_bool(payload["is_correct"])
                    if verdict is None:
                        return None, "裁判 is_correct 不是可识别的布尔值"
                    return verdict, (
                        f"{payload.get('reason', '')}"
                        "（⚠ 判词含未转义引号或混用字符串引号，规范化后解析）"
                    )
            continue
        if isinstance(payload, dict) and "is_correct" in payload:
            verdict = _as_bool(payload["is_correct"])
            if verdict is None:
                return None, "裁判 is_correct 不是可识别的布尔值"
        verdict = _as_bool(payload.get("is_correct")) if isinstance(payload, dict) else None
        if verdict is not None:
            return verdict, str(payload.get("reason", ""))

    # 严格路径全灭 ⇒ 兜底：把字符串里的裸引号转义掉再来一遍（见 `_escape_stray_quotes`）。
    if 0 <= head < tail:
        try:
            payload = json.loads(_repair_mmb_judge_strings(text[head : tail + 1]))
        except json.JSONDecodeError:
            payload = None
        verdict = _as_bool(payload.get("is_correct")) if isinstance(payload, dict) else None
        if verdict is not None:
            reason = str(payload.get("reason", ""))
            return verdict, f"{reason}（⚠ 判词含未转义引号，转义后解析）"
    return None, "裁判响应里没有可解析的 is_correct"


def _repair_mmb_judge_strings(text: str) -> str:
    """Normalize only string syntax; the literal verdict remains authoritative."""
    pattern = r"""("(?:note|reason)"\s*:\s*)('(?:[^\\]|\\.)*?')(?=\s*[,}])"""

    def normalize_note(match: re.Match) -> str:
        try:
            value = ast.literal_eval(match[2])
        except (SyntaxError, ValueError):
            # Judges can use an outer single-quoted note containing unescaped
            # apostrophes. Only normalize this prose field; the boolean verdict
            # remains outside it and must still parse as an explicit decision.
            value = match[2][1:-1]
        return match[1] + json.dumps(value, ensure_ascii=False)

    return _escape_stray_quotes(re.sub(pattern, normalize_note, text))


# Native question category selects the QA format; gold remains judge-only.
# Evidence and matched-budget comparison: reports/optimization-20261007.md.
MMB_MCD_ANSWER_CONTRACT = "medmemorybench-clinical-evidence-v1"
MMB_MCD_ANSWER_PROMPT = (
    "You are an assistant answering from the patient's retrieved visit history.\n"
    """
Rules:
1. Use only information in the memories. Do not invent medical history,
   measurements, diagnoses, dates, medications, or causal mechanisms.
2. When asked why symptoms or events are related, explain the supported links
   across visits. Include the relevant patient-specific facts, dates, medication
   details and measurements that support those links. A generic conclusion alone
   is insufficient. Keep the explanation focused on the question.
3. Distinguish current facts from earlier facts and statements of uncertainty.
   If a necessary link is missing, state that limitation instead of inventing it.
4. If the question provides answer options, return only the letters of all
   selected options. For a direct name, value, date or status question, return
   only the requested information.
5. If the history provides no answer, reply exactly: Cannot determine from the memories.

Memories:
{memories}

Question: {question}

Answer:"""
)

#: `multiple_choice` 的作答格式：judge 只解析选项字母，prompt 就必须只要字母。
MMB_MCQ_ANSWER_PROMPT = (
    "You are an assistant answering from the patient's retrieved visit history.\n"
    "Rules:\n"
    "1. Use only information in the memories. Do not use outside knowledge.\n"
    "2. Return only the letters of all selected options (for example: A, or A,C).\n"
    "   Do not include explanations or the option text.\n"
    "3. If the history provides no answer, reply exactly: Cannot determine from the memories.\n\n"
    "Memories:\n{memories}\n\nQuestion: {question}\n\nAnswer:"
)
