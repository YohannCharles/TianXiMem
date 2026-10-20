#!/usr/bin/env python3
"""官方采集（`official-capture`）的 answer / evaluate —— **一个 run 里按题分派到 8 个数据集**。

## 为什么需要它

其余每个数据集都是"一个数据集一套 prompt + 一套判分"，所以 harness 的
`run_judge` 按 `dataset` 选一份 pipeline 就够了。官方采集这份**不是**：
一个 run 里的题**混着十几个数据集**（locomo / scriptmem / personamem / beam /
clbench / mquake / corporatebench / medmemorybench / memtrapbench / docpp /
musique / hybridqa / feverous / halumem），每题该用哪套 prompt 由**题自己的归属**决定
（[`build_official_items`](./judge_items.py) 把它放在 `item["dataset"]`）。

⇒ 本文件是**分派层**，不是第二份实现：

| 家族 | answer prompt | judge |
| --- | --- | --- |
| `locomo-refined` | `pipeline_locomo-refined.py` 的 answer prompt | 同文件 accuracy prompt + LLM |
| `scriptmem` | `pipeline_scriptmem.py` 的 answer prompt | `predicted_letters` + `score_item` |
| `beam` | `pipeline_beam.py#render_answer_prompt` | 同文件的 rubric 批量裁判（三点制均分） |
| `clbench` | `clb_pipeline.py#build_answer_prompt` | 同文件 rubric 裁判（全有全无 + 比例分） |
| `personamem-v2` | `pipeline_v2_personamem.py#official_generative_messages` | **官方 `evaluate_narrow`**（`JUDGE_PROMPT_NARROW_POSITIVE` / `_NEGATIVE`，见下） |
| `hybridqa` | `corpusqa_pipeline.py#render_answer_prompt` | 同文件 `score_answer`（**官方归一化 EM + token-F1**，`is_correct = EM`） |
| `feverous` | 本文件 `_FEVEROUS_LABEL_PROMPT`（**只要标签**） | 本文件 `_feverous_judge`（**label-only**，理由见该函数） |
| `musique` | `extra_pipeline.py#render_answer_prompt`（原生 Full 契约） | `extra_pipeline.py#judge_one`（规则：归一化精确匹配） |
| 上表之外的 6 个 | `extra_pipeline.py#render_answer_prompt` | `extra_pipeline.py#judge_one` |

**每个数据集用自己的 prompt 与判分**：`judge_kind` 只是套件里的语义声明，
**不再是**"没有专属分支时借别的家族判分"的兜底开关。`supports()` 与 `_judge()`
共用 [`_JUDGE_KINDS`](#) 一张表——预检说"能判"的组合，运行时一定走对应分支。

**prompt 与判分逻辑一行都不重写**：归档那几份用 `sys.path` + `import` 直接调
（文件名带连字符的用 `importlib`），`official-extra` 那六个走 `extra_pipeline` 的公开入口。

## 口径说明

1. **`personamem-v2` 走官方 narrow 口径**（2026-10-19 改回）：采集题是开放题，官方
   `evaluate-narrow` 用 `preference` + 模型回答问裁判要 0–1 分。`preference` 不在采集
   search 里，但 `official-gold-extra` 已从 `benchmark.csv` 按 query 逐字匹配补上
   （114/114 全部命中）⇒ **prompt 与打分函数逐字用归档那份**，只在外面加一层
   「0–1 分 → `is_correct`」的二值化。
   ⚠ **阈值是我们定的**：`PERSONAMEM_NARROW_PASS = 0.5`（与 `BEAM` 的"均分 == 1.0"
   不是一回事，别混）。原始分进 `partial` 与 `judge_response`，换阈值不用重跑裁判。
2. **`beam` 的二值化**：官方只给逐条 rubric 的均分（`llm_judge_score`），
   `is_correct` 是我们定的 `均分 == 1.0`——与 `judge.py` 的 `_read_beam_labels` **同一口径**。

## 续跑

`answer` 追加 + 按 id 跳过；`evaluate` **按 id 合并**（已判过的不再判）——
判分要真金白银地调网关，一次 Full 重放几千题，被打断后不该从头再来一遍。
⚠ 与归档那几份（`evaluate` 一律覆盖）**故意不同**。
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
from pathlib import Path

# ⚠ 本文件是**当脚本跑的**（`run_judge` 起 subprocess）：那时 `sys.path[0]` 是
# `eval/harness/`，仓库根不在上面 ⇒ `import eval.*` 会 `ModuleNotFoundError`。
# 与 [`personamem_pipeline.py`](./personamem_pipeline.py) 同一处置：把仓库根与本目录都补上。
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from eval.harness.pipeline_cli import build_pipeline_parser  # noqa: E402
from eval.jsonl_io import jsonl_line, read_jsonl, write_line  # noqa: E402

#: 走 `extra_pipeline` 的六个家族（它自己那套 prompt 与判分）。
_EXTRA_FAMILIES = frozenset(
    {
        "mquake-remastered",
        "memtrapbench",
        "corporatebench",
        "medmemorybench",
        "tempreason",
        "docpp",
    }
)

#: **判分在本文件里**的采集家族；PersonaMem 的答案 prompt 另走官方 generative，
#: HaluMem / FEVEROUS 没有官方 answer pipeline，仍由 `_answer_prompt` 的本地分派处理。
_LOCAL_JUDGE_FAMILIES = frozenset({"personamem-v2", "halumem", "feverous"})


#: 家族 → **真正接线的**判分语义。`supports()` 只认这张表。
#:
#: ⚠ 这是一张**分派镜像**，不是"允许的取值域"：`_judge()` 里没有对应分支的家族
#: 绝不能出现在这里。历史实现按 `judge_kind ∈ {exact, llm}` 兜底，于是
#: `musique` / `hybridqa` 在预检里"能判"、运行时却落到 `judge_tempreason` 的别名包含
#: （2026-10-20 审计）。新增家族时**两边一起改**，测试逐组合钉住。
_JUDGE_KINDS: dict[str, frozenset[str]] = {
    "locomo-refined": frozenset({"llm"}),
    "scriptmem": frozenset({"letters"}),
    "beam": frozenset({"rubric"}),
    "clbench": frozenset({"rubric"}),
    "personamem-v2": frozenset({"narrow"}),
    "halumem": frozenset({"llm"}),
    "feverous": frozenset({"exact"}),
    "hybridqa": frozenset({"exact"}),
    "musique": frozenset({"exact"}),
    "mquake-remastered": frozenset({"exact"}),
    "corporatebench": frozenset({"exact"}),
    "medmemorybench": frozenset({"exact"}),
    "tempreason": frozenset({"exact"}),
    "docpp": frozenset({"rubric"}),
    "memtrapbench": frozenset({"rubric"}),
}


def supports(dataset: str | None, judge_kind: str | None) -> bool:
    """这个家族 + 判分语义**会不会真的被判**（而不是落到 `JUDGE_ERROR`）。

    它是 `_judge()` 分派表的**纯函数镜像**——给跑批前的预检用，避免"跑了 8,783 题、
    最后一天才发现某族全是 `JUDGE_ERROR`"。实现直接查 `_JUDGE_KINDS`，
    与 `_judge()` 的显式分支一一对应（**没有**与家族无关的兜底）。

    ⚠ 新增一个家族时**两边都要改**；`tests/test_official_capture.py` 里有一个
    逐家族对照用例（拿套件里真实出现过的组合逐个断言）。
    """
    return str(judge_kind or "") in _JUDGE_KINDS.get(str(dataset or ""), frozenset())


#: FEVEROUS 的三分类词表。**不能用别名包含判**——`SUPPORTS` 会命中任何含 "supports"
#: 的句子（而 `NOT ENOUGH INFO` 里也含 `SUPPORT` 的反义表述），必然假阳性。
_FEVEROUS_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NOT ENOUGH INFO", ("not enough info", "not enough information", "insufficient")),
    ("REFUTES", ("refutes", "refuted", "refute")),
    ("SUPPORTS", ("supports", "supported", "support")),
)


def _feverous_judge(item: dict, generated: str) -> dict:
    """FEVEROUS：事实核查的**三分类**（SUPPORTS / REFUTES / NOT ENOUGH INFO）。

    口径：金标是 dev challenge 那一行的 `label`；判分取模型输出里**最先出现**的那个
    标签词（顺序敏感：先查 NOT ENOUGH INFO，再 REFUTES，最后 SUPPORTS——
    否则 "does not support" 会被读成 SUPPORTS）。
    ⚠ **采集重放仍是 label-only**：作答 prompt（`_FEVEROUS_LABEL_PROMPT`）只要一个标签。
    本地 `corpusqa_pipeline` 那条路才要求证据 id 并评严格证据组——采集正文里没有可解析的
    原生元素 id（实测 0/53027 个方括号片段是合法 id），那里评证据必然全 0。
    """
    extra = _extra()
    qid = str(item["id"])
    gold = [str(g).strip().upper() for g in (item.get("gold_answers") or [])]
    if not gold:
        return extra.verdict(qid, False, "FEVEROUS：无金标")
    text = (generated or "").lower()
    said = next((name for name, keys in _FEVEROUS_LABELS if any(k in text for k in keys)), None)
    if said is None:
        return extra.verdict(qid, False, f"FEVEROUS：回答里没有可识别的判定（金标 {gold[0]}）")
    return extra.verdict(qid, said == gold[0], f"FEVEROUS：判为 {said}，金标 {gold[0]}")


#: 归档那几份的模块名（`benchmark_dir()` 下）。`pipeline_locomo-refined` 带连字符，
#: 只能走 `importlib`（见 `_archived`）。
_ARCHIVED = {
    "locomo-refined": "pipeline_locomo-refined",
    "scriptmem": "pipeline_scriptmem",
    "beam": "pipeline_beam",
    "clbench": "clb_pipeline",
    "personamem-v2": "pipeline_v2_personamem",
}


def _extra():
    """`extra_pipeline` 的**公开入口**（同一份实现，别在这里抄第二份 prompt）。"""
    from eval.harness import extra_pipeline

    return extra_pipeline


#: HaluMem 的裁判 prompt 在**上游文件**里（我们只读、不复制——复制一份就是第二个家）。
#: 归档路径与 sha256 记在 [`tools/recover_halumem.py`](../../tools/recover_halumem.py) 的文件头。
_HALU_TEMPLATE_FILE = "halumem/eval_tools.py"
_HALU_TEMPLATE_NAME = "EVALUATION_PROMPT_FOR_QUESTION"


@functools.lru_cache(maxsize=4)
def _upstream_template(relative: str, name: str) -> str:
    """从归档的上游脚本里读一个**字符串常量**（纯数据文件，不 import）。

    与 [`extra_pipeline._docpp_prompts`](./extra_pipeline.py) 同一手法：Doc-PP 那份的坑
    （`\"\"\"…\"\"\".strip()` 里字符串是接收者）照旧适用。

    ⚠ 带 `lru_cache`：它是**每题都要读一次**的（HaluMem 一族有 1,481 题），
    而文件内容在一个 run 里不会变。
    """
    import ast

    from eval.datasets.layout import archive_file
    from eval.datasets.registry import benchmark_dir

    path = archive_file(benchmark_dir(), relative)
    if not path.exists():
        raise SystemExit(f"缺 {path}——上游那份裁判 prompt 没归档（见 tools/recover_halumem.py）")
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = getattr(node, "targets", [])
        if not isinstance(node, ast.Assign) or not targets:
            continue
        if getattr(targets[0], "id", "") != name:
            continue
        value = node.value
        if isinstance(value, ast.Call) and getattr(value.func, "attr", "") == "strip":
            value = getattr(value.func, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    raise SystemExit(f"{path} 里没有字符串常量 {name}")


def _halumem_judge(item: dict, generated: str, *, max_tokens: int) -> dict:
    """HaluMem：**上游只发布一份 LLM 裁判**（`Correct | Hallucination | Omission`）。

    逐字规则在归档的那份 `eval_tools.py` 里；这里只做三件事：填三个槽
    （参考回答 / 关键记忆点 / 系统作答）、发出去、解析 `evaluation_result`。
    **只有 `Correct` 算对**（`Omission` 也是错——它有独立的语义，别混进正确率）。
    """
    extra = _extra()
    qid = str(item["id"])
    gold = [str(g) for g in (item.get("gold_answers") or [])]
    if not gold:
        return extra.verdict(qid, False, "HaluMem：无金标")
    native = item.get("gold_answer") or {}
    evidence = native.get("evidence") if isinstance(native, dict) else None
    points = (
        "\n".join(str(entry.get("memory_content", "")) for entry in evidence or []).strip()
        or "(none)"
    )
    prompt = _upstream_template(_HALU_TEMPLATE_FILE, _HALU_TEMPLATE_NAME).format(
        question=item["question"],
        reference_answer=gold[0],
        key_memory_points=points,
        response=generated or "(empty)",
    )
    raw = extra.chat(
        *extra.config("judge"), prompt, max_tokens=max_tokens, timeout=extra.JUDGE_TIMEOUT
    )
    label = _parse_three_way(raw)
    if label is None:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": f"裁判响应里没有 Correct/Hallucination/Omission｜{raw[:2000]}",
        }
    return {
        "id": qid,
        "is_correct": label == "Correct",
        "label": label,
        "judge_response": raw[:4000],
    }


def _parse_three_way(raw: str) -> str | None:
    """取 HaluMem 的三分类结果（JSON 的 `evaluation_result`，回退整段扫）。"""
    text = raw or ""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            result = str(payload.get("evaluation_result", "")).strip().title()
            if result in {"Correct", "Hallucination", "Omission"}:
                return result
    lowered = text.lower()
    for label in ("hallucination", "omission", "correct"):
        if label in lowered:
            return label.title()
    return None


def _archived(family: str):
    """按家族名 import 归档那份 pipeline——**懒加载**（模块级 import 会在没注入
    `PYTHONPATH` 的场合就炸，而单元测试不该需要网关配置）。"""
    module_name = _ARCHIVED[family]
    #: `personamem_pipeline.py` 同款处置：把 `benchmark_dir()` 塞进 `sys.path`，
    #: 归档脚本模块级那句 `from api_config import ...` 由 `run_judge` 的 `PYTHONPATH` 满足。
    from eval.datasets.layout import upstream_aml_dir
    from eval.datasets.registry import benchmark_dir

    root = str(upstream_aml_dir(benchmark_dir(), f"{module_name}.py"))
    if root not in sys.path:
        sys.path.insert(0, root)
    if "-" in module_name:  # 文件名带连字符：只能按路径加载
        import importlib.util

        path = Path(root) / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name.replace("-", "_"), path)
        assert spec and spec.loader, f"加载不了 {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import importlib

    return importlib.import_module(module_name)


#: FEVEROUS 采集重放的作答 prompt：**只要标签**。
#:
#: 本地 `corpusqa_pipeline` 那份要求 label + 原生证据 ID（那是 FEVEROUS 自己的评分标准），
#: 但采集正文里没有可解析的原生元素 ID（实测 0/53027 个方括号片段合法），
#: `_feverous_judge` 也只看标签 ⇒ 在采集路径上要 ID 是**无效复杂度**（模型多写一串没人评）。
_FEVEROUS_LABEL_PROMPT = """\
Verify the following claim using only the retrieved evidence below.
Return exactly one label and nothing else: SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Retrieved evidence:
{memories}

Claim: {claim}

Label:"""


def _feverous_claim(item: dict) -> str:
    """取采集 query 里的 claim 本体——**去掉前面那段"返回证据 ID"的指令**。"""
    text = str(item.get("question") or "")
    marker = "Claim:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def _answer_prompt(item: dict) -> str:
    """按家族给答案 prompt——**每一支都指回它的实现**（本文件不复制任何 prompt 正文）。"""
    family = item.get("dataset")
    if family == "locomo-refined":
        return _archived(family).render_answer_prompt(item)
    if family == "beam":
        return _archived(family).render_answer_prompt(item)
    if family == "scriptmem":
        # 那一份**没有** `retrieved_context` 回退（只认 speaker_1_memories）——
        # 这里显式把命中注进它读的键，而不是改归档。
        return _archived(family).render_answer_prompt(
            {
                **item,
                "speaker_1_name": "speaker 1",
                "speaker_1_memories": item.get("retrieved_context") or "",
                "speaker_2_memories": "",
            }
        )
    if family == "hybridqa":
        # HybridQA 的作答 prompt 来自 `corpusqa_pipeline.py`（"只回短答案"），
        # **不能**退回通用 ANSWER_PROMPT（那一份允许 `Cannot determine from the memories.`）。
        from eval.harness import corpusqa_pipeline

        return corpusqa_pipeline.render_answer_prompt(item)
    if family == "feverous":
        # 采集路径只评标签 ⇒ 只要求标签；本地 `corpusqa_pipeline` 那条路仍要证据 ID。
        return _FEVEROUS_LABEL_PROMPT.format(
            memories=item.get("retrieved_context") or "(no memories)",
            claim=_feverous_claim(item),
        )
    if family == "clbench":
        # ⚠ **不能直接把我们的 item 丢给官方那份 `build_answer_prompt`**（2026-10-06 实测）：
        #   它读的是 `system_prompt` 与 `retrieval.selected[*].text`，
        #   而我们的项里**两个键都没有** ⇒ 渲染出来是 `(no memories)` + 空 system
        #   ⇒ 模型**盲答**（实测 74 题全 0，而检索其实带回了 1.7k 字符）。
        #   ⇒ 按官方认的形状把命中塞回去（我们的命中已经渲染成一整串，因此只放一条）。
        native = item.get("gold_answer") or {}
        return _archived(family).build_answer_prompt(
            {
                "system_prompt": native.get("system_prompt") or "",
                "question": item.get("question") or "",
                "qa_type": item.get("task") or "",
                "options": item.get("options") or [],
                "retrieval": {"selected": [{"text": item.get("retrieved_context") or ""}]},
            }
        )
    if family == "docpp":
        # ⚠ **采集里那串 query 本身就是官方答案 prompt**（system prompt + 政策 + 用户问题，
        #   逐字来自 `dataset/.upstream/doc-pp/prompts/evaluate_model.py`）——上游是把**文档正文**
        #   另外交给模型的，而这里文档必须由检索提供 ⇒ 补上记忆块。
        #   拼接文案是**我们定的**（上游模板没有记忆占位符）。
        return (
            f"{item['question']}\n\n"
            f"# Retrieved document content\n{item.get('retrieved_context') or '(no memories)'}\n\n"
            "Now answer the user question above, using only the retrieved document content."
        )
    if family == "personamem-v2":
        # 官方采集中的 PersonaMem 行是开放题，只有独立 PersonaMem-v2 题库才有
        # `correct_answer` / `incorrect_answers`，所以这里必须走官方 generative
        # prompt，而不是把平铺采集项误送到官方 MCQ 适配器。
        module = _archived(family)
        messages = module.official_generative_messages(
            {
                **item,
                "chat_history": [
                    {
                        "role": "user",
                        "content": (
                            "Retrieved memories (in retrieval order):\n"
                            + (item.get("retrieved_context") or "(no memories)")
                        ),
                    }
                ],
            }
        )
        # `extra_pipeline.chat` 的历史接口接收单个字符串；保留官方消息正文和顺序，
        # 并显式保留角色边界，避免重新编写官方 prompt 文案。
        return "\n\n".join(
            f"{message.get('role', 'user').title()}: {message.get('content', '')}"
            for message in messages
        )
    # 其余没有官方 answer prompt 的家族走我们自写的专用/通用提示词。
    return _extra().render_answer_prompt(item)


#: PersonaMem-v2 的 narrow 分**二值化阈值**。官方只回 0–1 连续分、不规定过线，
#: 所以阈值是我们定的（与 `BEAM` 的"均分 == 1.0"不是一回事，别混）。
#: ⚠ 原始分进 `partial` 与 `judge_response` ⇒ **换阈值不用重跑裁判**。
PERSONAMEM_NARROW_PASS: float = 0.5


def _narrow_judge(item: dict, generated: str, *, max_tokens: int) -> dict:
    """PersonaMem-v2 开放题——**官方 `evaluate_narrow` 口径**（2026-10-19 改回）。

    上游 `pipeline_v2_personamem.py` 的 `evaluate-narrow`：
    `preference` 以 `do not` 开头用 `JUDGE_PROMPT_NARROW_NEGATIVE`，否则
    `JUDGE_PROMPT_NARROW_POSITIVE`；裁判回 `\\boxed{score}`（0–1），
    `extract_boxed_score` 解析。**prompt 与解析都直接调归档那份，不重写。**

    `preference` 来自 `item["gold_answer"]["preference"]`（套件 `gold_native` 带过来的，
    源头是 `benchmark.csv`）。⚠ **缺 `preference` 就响亮报 `JUDGE_ERROR`**，不静默
    退回"拿参考回答当标准"——那会**悄悄换掉官方口径**，而分数看起来完全正常。
    """
    extra = _extra()
    qid = str(item["id"])
    native = item.get("gold_answer")
    preference = ""
    if isinstance(native, dict):
        preference = str(native.get("preference") or "").strip()
    if not preference:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": (
                "PersonaMem-v2 走官方 narrow 口径，但这一行没有 `preference`"
                "（见 tools/recover_official_gold.py 的 load_personamem）"
            ),
        }
    if not (generated or "").strip():
        return extra.verdict(qid, False, "模型输出为空")
    module = _archived("personamem-v2")
    template = (
        module.JUDGE_PROMPT_NARROW_NEGATIVE
        if preference.lower().startswith("do not")
        else module.JUDGE_PROMPT_NARROW_POSITIVE
    )
    prompt = template.format(
        user_query=item.get("question") or "",
        preference=preference,
        model_response=generated,
    )
    raw = extra.chat(
        *extra.config("judge"), prompt, max_tokens=max_tokens, timeout=extra.JUDGE_TIMEOUT
    )
    score = module.extract_boxed_score(raw)
    return {
        "id": qid,
        "is_correct": score >= PERSONAMEM_NARROW_PASS,
        "label": f"SCORE={score:.2f}",
        "judge_response": raw,
        "partial": score,
    }


def _judge(item: dict, generated: str, *, max_tokens: int) -> dict:
    """一题的判分——**按家族分派**，每个分支的口径出处写在下面。"""
    family = item.get("dataset")
    qid = str(item["id"])
    extra = _extra()
    if family == "musique":
        # 原生 MuSiQue-Full（采集行没有 category，不推断链式档）：规则判分，
        # 可答题归一化精确匹配别名表，不可答题严格匹配拒答标记。
        return extra.judge_one(
            "musique", qid, item, generated, judge=extra.config("judge"), max_tokens=max_tokens
        )
    if family == "hybridqa":
        from eval.harness import corpusqa_pipeline

        # 官方归一化 EM + token-F1；is_correct = EM，F1 只作指标（上游没有过线阈值）。
        return corpusqa_pipeline.score_answer(item, generated)
    if family in _EXTRA_FAMILIES or family in _LOCAL_JUDGE_FAMILIES:
        if family == "personamem-v2":
            return _narrow_judge(item, generated, max_tokens=max_tokens)
        if family == "halumem":
            return _halumem_judge(item, generated, max_tokens=max_tokens)
        if family == "feverous":
            # 采集正文里没有可解析的原生元素 ID（实测 0/53027 个方括号片段是合法 ID），
            # 严格证据组判分在这条路上必然全 0 ⇒ 保留已声明的 label-only 口径。
            return _feverous_judge(item, generated)
        return extra.judge_one(
            family, qid, item, generated, judge=extra.config("judge"), max_tokens=max_tokens
        )

    module = _archived(family) if family in _ARCHIVED else None
    if module is None:
        return _generic_judge(item, generated, max_tokens=max_tokens)
    if family == "locomo-refined":
        prompt = module.render_accuracy_prompt(item, generated)
        raw = extra.chat(
            *extra.config("judge"), prompt, max_tokens=max_tokens, timeout=extra.JUDGE_TIMEOUT
        )
        try:
            label = module.parse_judge_label(raw)
        except (ValueError, json.JSONDecodeError):
            return {
                "id": qid,
                "is_correct": False,
                "label": "JUDGE_ERROR",
                "judge_response": f"裁判响应解析失败｜{raw[:2000]}",
            }
        return {
            "id": qid,
            "is_correct": label == "CORRECT",
            "label": label,
            "judge_response": raw,
        }

    if family == "scriptmem":
        qa_type = str(item.get("task") or "")
        gold = [str(letter).upper() for letter in (item.get("gold_answers") or [])]
        predicted, malformed = module.predicted_letters(generated, qa_type)
        score = module.score_item(qa_type, gold, predicted, malformed)
        return {
            "id": qid,
            "is_correct": score == 1.0,
            "label": f"LETTERS={''.join(predicted) or '-'}/{''.join(gold)}",
            "judge_response": f"ScriptMem/{qa_type}：预测 {predicted}，金标 {gold}，"
            f"malformed={malformed}",
        }

    if family == "beam":
        native = item.get("gold_answer") or {}
        rubrics = module.rubric_items({**item, "rubric": native.get("rubric") or []})
        prompt = module.render_batch_judge_prompt(item["question"], generated, rubrics)
        raw = extra.chat(
            *extra.config("judge"), prompt, max_tokens=max_tokens, timeout=extra.JUDGE_TIMEOUT
        )
        try:
            scores = module.parse_rubric_scores(raw, len(rubrics))
        except (ValueError, json.JSONDecodeError):
            return {
                "id": qid,
                "is_correct": False,
                "label": "JUDGE_ERROR",
                "judge_response": f"{len(rubrics)} 条 rubric 解析失败｜{raw[:2000]}",
            }
        mean = sum(entry["score"] for entry in scores) / len(scores)
        return {
            "id": qid,
            # 二值化口径与 `judge.py::_read_beam_labels` 一致（见模块 docstring）。
            "is_correct": mean >= 1.0,
            "label": f"SCORE={mean:.2f}",
            "judge_response": json.dumps(scores, ensure_ascii=False)[:4000],
            "partial": mean,
        }

    if family == "clbench":
        return _clbench_judge(item, generated)

    return {
        "id": qid,
        "is_correct": False,
        "label": "JUDGE_ERROR",
        "judge_response": f"没有为数据集 {family!r} 配判分（见模块 docstring 的分派表）",
    }


#: CL-Bench 每批判几条 rubric。**30 是量出来的**（2026-10-06，同一道题、同一个网关）：
#: 10 条 → 12.1 秒 · 30 条 → 13.5 秒 · **109 条 → 360 秒还不返回**（撞网关的源站时限）。
#: ≤30 条时**与归档那份的"一次判完"逐字等价**（只有一次调用）。
_CLBENCH_CHUNK = 30


def _clbench_judge(item: dict, generated: str) -> dict:
    """CL-Bench：**拆批判**（每批 `_CLBENCH_CHUNK` 条 rubric），**每批都通过才算对**。

    ## 为什么不是"一次判完"

    官方那份 `evaluate_rubric_clbench` 把**全部** rubric 塞进一次调用，
    而它的判据本来就是**逐条独立判定、全中才算 1**（prompt 原文：
    *"For every requirement in the 【Rubrics】, verify one by one"* / *"strict, all-or-nothing"*）。
    ⇒ 拆批**不改变判据**（同一组条件被拆成几组"全都要满足"），只是让每次调用的规模落在
    网关扛得住的范围内。⚠ 这是**已声明的偏离**（记在 docs/decisions.md **D34**）：
    批与批之间互相看不见，所以**同一道题的判定可能与"一次判完"不同**——
    实测 ≤30 条的那 128 道走的是同一条路（只有一批），**不受影响**。

    ## 判分语义

    * 所有批次的 `rubric_clbench_score` 都是 1 ⇒ `CORRECT`；
    * 任一批**判不出来**（超时 / JSON 解析失败）⇒ `JUDGE_ERROR`（**不冒充 WRONG**）——
      归档那份对这两种情况一律记 0，分不出"没测到"与"答得不对"，我们在这一层分开；
    * `partial` = 全部批次里 `yes` 的条数 / 总条数（比单批的比例更接近真实进展）。
    """
    import asyncio

    import httpx

    extra = _extra()
    module = _archived("clbench")
    qid = str(item["id"])
    rubrics = list(item.get("gold_rubric") or [])
    if not rubrics:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": "CL-Bench：这条题没有 rubric（套件里是空表）",
        }
    base, key, model = extra.config("judge")
    batches = [rubrics[i : i + _CLBENCH_CHUNK] for i in range(0, len(rubrics), _CLBENCH_CHUNK)]

    async def _run() -> list[dict]:
        out: list[dict] = []
        async with httpx.AsyncClient(timeout=extra.JUDGE_TIMEOUT) as client:
            for batch in batches:
                out.append(
                    await module.evaluate_rubric_clbench(
                        client,
                        base_url=base,
                        api_key=key,
                        model=model,
                        rubrics=batch,
                        predicted_answer=generated,
                        # 每批**只试一次**：失败是结构性的（撞源站时限），重试只是把
                        # "一条判不出来"变成"多花几分钟判不出来"。
                        max_retries=1,
                    )
                )
        return out

    payloads = asyncio.run(_run())
    passed = 0
    broken = False
    yes = 0
    total = 0
    for payload in payloads:
        reason = str(payload.get("rubric_clbench_rationale") or "")
        statuses = list(payload.get("rubric_clbench_requirement_status") or [])
        if "API call failed" in reason or "JSON parse failed" in reason or not statuses:
            broken = True
            continue
        total += len(statuses)
        yes += sum(1 for status in statuses if str(status).strip().lower().startswith("y"))
        if float(payload.get("rubric_clbench_score") or 0.0) >= 1.0:
            passed += 1
    correct = not broken and passed == len(batches)
    return {
        "id": qid,
        "is_correct": correct,
        "label": "JUDGE_ERROR" if broken else ("CORRECT" if correct else "WRONG"),
        "judge_response": json.dumps(
            {
                "batches": len(batches),
                "passed": passed,
                "requirement_yes": yes,
                "requirement_total": total,
                "details": [str(p.get("rubric_clbench_rationale") or "")[:200] for p in payloads],
            },
            ensure_ascii=False,
        )[:4000],
        "partial": (yes / total) if total else None,
    }


def _generic_judge(item: dict, generated: str, *, max_tokens: int) -> dict:
    """**没有专属判分的家族**——响亮报 `JUDGE_ERROR`，绝不借别的家族的判分顶替。

    历史实现按 `judge_kind` 兜底：`exact` 直接借 `judge_tempreason` 的别名包含、
    `llm` 借一份通用参考回答裁判。前者是**接线错误**——MuSiQue / HybridQA 因此
    被 TempReason 口径静默判分（2026-10-20 审计）。⇒ 现在：

    * 没有显式分支 ⇒ `JUDGE_ERROR`（`supports()` 也会在预检阶段判 False）；
    * `exact` 的别名包含只写进 `diagnostics`，**不参与 `is_correct`**。
    """
    kind = str(item.get("judge_kind") or "")
    diagnostics: dict[str, object] = {}
    if kind == "exact":
        ok, why = _extra().judge_tempreason(generated, item.get("gold_answers") or [])
        diagnostics = {"alias_contains": ok, "alias_reason": why}
    return {
        "id": str(item["id"]),
        "is_correct": False,
        "label": "JUDGE_ERROR",
        "judge_response": (
            f"{item.get('dataset')!r} / judge_kind={kind!r} 没有接线的专属判分；"
            "别名包含只作诊断，不进入正式分数"
        ),
        "diagnostics": diagnostics,
    }


def cmd_answer(args: argparse.Namespace) -> int:
    items = read_jsonl(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {str(row["id"]) for row in read_jsonl(output)}
    extra = _extra()
    base, key, model = extra.config("answer")
    with contextlib.nullcontext(output.open("a", encoding="utf-8")) as handle:
        for item in items:
            if str(item["id"]) in done:
                continue
            policy_path = os.environ.get("OFFICIAL_ANSWER_CONTEXT_POLICY")
            if policy_path:
                from eval.datasets.official_groups import write_json
                from eval.harness.model_context import fit_retrieved_context

                policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
                if model != policy["model"]:
                    raise ValueError("Answer context policy model differs from selected model")
                item, audit = fit_retrieved_context(
                    item, _answer_prompt, policy, max_tokens=args.max_tokens
                )
                actual_path = output.parent / "answer-context-input.jsonl"
                actual_path.write_text(jsonl_line(item), encoding="utf-8")
                audit.update(original_input=str(args.input), actual_input=str(actual_path))
                write_json(output.parent / "answer-context-audit.json", audit)
            generated = extra.chat(
                base,
                key,
                model,
                _answer_prompt(item),
                max_tokens=args.max_tokens,
                timeout=extra.ANSWER_TIMEOUT,
            )
            write_line(handle, {"id": str(item["id"]), "generated_answer": generated})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    items = {str(item["id"]): item for item in read_jsonl(Path(args.input))}
    answers = {
        str(row["id"]): row.get("generated_answer", "") for row in read_jsonl(Path(args.answers))
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # ⚠ **按 id 合并**：判分要调网关，几千题的一轮不该因为重启从头再判一遍。
    #   与归档那几份（`evaluate` 覆盖）故意不同，见模块 docstring。
    existing = {str(row["id"]) for row in read_jsonl(output)}
    with contextlib.nullcontext(output.open("a", encoding="utf-8")) as handle:
        for qid, item in items.items():
            if qid in existing:
                continue
            write_line(handle, _judge(item, answers.get(qid, ""), max_tokens=args.max_tokens))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI 形状与归档那几份一致：`--max-tokens` 挂在**两个子命令上**（见 `pipeline_cli`）。"""
    return build_pipeline_parser(
        __doc__.splitlines()[0],
        # AML Answer 阶段的官方输出预算：8,192 tokens。
        default_max_tokens=8192,
        max_tokens_help="answer / 裁判各自的输出上限",
    )


def main() -> int:
    args = build_parser().parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
