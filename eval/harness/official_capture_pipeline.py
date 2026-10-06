#!/usr/bin/env python3
"""官方采集（`official-capture`）的 answer / evaluate —— **一个 run 里按题分派到 8 个数据集**。

## 为什么需要它

其余每个数据集都是"一个数据集一套 prompt + 一套判分"，所以 harness 的
`run_judge` 按 `dataset` 选一份 pipeline 就够了。官方采集这份**不是**：
一个 run 里的题**混着 8 个数据集**（locomo / scriptmem / personamem / beam /
mquake / corporatebench / medmemorybench / memtrapbench），每题该用哪套 prompt
由 **题自己的归属**决定（[`build_official_items`](./judge.py) 把它放在 `item["dataset"]`）。

⇒ 本文件是**分派层**，不是第二份实现：

| 家族 | answer prompt | judge |
| --- | --- | --- |
| `locomo-refined` | `pipeline_locomo-refined.py` 的 answer prompt | 同文件 accuracy prompt + LLM |
| `scriptmem` | `pipeline_scriptmem.py` 的 answer prompt | `predicted_letters` + `score_item` |
| `beam` | `pipeline_beam.py#render_answer_prompt` | 同文件的 rubric 批量裁判（三点制均分） |
| `clbench` | `clb_pipeline.py#build_answer_prompt` | 同文件 rubric 裁判（全有全无 + 比例分） |
| `personamem-v2` | 通用 answer prompt | **归一化文本匹配**（见下） |
| 上表之外的 6 个 | `extra_pipeline.py#render_answer_prompt` | `extra_pipeline.py#judge_one` |

**prompt 与判分逻辑一行都不重写**：归档那几份用 `sys.path` + `import` 直接调
（文件名带连字符的用 `importlib`），`official-extra` 那六个走 `extra_pipeline` 的公开入口。

## 两处口径是**我们定的**，写在这里免得被当成官方口径

1. **`personamem-v2` 的判分**：上游对开放题走 `evaluate_narrow` 的 **LLM 裁判**，
   而它要 `item["preference"]`（偏好元数据）——**采集的 search 请求里没有这个字段**
   （那份 pipeline 根本不读检索字段，题与偏好都在它自己的题库里）。
   ⇒ 这里换成**同一类但不看偏好**的 LLM 裁判：拿 `correct_answer` 整段当参考回答，
   只判"意思对不对"（prompt 在 `_NARROW_JUDGE_PROMPT`）。**分数只在仓内前后比。**
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
import sys
from pathlib import Path

# ⚠ 本文件是**当脚本跑的**（`run_judge` 起 subprocess）：那时 `sys.path[0]` 是
# `eval/harness/`，仓库根不在上面 ⇒ `import eval.*` 会 `ModuleNotFoundError`。
# 与 [`personamem_pipeline.py`](./personamem_pipeline.py) 同一处置：把仓库根与本目录都补上。
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

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

#: **判分在本文件里、但答案 prompt 仍走通用那条**的家族（它们没有归档的 AML pipeline）。
_SIMPLE_JUDGE_FAMILIES = frozenset({"personamem-v2", "halumem", "feverous"})

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
    ⚠ **证据 id 的 F1 不在这里算**（上游有 `feverous_scorer`）——那要模型的证据引用，
    而我们的答案 prompt 不要求它输出证据 id。**这条写在这里，免得被当成"全口径"**。
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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def _write_line(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


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
    # `personamem-v2` 与六个 `official-extra` 家族都走我们自写那套通用 prompt
    # （`extra_pipeline.render_answer_prompt` 对未知 dataset 就落回 `ANSWER_PROMPT`）。
    return _extra().render_answer_prompt(item)


#: PersonaMem-v2 的金标是**整段参考回答**（不是答案串）⇒「包含」几乎不可能成立。
#: 这道 LLM 裁判是**我们定的口径**（上游 `evaluate_narrow` 要 `preference` 元数据，
#: 而采集的 search 请求里没有那个字段）。判据只有一条：**意思对不对**。
_NARROW_JUDGE_PROMPT = """\
You are grading whether a model's answer to a user's question is correct.

Grade CORRECT if the answer conveys the same key information as the reference answer \
(it does not need the same wording). Grade WRONG if it misses the key information, \
contradicts the reference, or refuses while the reference contains the answer.

Question: {question}

Reference answer: {gold}

Model answer: {generated}

Reply with a single JSON object: {{"label": "CORRECT" or "WRONG", "why": "<one short sentence>"}}"""


def _narrow_judge(item: dict, generated: str, *, max_tokens: int) -> dict:
    """PersonaMem-v2 开放题的 LLM 裁判（见常量注释）。"""
    extra = _extra()
    gold = [str(g) for g in (item.get("gold_answers") or [])]
    qid = str(item["id"])
    if not gold:
        return extra.verdict(qid, False, "无金标")
    if not (generated or "").strip():
        return extra.verdict(qid, False, "模型输出为空")
    prompt = _NARROW_JUDGE_PROMPT.format(
        question=item["question"], gold=gold[0], generated=generated
    )
    raw = extra.chat(
        *extra.config("judge"), prompt, max_tokens=max_tokens, timeout=extra.JUDGE_TIMEOUT
    )
    label = _parse_label(raw)
    if label is None:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": f"裁判响应里没有 CORRECT/WRONG｜{raw[:2000]}",
        }
    return {"id": qid, "is_correct": label == "CORRECT", "label": label, "judge_response": raw}


def _parse_label(raw: str) -> str | None:
    """从裁判响应里取 `CORRECT` / `WRONG`（JSON 里的 `label`，取不到就整段扫）。"""
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
            label = str(payload.get("label", "")).upper()
            if label in {"CORRECT", "WRONG"}:
                return label
    upper = text.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return "CORRECT"
    if "WRONG" in upper and "CORRECT" not in upper:
        return "WRONG"
    return None


def _judge(item: dict, generated: str, *, max_tokens: int) -> dict:
    """一题的判分——**按家族分派**，每个分支的口径出处写在下面。"""
    family = item.get("dataset")
    qid = str(item["id"])
    extra = _extra()
    if family in _EXTRA_FAMILIES or family in _SIMPLE_JUDGE_FAMILIES:
        if family == "personamem-v2":
            return _narrow_judge(item, generated, max_tokens=max_tokens)
        if family == "halumem":
            return _halumem_judge(item, generated, max_tokens=max_tokens)
        if family == "feverous":
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
    """**没有专属判分的家族**——按套件里那一个语义字段 `judge_kind` 二选一。

    这样后面再补一族金标（HaluMem / HybridQA / MuSiQue / FEVEROUS …）时，
    **接线只需要在套件里写对 `judge_kind`**，不需要改本文件：

    | `judge_kind` | 怎么判 |
    | --- | --- |
    | `exact` | 别名包含（`extra_pipeline.judge_tempreason` 的那套归一化）——短答案串 |
    | `llm` | 拿参考回答当标准的 LLM 裁判（`_narrow_judge`）——整段参考回答 |
    | 其它 | `JUDGE_ERROR`（**说出来**，别静默判 0） |
    """
    extra = _extra()
    kind = str(item.get("judge_kind") or "")
    if kind == "exact":
        ok, why = extra.judge_tempreason(generated, item.get("gold_answers") or [])
        return extra.verdict(str(item["id"]), ok, why)
    if kind == "llm":
        return _narrow_judge(item, generated, max_tokens=max_tokens)
    return {
        "id": str(item["id"]),
        "is_correct": False,
        "label": "JUDGE_ERROR",
        "judge_response": (
            f"{item.get('dataset')!r} / judge_kind={kind!r} 没有对应的判分"
            "（见 _generic_judge 的表）"
        ),
    }


def cmd_answer(args: argparse.Namespace) -> int:
    items = _read_jsonl(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {str(row["id"]) for row in _read_jsonl(output)}
    extra = _extra()
    base, key, model = extra.config("answer")
    with contextlib.nullcontext(output.open("a", encoding="utf-8")) as handle:
        for item in items:
            if str(item["id"]) in done:
                continue
            generated = extra.chat(
                base,
                key,
                model,
                _answer_prompt(item),
                max_tokens=args.max_tokens,
                timeout=extra.ANSWER_TIMEOUT,
            )
            _write_line(handle, {"id": str(item["id"]), "generated_answer": generated})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    items = {str(item["id"]): item for item in _read_jsonl(Path(args.input))}
    answers = {
        str(row["id"]): row.get("generated_answer", "") for row in _read_jsonl(Path(args.answers))
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # ⚠ **按 id 合并**：判分要调网关，几千题的一轮不该因为重启从头再判一遍。
    #   与归档那几份（`evaluate` 覆盖）故意不同，见模块 docstring。
    existing = {str(row["id"]) for row in _read_jsonl(output)}
    with contextlib.nullcontext(output.open("a", encoding="utf-8")) as handle:
        for qid, item in items.items():
            if qid in existing:
                continue
            _write_line(handle, _judge(item, answers.get(qid, ""), max_tokens=args.max_tokens))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI 形状与归档那几份一致：`--max-tokens` 挂在**两个子命令上**。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-tokens", type=int, default=1024, help="answer / 裁判各自的输出上限")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    answer = sub.add_parser("answer", parents=[common])
    answer.add_argument("--input", required=True)
    answer.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--answers", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
