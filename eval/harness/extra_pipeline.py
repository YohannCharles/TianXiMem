#!/usr/bin/env python3
"""我们自写的五份 `official-extra` 数据集的 `answer` / `evaluate`——**官方没有发布它们的 pipeline**。

CLI 与归档 pipeline **同形**（harness 的 `run_judge` 就是按这个形状调的）：

    python extra_pipeline.py answer --input items.jsonl --output answers.jsonl [--max-tokens N]
    python extra_pipeline.py evaluate --input items.jsonl --answers answers.jsonl \
        --output labels.jsonl

输入项由 [`judge._build_extra_items`](./judge.py) 产出：
`{id, dataset, question, gold_answer, retrieved_context, category}`

输出：

    answer   → `{"id", "generated_answer"}`（**追加**模式，临床多跳另带契约与输入指纹）
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
   **MuSiQue 两套并存**（默认走快速规则判分）：
   `--musique-judge rule`（默认）走原来那套归一化精确匹配，`--musique-judge llm`
   显式启用模型判"语义是否等价"。
3. **失败不静默**：判分出错时 `label` 带 `JUDGE_ERROR`、`is_correct=False`，
   原始响应留着——与 CL-Bench"API/JSON 失败一律记 0"同一处置，但**在 label 里说明白**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ⚠ 归档目录走 D16 的唯一入口（`benchmark_dir()` 读 `TIANXIMEM_BENCHMARK_DIR`，
#   默认 `dataset`）。subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根，
#   与 `tools/*.py` 同一处置。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── façade：`official-extra` 已按家族拆开（2026-10-15），这里**重新导出**搬走的名字 ──
# `from eval.harness.extra_pipeline import X` 全线照旧（消费方遍布 tests/、tools/、
# `corpusqa_pipeline.py` 与 `official_capture_pipeline.py`——后者按模块属性取名字）。
#
# ⚠ **模型调用的编排刻意留在本层**（`cmd_answer` / `cmd_evaluate` / `_judge_one` /
#   `_judge_musique` / `_judge_medmemorybench` / `_judge_docpp` / `_call_judge`）：
#   消费方直接 `monkeypatch.setattr(extra_pipeline, "_chat" / "_config" / "_call_judge" /
#   "_judge_one", …)` 把模型调用换成桩，而 bare-name 在**调用处所在模块**的字典里解析。
#   编排搬进家族模块会让这些补丁**静默失效**（家族模块又不许 import 本层，绕不回来）
#   ⇒ 家族模块里只放纯函数 / 模板 / 解析器；"怎么调模型、怎么记 JUDGE_ERROR"留在本层。
#
# 分层：`extra_base` ← `extra_<家族>`（六个）← 本层。依赖方向单向，家族不许 import 本层。
from eval.datasets.musique import ANSWER_CONTRACT as MUSIQUE_ANSWER_CONTRACT  # noqa: E402
from eval.datasets.musique import DIRECT_ANSWER_CONTRACT as MUSIQUE_DIRECT_CONTRACT  # noqa: E402
from eval.harness.corporatebench_pipeline import (  # noqa: E402
    build_answer_prompt as build_corporatebench_answer_prompt,
)
from eval.harness.corporatebench_pipeline import (  # noqa: E402
    judge_corporatebench,
    score_corporatebench,
)
from eval.harness.extra_base import (  # noqa: E402, F401
    ANSWER_PROMPT,
    ANSWER_TIMEOUT,
    JUDGE_TIMEOUT,
    _as_bool,
    _chat,
    _config,
    _contains,
    _escape_stray_quotes,
    _repair_memtrap_string_escapes,
    _verdict,
    normalize,
)
from eval.harness.extra_docpp import (  # noqa: E402, F401
    DOCPP_PROMPTS,
    _docpp_prompts,
    build_docpp_judge_prompt,
    judge_docpp_verdict,
)
from eval.harness.extra_medmemorybench import (  # noqa: E402, F401
    _MMB_OPTION_PATTERNS,
    _MMB_PUNCT,
    MMB_CODE_DIR,
    MMB_LLM_TYPES,
    MMB_MCD_ANSWER_CONTRACT,
    MMB_MCD_ANSWER_PROMPT,
    MMB_MCD_MAX_TOKENS,
    MMB_PROMPTS,
    _mmb_mcd_params,
    _mmb_metadata_info,
    _mmb_prompt_templates,
    _repair_mmb_judge_strings,
    build_mmb_judge_prompt,
    judge_mmb_llm_verdict,
    judge_mmb_option_match,
    judge_mmb_string_contain,
    mmb_normalize,
    mmb_option_letters,
)
from eval.harness.extra_memtrapbench import (  # noqa: E402, F401
    MEMTRAP_ANSWER_PROMPT,
    MEMTRAP_DIMENSIONS,
    MEMTRAP_JUDGE_PROMPT,
    MEMTRAP_PASS_MEAN,
    judge_memtrapbench_scores,
)
from eval.harness.extra_mquake import (  # noqa: E402, F401
    MQUAKE_ANSWER_PROMPT,
    judge_mquake,
)
from eval.harness.extra_musique import (  # noqa: E402, F401
    DEFAULT_MUSIQUE_JUDGE,
    MUSIQUE_ANSWER_PROMPT,
    MUSIQUE_DIRECT_ANSWER_PROMPT,
    MUSIQUE_JUDGE_LLM,
    MUSIQUE_JUDGE_MODES,
    MUSIQUE_JUDGE_PROMPT,
    MUSIQUE_JUDGE_RULE,
    build_musique_judge_prompt,
    judge_musique,
    judge_musique_llm_verdict,
)
from eval.harness.extra_tempreason import (  # noqa: E402, F401
    TEMPR_ANSWER_PROMPT,
    judge_tempreason,
)
from eval.harness.pipeline_cli import build_pipeline_parser  # noqa: E402
from eval.jsonl_io import read_jsonl, write_line  # noqa: E402

__all__ = [
    "DEFAULT_MUSIQUE_JUDGE",
    "MEMTRAP_PASS_MEAN",
    "MUSIQUE_JUDGE_MODES",
    "build_docpp_judge_prompt",
    "build_musique_judge_prompt",
    "judge_docpp_verdict",
    "judge_musique_llm_verdict",
    "judge_tempreason",
    "judge_corporatebench",
    "judge_memtrapbench_scores",
    "judge_mquake",
    "judge_musique",
    "normalize",
]


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
    if item.get("dataset") == "musique":
        # ⭐ 只有**链式多跳那一档**换 prompt（它没有拒答题）；原生可答/不可答变体走 v1，照旧。
        template = {
            MUSIQUE_DIRECT_CONTRACT: MUSIQUE_DIRECT_ANSWER_PROMPT,
            MUSIQUE_ANSWER_CONTRACT: MUSIQUE_ANSWER_PROMPT,
        }.get(item.get("answer_contract"), template)
    if item.get("dataset") == "corporatebench":
        return build_corporatebench_answer_prompt(item, scalar_template=ANSWER_PROMPT)
    if _is_mmb_clinical(item):
        template = MMB_MCD_ANSWER_PROMPT
    return template.format(
        memories=item.get("retrieved_context") or "(no memories)",
        question=item["question"],
    )


def _is_mmb_clinical(item: dict) -> bool:
    return (
        item.get("dataset") == "medmemorybench"
        and item.get("category") == "multi_hop_clinical_deduction"
    )


def _mmb_answer_metadata(item: dict, config: tuple[str, str, str], max_tokens: int) -> dict:
    if not _is_mmb_clinical(item):
        return {}
    payload = {
        "contract": MMB_MCD_ANSWER_CONTRACT,
        "prompt": render_answer_prompt(item),
        "base": config[0],
        "model": config[2],
        "max_tokens": max_tokens,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return {
        "answer_contract": MMB_MCD_ANSWER_CONTRACT,
        "input_fingerprint": fingerprint,
        "answer_max_tokens": max_tokens,
    }


def _validate_mmb_answers(
    items: list[dict], answers: dict, config: tuple[str, str, str], max_tokens: int | None = None
) -> None:
    """Reject old or changed clinical answers before any answer/judge API call."""
    for item in items:
        previous = answers.get(item["id"])
        if previous is None:
            continue
        if not (
            _is_mmb_clinical(item) or previous.get("answer_contract") == MMB_MCD_ANSWER_CONTRACT
        ):
            continue
        budget = max_tokens if max_tokens is not None else previous.get("answer_max_tokens")
        if type(budget) is not int or budget <= 0:
            raise ValueError("MedMemoryBench clinical answer has no valid budget; use a new run-id")
        expected = _mmb_answer_metadata(item, config, budget)
        if not expected or any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "MedMemoryBench clinical prompt/input/model/budget changed; use a new run-id"
            )


def cmd_answer(args: argparse.Namespace) -> int:
    items = read_jsonl(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {row["id"]: row for row in read_jsonl(output)}
    config = _config("answer")
    _validate_mmb_answers(items, previous, config, args.max_tokens)
    base, key, model = config
    with output.open("a", encoding="utf-8") as handle:
        for item in items:
            if item["id"] in previous:
                continue
            # **逐数据集挑 prompt**（判据是 item 里那个 `dataset` 字段，与 evaluate 同一口径）。
            # 分派理由见各自模板注释及 CorporateBench 的独立适配器。
            prompt = render_answer_prompt(item)
            generated = _chat(
                base, key, model, prompt, max_tokens=args.max_tokens, timeout=ANSWER_TIMEOUT
            )
            row = {"id": item["id"], "generated_answer": generated}
            row.update(_mmb_answer_metadata(item, config, args.max_tokens))
            write_line(handle, row)
            previous[item["id"]] = row
    return 0


def _judge_one(
    dataset: str | None,
    qid: str,
    item: dict,
    generated: str,
    *,
    judge: tuple[str, str, str],
    max_tokens: int,
    musique_judge: str = DEFAULT_MUSIQUE_JUDGE,
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
        return _judge_musique(
            qid, item, generated, judge=judge, max_tokens=max_tokens, mode=musique_judge
        )
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


def _judge_musique(
    qid: str,
    item: dict,
    generated: str,
    *,
    judge: tuple[str, str, str],
    max_tokens: int,
    mode: str,
) -> dict:
    """MuSiQue 判分：**可答默认走规则匹配 / 可显式切换模型裁判**。

    ⛔ **拒答题永远走规则**——见下。

    ⚠ `mode` 只影响**可答**那一支——`answerable=false` 的题问的是"有没有按契约拒答"，
    那是一个**输出格式**判据，交给模型判只会引入抖动（且会改动别处的 refusal 口径）。
    """
    if mode not in MUSIQUE_JUDGE_MODES:
        raise ValueError(f"`--musique-judge` 只能是 {MUSIQUE_JUDGE_MODES}，收到 {mode!r}")
    gold = item.get("gold_answer")
    contract = str(item.get("answer_contract") or MUSIQUE_ANSWER_CONTRACT)
    unanswerable = isinstance(gold, dict) and gold.get("answerable") is False

    if mode == MUSIQUE_JUDGE_RULE or unanswerable:
        ok, why = judge_musique(generated, gold, contract=contract)
        return _verdict(qid, ok, why)

    if not (generated or "").strip():
        # 空输出不必花一次裁判调用（与 MedMemoryBench / Doc-PP 同一处置）。
        return _verdict(qid, False, "MuSiQue/llm：模型输出为空")

    prompt = build_musique_judge_prompt(item["question"], gold, generated)
    raw = _call_judge(judge, prompt, max_tokens=max_tokens)
    ok, why = judge_musique_llm_verdict(raw)
    if ok is None:
        return {
            "id": qid,
            "is_correct": False,
            "label": "JUDGE_ERROR",
            "judge_response": f"{why}｜{raw[:2000]}",
        }
    # ⚠ 裁判**模型名**写进 `judge_response`：它是这次分数的配置指纹（换模型 = 换口径），
    #   而 label 的取值域被测试与台账钉着，别往里塞。
    return {
        "id": qid,
        "is_correct": ok,
        "label": "Musique/llm:" + ("CORRECT" if ok else "WRONG"),
        "judge_response": f"[{judge[2]}] {why}",
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
    answer_rows = {row["id"]: row for row in read_jsonl(Path(args.answers))}
    if any(
        _is_mmb_clinical(item)
        or answer_rows.get(item["id"], {}).get("answer_contract") == MMB_MCD_ANSWER_CONTRACT
        for item in items.values()
    ):
        _validate_mmb_answers(list(items.values()), answer_rows, _config("answer"))
    answers = {qid: row.get("generated_answer", "") for qid, row in answer_rows.items()}
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
                musique_judge=getattr(args, "musique_judge", DEFAULT_MUSIQUE_JUDGE),
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
    def add_musique_judge(evaluate: argparse.ArgumentParser) -> None:
        evaluate.add_argument(
            "--musique-judge",
            choices=MUSIQUE_JUDGE_MODES,
            default=DEFAULT_MUSIQUE_JUDGE,
            help=(
                "MuSiQue 可答题的判分方式（拒答题恒走 rule）："
                f"`{MUSIQUE_JUDGE_RULE}` = 归一化精确匹配（默认，快速且复现历史数字）；"
                f"`{MUSIQUE_JUDGE_LLM}` = 模型判语义等价。"
            ),
        )

    return build_pipeline_parser(
        __doc__.splitlines()[0],
        default_max_tokens=512,
        max_tokens_help="answer / 裁判各自的输出上限",
        answer_help="作答（**追加**模式，按 id 跳过已完成的题）",
        evaluate_help="判分（**覆盖**模式）",
        evaluate_extra=add_musique_judge,
    )


def main() -> int:
    args = build_parser().parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
