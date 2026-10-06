"""HybridQA / FEVEROUS 本地文本检索适配：Search-only 作答，固定上游函数判分。

新入口要求显式 answer_contract。采集重放继续走原有分派层，不改变其 prompt/评分。
FEVEROUS 只接受答案 prompt 可见的元素 ID；空检索不会补候选页或 gold evidence。
v3 对 JSON/标签/不可见 ID 错误最多做一次模型校正，只使用相同 Search 文本并保留两次输出。
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.datasets.aml.feverous import ANSWER_CONTRACT as AML_FEVEROUS_CONTRACT
from eval.datasets.aml.feverous import visible_json_evidence
from eval.datasets.feverous import ANSWER_CONTRACT as FEVEROUS_CONTRACT
from eval.datasets.feverous import LABELS, evidence_triplet
from eval.datasets.feverous import SCORER as FEVEROUS_SCORER
from eval.datasets.hybridqa import ANSWER_CONTRACT as HYBRIDQA_CONTRACT
from eval.datasets.hybridqa import SCORER as HYBRIDQA_SCORER
from eval.datasets.layout import archive_file
from eval.datasets.registry import benchmark_dir
from eval.harness.extra_pipeline import ANSWER_TIMEOUT, _chat, _config, _read_jsonl, _write_line

_CONTRACTS = {
    "hybridqa": {HYBRIDQA_CONTRACT},
    "feverous": {FEVEROUS_CONTRACT, AML_FEVEROUS_CONTRACT},
}
# 候选整页可形成远长于普通 QA 的上下文；仅本地 FEVEROUS 作答扩展超时。
FEVEROUS_ANSWER_TIMEOUT = 360.0
_HYBRID_PROMPT = """Use only the retrieved table rows and linked passages to answer the question.
Return only the short answer, without an explanation.
If evidence is insufficient, return INSUFFICIENT_EVIDENCE.

Retrieved evidence:
{memories}

Question: {question}
Answer:"""
_FEVEROUS_PROMPT = """Verify the claim using only the retrieved Wikipedia evidence below.
Return exactly one JSON object with two fields: label (a string) and evidence (a list of strings).
label must be exactly SUPPORTS, REFUTES, or NOT ENOUGH INFO.
evidence must list the bracketed element IDs that establish the verdict, without the brackets.
Copy each complete ID verbatim from the retrieved evidence, including its page title, spaces,
punctuation, accents, and numeric suffix. Never infer a cell or sentence number.
The element type is part of the ID: header_cell and cell are different; preserve it exactly.
Return at most five sentence IDs and twenty-five table/list IDs.
Do not invent IDs or use background knowledge. If you cannot copy a relevant visible ID,
use an empty evidence list. An empty list does not prove SUPPORTS or REFUTES.
NOT ENOUGH INFO may cite retrieved evidence showing why the claim cannot be verified;
if there is no evidence, return an empty evidence list.

Retrieved evidence:
{memories}

Claim: {question}
JSON:"""

_FEVEROUS_JSON_PROMPT = """Verify the claim using only the retrieved Wikipedia JSON below.
Return exactly one JSON object with label and evidence fields.
label must be exactly SUPPORTS, REFUTES, or NOT ENOUGH INFO.
evidence is a list of complete native element ID strings. For a complete visible page JSON,
join its exact title, an underscore, and the visible sentence key or cell/item id.
For a table caption, join the title with _table_caption_ and the visible table number.
Preserve spaces, accents, underscores, header_cell versus cell, and numeric suffixes exactly.
Only complete parseable retrieved page JSON or explicit bracketed IDs may establish an ID.
Incomplete, reordered or missing JSON fragments cannot establish an ID; do not fill the gaps.
Return at most five sentence IDs and twenty-five table/list IDs.
Do not use outside knowledge or invent IDs. Use an empty evidence list if no valid ID is visible.
NOT ENOUGH INFO may cite relevant retrieved evidence showing why the claim cannot be verified.

Retrieved evidence:
{memories}

Claim: {question}
JSON:"""


def render_answer_prompt(item: dict) -> str:
    dataset = item["dataset"]
    if dataset not in _CONTRACTS or item.get("answer_contract") not in _CONTRACTS[dataset]:
        raise ValueError("Corpus QA: missing/unsupported local answer contract")
    if not isinstance(item.get("retrieved_context"), str):
        raise ValueError("Corpus QA: retrieved_context must be present, including empty results")
    template = _HYBRID_PROMPT if dataset == "hybridqa" else _FEVEROUS_PROMPT
    if item.get("answer_contract") == AML_FEVEROUS_CONTRACT:
        if item.get("input_contract") != "aml-v1":
            raise ValueError("AML JSON answer contract requires an explicit AML input contract")
        template = _FEVEROUS_JSON_PROMPT
    return template.format(
        memories=item["retrieved_context"] or "(no memories)", question=item["question"]
    )


def _scorer(dataset: str) -> dict:
    name = HYBRIDQA_SCORER if dataset == "hybridqa" else FEVEROUS_SCORER
    path = archive_file(benchmark_dir(), name)
    from eval.datasets.manifest import MANIFEST

    source = path.read_bytes()
    expected = next(entry["sha256"] for entry in MANIFEST if entry["name"] == name)
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError(f"Corpus QA: upstream scorer sha256 mismatch: {path}")
    # HybridQA 脚本尾部直接读 sys.argv；只编译未经改动的函数/import 节点。
    tree = ast.parse(source.decode("utf-8"), filename=str(path))
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ],
        type_ignores=[],
    )
    namespace: dict = {}
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace


def visible_evidence(context: str, *, json_pages: bool = False) -> set[str]:
    result = set()
    for candidate in re.findall(r"\[([^\[\]\n]+)\]", context):
        try:
            evidence_triplet(candidate)
        except ValueError:
            continue
        result.add(candidate)
    if json_pages:
        result.update(visible_json_evidence(context))
    return result


def parse_feverous_answer(
    generated: str, context: str, *, json_pages: bool = False
) -> tuple[str, list[list[str]]]:
    text = generated.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    obj = json.loads(text)
    if not isinstance(obj, dict) or obj.get("label") not in LABELS:
        raise ValueError("FEVEROUS answer must contain a valid label")
    evidence = obj.get("evidence")
    if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
        raise ValueError("FEVEROUS evidence must be a list of element ID strings")
    allowed = visible_evidence(context, json_pages=json_pages)
    invisible = sorted(set(evidence) - allowed)
    if invisible:
        raise ValueError(
            "FEVEROUS answer cites evidence outside the retrieved context: "
            + json.dumps(invisible, ensure_ascii=False)
        )
    return obj["label"], [evidence_triplet(e) for e in dict.fromkeys(evidence)]


def score_answer(item: dict, generated: str) -> dict:
    dataset = item["dataset"]
    scorer = _scorer(dataset)
    if dataset == "hybridqa":
        answer = item["gold_answer"]["answer"]
        exact = float(scorer["compute_exact"](answer, generated))
        f1 = float(scorer["compute_f1"](answer, generated))
        metrics = {"exact_match": exact, "f1": f1}
        correct = bool(exact)
        why = f"upstream answer EM={exact:g}, token F1={f1:g}"
    else:
        try:
            label, evidence = parse_feverous_answer(
                generated,
                item["retrieved_context"],
                json_pages=item.get("answer_contract") == AML_FEVEROUS_CONTRACT,
            )
        except (ValueError, TypeError) as error:
            return {
                "id": item["id"],
                "is_correct": False,
                "label": "ANSWER_ERROR",
                "judge_response": str(error),
                "metrics": {
                    "strict_score": 0.0,
                    "label_accuracy": 0.0,
                    "evidence_precision": 0.0,
                    "evidence_recall": 0.0,
                },
            }
        gold = item["gold_answer"]
        instance = {
            "label": gold["label"],
            "evidence": gold["evidence"],
            "predicted_label": label,
            "predicted_evidence": evidence,
        }
        scorer["truncate_evidence"](instance, max_evidence=5, max_evidence_cell=25)
        correct = bool(scorer["is_strictly_correct"](instance))
        metrics = {
            "strict_score": float(correct),
            "label_accuracy": float(scorer["is_correct_label"](instance)),
            "evidence_precision": float(scorer["evidence_macro_precision"](instance)[0]),
            "evidence_recall": float(scorer["evidence_macro_recall"](instance)[0]),
        }
        # 上游 feverous_score 在 precision=recall=0 时除零；复用其逐题函数，
        # 汇总的 harmonic F1 在 run_record 中对零分母返回 0，不修改上游源码。
        why = json.dumps(
            {
                "predicted_label": label,
                "predicted_evidence": instance["predicted_evidence"],
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    return {
        "id": item["id"],
        "is_correct": correct,
        "label": "CORRECT" if correct else "WRONG",
        "judge_response": why,
        "metrics": metrics,
    }


def _fingerprint(item: dict) -> str:
    raw = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _items(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Corpus QA: duplicate input IDs")
    for row in rows:
        render_answer_prompt(row)
    return rows


def repair_id_guide(generated: str, context: str, *, json_pages: bool = False) -> list[str]:
    """Visible spelling hints only; these do not establish relevance or replace IDs."""
    allowed = sorted(visible_evidence(context, json_pages=json_pages))
    try:
        obj = json.loads(generated)
    except ValueError:
        return []
    evidence = obj.get("evidence") if isinstance(obj, dict) else None
    if not isinstance(evidence, list):
        return []
    hints = set()
    for element_id in evidence[:30]:
        if isinstance(element_id, str) and element_id not in allowed:
            # 页内相邻的 cell 数字只差一位，可能挤掉 header_cell 拼写提示。
            # 两种类型都只在完整 ID 确实可见时列出，不自动替换预测。
            cell = re.search(r"_(header_cell|cell)_(\d+_\d+_\d+)$", element_id)
            if cell:
                kind = "cell" if cell[1] == "header_cell" else "header_cell"
                alternative = element_id[: cell.start()] + f"_{kind}_{cell[2]}"
                if alternative in allowed:
                    hints.add(alternative)
            hints.update(difflib.get_close_matches(element_id, allowed, n=3, cutoff=0.6))
    return sorted(hints)


def generate_answer(item: dict, base: str, key: str, model: str, max_tokens: int) -> dict:
    """Bounded format repair; validation never reads gold or invokes the scorer."""
    original_prompt = render_answer_prompt(item)
    prompt = original_prompt
    attempts = []
    for _ in range(2 if item["dataset"] == "feverous" else 1):
        generated = _chat(
            base,
            key,
            model,
            prompt,
            max_tokens=max_tokens,
            timeout=FEVEROUS_ANSWER_TIMEOUT if item["dataset"] == "feverous" else ANSWER_TIMEOUT,
        )
        attempt = {"generated_answer": generated}
        attempts.append(attempt)
        if item["dataset"] != "feverous":
            break
        try:
            parse_feverous_answer(
                generated,
                item["retrieved_context"],
                json_pages=item.get("answer_contract") == AML_FEVEROUS_CONTRACT,
            )
        except (ValueError, TypeError) as error:
            attempt["validation_error"] = str(error)
            id_rule = (
                "Every evidence ID must use the exact visible spelling "
                "permitted by the original instructions. "
                if item.get("answer_contract") == AML_FEVEROUS_CONTRACT
                else "Every evidence ID must be copied verbatim from a visible bracketed ID. "
            )
            prompt = (
                original_prompt
                + "\n\nYour previous response was invalid:\n"
                + generated
                + "\nValidation error: "
                + str(error)
                + "\nVisible IDs with similar spellings (from the same retrieved text): "
                + json.dumps(
                    repair_id_guide(
                        generated,
                        item["retrieved_context"],
                        json_pages=item.get("answer_contract") == AML_FEVEROUS_CONTRACT,
                    ),
                    ensure_ascii=False,
                )
                + "\nReturn a corrected JSON object using only the same retrieved evidence. "
                + id_rule
                + "The previous invalid IDs must not appear in your new response. "
                "Preserve header_cell versus cell. The spelling hints alone do not prove "
                "the claim; select evidence based on its text. "
                "Use an empty evidence list if you cannot identify a valid relevant ID. "
                "Do not guess another element number. Return only JSON."
            )
        else:
            break
    return {"generated_answer": generated, "answer_attempts": attempts}


def cmd_answer(args: argparse.Namespace) -> int:
    items = _items(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(output)
    if len({row["id"] for row in existing}) != len(existing):
        raise ValueError("Corpus QA: duplicate existing answer IDs")
    by_id = {item["id"]: item for item in items}
    for row in existing:
        if row["id"] not in by_id or row.get("input_fingerprint") != _fingerprint(by_id[row["id"]]):
            raise ValueError("Corpus QA: existing answers use another input; choose a new run-id")
    done = {row["id"] for row in existing}
    base, key, model = _config("answer")
    with output.open("a", encoding="utf-8") as handle:
        for item in items:
            if item["id"] in done:
                continue
            answer = generate_answer(item, base, key, model, args.max_tokens)
            _write_line(
                handle,
                {
                    "id": item["id"],
                    **answer,
                    "input_fingerprint": _fingerprint(item),
                },
            )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    items = _items(Path(args.input))
    raw_answers = _read_jsonl(Path(args.answers))
    answers = {row["id"]: row for row in raw_answers}
    if len(answers) != len(raw_answers) or set(answers) != {item["id"] for item in items}:
        raise ValueError("Corpus QA: input/answer IDs must match exactly")
    for item in items:
        if answers[item["id"]].get("input_fingerprint") != _fingerprint(item):
            raise ValueError("Corpus QA: answer input fingerprint changed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            answer = answers[item["id"]]
            _write_line(handle, score_answer(item, answer["generated_answer"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("answer", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--input", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--max-tokens", type=int, default=1024)
        if command == "evaluate":
            child.add_argument("--answers", required=True)
    args = parser.parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
