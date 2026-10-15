"""Doc-PP 家族的 evaluate：上游 `prompts/judge_evaluation.py` 两条判据的 prompt 与解析。

拆自 `extra_pipeline.py`（2026-10-15，只搬位置）：那层现在是薄 façade、把本模块的名字
全部重新导出。走 `with_policy=false` 那两条路的理由见下面那段注释。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# ⚠ 与 `extra_pipeline.py` 同一处置：subprocess 的 `sys.path[0]` 是本目录 ⇒ 补一条仓库根。
#   本目录的模块既被 `eval.harness.*` 包内引用、也被当脚本/顶层模块跑，两种跑法都要能
#   解析到 `eval.*`（包内 import 时仓库根已在路径上，这里是幂等的）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets.registry import benchmark_dir  # noqa: E402

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
