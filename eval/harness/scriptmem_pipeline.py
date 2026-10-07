#!/usr/bin/env python3
"""ScriptMem 的 `answer` / `evaluate` 适配器——**判分直接 import 归档里的官方那份**。

## 为什么需要适配器

归档的 `pipeline_scriptmem.py` 与其余几份**形状不同**：

```bash
# 它的 answer 正常
python pipeline_scriptmem.py answer --input … --output …
# 它的 evaluate **既不吃 --input 也不吃 --answers**
python pipeline_scriptmem.py evaluate --data-dir data/raw --submission <提交文件> --details …
```

`run_judge` 是按"`answer` → `evaluate`（`--input/--answers/--output`）"这条通用形状调的，
所以这里放一层薄适配器把两边接上。**判分逻辑一行都不重写**：

```python
sys.path.insert(0, str(benchmark_dir()))
import pipeline_scriptmem as official          # ← 归档里那份，逐字
official.gold_letters(...) · official.predicted_letters(...) · official.score_item(...)
```

⚠ 那份官方模块**模块级**就 `from api_config import (...)` ⇒ import 它需要 `api_config` 可见。
运行期没问题（`run_judge` 起子进程时会注入 `PYTHONPATH`，见 [`judge.py`](./judge.py) 的
`_subprocess_env`）；**单元测试要自己把 `eval/harness` 塞进 `sys.path`**（见
[`../../tests/test_datasets_extra.py`](../../tests/test_datasets_extra.py)）。

## 输出（与其余 pipeline 同形）

    answer   → {"id", "generated_answer"}（**追加**模式，按 id 跳过已完成的题）
    evaluate → {"id", "is_correct", "label", "judge_response"}（**覆盖**模式）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ⚠ 与 `extra_pipeline.py` 同一处置：补仓库根 + 用 D16 的唯一入口拿归档目录。
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from eval.datasets.registry import benchmark_dir  # noqa: E402
from eval.harness.pipeline_cli import build_pipeline_parser  # noqa: E402
from eval.jsonl_io import read_jsonl, write_line  # noqa: E402


def _official():
    """归档里那份官方 pipeline——**放在函数里 import**：只有真跑时才需要它。"""
    path = str(benchmark_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    import pipeline_scriptmem  # type: ignore[import-not-found]

    return pipeline_scriptmem


def _chat(prompt: str) -> str:
    import httpx

    from eval.harness.api_config import ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL

    if not ANSWER_API_BASE:
        raise SystemExit("端点未配置（AML_BASE_URL）——本脚本不能离线跑模型调用")
    response = httpx.post(
        f"{ANSWER_API_BASE.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {ANSWER_API_KEY}"},
        json={
            "model": ANSWER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def cmd_answer(args: argparse.Namespace) -> int:
    official = _official()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {str(row.get("id")) for row in read_jsonl(output)}
    with output.open("a", encoding="utf-8") as handle:
        for item in read_jsonl(Path(args.input)):
            ident = str(item["id"])
            if ident in done:
                continue
            generated = _chat(official.render_answer_prompt(item))
            write_line(handle, {"id": ident, "generated_answer": generated})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """逐题判分——**三个函数都来自官方那份**（`score_item` 里那三条规则：单选相等、
    多选集合相等、排序按序）。"""
    official = _official()
    items = {str(item["id"]): item for item in read_jsonl(Path(args.input))}
    answers = {
        str(row["id"]): str(row.get("generated_answer", ""))
        for row in read_jsonl(Path(args.answers))
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for qid, item in items.items():
            qa_type = str(item.get("qa_type") or "")
            gold = official.gold_letters(item.get("gold_answer"))
            predicted, malformed = official.predicted_letters(answers.get(qid, ""), qa_type)
            score = official.score_item(qa_type, gold, predicted, malformed)
            ok = score >= 1.0
            why = f"ScriptMem/{qa_type}：gold={gold} predicted={predicted}" + (
                "（**malformed**：解析不出选项字母）" if malformed else ""
            )
            write_line(
                handle,
                {
                    "id": qid,
                    "is_correct": ok,
                    "label": "CORRECT" if ok else "WRONG",
                    "judge_response": why,
                },
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    return build_pipeline_parser(
        __doc__.splitlines()[0],
        default_max_tokens=512,
        max_tokens_help="**收下但不用**——形状对齐用",
    )


def main() -> int:
    args = build_parser().parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
