"""包住 AML pipeline 的裁判——**subprocess 调用，不 import 它们**。

## 为什么只能 subprocess

`pipeline_locomo-refined.py` 的**文件名里带连字符**，无法 `import`；它和
`pipeline_longmemeval-s.py` 都是带 argparse 子命令的独立脚本。
harness 要么 subprocess 调它、要么**复制它的逻辑**——后者会让
"契约以 pipeline 代码为准"这条失去意义（复制的当天就对不上了）。

## 注入格式：**键名是 `speaker_1_memories`**

§17.3 的 **V3** 记着"用什么键名注入没有明文"。**读 pipeline 源码即可回答**：

```python
fallback_memories = item.get("retrieved_context", item.get("memories", ""))
values = {
    "speaker_1_name": item.get("speaker_1_name", "speaker 1"),
    "speaker_1_memories": item.get("speaker_1_memories", fallback_memories),
    "speaker_2_name": item.get("speaker_2_name", "speaker 2"),
    "speaker_2_memories": item.get("speaker_2_memories", ""),
    ...
}
```

⇒ **主字段是 `speaker_1_memories`**（回退链 `speaker_1_memories → retrieved_context →
memories`）；**`speaker_2_memories` 没有回退，缺即空串**——这条不对称是
[`../CLAUDE.md`](../CLAUDE.md) 里"退化不对称"的出处。

> ⚠ **仍然未知的是"怎么分"**，不是"用哪个键"：真实 AML 侧按什么把我们的返回列表
> 切成两个 speaker 块，**归档里看不到**（§17.1 的 S1 是同一类问题的记忆版）。
> ⇒ 本模块把**全部命中的 `content` 拼成一个字符串放进 `speaker_1_memories`**，
> `speaker_2_memories` 留空。**这是一个显式的代理假设**，不是事实——
> `memory_field` 参数就是给它留的出口。**别把它当成已验证的契约。**

## 记住这条不对称（会改变结果）

| 数据集 | `evaluate` 步遇到 ID 集合不一致时 |
| --- | --- |
| `pipeline_locomo-refined.py` / `pipeline_longmemeval-s.py` | **`raise SystemExit`——硬失败** |
| `clb_pipeline.py` | `answers.get(ident, {})`——**缺答案一律判 0** |

⇒ **同一个 harness 缺陷（漏跑几题）在两份数据集上表现完全不同**：
一份崩掉，一份静默掉分。**别把后者当成"模型变差了"**（`../CLAUDE.md`）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from eval.datasets import Sample
from eval.datasets.locomo import PIPELINE as LOCOMO_PIPELINE
from eval.datasets.longmemeval import PIPELINE as LME_PIPELINE

from .driver import SearchHit

__all__ = [
    "JudgeResult",
    "MEMORY_FIELD",
    "build_input_items",
    "render_memories",
    "run_judge",
    "pipeline_for",
]

#: 命中列表的注入位置（见模块 docstring 的 V3 讨论）。
MEMORY_FIELD = "speaker_1_memories"

#: 数据集 → 归档 pipeline 文件名。**数据集专属文件名的知识止步于 `eval/datasets/`**（D16）
#: ——这里的映射就是那条边界在 harness 侧的接口。
_PIPELINES = {
    "locomo-refined": LOCOMO_PIPELINE,
    "longmemeval-s": LME_PIPELINE,
}


def pipeline_for(bench_dir: Path, dataset: str) -> Path:
    """取归档里的 pipeline 脚本路径。"""
    try:
        name = _PIPELINES[dataset]
    except KeyError:
        raise ValueError(f"未知数据集 {dataset!r}——没有对应的归档 pipeline") from None
    return Path(bench_dir) / name


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """一题的判分结果——`judge_response` 原样留着，**判分失败时它是唯一的线索**。"""

    qid: str
    is_correct: bool
    label: str
    judge_response: str
    generated_answer: str


def render_memories(hits: list[SearchHit]) -> str:
    """把命中拼成注入文本。

    **只取 `content`**——这是 S1 那条代理假设的落点（"AML 从返回项里取哪个字段"）。
    用 `"\\n"` 拼接而不是别的分隔符：归档的 `memory_text()` 对列表正是 `"\\n".join(...)`，
    所以"我们自己拼"与"把列表交给它拼"**逐字相同**。
    """
    return "\n".join(hit.content for hit in hits)


def build_input_items(
    sample: Sample,
    hits_by_qid: dict[str, list[SearchHit]],
    *,
    memory_field: str = MEMORY_FIELD,
) -> list[dict]:
    """构造 pipeline `answer` 步读的 JSONL 项。

    字段名**以 pipeline 代码为准、不照 readme**（`generated_answer` 才是 stage 间的规范
    字段；readme 写的 `predicted_answer` / `hypothesis` **没有 pipeline 读**）。

    `gold_answer` 直接放 `Question.gold` 的原始形态（LoCoMo 是 `list[str]`）——
    归档的 `gold_answer()` 走 `memory_text()`，**列表由它自己用 `\\n` 拼**，
    在这里预先拼会改变那条链路的行为。
    """
    items = []
    for question in sample.questions:
        hits = hits_by_qid.get(question.qid, [])
        item: dict = {
            "id": question.qid,
            "question": question.question,
            "gold_answer": question.gold,
            "speaker_1_name": sample.speaker_names[0],
            "speaker_2_name": sample.speaker_names[1],
            # `speaker_2_memories` **没有回退**，缺即空串——这里就照实留空，
            # 让不对称原样暴露在 prompt 里，而不是我们自己造一个"看起来更合理"的形状。
            "speaker_2_memories": "",
        }
        item[memory_field] = render_memories(hits)
        items.append(item)
    return items


def _subprocess_env() -> dict[str, str]:
    """注入 `PYTHONPATH` 让 `from api_config import (...)` 找到本目录那份。

    `sys.path.insert(0, <不存在路径>)` 不会中断 import——它只是塞进一个没有该模块的
    条目，import 继续往后找到这里。归档因此**保持只读**。
    """
    env = dict(os.environ)
    harness_dir = str(Path(__file__).resolve().parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = harness_dir + (os.pathsep + existing if existing else "")
    return env


def _run(pipeline: Path, argv: list[str], *, timeout: float | None) -> None:
    completed = subprocess.run(
        [sys.executable, str(pipeline), *argv],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{pipeline.name} {' '.join(argv[:1])} 失败（exit {completed.returncode}）\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )


def run_judge(
    pipeline: Path,
    items: list[dict],
    out_dir: Path,
    *,
    max_tokens: int = 256,
    timeout: float | None = None,
) -> list[JudgeResult]:
    """跑 `answer` → `evaluate` 两步，返回逐题结果。

    ⚠ **`out_dir` 必须由调用方建**——`pipeline_locomo-refined.py` **不会
    `mkdir(parents=True)`**（`clb_pipeline.py` 会）。文件不存在时它以 `"a"` 模式打开，
    所以目录缺了就是 `FileNotFoundError`。

    ⚠ **超时默认 `None`**：Full run 要连续跑 0.5–2 天，一个"看着合理"的超时会把
    **慢**伪装成**失败**（§15）。要设就设得比 `Full` 的时长还宽。

    **两步之间的不对称**：`answer` 以**追加**模式打开并按 `id` 跳过已完成的题
    （所以中断后重跑是安全的）；`evaluate` 以 `"w"` **覆盖**打开。
    而两个 pipeline 的 `evaluate` 都在 ID 集合不一致时 **`raise SystemExit`**——
    **漏跑几题是硬失败，不是静默掉分**。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.jsonl"
    answers_path = out_dir / "answers.jsonl"
    labels_path = out_dir / "labels.jsonl"

    with input_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    # `--model` / `--base-url` / `--api-key-env` 都**是死的**：这两个子命令在协程开头
    # 就用 `api_config` 的常量重写 `args.*`。所以这里一个都不传（传了只会让人以为生效了）。
    _run(
        pipeline,
        [
            "answer",
            "--input",
            str(input_path),
            "--output",
            str(answers_path),
            "--max-tokens",
            str(max_tokens),
        ],
        timeout=timeout,
    )
    _run(
        pipeline,
        [
            "evaluate",
            "--input",
            str(input_path),
            "--answers",
            str(answers_path),
            "--output",
            str(labels_path),
            "--max-tokens",
            str(max_tokens),
        ],
        timeout=timeout,
    )

    generated = {
        row["id"]: row["generated_answer"]
        for row in (
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    results = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            results.append(
                JudgeResult(
                    qid=row["id"],
                    is_correct=row["is_correct"],
                    label=row["label"],
                    judge_response=row["judge_response"],
                    generated_answer=generated.get(row["id"], ""),
                )
            )
    return results
