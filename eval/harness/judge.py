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
import time
from pathlib import Path

# ── façade：搬去 judge_items / judge_results 的名字在这层**重新导出** ────────
# `from eval.harness.judge import X` 全线照旧（消费方遍布 tests/ tools/ experiments/）。
# ⚠ 重新导出的是**名字的快照**：`monkeypatch.setattr(judge, "X", ...)` 打不到搬走后的
#   真实现——要补丁就去补 `judge_items.X` / `judge_results.X`（见 tests/test_harness.py）。
from eval.harness.judge_items import (  # noqa: F401
    _ADAPTER_PIPELINES,
    _LOCAL_PIPELINES,
    _PIPELINES,
    DATE_HEADER,
    DATE_MODES,
    DATE_PREFIX,
    EXTRA_DATASETS,
    MEMORY_FIELD,
    PLATFORM_TOKEN_PREFIX,
    PLATFORM_TOKENIZER,
    _annotated,
    _build_beam_items,
    _build_clbench_items,
    _build_extra_items,
    _build_personamem_items,
    _date_header,
    _encode,
    _encoder,
    _prefix_within_budget,
    _prepend_execution_record,
    _warn_truncated,
    _weekday,
    build_input_items,
    build_official_items,
    pipeline_for,
    render_memories,
    truncate_to_platform_prefix,
)
from eval.harness.judge_results import (  # noqa: F401
    JudgeResult,
    _read_beam_labels,
    _read_clbench_labels,
    _recover_beam_labels,
    _sanitize_jsonl,
)
from eval.jsonl_io import jsonl_line as _jsonl_line

from .annotate import MARKS
from .api_config import ENV_ENDPOINT_INDEX

__all__ = [
    "DATE_HEADER",
    "DATE_MODES",
    "DATE_PREFIX",
    "MARKS",
    "JudgeResult",
    "MEMORY_FIELD",
    "build_input_items",
    "build_official_items",
    "render_memories",
    "run_judge",
    "pipeline_for",
]


def _subprocess_env(endpoint_index: int | None = None) -> dict[str, str]:
    """注入 `PYTHONPATH` 让 `from api_config import (...)` 找到本目录那份。

    `sys.path.insert(0, <不存在路径>)` 不会中断 import——它只是塞进一个没有该模块的
    条目，import 继续往后找到这里。归档因此**保持只读**。

    `endpoint_index` 非 None ⇒ 一并写进 `AML_ENDPOINT_INDEX`：子进程里 import 的
    `api_config` 据此选中那一号对话端点（`memory` / `memory2` 是同一个模型、两把 key，
    **D39**）。⚠ 它是**每个子进程**唯一的端点开关——父进程 `os.environ` 是共享的，
    在线程池里改不了（见 `api_config` 的模块 docstring）。
    """
    env = dict(os.environ)
    if endpoint_index is not None:
        env[ENV_ENDPOINT_INDEX] = str(endpoint_index)
    harness_dir = str(Path(__file__).resolve().parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = harness_dir + (os.pathsep + existing if existing else "")
    return env


def _run(
    pipeline: Path,
    argv: list[str],
    *,
    timeout: float | None,
    attempts: int = 3,
    endpoint_index: int | None = None,
) -> None:
    """跑一个 pipeline 子命令，**瞬时故障自动重试**。

    ## 为什么必须重试（2026-09-26，一次真实事故）

    LongMemEval 那轮跑到第 47 题时，**判分侧一次 TLS 握手失败**
    （`httpcore ... start_tls` → `httpx.ConnectError`）就把**两小时的跑批整个打死**，
    而**没有任何东西会重试**。那一轮实际被同类故障打断过 11 次。

    ⚠ **重试是安全的**，因为两个子命令都幂等：`answer` 以**追加**模式打开、按 `id`
    跳过已完成的题；`evaluate` 以 `"w"` **覆盖**打开。⇒ 重跑一次最坏是白跑一遍。
    ⇒ **不重试的代价（整轮作废）远大于重试的代价（几十秒）**。

    ⚠ **重试次数要有界**：真 bug（形状不符、数据坏了）重试 3 次还是失败 ⇒ 照样抛，
    错误信息原样带出去——**不把"一直失败"伪装成"在重试"**。

    `endpoint_index` 透传给 `_subprocess_env()`：它决定这个子进程打哪个对话端点
    （`None` = 不注入 ⇒ 子进程用主端点）。
    """
    last: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            [sys.executable, str(pipeline), *argv],
            env=_subprocess_env(endpoint_index),
            capture_output=True,
            text=True,
            # ⚠ **父进程必须按 UTF-8 解子进程的管道**（2026-10-13 实测踩过）：
            #   `text=True` 单独用会走**locale 编码**，Windows 上是 GBK——而 pipeline
            #   子进程写的是 UTF-8（中文与全角标点），于是 `subprocess` 的读线程抛
            #   `UnicodeDecodeError`、`stdout`/`stderr` 双双是 `None`，
            #   真正的失败原因（下面那句 `[-4000:]`）**反被一个 TypeError 顶掉**，
            #   而且**整轮跑批在判分那一步作废**（Search 的 26 分钟白跑）。
            #   `errors="replace"` 只影响日志文本，不碰任何数据文件。
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if completed.returncode == 0:
            if attempt > 1:
                print(
                    f"  （{pipeline.name} {' '.join(argv[:1])}：第 {attempt} 次尝试成功）",
                    flush=True,
                )
            return
        last = RuntimeError(
            f"{pipeline.name} {' '.join(argv[:1])} 失败（exit {completed.returncode}）\n"
            f"{(completed.stderr or '').strip().split(chr(10))[-1][:1600]}\n"
            f"--- stdout ---\n{(completed.stdout or '')[-4000:]}\n"
            f"--- stderr ---\n{(completed.stderr or '')[-4000:]}"
        )
        if attempt < attempts:
            wait = 5 * attempt
            print(
                f"  ⚠ {pipeline.name} {' '.join(argv[:1])} "
                f"第 {attempt}/{attempts} 次失败（exit {completed.returncode}），"
                f"{wait}s 后重试（该步骤幂等）",
                flush=True,
            )
            time.sleep(wait)
    raise last  # type: ignore[misc] —— attempts ≥ 1，循环至少设过一次


#: 判分子进程的**上限**（秒）。`run_judge` 的 `timeout=None` 视为取它（见 `run_judge` 的 docstring）。
#: ⚠ **1200s（20 分钟）**：**不返回就记题号、跳过**（用户 2026-10-14 定，所有数据集统一）。
#: 口径是「**一个用户的某一步**（`answer` 或 `evaluate`）**整体**超过 20 分钟」——
#: ⛔ 不是"单题 20 分钟"（归档 pipeline 本来就按用户整批跑）。20 分钟 > 最慢的合法 clbench 子进程。
DEFAULT_JUDGE_TIMEOUT: float = 1200.0
PLATFORM_OUTPUT_TOKENS: int = 8_192


def run_judge(
    pipeline: Path,
    items: list[dict],
    out_dir: Path,
    *,
    dataset: str = "locomo-refined",
    max_tokens: int = PLATFORM_OUTPUT_TOKENS,
    timeout: float | None = None,
    endpoint_index: int | None = None,
) -> list[JudgeResult]:
    """跑 `answer` → `evaluate` 两步，返回逐题结果。

    ⚠ **`out_dir` 必须由调用方建**——`pipeline_locomo-refined.py` **不会
    `mkdir(parents=True)`**（`clb_pipeline.py` 会）。文件不存在时它以 `"a"` 模式打开，
    所以目录缺了就是 `FileNotFoundError`。

    ⚠ **超时默认 `DEFAULT_JUDGE_TIMEOUT`（1200s = 20 分钟）**（2026-10-14 定）：之前是"`None` = 不超时"，
    但实测一个**真挂住**的子进程会**永久阻塞**那个 worker（clbench 一条 0 字节挂了 57 分钟）。
    ⇒ 现在 20 分钟不返回就走调用方的「记下题号、跳过」。口径是**一个用户的某一步整体**，
    ⛔ 不是"单题"（归档 pipeline 按用户整批跑）。

    **两步之间的不对称**：`answer` 以**追加**模式打开并按 `id` 跳过已完成的题
    （所以中断后重跑是安全的）；`evaluate` 以 `"w"` **覆盖**打开。
    而两个 pipeline 的 `evaluate` 都在 ID 集合不一致时 **`raise SystemExit`**——
    **漏跑几题是硬失败，不是静默掉分**。

    `endpoint_index` 是**这次调用**用哪一号对话端点（D39）：两步都用同一个
    （同一次调用内不切换），而不同 sample 之间由调用方轮转——派发在
    [`../experiments/run.py`](../experiments/run.py) 的判分线程池。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.jsonl"
    answers_path = out_dir / "answers.jsonl"
    labels_path = out_dir / "labels.jsonl"

    with input_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(_jsonl_line(item))

    # ★ 续跑前先修掉上次被强杀留下的半行——否则 `answer` 步读它就 `JSONDecodeError`，
    #   而表现是"**卡在同一道题**"（详见 `_sanitize_jsonl` 的事故记录）。
    _sanitize_jsonl(answers_path)

    # `--model` / `--base-url` / `--api-key-env` 都**是死的**：这两个子命令在协程开头
    # 就用 `api_config` 的常量重写 `args.*`。所以这里一个都不传（传了只会让人以为生效了）。
    # ⚠ **`--max-tokens` 三份收得都不一样**（**按 pipeline 代码为准**）：
    #   · `clb_pipeline.py`：**两个子命令都不收** ⇒ 一个都别传，传了会 `unrecognized arguments`
    #   · `pipeline_beam.py`：`answer` 收 `--max-tokens`，`evaluate` 收的是 `--judge-max-tokens`
    #     ⇒ **只给 answer**（evaluate 那个我们沿用它的默认值）
    #   · 其余：两个子命令都收
    answer_extra = [] if dataset == "clbench" else ["--max-tokens", str(max_tokens)]
    evaluate_extra = ["--max-tokens", str(max_tokens)] if dataset not in ("clbench", "beam") else []
    # ⚠ `None` ⇒ 取上限（`DEFAULT_JUDGE_TIMEOUT`）：真挂住的子进程不能永久阻塞 worker。
    timeout = timeout if timeout is not None else DEFAULT_JUDGE_TIMEOUT
    _run(
        pipeline,
        ["answer", "--input", str(input_path), "--output", str(answers_path), *answer_extra],
        timeout=timeout,
        endpoint_index=endpoint_index,
    )
    if dataset == "beam":
        # ⚠ 归档 `pipeline_beam.py evaluate` 要求"每个 rubric 索引**恰好一次**"；模型偶尔
        #   不合规就整批 `ValueError`（2026-10-14 实测：第一个 beam 用户就撞上，整轮中止）。
        #   逐 rubric 单条模板恢复（`beam_judge.py`，产出与 evaluate **相同的标签行**）
        #   ⇒ 不因一条不合规丢掉整轮；恢复也失败才响亮抛出。
        try:
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
                    *evaluate_extra,
                ],
                timeout=timeout,
                endpoint_index=endpoint_index,
            )
        except RuntimeError:
            _recover_beam_labels(
                pipeline, items, answers_path, labels_path, out_dir, endpoint_index
            )
        return _read_beam_labels(answers_path, labels_path)

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
            *evaluate_extra,
        ],
        timeout=timeout,
        endpoint_index=endpoint_index,
    )

    if dataset == "clbench":
        return _read_clbench_labels(answers_path, labels_path)

    generated = {
        row["id"]: row["generated_answer"]
        for row in (
            # ⚠ 同样按 `"\n"` 切：答案文本里出现 `U+2028` 时，`splitlines()`
            #   会把这一条劈开 ⇒ 答案对不上号（而**不会报错**）。
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").split("\n")
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
                    partial=row.get("partial"),
                    metrics=row.get("metrics"),
                )
            )
    return results

