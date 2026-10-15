"""判分**产物**那一侧：结果类型 + 逐家族的标签回读（含续跑前的半行清理）。

`judge.py` 拆出来的第三块（2026-10-15）。"结果是什么形状、怎么从磁盘上的
`answers.jsonl` / 标签文件把它读回来"在这里；"怎么起子进程跑 pipeline"留在 `judge.py`。

⚠ `JudgeResult` 的**家在这里**（不是 `judge.py`）：读回函数是构造它的那一层，
`run_judge` 也要用它 ⇒ 定义放在这里，依赖方向单向（`judge` → `judge_results`）。
消费方照旧 `from eval.harness.judge import JudgeResult`（facade 重新导出）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eval.jsonl_io import LINE_BREAKS as _LINE_BREAKS
from eval.jsonl_io import jsonl_line as _jsonl_line


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """一题的判分结果——`judge_response` 原样留着，**判分失败时它是唯一的线索**。"""

    qid: str
    is_correct: bool
    label: str
    judge_response: str
    generated_answer: str
    #: **部分分**（0–1），只有**判分方本来就给出它**的数据集才有；给不出就是 `None`。
    #:
    #: 为什么需要它：CL-Bench 的官方分是**全有全无**的（`clb_pipeline.py` 的判分 prompt
    #: 原文："strict, all-or-nothing … The final score is binary"）——一道题从 0 翻到 1
    #: 要**每一条** rubric 都满足，**中间的所有进展都看不见**。实测某题
    #: `score=0.0` 而 `requirement_ratio=0.50`（14 条里满足 7 条）⇒ 它和"一条都没满足"
    #: 在二值分下**完全一样**。
    #:
    #: ⛔ **它不进 `overall`**：榜分是二值那一列，`partial` 只是**本地仪器**，
    #: 用来让"改进有没有效果"在低分辨率的数据集上也能被看见（见 `summarize` 的
    #: `partial_credit`）。CorporateBench 在这里传递标量 EM / 列表 set-F1，
    #: 同时在 run record 的 `scores.dataset_score` 记录其数据集指标。
    partial: float | None = None
    #: 数据集原生指标；不替代 is_correct/overall，按各自的聚合公式记录。
    metrics: dict[str, float] | None = None


def _sanitize_jsonl(path: Path) -> int:
    """把 `answers.jsonl` 修成**归档读得下去**的样子；返回改动了几行。

    **两件事一起做**（都只针对这个文件，且都保持语义不变）：

    | 症状 | 处置 |
    | --- | --- |
    | 末尾有**半行 JSON**（上次被强杀留下的） | **丢掉**——对应的题会被重新生成 |
    | 某行里含 `U+2028` 之类的字符（**归档自己写的**，我们只读不写） | **就地转义**成 `\u2028` |

    ⚠ 第二条是**我们控制不了写入侧**时的唯一办法：`answers.jsonl` 由归档的 `answer()`
    以 `json.dumps(..., ensure_ascii=False)` 写、又由它自己的 `rows()`（`splitlines()`）
    读 —— 万一**模型生成的答案**里带 `U+2028`，**归档会在下一次续跑时崩在自己写的文件上**。
    我们改不了归档，所以只能在它读之前把文件修好。转义是安全的：`json.loads` 会解回
    同一个字符，**语义一字不变**。

    ## 为什么需要它（2026-09-26，一次真实事故，比网络抖动更隐蔽）

    那次事故的真实链条是这样的：

    1. 判分侧**一次 TLS 抖动**把 runner 打死；
    2. 死的时候 pipeline 正以**追加**模式写 `answers.jsonl`（逐行 `write` + `flush`，
       见 `pipeline_longmemeval-s.py` 的 `answer()`）⇒ 留下**半行 JSON**；
    3. 而同一个 `answer()` 开头会 `rows(output)` **读它来跳过已完成的题**
       （`done = {item["id"] for item in rows(output)}`）⇒ `JSONDecodeError`；
    4. ⇒ **之后每次续跑都在同一个样本上确定性失败**——表现为"卡在同一题"，
       而**看起来像网络问题**（第一次确实是），于是排查方向全错。

    ⚠ **只在末尾删**：中途的坏行说明别的问题（磁盘、并发写），不该被静默吞掉。
    ⚠ 删掉的只有**半行**——它对应的那道题会被重新生成（`answer` 是追加 + 跳过已完成）。
    """
    if not path.exists():
        return 0
    # ⚠ **按 `"\n"` 切，不用 `splitlines()`**：这个文件是归档按 `+ "\n"` 写出来的，
    #   而 `splitlines()` 会把正文里的 `U+2028` 也当换行——那会把一条**完整**的记录
    #   看成两条坏的，于是被"修"掉（该记录会被重新生成，白花一次调用）。
    raw = path.read_text(encoding="utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()  # 文件以 `"\n"` 结尾 ⇒ `split` 会多出一个空元素，重写时要还回去

    # ① 末尾的半行：从后往前丢（只丢**解析不了**的）
    keep = len(raw)
    while keep > 0 and raw[keep - 1].strip():
        try:
            json.loads(raw[keep - 1])
            break
        except json.JSONDecodeError:
            keep -= 1
    dropped = len(raw) - keep

    # ② 会被 `splitlines()` 劈开的字符：**就地转义**（语义不变）
    escaped = 0
    fixed: list[str] = []
    for line in raw[:keep]:
        if line.strip() and any(char in line for char in _LINE_BREAKS):
            try:
                line = _jsonl_line(json.loads(line)).rstrip("\n")
                escaped += 1
            except json.JSONDecodeError:  # pragma: no cover —— 上面刚验过能解析
                pass
        fixed.append(line)

    if dropped or escaped:
        path.write_text("".join(f"{line}\n" for line in fixed), encoding="utf-8")
        print(
            f"  ⚠ 修复 `{path.name}`：丢掉末尾 {dropped} 行不完整的 JSON、"
            f"转义 {escaped} 行里的行分隔符字符",
            flush=True,
        )
    return dropped + escaped


def _read_clbench_labels(answers_path: Path, labels_path: Path) -> list[JudgeResult]:
    """读 CL-Bench 的两份产物——**字段名与那两份完全不同**。

    | 文件 | 谁写的 / 读什么 |
    | --- | --- |
    | `answers.jsonl` | `answer` 步写 **`model_output`**（不是 `generated_answer`），
    行键是 **`idx`** |
    | `labels.jsonl` | `evaluate` 步写 `rubric_clbench_score` / `_rationale` /
    `_requirement_status` / `_requirement_ratio` |

    ⚠ **它是严格全有全无**：`score` 只有 0 或 1 ⇒ 映射成 `is_correct = score >= 1.0`，
    并把 `requirement_ratio`（满足了几成要求）一并留进 `judge_response`——**分档信息在
    二值化之后会丢**，而"差一点就满分"与"完全跑偏"是两回事。
    """
    generated = {
        str(row.get("idx", row.get("id", ""))): str(row.get("model_output") or "")
        for row in (
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
    }
    results: list[JudgeResult] = []
    for line in labels_path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = str(row.get("idx", row.get("id", "")))
        score = float(row.get("rubric_clbench_score") or 0.0)
        ratio = float(row.get("rubric_clbench_requirement_ratio") or 0.0)
        results.append(
            JudgeResult(
                qid=ident,
                is_correct=score >= 1.0,
                label="CORRECT" if score >= 1.0 else "WRONG",
                judge_response=(
                    f"[rubric score={score:.0f} ratio={ratio:.2f}] "
                    f"{str(row.get('rubric_clbench_rationale') or '')[:1500]}"
                ),
                generated_answer=generated.get(ident, ""),
                # **部分分**：归档裁判本来就写 `rubric_clbench_requirement_ratio`
                # （满足条数 / 总条数）。二值分把它扔了，`partial_credit` 把它捡回来。
                partial=ratio,
            )
        )
    return results


def _read_beam_labels(answers_path: Path, labels_path: Path) -> list[JudgeResult]:
    """读 BEAM 的判分产物——**它既没有 `is_correct` 也没有 `label`**。

    `pipeline_beam.py` 的 `evaluate` 写的是 `llm_judge_score`（**逐条 rubric 三点制
    0 / 0.5 / 1 的均分**）与逐条 `rubric_scores`。

    ⇒ **二值化口径是我们定的**（数据集与官方 pipeline 都只报均分）：
    **均分 == 1.0 才算 CORRECT**（逐条全中），真分原样进 `label`（`SCORE=0.50`）与
    `judge_response` ⇒ **换阈值不用重跑裁判**——与 `extra_pipeline` 里 MemTrapBench 同一条处置。

    ⚠ **`max_tokens` 从 256 提到 1024**（2026-10-03）：256 会把**裁判自己的回答**截断——
    MemTrapBench 实测 **129/250 条的 JSON 断在中间**（129/129 根括号未闭合），
    而解析失败一律记错 ⇒ **那一半分数是截断的产物，不是模型不会**。
    它同时传给 `answer` 与 `evaluate` 两步（见 `run_judge` 末尾）。


    ⚠ 它的 `evaluate` 以**追加**模式写、按 `llm_judge_score` 跳过已完成（与其余几份
    `"w"` 覆盖不同）⇒ 被强杀会留半行，所以读之前先 `_sanitize_jsonl`。
    """
    generated = {
        str(row["id"]): str(row.get("generated_answer") or "")
        for row in (
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
    }
    _sanitize_jsonl(labels_path)
    results: list[JudgeResult] = []
    for line in labels_path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = str(row["id"])
        score = float(row.get("llm_judge_score") or 0.0)
        results.append(
            JudgeResult(
                qid=ident,
                is_correct=score >= 1.0,
                label=f"SCORE={score:.2f}",
                judge_response=str(row.get("judge_response") or ""),
                generated_answer=generated.get(ident, ""),
            )
        )
    return results


def _recover_beam_labels(
    pipeline: Path,
    items: list[dict],
    answers_path: Path,
    labels_path: Path,
    out_dir: Path,
    endpoint_index: int | None,
) -> None:
    """`pipeline_beam.py evaluate` 失败后的**逐 rubric 恢复**（[`beam_judge`][eval.harness.beam_judge]）。

    上游 batch 裁判一次要模型给**每一个 rubric 索引恰好一个分**；模型偶尔不合规就整批
    `ValueError`。这里改用**逐 rubric 单条模板**重判（每条都要显式有效分），产出与
    `evaluate` **相同的标签行**（`id` / `llm_judge_score` / `judge_response`），覆盖写
    `labels_path` 后交给 `_read_beam_labels` 读。恢复目录可续跑（逐 rubric 落 `-completed.json`）。
    """
    from eval.harness import beam_judge

    generated = {
        str(row["id"]): str(row.get("generated_answer") or "")
        for row in (
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
    }
    # ⚠ **每个 item 一个子目录**：`judge_independently` 的产物名是 `{rubric_index}-completed.json`，
    #   多个 item 共用一个目录会**同名相撞** ⇒ `Saved rubric judgment has different input`。
    recovery_root = out_dir / "beam-recovery"
    labels_path.write_text(
        "".join(
            _jsonl_line(
                beam_judge.judge_independently(
                    pipeline,
                    item,
                    generated.get(str(item["id"]), ""),
                    recovery_root / str(item["id"]).replace("/", "_").replace("\\", "_"),
                    endpoint_index or 0,
                )
            )
            for item in items
        ),
        encoding="utf-8",
    )
