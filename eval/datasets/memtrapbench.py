"""MemTrapBench 的加载层（记忆陷阱题——语料是**真对话**，判分要 LLM）。

## 这份数据长什么样（2026-09-30 实测，`benchmark_data/memtrapbench/`）

```text
memtrapbench/<场景>/<名字>.json   10 个文件，其中 6 个可评测（共 1050 条）
                                  其余 4 个是**种子**（条目只有 `domain` + `id`）⇒ 跳过并报数
  context_history   [{turn, role, content}]  —— 20–40 轮的真对话（user/assistant 交替）
  final_trigger     自包含的新问题（理论上无历史也能读）
  gold_standard     判分要点（自由文本，**不是**答案串）
  expected_failure_output  踩了陷阱会长什么样（红队样本）
  test_type         red / green
  poisoned_fact / objective_truth   safety 那一族特有
```

**本数据集是六份里唯一"记忆就是对话"的**——所以**不用** MQuAKE 那种"一条一个 session"
的约定：直接按原对话投喂，切批照旧的 20 条那一路（§6.5）。

## 判分口径（**我们定的**，官方只给均分）

官方 `runners/` 用 LLM 裁判在 4 个维度上各打 0–5，`summarize_judge_scores.py` 只算**均分**，
**没有过/不过线**。我们的 `JudgeResult` 需要二元 ⇒ `extra_pipeline.py` 里定了一条阈值，
并把**四个原始分**原样存进 `judge_response`——**阈值可以事后重算，不用重跑裁判**。
"""

from __future__ import annotations

import json
from pathlib import Path

from .preprocess import Message, Question, Sample, Session
from .sampling import stratified_sample

__all__ = ["DATA_DIR", "PIPELINE", "SCENARIOS", "USER_PREFIX", "load_memtrapbench"]

DATA_DIR = "memtrapbench"

#: 子目录（仓库把数据集放在 `memtrapbench/memtrapbench/` 下）。
DATA_SUBDIR = "memtrapbench"

#: 本数据集**没有官方 AML pipeline**——这条指向我们自己的实现。
PIPELINE = "extra_pipeline.py"

USER_PREFIX = "mtb-"

#: 场景目录。⚠ **哪些条目是种子不能看文件名**——`trauma/hurt.json` 只装种子却不叫 `_seed`；
#: 判据是**有没有 `context_history`**（见 `load_memtrapbench`）。
SCENARIOS = ("cognitive_bias", "safety", "task_boundary", "trauma")


def _items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name}：顶层不是列表")
    return [item for item in data if isinstance(item, dict)]


def load_memtrapbench(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 MemTrapBench——**一条陷阱题一个 `Sample`**（1 条记忆 : 1 道题）。

    ⚠ 部分跑要加 `spread=True`：文件按场景目录排，否则 `[:limit]` 只取到 `cognitive_bias`。
    """
    root = Path(bench_dir) / DATA_DIR / DATA_SUBDIR
    if not root.is_dir():
        raise FileNotFoundError(f"缺 {root}（`make fetch-data` 取回 official-extra 那一档）")

    records: list[tuple[str, Path, dict]] = []
    seeds = 0
    per_scenario: dict[str, int] = dict.fromkeys(SCENARIOS, 0)
    for scenario in SCENARIOS:
        for path in sorted((root / scenario).glob("*.json")):
            for item in _items(path):
                # ⚠ **判据是内容不是文件名**：种子条目只有 `domain` + `id`
                #   （没有 `context_history`）。文件名靠不住——`trauma/hurt.json` 是个
                #   只装种子的文件，名字里却没有 `_seed`（2026-09-30 冒烟时它就在加载阶段炸了）。
                if not item.get("context_history"):
                    seeds += 1
                    continue
                records.append((scenario, path, item))
                per_scenario[scenario] += 1
    # ⚠ **四个场景一个都不许空**（2026-10-04）。`glob` 对**不存在的目录**返回空迭代器、
    #   **一声不吭** ⇒ 少一个场景目录 = 静默少 150–350 题，而分数看着完全正常、
    #   只是**不可比**（"分数看起来完全正常、却不可比"正是本仓最防的一类）。
    #   上游四个场景各有 350 / 200 / 350 / 150 题，**没有"合法的空场景"**。
    empty = [name for name, count in per_scenario.items() if not count]
    if empty:
        raise FileNotFoundError(
            f"MemTrapBench 这些场景目录里一题都没有：{' / '.join(empty)}（root={root}）"
            "——`make fetch-data` 取回 official-extra 那一档"
        )
    if seeds:
        # 报一声（同 LME 加载器"跳过空轮并说明"的处置）：静默少读会被当成"题目就这么多"。
        print(f"  （MemTrapBench：跳过 {seeds} 条种子——它们没有 context_history，不是题）")
    if limit is not None:
        records = (
            stratified_sample(records, limit, key=lambda r: r[0]) if spread else records[:limit]
        )

    samples: list[Sample] = []
    for scenario, path, item in records:
        trigger = item.get("final_trigger")
        gold = item.get("gold_standard")
        if not trigger or not gold:
            # 到这里说明**它有 `context_history`（是题）却缺题面/判分要点** ⇒ 真坏了，响亮失败。
            raise ValueError(f"{path.name}:{item.get('id')}：final_trigger 或 gold_standard 为空")
        turns = item["context_history"]
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{path.stem}-{item.get('id')}",
                dataset="memtrapbench",
                sessions=(
                    Session(
                        session_id=f"{path.stem}-{item.get('id')}",
                        # ⚠ 这份数据**没有时间戳** ⇒ `event_time` 为 NULL、`created_at` 发空串
                        #   （§11.3 那条有定义的降级路径）。**不是漏读**。
                        messages=tuple(
                            Message(
                                role=str(turn.get("role") or "user"),
                                content=str(turn.get("content") or ""),
                            )
                            for turn in turns
                        ),
                    ),
                ),
                questions=(
                    Question(
                        qid=f"{path.stem}-{item.get('id')}",
                        question=str(trigger),
                        # gold 一整个 dict 交给 `extra_pipeline.py` 的裁判：
                        # 判分要的是 gold_standard，红队样本还要知道期望的失败形态。
                        gold={
                            "gold_standard": gold,
                            "test_type": item.get("test_type"),
                            "expected_failure_output": item.get("expected_failure_output"),
                        },
                        category=scenario,
                    ),
                ),
                speaker_names=("user", "assistant"),
            )
        )
    return samples
