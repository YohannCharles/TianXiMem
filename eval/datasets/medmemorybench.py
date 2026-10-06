"""MedMemoryBench（中文档）的加载层——**检查点式**，记忆只喂到该次问诊为止。

## 这份数据长什么样（2026-09-30 实测，`dataset/medmemorybench/`）

```text
data/zh/dialogues.parquet    31,976 轮 / 2,020 次问诊 / 20 个 persona
                             每轮：persona_id / session_id / turn / role / content /
                                   event_id / event_info{event,type,date} / knowledge_points
data/zh/queries.parquet      1,939 题，6 类；`session_id` **全是 10 的倍数**（10…100）
                             答案在 `answers`（JSON 串）：每条带 content / is_correct / explanation
```

## ⚠ 形状是**照上游评测循环复刻**的（不是我们发明）

上游 `benchmarks/medmemorybench/dataset.py` 的 `_get_independent_units()`
+ `configs/dataset_config/medmemorybench.yaml` 的 `evaluation_interval: 10`：
**每 10 次问诊提问一次**，问的就是"到这次为止"积累的记忆。

⇒ 本加载器：**一个 `Sample` = 一个（persona, 检查点）**，`sessions` 是该 persona
**到该检查点为止的全部问诊**，`questions` 是该检查点的题。

⚠ **为什么不把 101 次问诊一次喂完再问全部题**：那样**读到未来**——`state_update`
那 195 道问的是"**目前**用药状态"，用全量语料问就等于让它看完全程再回答早期时点。
项目对"读未来"的处置有先例（官方流量重放那条：不做 ts 截断 = 指标虚高）。

**代价**：按检查点各起一个 `Sample` ⇒ 投喂量约 **5.4×**（同一段对话会在后续检查点里再喂一次）。
⚠ 这是**故意的**：`embedding` 缓存按内容哈希 ⇒ **重复的正文不重付钱**，只多花时间；
要省时间可以只跑 `--limit 1`（一个检查点 ≈ 10 次问诊）。

## 判分**不在这里**——口径在上游代码里

`dataset/.upstream/medmemorybench/metrics/`（我们另下的 GitHub 仓）：
`entity_exact_match→string_contain`、`multiple_choice→option_match`、
其余四类 → LLM 裁判（`llm_judge` / 多跳那条是 `llm_judge_mcd`）。
⇒ 判分住在 [`../harness/extra_pipeline.py`](../harness/extra_pipeline.py)，**照那份代码实现**。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from .preprocess import Message, Question, Sample, Session
from .sampling import stratified_sample

__all__ = [
    "CODE_DIR",
    "DATA_DIR",
    "PIPELINE",
    "SHAPE_NOTE",
    "USER_PREFIX",
    "load_medmemorybench",
]

DATA_DIR = "medmemorybench"

#: 我们归档的是 **zh 干净档**（`dialogues_with_noise` / `noise_sessions` 没取）。
DATA_SUBDIR = "data/zh"

#: 判分口径的来源（上游 GitHub 仓，另下一份）——**loader 不读它，只是指路**。
CODE_DIR = "medmemorybench-code"

#: 本数据集没有官方 AML pipeline ⇒ 走我们自写的实现。
PIPELINE = "extra_pipeline.py"

USER_PREFIX = "mmb-"

#: 写进**数据指纹**的 `note`——检查点前缀的投喂量是 5.4×，改它等于换数据集。
SHAPE_NOTE = (
    "⚠ **形状是照上游评测循环复刻的**：一个 Sample = 一个（persona, 检查点），"
    "记忆只喂到该检查点为止（上游 `evaluation_interval: 10`）——**不读未来**；"
    "代价是同一段对话在后续检查点会被再喂一次（投喂量 ≈ 5.4×，embedding 有缓存不重付钱）"
)


def _rows(path: Path, columns: list[str] | None = None, batch: int = 2048):
    for chunk in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=columns):
        yield from chunk.to_pylist()


def _json(value, fallback):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def _epoch_ms(day: str | None) -> int | None:
    if not day:
        return None
    try:
        moment = datetime.strptime(str(day)[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    return int(moment.timestamp() * 1000)


def _sessions(turns: list[dict], persona: int, *, up_to: int) -> tuple[Session, ...]:
    """该 persona 到 `up_to` 为止的每一次问诊一个 `Session`（时间戳取 `event_info.date`）。

    ⚠ 日期是**会话级**的（同一次问诊共享一个 `event_info`）⇒ 会话内没有区分度，
    与 LoCoMo / LongMemEval 同一条形（§6.1）。
    """
    grouped: dict[int, list[dict]] = {}
    for row in turns:
        session_id = int(row["session_id"])
        if session_id <= up_to:
            grouped.setdefault(session_id, []).append(row)
    sessions: list[Session] = []
    for session_id in sorted(grouped):
        ordered = sorted(grouped[session_id], key=lambda r: int(r.get("turn") or 0))
        info = _json(ordered[0].get("event_info"), {}) if ordered else {}
        stamp = _epoch_ms(info.get("date"))
        sessions.append(
            Session(
                session_id=f"p{persona}-s{session_id}",
                messages=tuple(
                    Message(
                        role=str(row.get("role") or "user"),
                        content=str(row.get("content") or ""),
                        timestamp_ms=stamp,
                    )
                    for row in ordered
                ),
            )
        )
    return tuple(sessions)


def load_medmemorybench(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 MedMemoryBench（zh）——**一个 `Sample` = 一个（persona, 检查点）**。

    ⚠ 部分跑要加 `spread=True`：sample 按 (persona, 检查点) 排序，`[:limit]` 只会取到
    1 号的头几个检查点。
    """
    root = Path(bench_dir) / DATA_DIR / DATA_SUBDIR
    dialogues = root / "dialogues.parquet"
    queries_path = root / "queries.parquet"
    for path in (dialogues, queries_path):
        if not path.exists():
            raise FileNotFoundError(f"缺 {path}（`make fetch-data` 取回 official-extra 那一档）")

    turns: dict[int, list[dict]] = {}
    for row in _rows(
        dialogues, columns=["persona_id", "session_id", "turn", "role", "content", "event_info"]
    ):
        turns.setdefault(int(row["persona_id"]), []).append(row)

    by_checkpoint: dict[tuple[int, int], list[Question]] = {}
    for row in _rows(queries_path):
        persona = int(row["persona_id"])
        checkpoint = int(row["session_id"])
        metadata = _json(row.get("metadata"), {})
        by_checkpoint.setdefault((persona, checkpoint), []).append(
            Question(
                # ⚠ **必须带 `persona` 前缀**：parquet 里的 `query_id` **只在 persona 内唯一**
                #   （实测 1,939 题只有 **100 个**唯一 `query_id`——20 个 persona × 100 个检查点题，
                #   `(persona_id, query_id)` 才唯一）。这与 `personamem` 的 `pm-{persona}-{index}`
                #   是同一条约定：**id 必须全局唯一**。
                # 不带前缀的后果**不报错**：产物按 `sample.user_id` 分目录 ⇒ 批内恰好不撞，
                #   所以比分不受影响；但**任何跨批按 id 配对的下游都会静默合并**
                #   （诊断工具读整个 run 目录 ⇒ 172 行压成 20 个键）。
                qid=f"mmb-{persona}-{row['query_id']}",
                question=str(row["question"]),
                # 一整个 dict 进 gold：判分要 answers（含 is_correct）**与** query_type
                # 才知道走哪条口径，多跳那一类还要 metadata 里的推理链。
                gold={
                    "query_type": str(row["query_type"]),
                    "answers": _json(row.get("answers"), []),
                    "metadata": metadata,
                },
                category=str(row["query_type"]),
            )
        )

    ordered = sorted(by_checkpoint)
    if limit is not None:
        # ⚠ 抽样键是 **`persona`**（`item[0]`）——`ordered` 的元素已经是 `(persona, checkpoint)`
        #   这个元组本身，再写 `item[0][0]` 是对 int 取下标 ⇒ `TypeError`。
        #   本加载器此前没有 `--spread` 的用例，所以这条路径一直没被跑过。
        # ⚠ **不 `--spread` 时偏得比别的数据集厉害**：`sorted(by_checkpoint)` 先按 persona 排，
        #   所以 `ordered[:limit]` 取到的是**同一个 persona 的最前面几个检查点**
        #   （`--limit 4` ⇒ persona 1 的检查点 10/20/30/40）——“跑一部分”变成“跑一个人”。
        #   而 `--spread` 按 persona 分 **20 组** ⇒ `--limit 4` 实际拿到 **20 个样本**
        #   （`stratified_sample` 每组至少 1 条）。**两个方向都要留意。**
        ordered = (
            stratified_sample(ordered, limit, key=lambda item: item[0])
            if spread
            else ordered[:limit]
        )

    samples: list[Sample] = []
    for persona, checkpoint in ordered:
        sessions = _sessions(turns.get(persona, []), persona, up_to=checkpoint)
        if not sessions:
            raise ValueError(f"persona {persona} 检查点 {checkpoint}：前缀里一个会话都没有")
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{persona:02d}-s{checkpoint:03d}",
                dataset="medmemorybench",
                sessions=sessions,
                questions=tuple(by_checkpoint[(persona, checkpoint)]),
                speaker_names=("患者", "医生"),
            )
        )
    return samples
