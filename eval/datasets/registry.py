"""数据指纹与**数据路径的唯一入口**。

## 路径口径（D16）

**数据路径一律经 `TIANXIMEM_BENCHMARK_DIR` 读取**——代码中不得硬编码 `benchmark_data/`
或 `eval/datasets/`。默认值是 `benchmark_data`（只读归档）；本地开发用 `.env` 指到
开发数据集即可。全仓只有 [`benchmark_dir()`][eval.datasets.registry.benchmark_dir] 一处
读这个变量，其余模块一律接 `Path` 参数——**这样测试才能指向临时目录**。

## 指纹为什么要记（§13）

§13 的记录格式要求 **数据指纹 = 数据集 + 版本 + 切批口径**。三样缺一不可：

| 项 | 不记的后果 |
| --- | --- |
| 文件内容（sha256） | 归档被换掉/重取后，两次 run 的数字被当成可比的 |
| **切批口径** | 本地只有"20 条消息"那一路（词数那一路复现不了，§6.5） |
| 实际用了几题 | `load_longmemeval(limit=...)` 这种调试截断会伪装成"全量跑完" |

**切批口径不写下来，就会有人以为线上也是 20 条。**

**它是常量、不是旋钮**：本地能复现的只有"≤20 条消息"这一路，
所以这里只有这一个值可记。§17.1 的 S2 清掉之前，线上真实口径**未知**，
——记成 `"20 messages (local repro only)"` 而不是 `"20 messages"`，差别就在这。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .beam import PARQUET as BEAM_PARQUET
from .clbench import CLBENCH_JSONL
from .corporatebench import DATA_DIR as CORP_DATA_DIR
from .corporatebench import KB_FILE as CORP_KB_FILE
from .corporatebench import QA_TYPES as CORP_QA_TYPES
from .corporatebench import SHAPE_NOTE as CORP_SHAPE_NOTE
from .locomo import QUESTIONS_JSONL, conversation_file
from .longmemeval import LME_JSON
from .medmemorybench import DATA_DIR as MMB_DATA_DIR
from .medmemorybench import DATA_SUBDIR as MMB_DATA_SUBDIR
from .medmemorybench import SHAPE_NOTE as MMB_SHAPE_NOTE
from .memtrapbench import DATA_DIR as MTB_DATA_DIR
from .mquake import DATA_DIR as MQUAKE_DATA_DIR
from .mquake import FILES as MQUAKE_FILES
from .mquake import SHAPE_NOTE as MQUAKE_SHAPE_NOTE
from .personamem import BENCHMARK_CSV as PM_CSV
from .personamem import DATA_DIR as PM_DATA_DIR
from .tempreason import DATA_DIR as TR_DATA_DIR
from .tempreason import FILES as TR_FILES

__all__ = ["benchmark_dir", "data_fingerprint", "file_fingerprint", "shape_note", "BATCHING_LOCAL"]

#: 本地唯一能复现的切批口径（§6.5：词数那一路复现不了，"Adapter 计的词"从未定义）。
BATCHING_LOCAL = "20 messages (local repro only; word-count path unreproducible)"

_HASH_CHUNK = 1 << 20


def benchmark_dir(env: dict[str, str] | None = None) -> Path:
    """`TIANXIMEM_BENCHMARK_DIR`——**本仓唯一的读取点**（D16）。"""
    source = os.environ if env is None else env
    return Path(source.get("TIANXIMEM_BENCHMARK_DIR") or "benchmark_data")


#: 形状是**我们造的**那几个数据集（不是数据集事实）——它们的约定要进数据指纹的 `note`。
_SHAPE_NOTES = {
    "mquake-remastered": MQUAKE_SHAPE_NOTE,
    "corporatebench": CORP_SHAPE_NOTE,
    "medmemorybench": MMB_SHAPE_NOTE,
}


def shape_note(dataset: str) -> str:
    """该数据集"形状是我们造的"那部分；没有就说空串。

    ⚠ 与 [`BATCHING_LOCAL`][eval.datasets.registry.BATCHING_LOCAL] 同一性质：
    **它是常量，不是旋钮**——改它会改分数，而两次 run 的 record 否则一模一样。
    """
    return _SHAPE_NOTES.get(dataset, "")


def file_fingerprint(path: Path) -> dict:
    """逐字节指纹。分块读——LME 那份是 277 MB，一次性读进来只为算哈希不划算。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()[:16]}


def _sources(bench_dir: Path, dataset: str) -> list[Path]:
    if dataset == "locomo-refined":
        return [bench_dir / QUESTIONS_JSONL, conversation_file(bench_dir)]
    if dataset == "longmemeval-s":
        return [bench_dir / LME_JSON]
    if dataset == "clbench":
        return [bench_dir / CLBENCH_JSONL]
    if dataset == "tempreason":
        return [bench_dir / TR_DATA_DIR / name for name in TR_FILES]
    if dataset == "personamem-v2":
        # 语料是 200 份 chat_history——**逐个进指纹太长**，这里只钉 CSV（题与答案都在里面）+
        # `n_samples` 一起构成"用哪一档"的记录。
        return [bench_dir / PM_DATA_DIR / PM_CSV]
    if dataset == "medmemorybench":
        root = bench_dir / MMB_DATA_DIR / MMB_DATA_SUBDIR
        return [root / "dialogues.parquet", root / "queries.parquet"]
    if dataset == "beam":
        return [bench_dir / BEAM_PARQUET]
    if dataset == "mquake-remastered":
        # 四份 parquet 一起进指纹——它们的 `case_id` 各自从 1 开始，少一份就换了题库。
        root = bench_dir / MQUAKE_DATA_DIR / "data"
        return [root / f"{stem}-00000-of-00001.parquet" for stem in MQUAKE_FILES]
    if dataset == "memtrapbench":
        root = bench_dir / MTB_DATA_DIR / "memtrapbench"
        return sorted(p for p in root.rglob("*.json") if not p.name.endswith("_seed.json"))
    if dataset == "corporatebench":
        root = bench_dir / CORP_DATA_DIR
        return [
            root / CORP_KB_FILE,
            *(root / "data" / qa / "zenith_questions.json" for qa in CORP_QA_TYPES),
        ]
    raise ValueError(f"未知数据集 {dataset!r}——没有登记源文件")


def data_fingerprint(
    bench_dir: Path,
    dataset: str,
    *,
    n_samples: int,
    n_questions: int,
    add_shape: str = "",
    note: str = "",
) -> dict:
    """写进 run record 的数据指纹（§13）。

    `n_samples` / `n_questions` 由调用方传入——它们是**加载后**的事实，
    而指纹只读文件字节；两者若对不上（例如 `limit=` 截断），`note` 里会说清楚。

    `add_shape` 由 runner 传入（[`../harness/add_shape.py`](../harness/add_shape.py)）——
    **本模块不 import harness**：加载层不该知道"谁在用 HTTP 发这些消息"。它空着就是
    "这一轮没声明形态"，与切批那种"常量"不同类，所以不加缺省值。
    """
    return {
        "dataset": dataset,
        "benchmark_dir": str(bench_dir),
        "files": [file_fingerprint(p) for p in _sources(bench_dir, dataset)],
        "n_samples": n_samples,
        "n_questions": n_questions,
        "batching": BATCHING_LOCAL,
        "add_shape": add_shape,
        "note": note,
    }
