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

from .clbench import CLBENCH_JSONL
from .locomo import QUESTIONS_JSONL, conversation_file
from .longmemeval import LME_JSON

__all__ = ["benchmark_dir", "data_fingerprint", "file_fingerprint", "BATCHING_LOCAL"]

#: 本地唯一能复现的切批口径（§6.5：词数那一路复现不了，"Adapter 计的词"从未定义）。
BATCHING_LOCAL = "20 messages (local repro only; word-count path unreproducible)"

_HASH_CHUNK = 1 << 20


def benchmark_dir(env: dict[str, str] | None = None) -> Path:
    """`TIANXIMEM_BENCHMARK_DIR`——**本仓唯一的读取点**（D16）。"""
    source = os.environ if env is None else env
    return Path(source.get("TIANXIMEM_BENCHMARK_DIR") or "benchmark_data")


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
    raise ValueError(f"未知数据集 {dataset!r}——没有登记源文件")


def data_fingerprint(
    bench_dir: Path, dataset: str, *, n_samples: int, n_questions: int, note: str = ""
) -> dict:
    """写进 run record 的数据指纹（§13）。

    `n_samples` / `n_questions` 由调用方传入——它们是**加载后**的事实，
    而指纹只读文件字节；两者若对不上（例如 `limit=` 截断），`note` 里会说清楚。
    """
    return {
        "dataset": dataset,
        "benchmark_dir": str(bench_dir),
        "files": [file_fingerprint(p) for p in _sources(bench_dir, dataset)],
        "n_samples": n_samples,
        "n_questions": n_questions,
        "batching": BATCHING_LOCAL,
        "note": note,
    }
