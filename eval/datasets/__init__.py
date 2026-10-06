"""评测数据加载层（PRD §12.2 / §12.3，D16 / D35）。

导出按需加载：仅下载或校验数据不需要先安装 pyarrow 等评测依赖。
loader/import 不联网；评测 CLI 在执行前显式调用 prepare.ensure_dataset。
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BATCHING_LOCAL": "registry",
    "Message": "preprocess",
    "Question": "preprocess",
    "Sample": "preprocess",
    "Session": "preprocess",
    "benchmark_dir": "registry",
    "data_fingerprint": "registry",
    "file_fingerprint": "registry",
    "load_beam": "beam",
    "load_clbench": "clbench",
    "load_corporatebench": "corporatebench",
    "load_feverous": "feverous",
    "load_halumem": "halumem",
    "load_hybridqa": "hybridqa",
    "load_locomo": "locomo",
    "load_longmemeval": "longmemeval",
    "load_medmemorybench": "medmemorybench",
    "load_memtrapbench": "memtrapbench",
    "load_mquake": "mquake",
    "load_musique": "musique",
    "load_personamem": "personamem",
    "load_tempreason": "tempreason",
    "shape_note": "registry",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
