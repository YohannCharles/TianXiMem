#!/usr/bin/env python3
"""兼容旧命令；数据清单、准备和校验已迁入 eval/datasets/（D35）。

新入口：python -m eval.datasets.prepare --dataset locomo-refined
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets.manifest import DELETED, MANIFEST  # noqa: E402,F401
from eval.datasets.prepare import _patch_async_open, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
