"""归档 pipeline 要 `import` 的那份适配器——**七个名字的唯一来源**。

## 为什么需要它

归档的 7 个 pipeline 都在模块顶部做这两件事（`benchmark_data/pipeline_locomo-refined.py:17`）：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api_config import (ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL,
                        JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_VERSION)
```

`__file__` 在 `benchmark_data/` 下 ⇒ `parents[2]` 解析到**仓库外一层**——那份 `api_config`
不存在，7 个 pipeline 全部 import 失败。

**处置（D12，见 [`CLAUDE.md`](./CLAUDE.md)）**：**不在仓库外创建它**。
本文件就是那一份，放在仓库内；由 harness 在 subprocess 里注入
`PYTHONPATH=<本目录>`。`sys.path.insert(0, <不存在的路径>)` 只是塞进一个没有该模块的
条目，**import 会继续往后找到 `PYTHONPATH` 里的这份**——归档保持只读，
`parents[2]` 那条脆弱路径被绕开，配置只有 `.env` 一份。

## 它只做一件事：把 `AML_*` 翻成归档要的七个名字

**两边命名不同，不是笔误**：归档读 `ANSWER_*` / `JUDGE_*`，我们 `.env` 里是 `AML_*`。

⚠ **`JUDGE_*` 缺省回落到 `ANSWER_*`**——网关只有一个对话模型（`Qwen/Qwen3.5-9B`），
归档 readme 记的裁判模型 `Qwen/Qwen3-14B` 我们**没有**。裁判弱于官方是已知代价，
但**全部实验固定同一个裁判，§13 的相对对照仍然成立**（`.env.example` 已记）。

⚠ **那七个名字里没有 embedding。** 归档 pipeline 是纯 LLM 的 answer/judge 驱动、
不做任何向量化——Qwen3-Embedding-8B 的配置不经过这里，它只属于 `src/tianxi_am`。
"""

from __future__ import annotations

import os

__all__ = [
    "ANSWER_API_BASE",
    "ANSWER_API_KEY",
    "ANSWER_MODEL",
    "JUDGE_API_BASE",
    "JUDGE_API_KEY",
    "JUDGE_MODEL",
    "JUDGE_VERSION",
]

_ANSWER_BASE = os.environ.get("AML_BASE_URL", "")
_ANSWER_KEY = os.environ.get("AML_API_KEY", "")

#: 答案模型端点。**必须是 memory2**（对话网关）——主网关只有 embedding + reranker。
ANSWER_API_BASE = _ANSWER_BASE
ANSWER_API_KEY = _ANSWER_KEY
ANSWER_MODEL = os.environ.get("AML_MODEL", "")

#: 裁判端点。留空则回落答案端点（只有一个模型可用，见模块 docstring）。
JUDGE_API_BASE = os.environ.get("AML_JUDGE_BASE_URL") or _ANSWER_BASE
JUDGE_API_KEY = os.environ.get("AML_JUDGE_API_KEY") or _ANSWER_KEY
JUDGE_MODEL = os.environ.get("AML_JUDGE_MODEL") or ANSWER_MODEL

#: 七个 pipeline 全部 `import` 它、**没有一处使用它**。保留只为 import 不失败。
JUDGE_VERSION = os.environ.get("AML_JUDGE_VERSION", "")
