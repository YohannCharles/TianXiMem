"""归档 pipeline 要 `import` 的那份适配器——**七个名字的唯一来源**。

## 为什么需要它

归档的 pipeline 都在模块顶部做这两件事
（`dataset/.upstream/aml/pipeline_locomo-refined.py:17`）：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api_config import (ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL,
                        JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_VERSION)
```

D35 后 `__file__` 在 `dataset/.upstream/aml/` 下，`parents[2]` 指向数据根目录。
旧平铺归档时它指向仓库外；两种目录都不能靠那条路径找到配置。

**处置（D12，见 [`CLAUDE.md`](./CLAUDE.md)）**：**不在仓库外创建它**。
本文件就是那一份，放在仓库内；由 harness 在 subprocess 里注入
`PYTHONPATH=<本目录>`。`sys.path.insert(0, <不存在的路径>)` 只是塞进一个没有该模块的
条目，**import 会继续往后找到 `PYTHONPATH` 里的这份**——归档保持只读，
`parents[2]` 那条脆弱路径被绕开，配置只有 `.env` 一份。

## 它只做一件事：把 `AML_*` 翻成归档要的七个名字

**两边命名不同，不是笔误**：归档读 `ANSWER_*` / `JUDGE_*`，我们 `.env` 里是 `AML_*`。

## 对话端点有**多个**，本进程用哪一个是**被指派**的（D39，2026-10-09）

`memory2` 与 `memory3` 提供的是**同一个** `Qwen/Qwen3.5-9B`（128K），但**各有各的 key**
（对调都是 401）⇒ 端点是 `(base_url, api_key)` **成对**的，不能只管 url。

端点表按序号后缀拼：`AML_BASE_URL` / `AML_API_KEY`（主）、`AML_BASE_URL_2` / `AML_API_KEY_2`、
以此类推，缺哪个停哪个。**本进程选第几个**由 `AML_ENDPOINT_INDEX` 给（0-based，越界按长度取模）；
**没给就是 0**——这条很关键：父进程（`run.py` 的前置检查与 `input-manifest.json` 的
`answer_base` / `judge_base`）读到的永远是**主端点**，于是轮转不会让续跑校验以为"输入变了"。

⚠ **为什么是"被指派"而不是"自己挑"**：本模块在**每个** subprocess 里被 import 一次，
而 `judge._subprocess_env()` 传给子进程的是**父进程 `os.environ` 的副本**——
所以序号只能由调用方经环境变量传进来。轮转的派发方是
[`../experiments/run.py`](../experiments/run.py) 的判分线程池，注入方是
[`./judge.py`](./judge.py) 的 `_run()`。

⚠ **`JUDGE_*` 缺省回落到所选端点**：两端点是同一个模型，裁判与答案没有理由分开配
（归档 readme 记的裁判 `Qwen/Qwen3-14B` 我们**没有**；裁判弱于官方是已知代价，
但**全部实验固定同一个裁判，§13 的相对对照仍然成立**）。

⚠ **那七个名字里没有 embedding。** 归档 pipeline 是纯 LLM 的 answer/judge 驱动、
不做任何向量化——Qwen3-Embedding-8B 的配置不经过这里，它只属于 `src/tianximem`。
"""

from __future__ import annotations

import os
from typing import Final

__all__ = [
    "ANSWER_API_BASE",
    "ANSWER_API_KEY",
    "ANSWER_MODEL",
    "ENDPOINTS",
    "ENDPOINT_INDEX",
    "ENV_ENDPOINT_INDEX",
    "JUDGE_API_BASE",
    "JUDGE_API_KEY",
    "JUDGE_MODEL",
    "JUDGE_VERSION",
]

#: 由调用方注入的"本进程用第几个端点"（见模块 docstring 的 D39 一节）。
#: ⚠ **不要放进 `.env`**：它描述的是"这一次调度"，不是这台机器的配置。
ENV_ENDPOINT_INDEX: Final[str] = "AML_ENDPOINT_INDEX"

#: 端点序号后缀。主端点没有后缀。**上限是显式的**——写死四个比"扫到 99"少一次
#: 静默拼错的机会（多配的端点没人读，本该是一次配置错误）。
_SUFFIXES: Final[tuple[str, ...]] = ("", "_2", "_3", "_4")


def _endpoint_pairs() -> tuple[tuple[str, str], ...]:
    """`((base_url, api_key), ...)`，顺序 = 后缀顺序。

    ⚠ **url 与 key 成对取**：两个对话端点各有一把 key，混用是 **401**，
    而 401 只会在第一次判分时出现（那时整轮的 embedding 与检索都已经付出去了）。
    """
    pairs: list[tuple[str, str]] = []
    for suffix in _SUFFIXES:
        base = (os.environ.get(f"AML_BASE_URL{suffix}") or "").strip()
        if not base:
            continue
        key = (os.environ.get(f"AML_API_KEY{suffix}") or "").strip()
        if not key:
            # ⚠ **响亮失败，而且要在这一层**（不是等轮转到它）：少一把 key 的端点会在
            #   第一次打到它的时候 401，而那时整轮的 Add / Search 与前面几题的判分
            #   都已经付出去了——这正是 `judge_preconditions()` 要避免的那类晚失败。
            #   端点表本身就短（≤4），所以这里检查的代价可以忽略。
            raise ValueError(
                f"配了 `AML_BASE_URL{suffix}` 却没配 `AML_API_KEY{suffix}`——"
                "两个对话端点各有各的 key，混用/漏填只会拿到 401（见 .env.example）"
            )
        pairs.append((base, key))
    # 一个都没配 ⇒ 留一个空端点：形状不变，失败与从前一模一样（`_chat` 会响亮报"端点未配置"）。
    return tuple(pairs) or (("", ""),)


def _selected_index(count: int) -> int:
    raw = (os.environ.get(ENV_ENDPOINT_INDEX) or "").strip()
    if not raw:
        return 0
    try:
        index = int(raw)
    except ValueError as exc:
        # ⚠ 响亮失败，不静默退回 0：那会让"我以为它在打 memory3"变成一个查不出来的问题。
        raise ValueError(
            f"{ENV_ENDPOINT_INDEX} 必须是整数（第几个端点，0-based），收到 {raw!r}"
        ) from exc
    return index % count


#: 全部对话端点，顺序稳定（父进程与子进程读到的是同一份）。派发方按它的长度决定并行度。
ENDPOINTS: Final[tuple[tuple[str, str], ...]] = _endpoint_pairs()

#: 本进程选了第几个（0 = 主端点）。
ENDPOINT_INDEX: Final[int] = _selected_index(len(ENDPOINTS))

_SELECTED_BASE, _SELECTED_KEY = ENDPOINTS[ENDPOINT_INDEX]

#: 答案模型端点。**两个对话网关之一**（`memory2` / `memory3`），见模块 docstring。
ANSWER_API_BASE = _SELECTED_BASE
ANSWER_API_KEY = _SELECTED_KEY
ANSWER_MODEL = os.environ.get("AML_MODEL", "")

#: 裁判端点。留空则回落本进程所选的那个答案端点（同一个模型，见模块 docstring）。
JUDGE_API_BASE = os.environ.get("AML_JUDGE_BASE_URL") or _SELECTED_BASE
JUDGE_API_KEY = os.environ.get("AML_JUDGE_API_KEY") or _SELECTED_KEY
JUDGE_MODEL = os.environ.get("AML_JUDGE_MODEL") or ANSWER_MODEL

#: 七个 pipeline 全部 `import` 它、**没有一处使用它**。保留只为 import 不失败。
JUDGE_VERSION = os.environ.get("AML_JUDGE_VERSION", "")
