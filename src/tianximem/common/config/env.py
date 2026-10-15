"""`.env` 的读取与环境来源的合成（`TIANXIMEM_ENV_FILE`）。

⚠ 只有 `env is None`（生产路径）读磁盘——那条边界的完整理由在包 docstring。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .constants import DEFAULT_ENV_FILE, ENV_FILE

# ── `.env` 的读取 ──────────────────────────────────────────────────────


def _parse_env_line(raw: str) -> tuple[str, str] | None:
    """把一行 `.env` 解析成 `(key, value)`；空行、注释行与非 `KEY=value` 行返回 `None`。

    支持我们**自己的** `.env.example` 用到的全部写法，不多支持：
    `KEY=value` · `export KEY=value` · 行内 `# 注释` · 引号包起来的值。

    ⚠ **`#` 前必须有空白才算注释**——否则 `KEY=a#b` 会被截成 `a`（这是 dotenv 的通例）。
    ⚠ **坏的行使值缺失 ⇒ 由 `validate()` 响亮失败**（"AML_EMB_API_KEY 没填"），
    所以这里**跳过**而不是报错：`.env` 是给人写的，一个多出来的空行不该让服务起不来。
    """
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, sep, value = line.partition("=")
    if not sep or not key.strip():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return key.strip(), value[1:-1]  # 引号包起来 ⇒ 原样，不剥注释
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            value = value[:index].strip()
            break
    return key.strip(), value


def _read_env_file(path: Path) -> dict[str, str]:
    """读一个 `.env`。**文件不存在 ⇒ 空 dict**（生产上可能直接用真实环境变量）。"""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw)
        if parsed is not None:
            values[parsed[0]] = parsed[1]
    return values


def _env_source(
    env: Mapping[str, str] | None,
    env_file: str | Path | None,
    /,
) -> Mapping[str, str]:
    """把「真实环境变量 + `.env`」合成一份来源映射；`env` 非 None ⇒ **原样返回，不碰磁盘**。"""
    if env is not None:
        return env
    # .env 是"这台机器的环境"的本地副本 ⇒ 排在真实环境变量**之下**，自动让位
    file_path = Path(
        env_file if env_file is not None else (os.environ.get(ENV_FILE) or DEFAULT_ENV_FILE)
    )
    return {**_read_env_file(file_path), **os.environ}
