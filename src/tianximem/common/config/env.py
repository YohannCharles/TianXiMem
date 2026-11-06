from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .constants import DEFAULT_ENV_FILE, ENV_FILE


def _parse_env_line(raw: str) -> tuple[str, str] | None:
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
        return key.strip(), value[1:-1]
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            value = value[:index].strip()
            break
    return key.strip(), value


def _read_env_file(path: Path) -> dict[str, str]:
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
    if env is not None:
        return env
    file_path = Path(
        env_file if env_file is not None else (os.environ.get(ENV_FILE) or DEFAULT_ENV_FILE)
    )
    return {**_read_env_file(file_path), **os.environ}
