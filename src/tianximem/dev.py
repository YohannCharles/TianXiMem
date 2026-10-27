"""Cross-platform TianXiMem runtime commands."""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY_ENV_FILE = _REPO_ROOT / "deploy" / ".env"


def _run(command: Sequence[str], *, cwd: Path = _REPO_ROOT) -> int:
    try:
        completed = subprocess.run(list(command), cwd=cwd, check=False)
    except FileNotFoundError:
        print(f"Command not found: {command[0]}", file=sys.stderr)
        return 127
    return completed.returncode


def _python(*args: str) -> list[str]:
    return [sys.executable, *args]


def _uv(*args: str) -> list[str]:
    executable = shutil.which("uv")
    if executable is None:
        raise SystemExit("uv is required and must be on PATH")
    return [executable, *args]


def _compose(*args: str) -> list[str]:
    return ["docker", "compose", "-f", "deploy/compose.yaml", *args]


def _config_check() -> int:
    from tianximem.common.config import (
        ENV_EMBED_API_KEY,
        ENV_EMBED_BASE_URL,
        ENV_LLM_API_KEY,
        ENV_LLM_BASE_URL,
        ENV_LLM_MODEL,
        ENV_PROFILE,
        load_config,
    )

    env = {
        ENV_EMBED_BASE_URL: "http://config-check.invalid/v1",
        ENV_EMBED_API_KEY: "config-check",
        ENV_LLM_BASE_URL: "http://config-check.invalid/v1",
        ENV_LLM_API_KEY: "config-check",
        ENV_LLM_MODEL: "config-check",
    }
    profiles = sorted(path.stem for path in (_REPO_ROOT / "configs").glob("*.yaml"))
    if not profiles:
        raise SystemExit("No YAML profiles found under configs/")
    for profile in profiles:
        config = load_config(
            {**env, ENV_PROFILE: profile},
            config_dir=_REPO_ROOT / "configs",
        )
        print(
            f"{profile}: router={config.router.enabled} "
            f"agentic={config.agentic.enabled}/{config.agentic.mode} "
            f"grounded={config.retrieval.grounded_evidence} "
            f"collection={config.storage.qdrant.collection}"
        )
    return 0


def _clean() -> int:
    targets = (
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "var/*.db",
        "var/*.db-journal",
        "var/*.db-wal",
        "var/*.db-shm",
    )
    for pattern in targets:
        for raw in glob.glob(str(_REPO_ROOT / pattern)):
            path = Path(raw)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    return 0


def _cmd_help(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    build_parser().print_help()
    return 0


def _cmd_sync(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(_uv("sync"))


def _cmd_lock(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(_uv("lock"))


def _cmd_format(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(_python("-m", "ruff", "format", "src"))


def _cmd_format_check(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(_python("-m", "ruff", "format", "--check", "src"))


def _cmd_lint(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(_python("-m", "ruff", "check", "src"))


def _cmd_typecheck(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(_python("-m", "mypy", "src"))


def _cmd_serve(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _run(
        _python(
            "-m",
            "uvicorn",
            "tianximem.service.app:create_app_from_env",
            "--factory",
            "--workers",
            "1",
        )
    )


def _cmd_config_check(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _config_check()


def _cmd_qdrant_up(args: argparse.Namespace, extra: list[str]) -> int:
    del args
    return _run(_compose("up", "-d", "qdrant", *extra))


def _cmd_qdrant_down(args: argparse.Namespace, extra: list[str]) -> int:
    del args
    return _run(_compose("stop", "qdrant", *extra))


def _cmd_image_build(args: argparse.Namespace, extra: list[str]) -> int:
    del args
    return _run(
        ["docker", "build", "-f", "deploy/Dockerfile", "-t", "tianximem:local", ".", *extra]
    )


def _cmd_up(args: argparse.Namespace, extra: list[str]) -> int:
    del args
    if not _DEPLOY_ENV_FILE.is_file():
        print("Missing deploy/.env; copy deploy/.env.example and fill in values.", file=sys.stderr)
        return 1
    return _run(_compose("up", "-d", "--build", *extra))


def _cmd_down(args: argparse.Namespace, extra: list[str]) -> int:
    del args
    return _run(_compose("down", *extra))


def _cmd_clean(args: argparse.Namespace, extra: list[str]) -> int:
    del args, extra
    return _clean()


_HANDLERS: dict[str, Callable[[argparse.Namespace, list[str]], int]] = {
    "help": _cmd_help,
    "sync": _cmd_sync,
    "lock": _cmd_lock,
    "format": _cmd_format,
    "format-check": _cmd_format_check,
    "lint": _cmd_lint,
    "typecheck": _cmd_typecheck,
    "serve": _cmd_serve,
    "config-check": _cmd_config_check,
    "qdrant-up": _cmd_qdrant_up,
    "qdrant-down": _cmd_qdrant_down,
    "image-build": _cmd_image_build,
    "up": _cmd_up,
    "down": _cmd_down,
    "clean": _cmd_clean,
}

_PASSTHROUGH_COMMANDS = frozenset(
    {
        "qdrant-up",
        "qdrant-down",
        "image-build",
        "up",
        "down",
    }
)


def _add_command(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(name, help=help_text, description=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tianximem.dev",
        description="TianXiMem runtime commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_command(subparsers, "help", "List commands")
    _add_command(subparsers, "sync", "Install the locked runtime dependencies")
    _add_command(subparsers, "lock", "Regenerate uv.lock")
    _add_command(subparsers, "format", "Format service source")
    _add_command(subparsers, "format-check", "Check service source formatting")
    _add_command(subparsers, "lint", "Run Ruff")
    _add_command(subparsers, "typecheck", "Run mypy")
    _add_command(subparsers, "serve", "Start the single-process API server")
    _add_command(subparsers, "config-check", "Validate every YAML profile")
    _add_command(subparsers, "qdrant-up", "Start Qdrant")
    _add_command(subparsers, "qdrant-down", "Stop Qdrant")
    _add_command(subparsers, "image-build", "Build the application image")
    _add_command(subparsers, "up", "Start the application and Qdrant")
    _add_command(subparsers, "down", "Stop the application stack")
    _add_command(subparsers, "clean", "Remove local cache files and the SQLite database")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in _PASSTHROUGH_COMMANDS:
        parser.error(f"`{args.command}` does not accept extra arguments: {' '.join(extra)}")
    return _HANDLERS[args.command](args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
