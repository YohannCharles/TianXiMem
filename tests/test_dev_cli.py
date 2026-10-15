from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tianximem.dev import _HANDLERS, build_parser, main


def test_parser_exposes_platform_neutral_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["config-check"])
    assert args.command == "config-check"
    assert parser.parse_args(["data-check", "--dataset", "hybridqa"]).dataset == ["hybridqa"]
    assert parser.parse_args(["sync"]).command == "sync"
    assert parser.parse_args(["eval"]).dataset == "locomo-refined"
    assert parser.parse_args(["order-probe"]).processes == 1


def test_config_check_uses_repository_profile(capsys) -> None:
    assert main(["config-check"]) == 0
    output = capsys.readouterr().out
    assert "profile=default" in output
    assert "router.enabled=True" in output
    assert "agentic.mode=plan" in output


def test_module_entrypoint_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tianximem", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "config-check" in result.stdout
    assert "data-check" in result.stdout

    forwarded = subprocess.run(
        [sys.executable, "-m", "tianximem", "help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert forwarded.returncode == 0
    assert forwarded.stdout == result.stdout  # `make help` 转发到的就是这一份


def test_run_directory_defaults_are_dated_in_python() -> None:
    """`ROUTER_OUT` / `COQ_OUT` 的默认落点在 Python 里算（`$(shell date)` 不许回来）。"""

    stamp = datetime.now().strftime("%Y%m%d")
    parser = build_parser()
    assert parser.parse_args(["router-eval"]).out == f"eval/reports/runs/router-{stamp}"
    assert parser.parse_args(["coq-eval"]).out == f"eval/reports/runs/coq-{stamp}"


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    """取出子命令表——argparse 没有公开 API，只有这组私有属性。"""

    action = next(
        candidate
        for candidate in parser._actions
        if isinstance(candidate, argparse._SubParsersAction)
    )
    assert action.choices  # 一个都没认出来 = 这段解析本身坏了，别让空集合静默通过
    return action


def test_every_command_has_help_text() -> None:
    """`help` 是说明的唯一入口 ⇒ 每个子命令都要有一行非空说明、能渲染、且接上了实现。"""

    parser = build_parser()
    subparsers = _subparsers_action(parser)
    assert len(subparsers.choices) >= 20

    assert set(subparsers.choices) == set(_HANDLERS)  # 命令与实现一一对应

    described = {choice.dest: choice.help for choice in subparsers._choices_actions}
    assert set(described) == set(subparsers.choices)
    for name, subparser in subparsers.choices.items():
        assert described[name].strip()
        # 渲染一遍：`help=` 里的坏字符（如裸 `%`）会让 `--help` 直接炸，这里就该先炸
        assert described[name] in subparser.format_help()
    parser.format_help()  # 父级清单要展开每个 `help=`，同样得能渲染


def test_extra_args_are_forwarded(monkeypatch) -> None:
    """`ARGS='…'` 的语义：多余参数**原样下传**给被调用的脚本。"""

    captured: list[list[str]] = []

    def fake_run(command, **kwargs) -> int:
        captured.append(list(command))
        return 0

    monkeypatch.setattr("tianximem.dev._run", fake_run)
    monkeypatch.setattr("tianximem.dev._uv", lambda *args: ["uv", *args])

    assert main(["eval", "--limit", "3"]) == 0
    command = captured[-1]
    assert "eval.experiments.run" in command
    assert command[command.index("--limit") + 1] == "3"

    assert main(["router-eval", "--out", "somewhere"]) == 0
    router_command = captured[-1]
    assert router_command[router_command.index("--out") + 1] == "somewhere"


def test_extra_args_are_rejected_without_passthrough(capsys) -> None:
    """没有下传去处的命令见到多余参数必须**报错**，不许静默丢掉。"""

    with pytest.raises(SystemExit) as exit_info:
        main(["clean", "--bogus"])
    assert exit_info.value.code == 2
    assert "不接受透传参数" in capsys.readouterr().err


#: 每条命令一条最小 argv（只有必填项要带上）。**必须覆盖 `_HANDLERS` 的每一个键**——
#: 少一条就说明新命令没进这张表，而不是"它不用测"。
_COMMAND_ARGV: list[list[str]] = [
    ["help"],
    ["sync"],
    ["lock"],
    ["format"],
    ["format-check"],
    ["lint"],
    ["typecheck"],
    ["config-check"],
    ["test"],
    ["serve"],
    ["check"],
    ["contract-check"],
    ["diagnose", "--run-id", "smoke"],
    ["probe-reranker"],
    ["data-check"],
    ["data-patch"],
    ["data-fetch"],
    ["router-eval"],
    ["coq-eval"],
    ["coq-decompose"],
    ["order-probe"],
    ["export-multihop"],
    ["eval"],
    ["baseline"],
    ["replay-official"],
    ["t1"],
    ["t2"],
    ["t2-dump"],
    ["qdrant-up"],
    ["qdrant-down"],
    ["image-build"],
    ["up"],
    ["down"],
    ["deploy-check"],
    ["smoke"],
    ["clean"],
]


@pytest.mark.parametrize("argv", _COMMAND_ARGV, ids=lambda argv: argv[0])
def test_every_command_reaches_a_child_process(argv, monkeypatch) -> None:
    """逐条走一遍 dispatch：命令名、必填项与 handler 必须对得上（错配就是 AttributeError）。"""

    monkeypatch.setattr("tianximem.dev._run", lambda command, **kwargs: 0)
    monkeypatch.setattr("tianximem.dev._uv", lambda *args: ["uv", *args])
    monkeypatch.setattr("tianximem.dev._clean", lambda: 0)
    # `up` 会先查 deploy/.env；指到一个一定存在的文件，别让它挡在 dispatch 前面
    monkeypatch.setattr("tianximem.dev._DEPLOY_ENV_FILE", Path(__file__))

    assert {command[0] for command in _COMMAND_ARGV} == set(_HANDLERS)
    assert main(argv) == (1 if argv[0] == "smoke" else 0)
