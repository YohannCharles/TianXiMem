"""`*_pipeline.py` 共用的**两子命令 CLI 形状**——一处实现。

## ⚠ 这个形状是**契约**，不是风格

`judge.run_judge` 按归档那几份的写法，把 `--max-tokens` **在子命令之后**传
（`… answer --input … --output … --max-tokens 256`）⇒ **两个子命令都要收它**。

⚠ 2026-09-30 的教训：一开始把它挂在**顶层** parser 上，而 argparse 的子解析器不认识它
⇒ 整轮以 `unrecognized arguments` **死在第一步**（还没跑到任何一道题），
看起来像别的问题。

⇒ 所以要复用这一份、而不是各写各的：形状改一处就够，**默认值按调用方给的传**
（`extra` / `scriptmem` 用 512、归档重放用 1024——那个差别是实测出来的，不要抹平）。
"""

from __future__ import annotations

import argparse


def build_pipeline_parser(
    description: str,
    *,
    default_max_tokens: int,
    max_tokens_help: str,
    answer_help: str | None = None,
    evaluate_help: str | None = None,
) -> argparse.ArgumentParser:
    """`answer` / `evaluate` 两个子命令，**都**带 `--max-tokens`。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-tokens", type=int, default=default_max_tokens, help=max_tokens_help)

    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="command", required=True)

    answer = sub.add_parser("answer", parents=[common], help=answer_help)
    answer.add_argument("--input", required=True)
    answer.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate", parents=[common], help=evaluate_help)
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--answers", required=True)
    evaluate.add_argument("--output", required=True)

    return parser
