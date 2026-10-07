#!/usr/bin/env python3
"""PersonaMem-v2：用检索记忆构造历史，复用官方 MCQ 选项与判分。

## 为什么需要适配器

归档的 `pipeline_v2_personamem.py` 的 CLI 与通用形状不同：

```bash
python pipeline_v2_personamem.py answer --input … --output … --mode mcq|generative
python pipeline_v2_personamem.py evaluate-mcq --answers … --output …     # ⚠ 不吃 --input
```

`run_judge` 调的是 `answer → evaluate`（带 `--input/--answers/--output`）⇒ 中间放一层薄适配器。
**判分逻辑一行都不重写**：直接把官方那两个函数拿过来调。

```python
official.official_mcq_messages(item)  # 历史由 Search 结果替换
official.evaluate_mcq(namespace)
```

⚠ 与 ScriptMem 那条适配器同一条注意事项：官方模块**模块级**就 `from api_config import (...)`，
运行期靠 `run_judge` 注入的 `PYTHONPATH` 找到它。

## 本地检索评测口径

归档原版只读 `chat_history`；本适配器将其替换成逐题 Search 返回的片段。
空检索保留为空，不回填原始对话；缺检索字段直接报错。选项与 MCQ 判分复用归档。
这改变了答案输入口径，旧完整历史成绩不能直接比较，旧答案必须换 run-id 重跑。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from eval.datasets.registry import benchmark_dir  # noqa: E402
from eval.jsonl_io import read_jsonl, write_line  # noqa: E402

MEMORY_INPUT_VERSION = "personamem-search-memory-v1"


def retrieved_history(item: dict) -> list[dict[str, str]]:
    """只用检索字段生成官方认的历史；显式空结果不得回退到原始历史。"""
    context = item.get("retrieved_context", item.get("speaker_1_memories"))
    if context is None:
        raise ValueError("PersonaMem-v2 缺少检索记忆字段；不能使用原始 chat_history 代替")
    if not isinstance(context, str):
        raise TypeError("PersonaMem-v2 检索记忆必须是字符串（空检索用空串）")
    return [
        {
            "role": "user",
            "content": "Retrieved memories (in retrieval order):\n" + (context or "(no memories)"),
        }
    ]


def _input_fingerprint(item: dict) -> str:
    values = {
        k: item.get(k)
        for k in ("question", "user_query", "persona_id", "correct_answer", "incorrect_answers")
    }
    values["chat_history"] = retrieved_history(item)
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _official():
    from eval.datasets.layout import upstream_aml_dir

    path = str(upstream_aml_dir(benchmark_dir()))
    if path not in sys.path:
        sys.path.insert(0, path)
    import pipeline_v2_personamem  # type: ignore[import-not-found]

    return pipeline_v2_personamem


def _fold_trailing_system(messages: list[dict]) -> list[dict]:
    """把**非开头**的 system 消息折进**前一条**消息（只换 `role`，文字一字不改）。

    ⚠ 为什么需要这一步：官方 `official_mcq_messages()` 把选项说明作为**第二条 system**
    **追加在 user 提问之后**，而本项目的网关要求 *"System message must be at the beginning."*
    ——2026-09-30 实测：原样发 ⇒ **HTTP 400**，去掉 system ⇒ 200。
    ⇒ 折叠**保住了顺序与文字**（选项仍在问题之后），只把 role 从 `system` 换成 `user`。
    两臂（任何 A/B）用的是同一份折叠，所以相对比较不受影响。
    """
    folded: list[dict] = []
    for message in messages:
        if message.get("role") == "system" and folded:
            previous = dict(folded[-1])
            previous["content"] = f"{previous['content']}\n\n{message['content']}"
            folded[-1] = previous
        else:
            folded.append(dict(message))
    return folded


def build_mcq_messages(item: dict) -> tuple[list[dict], dict[str, str], str]:
    """保留官方选项与提问，替换历史输入后折叠尾部 system。"""
    memory_item = {**item, "chat_history": retrieved_history(item)}
    messages, mapping, correct_letter = _official().official_mcq_messages(memory_item)
    return _fold_trailing_system(messages), mapping, correct_letter


def cmd_answer(args: argparse.Namespace) -> int:
    """Search 片段 → 官方 MCQ 消息 → 模型选字母；拒绝复用旧口径答案。"""
    # 官方的选项构造与判分保持不变；答案消息改用逐题检索历史。
    from pathlib import Path as _Path

    from eval.harness.api_config import ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL

    output = _Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    items = read_jsonl(Path(args.input))
    fingerprints = {
        str(item.get("id", i)): _input_fingerprint(item) for i, item in enumerate(items)
    }
    done = set()
    for row in read_jsonl(output):
        ident = str(row.get("id"))
        if row.get("memory_input_version") != MEMORY_INPUT_VERSION or row.get(
            "input_fingerprint"
        ) != fingerprints.get(ident):
            raise ValueError("PersonaMem-v2 已有答案口径或输入不一致；请使用新 run-id 重跑")
        done.add(ident)

    async def run() -> None:
        import httpx

        with output.open("a", encoding="utf-8") as handle:
            async with httpx.AsyncClient(timeout=120.0) as client:
                for index, item in enumerate(items):
                    ident = str(item.get("id", index))
                    if ident in done:
                        continue
                    messages, mapping, correct_letter = build_mcq_messages(item)
                    response = await client.post(
                        f"{ANSWER_API_BASE.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {ANSWER_API_KEY}"},
                        json={
                            "model": ANSWER_MODEL,
                            "messages": messages,
                            "temperature": 0,
                            "max_tokens": 512,
                        },
                    )
                    response.raise_for_status()
                    generated = response.json()["choices"][0]["message"]["content"]
                    write_line(
                        handle,
                        {
                            "id": ident,
                            "mode": "mcq",
                            "generated_answer": generated,
                            "option_mapping": mapping,
                            "correct_letter": correct_letter,
                            "prompt_source": "PersonaMem-v2 inference.py"
                            "（Search 记忆替代历史；尾部 system 已折叠）",
                            "memory_input_version": MEMORY_INPUT_VERSION,
                            "input_fingerprint": fingerprints[ident],
                            "model": ANSWER_MODEL,
                        },
                    )

    asyncio.run(run())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """**直接调官方的 `evaluate_mcq`**，再把它的结果翻成 harness 认的 labels 形状。

    官方那一行的字段是 `id / predicted_letter / gold_letter / predicted_answer /
    gold_answer / is_correct`；这里只做搬运 + 加一个能读的 `label`。
    """
    official = _official()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_name("_personamem_official.jsonl")
    official.evaluate_mcq(SimpleNamespace(answers=args.answers, output=str(scratch)))

    with output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(scratch):
            ok = bool(row.get("is_correct"))
            write_line(
                handle,
                {
                    "id": str(row["id"]),
                    "is_correct": ok,
                    "label": f"letter={row.get('predicted_letter')}/{row.get('gold_letter')}",
                    "judge_response": (
                        f"PersonaMem-v2/mcq："
                        f"predicted={str(row.get('predicted_answer'))[:120]!r} "
                        f"gold={str(row.get('gold_answer'))[:120]!r}"
                    ),
                },
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-tokens", type=int, default=512, help="**收下但不用**——形状对齐用")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    answer = sub.add_parser("answer", parents=[common])
    answer.add_argument("--input", required=True)
    answer.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--answers", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return cmd_answer(args) if args.command == "answer" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
