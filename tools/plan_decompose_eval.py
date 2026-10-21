"""Plan 模式的**拆解**单独评测——**只跑拆解那一次调用**，不检索、不答题、不判分。

```bash
uv run --env-file .env python tools/plan_decompose_eval.py \
    --manifest eval/coq-musique.jsonl \
    --out eval/reports/runs/plan-decompose-
```

## 它回答什么

**"这份拆解提示词拆出来的计划有多长？"**——拿它与 MuSiQue 的**真值跳数**
（`question_decomposition` 的长度）比。⚠ 真值**不进提示词**，只在事后比对。

它补的是"整轮跑批看不见的那一层"：`PlanAgent` 里唯一决定"能走几跳"的就是这一次调用，
而整轮跑批里它被检索、答案、判分三层盖住（50 题要 12 分钟检索 ＋ 判分）。
**把检索整个拿掉之后，这一件事 3 分钟就能量。**

## 三条纪律

1. **用的是线上那一份提示词**（`agent/plan_prompts.py` 的 `PLAN_DECOMPOSE_PROMPT`）
   与**线上那一个解析器**（`parse_plan`）——
   ⛔ 不在这里另写一份，否则量的是源码之外的东西。
2. **版本不匹配就拒绝续跑**（`run.json` 里记着提示词版本与模型）：换了提示词还去续跑，
   会把两版的记录**混进同一张表**，而那就是一份看不出来的假结果。
3. ⛔ **不拿真值改提示词**：`--compare` 只读 `dataset/` 的金标用来**事后**算差。

## 产物（都在 `--out` 下，含数据集原文 ⇒ `runs/` 是 gitignored）

| 文件 | 是什么 |
| --- | --- |
| `run.json` | 指纹：提示词版本、模型、`max_tokens`、清单、题数 |
| `decompositions.jsonl` | 逐题一行（题号 / 真值跳数 / 计划 / 原始输出 / 耗时） |
| `decompositions.md` | 给人读的那一份（逐题计划 ＋ 汇总表） |
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# ⚠ **仓库根要上 `sys.path`**：`eval.*` 是仓库里的一个包，而 `uv run python tools/x.py`
#   的 `sys.path[0]` 是 `tools/`。同 [`coq_eval.py`](./coq_eval.py)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠ 清单与选题逻辑**从 `coq_eval` 复用**：两份清单读法必然漂，
#   而"跑的是不是同一批题"是这两份结果能不能并排看的前提。
from tools.coq_eval import DATASETS, _load_manifest, _select  # noqa: E402

from tianximem.agent.plan import render_plan_decompose_prompt  # noqa: E402
from tianximem.agent.plan_parse import InvalidPlanOutput, parse_plan  # noqa: E402
from tianximem.agent.plan_prompts import PLAN_DECOMPOSE_PROMPT_VERSION  # noqa: E402
from tianximem.common.config import AppConfig, load_config  # noqa: E402

DEFAULT_DATASET = "musique-haystack"


def _force_utf8_stdio() -> None:
    """Windows 控制台默认 GBK ⇒ 结尾那几行带 `⚠` 的说明会在**打印时**炸掉。"""
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _parse(raw: str) -> tuple[list[dict] | None, str | None, str | None]:
    """模型原文 → `(steps, extraction, error)`。**线上那一个解析器**（`parse_plan`）。

    ⚠ **只记原因，不记原文**（与 `InvalidPlanOutput` 同一条纪律：原文是模型读着
    不可信输入写出来的东西）。原文落在 `decompositions.jsonl`（gitignored）里供复核。

    ⚠ 每步多记两样：**模型声明的那一份 `depends_on`** 与**程序是否修过它**——
    规范化是"依赖以占位为准"那条规则的落点，而它必须能被数出来（不是静默改写）。
    """
    try:
        plan = parse_plan(raw, max_steps=4)
    except InvalidPlanOutput as exc:
        return None, None, f"InvalidPlanOutput: {exc}"
    steps = [
        {
            "id": s.id,
            "query": s.query,
            "depends_on": list(s.depends_on),
            "declared_dependencies": list(s.declared_dependencies),
            "repaired": s.dependencies_repaired,
        }
        for s in plan.steps
    ]
    return steps, plan.extraction, None


def summarize(rows: list[dict]) -> dict:
    """把逐题记录汇成报告要的那几个数（**纯函数**，用例钉着）。"""
    ok = [row for row in rows if row["plan_steps"] is not None]
    deltas = [row["plan_steps"] - row["gold_hops"] for row in ok]
    return {
        "n": len(rows),
        "invalid": sum(1 for row in rows if row["plan_steps"] is None),
        "exact": sum(1 for delta in deltas if delta == 0),
        "under": sum(1 for delta in deltas if delta < 0),
        "over": sum(1 for delta in deltas if delta > 0),
        "mean_gold": round(sum(row["gold_hops"] for row in ok) / len(ok), 3) if ok else None,
        "mean_plan": round(sum(row["plan_steps"] for row in ok) / len(ok), 3) if ok else None,
        #: `(真值跳数, 计划步数) → 题数`——**这张交叉表比均值有用**：
        #: 均值会把"3 跳拆 2 步"和"3 跳拆 4 步"混成一个数。
        "pairing": dict(
            sorted(Counter((row["gold_hops"], row["plan_steps"]) for row in ok).items())
        ),
        #: 有几步的 `depends_on` **被程序规范化过**（声明与占位不一致）。
        "repaired_steps": sum(1 for row in ok for step in row["steps"] if step.get("repaired")),
        "repaired_questions": sum(
            1 for row in ok if any(step.get("repaired") for step in row["steps"])
        ),
        "errors": dict(Counter(row["error"] for row in rows if row["error"])),
    }


def render_markdown(rows: list[dict], summary: dict, fingerprint: dict) -> str:
    lines = [
        f"# Plan 拆解单独评测（{fingerprint['prompt_version']}，{len(rows)} 题）",
        "",
        f"模型 `{fingerprint['model']}` · `max_tokens={fingerprint['max_tokens']}` · "
        f"清单 `{fingerprint['manifest']}` · 真值 = MuSiQue `question_decomposition` 的长度"
        "（⛔ 不进提示词）",
        "",
        "## 汇总",
        "",
        "| 量 | 值 |",
        "| --- | ---: |",
        f"| 题数 | {summary['n']} |",
        f"| 解析失败（作废） | **{summary['invalid']}** |",
        f"| 步数与真值**相等** | **{summary['exact']}** |",
        f"| **少拆** | **{summary['under']}** |",
        f"| 多拆 | {summary['over']} |",
        f"| 真值跳数均值 | {summary['mean_gold']} |",
        f"| **计划步数均值** | **{summary['mean_plan']}** |",
        "",
        "交叉表（真值跳数 → 计划步数）：",
        "",
        "| 真值 \\\\ 计划 | "
        + " | ".join(str(k) for k in sorted({p for _, p in summary["pairing"]}))
        + " |",
        "| --- | " + " | ".join("---:" for _ in sorted({p for _, p in summary["pairing"]})) + " |",
    ]
    plans = sorted({p for _, p in summary["pairing"]})
    golds = sorted({g for g, _ in summary["pairing"]})
    for gold in golds:
        cells = [str(summary["pairing"].get((gold, plan), 0)) for plan in plans]
        lines.append(f"| **{gold} 跳** | " + " | ".join(cells) + " |")
    lines += ["", "## 逐题", ""]
    for row in rows:
        head = f"### {row['qid']} — 真值 {row['gold_hops']} 跳"
        if row["plan_steps"] is None:
            lines += [head + f" · ❌ **解析失败**（{row['error']}）", ""]
            continue
        delta = row["plan_steps"] - row["gold_hops"]
        verdict = "相等" if delta == 0 else ("少拆" if delta < 0 else "多拆")
        lines += [head + f" → 计划 **{row['plan_steps']} 步**（{verdict}）", ""]
        lines += [
            f"{step['id']}: {step['query']}   (depends_on={step['depends_on']})"
            for step in row["steps"]
        ]
        lines.append("")
    return "\n".join(lines)


def _gold_hops(bench_dir: Path, qids: list[str], dataset: str) -> dict[str, int]:
    """`question_id → 真值跳数`。**只读**，且**只用在后事比对**。

    ⚠ 读不到就**响亮失败**：缺真值的那一题会让"少拆/相等"这类判断**静默算错**。
    """
    if dataset != "musique-haystack":
        raise SystemExit(f"✗ 本工具只对 musique-haystack 定义了真值跳数，收到 {dataset!r}")
    from eval.datasets.musique import read_rows

    by_id = {str(row["id"]): row for row in read_rows(bench_dir)}
    out: dict[str, int] = {}
    for qid in qids:
        row = by_id.get(qid)
        if row is None:
            raise SystemExit(f"✗ 清单里的 {qid} 不在数据集里（清单与 dataset/ 对不上了？）")
        hops = row.get("question_decomposition") or []
        if not hops:
            raise SystemExit(f"✗ {qid} 没有 question_decomposition ⇒ 没有真值跳数")
        out[qid] = len(hops)
    return out


def _client(config: AppConfig):
    """按**服务侧同一条路**拿对话客户端（⛔ 不自己拼 HTTP）。

    ⚠ `build_llm` 只在"某个消费者开着"时才给客户端 ⇒ 这里把 `agentic.enabled` 打开再问它。
    那**不是**在跑 agentic，只是借用同一个装配函数（同一条超时/端点口径）。
    """
    from dataclasses import replace

    from tianximem.service.app import build_llm

    enabled = replace(config, agentic=replace(config.agentic, enabled=True))
    client = build_llm(enabled, timeout=config.agentic.timeout_seconds)
    if client is None:
        raise SystemExit(
            "✗ 拿不到对话客户端：`.env` 里的 `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL` 不全"
        )
    return client


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Plan 拆解单独评测（不检索、不答题）")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    parser.add_argument("--manifest", default=None, help="默认按 --dataset 取")
    parser.add_argument("--out", required=True, help="产物目录（逐题落盘，可续跑）")
    parser.add_argument("--bench-dir", default=None, help="默认取 TIANXIMEM_BENCHMARK_DIR")
    parser.add_argument(
        "--limit", type=int, default=None, help="只跑前 N 题（冒烟用；正式跑不要传）"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略已落盘的记录、整批重跑（默认会把已完成的题跳过）",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="**不调模型**：拿已落盘的 `raw` 用当前解析器重算一遍并重写产物（改了解析规则时用）",
    )
    args = parser.parse_args(argv)

    bench_dir = Path(args.bench_dir or os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest or DATASETS[args.dataset]["manifest"])

    config = load_config()
    if config.agentic.plan_decompose_prompt_version != PLAN_DECOMPOSE_PROMPT_VERSION:
        # ⚠ 与 `PlanAgent.__init__` **同一条断言**：配置里声明的版本与代码里的那份
        #   不符 ⇒ 这份产物"用的是哪版提示词"就没有答案了。
        raise SystemExit(
            f"✗ `agentic.plan_decompose_prompt_version` = "
            f"{config.agentic.plan_decompose_prompt_version!r}，"
            f"代码里是 {PLAN_DECOMPOSE_PROMPT_VERSION!r} ⇒ 先对齐再跑。"
        )
    fingerprint = {
        "prompt_version": PLAN_DECOMPOSE_PROMPT_VERSION,
        "model": config.llm_model,
        "max_tokens": config.agentic.plan_max_output_tokens,
        "manifest": str(manifest_path),
        "dataset": args.dataset,
    }

    wanted = _load_manifest(manifest_path)
    samples = _select(bench_dir, wanted, dataset=args.dataset)
    questions = [(q.qid, q.question) for sample in samples for q in sample.questions]
    if args.limit is not None:
        questions = questions[: args.limit]
    qids = [qid for qid, _ in questions]
    gold = _gold_hops(bench_dir, qids, args.dataset)

    # ⚠ **指纹不匹配就拒绝续跑**（换一版提示词再续跑 = 两版记录混进同一张表，
    #   而那张表看起来完全正常）。
    fingerprint_path = out_dir / "run.json"
    if fingerprint_path.exists() and not args.fresh:
        old = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if old != fingerprint:
            raise SystemExit(
                "✗ 这个目录里已有**另一份指纹**的记录 ⇒ 拒绝续跑（否则两版会混在一起）：\n"
                f"  已有：{json.dumps(old, ensure_ascii=False)}\n"
                f"  本次：{json.dumps(fingerprint, ensure_ascii=False)}\n"
                "  ⇒ 换一个 `--out`，或加 `--fresh` 整批重跑。"
            )
    fingerprint_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    rows_path = out_dir / "decompositions.jsonl"
    rows: list[dict] = []
    if rows_path.exists() and not args.fresh:
        for line in rows_path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                rows.append(json.loads(line))

    if args.reparse:
        # ⭐ **改了解析规则时用这个**：模型输出没变，变的是"怎么读它"⇒
        #   重解析**恰好**隔离出解析规则那一个变量（⛔ 不重跑模型，所以连
        #   "网关不保证确定性"这条噪声都没有）。
        #   ⚠ 它**不调模型**，所以提示词版本那一条指纹照旧要匹配（否则量的是另一版输出）。
        if not rows:
            raise SystemExit(f"✗ {rows_path} 里没有可重解析的记录")
        changed = 0
        for row in rows:
            if not row.get("raw"):
                continue
            steps, extraction, error = _parse(row["raw"])
            row["steps"] = steps or []
            row["extraction"] = extraction
            row["error"] = error
            row["plan_steps"] = None if steps is None else len(steps)
            changed += 1
        rows_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        summary = summarize(rows)
        (out_dir / "decompositions.md").write_text(
            render_markdown(rows, summary, fingerprint), encoding="utf-8"
        )
        _report(summary, out_dir, seconds=0.0, reparsed=changed)
        return 0

    done = {row["qid"] for row in rows}
    todo = [(qid, question) for qid, question in questions if qid not in done]
    print(
        f"清单 {manifest_path}：{len(questions)} 题"
        f"，已有 {len(done)} 条记录，本次跑 {len(todo)} 题\n"
        f"提示词 {PLAN_DECOMPOSE_PROMPT_VERSION} · 模型 {config.llm_model} · "
        f"max_tokens={config.agentic.plan_max_output_tokens}",
        flush=True,
    )

    client = _client(config) if todo else None
    started = time.monotonic()
    try:
        with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
            for position, (qid, question) in enumerate(todo, start=1):
                # ⛔ **用的是线上那一份提示词**（`render_plan_decompose_prompt`
                #   就是执行器调的那一个）。
                prompt = render_plan_decompose_prompt(original_question=question)
                call_started = time.monotonic()
                error: str | None = None
                steps: list[dict] | None = None
                extraction: str | None = None
                raw = ""
                try:
                    raw = client.complete(  # type: ignore[union-attr]
                        prompt,
                        max_tokens=config.agentic.plan_max_output_tokens,
                        temperature=config.agentic.temperature,
                    )
                except Exception as exc:  # noqa: BLE001 —— 端点坏掉不该让整批消失
                    error = f"{type(exc).__name__}"
                if error is None:
                    steps, extraction, error = _parse(raw)
                row = {
                    "qid": qid,
                    "gold_hops": gold[qid],
                    "plan_steps": None if steps is None else len(steps),
                    "steps": steps or [],
                    "extraction": extraction,
                    "error": error,
                    "raw": raw,
                    "latency_ms": round((time.monotonic() - call_started) * 1000, 1),
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                flag = (
                    "❌"
                    if steps is None
                    else (
                        "="
                        if row["plan_steps"] == gold[qid]
                        else ("↓" if row["plan_steps"] < gold[qid] else "↑")
                    )
                )
                print(
                    f"  [{position}/{len(todo)}] {qid} 真值 {gold[qid]} → "
                    f"{row['plan_steps'] or '作废'} {flag}",
                    flush=True,
                )
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    summary = summarize(rows)
    (out_dir / "decompositions.md").write_text(
        render_markdown(rows, summary, fingerprint), encoding="utf-8"
    )
    _report(summary, out_dir, seconds=time.monotonic() - started)
    return 0


def _report(summary: dict, out_dir: Path, *, seconds: float, reparsed: int = 0) -> None:
    """打印汇总。⚠ **落盘必须在调用它之前**，避免打印失败导致产物缺失。"""
    print("\n════ 汇总 ════")
    print(f"  题数 {summary['n']} · 解析失败 **{summary['invalid']}**")
    print(f"  相等 {summary['exact']} · **少拆 {summary['under']}** · 多拆 {summary['over']}")
    print(f"  真值跳数均值 {summary['mean_gold']} → **计划步数均值 {summary['mean_plan']}**")
    print(f"  交叉表 {summary['pairing']}")
    print(
        f"  ⚠ `depends_on` 被程序规范化的步数 **{summary['repaired_steps']}**"
        f"（涉及 {summary['repaired_questions']} 题）"
    )
    if summary["errors"]:
        print(f"  ⚠ 失败原因 {summary['errors']}")
    print(f"\n产物：{out_dir}（run.json / decompositions.jsonl / decompositions.md）")
    if reparsed:
        print(
            f"⚠ **本次是 `--reparse`**：模型输出沿用已落盘的那一份，只重算了 {reparsed} 题的解析。"
        )
    print(f"⚠ 含数据集原文 ⇒ 只在 runs/ 下（gitignored）；本次 {seconds:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
