"""Detached runner: retain failures, audit artifacts and write a final local report."""

from __future__ import annotations

import json
import os
import traceback

import audit
import run_suite

ROOT = run_suite.ROOT
OUT = run_suite.OUT
VAR = run_suite.VAR
REPORT = ROOT / "eval/reports/facts-benchmark-20261005.md"


def percent(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def render_report(summary: dict | None, error: str | None) -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    state = json.loads((OUT / "status.json").read_text())
    entries = {(r["facts"], r["dataset"]): r for r in state["runs"]}
    finished = all(entries.get(("on", ds), {}).get("state") == "complete" for ds in run_suite.SETS)
    lines = [
        f"事实抽取后的七数据集固定题集重跑（{'完成' if finished else '未全部完成'}）",
        "",
        f"结果更新于 `{run_suite.now()}`。产品固定为 `{manifest['commit']}`，",
        "用户确认按历史相同题集重跑；不是原始数据全量，不消耗 AML Smoke/Full 配额。",
        "回答/裁判模型 Qwen/Qwen3.5-9B，答案 temperature 0、上限 1024，",
        "HTTP Add/Search、top_k 100、official Add、邻接半径 0、rerank 关闭。",
        "",
        "| 数据集 | 题数 | 历史完全正确精度 | 当前完全正确精度 | 差值 | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for meta in manifest["datasets"]:
        entry = entries.get(("on", meta["dataset"]), {})
        complete = entry.get("state") == "complete"
        current = entry.get("scores", {}).get("overall") if complete else None
        old = meta["historical_overall"]
        delta = "—" if current is None or old is None else f"{100 * (current - old):+.2f}pt"
        label = "完成" if complete else "未完成（不能填作 0%）"
        lines.append(
            f"| {meta['dataset']} | {meta['n_questions']} | {percent(old)} | "
            f"{percent(current)} | {delta} | {label} |"
        )
    lines += [
        "",
        "七份共 2,435 题，候选题在本轮重新生成答案。后台迁移保留本轮已完成的答案，",
        "只复用本轮真实 HTTP 检索响应；续跑前核对完整评分输入一致，不复用历史基准答案。",
        "六份有完整历史参照的数据，题号集合与共同原始文件字节指纹一致。",
        "",
        "CorporateBench 历史答案提示词不同，历史差值不能全部归因于事实抽取。",
        "本轮额外的 250 题开关对照使用当前相同提示词、同一原文库与向量集合：",
        "",
        "| CorporateBench 当前开关对照 | 完全正确精度 | 标量 EM / 列表 set-F1 均分 |",
        "| --- | ---: | ---: |",
    ]
    for arm, name in (("off", "关闭共同取证"), ("on", "开启共同取证")):
        entry = entries.get((arm, "corporatebench"), {})
        scores = entry.get("scores", {}) if entry.get("state") == "complete" else {}
        lines.append(
            f"| {name} | {percent(scores.get('overall'))} | "
            f"{percent(scores.get('dataset_score', {}).get('mean'))} |"
        )
    lines += [
        "",
        "各数据集裁判不同，不计算跨数据集平均值，也不把本地精度解释为 AML 榜分。",
        "历史比较还包含独立新建索引和模型/裁判波动；小幅差异不足以单次定论。",
        "",
    ]
    if summary:
        lines += [
            "| 本轮范围核对 | 答案 / 应有题数 | 判分条数 | 裁判错误 | 返回单条事实的题数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for entry in summary["runs"]:
            arm = "on" if entry["facts_enabled"] else "off"
            lines.append(
                f"| {arm} {entry['dataset']} | {entry['n_answers']}/{entry['n_expected']} | "
                f"{entry['n_labels']} | {entry['judge_errors']} | "
                f"{entry['queries_returning_atomic_facts']} |"
            )
        lines += [
            "",
            f"核对了 {summary['derived_facts']} 条派生事实的逐字原文出处与用户归属，",
            "并核对实际返回事实的正文、用户隔离、top_k 和完成题号集合。",
            "物理扫描覆盖不代表语义完整，返回原文的事实路径未计入单条事实题数。",
            "",
        ]
    if error:
        lines += [
            "自动核对或运行存在错误，详见 `var/facts-benchmark-20261005/background.log`。",
            "",
        ]
    for entry in state["runs"]:
        if entry.get("state") != "complete":
            lines += [
                f"未完成：`{entry['run_id']}`，退出码 `{entry.get('exit_code')}`；",
                "日志 `var/facts-benchmark-20261005/"
                f"run-{entry['facts']}-{entry['dataset']}.log`。",
                "",
            ]
    lines += [
        "产品、加载器、答案/裁判代码、pipeline 和配置指纹在",
        "`runs/facts-benchmark-20261005/manifest.json`；运行中变更会停止队列。",
        "独立新建原文库和向量集合，embedding 缓存作一致快照复制，既有服务与产物保留。",
        "逐题检索、输入、答案、判分以及失败产物全部保留。",
        "",
        "实时状态：`runs/facts-benchmark-20261005/status.json`。",
        "来源核对：`runs/facts-benchmark-20261005/audit-summary.json`。",
        "原文库/最终 SQLite 快照、日志：`var/facts-benchmark-20261005/`。",
        "执行脚本：`runs/facts-benchmark-20261005/run_suite.py` 与 `supervise.py`。",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    ledger = ROOT / "eval/reports/ledger.md"
    link = "[本轮完整结果与未完成记录](facts-benchmark-20261005.md)"
    content = ledger.read_text()
    if link not in content:
        entry = (
            "### 2026-10-05 事实抽取后的七数据集固定题集重跑\n\n"
            "`facts-benchmark-20261005`：按用户确认的历史同题集串行重跑，"
            "额外保留 CorporateBench 当前提示词下的事实开关对照。"
            f"完成范围、逐题成绩、来源核对和失败状态见 {link}。\n\n"
        )
        ledger.write_text(content.replace("## 基准\n\n", "## 基准\n\n" + entry, 1))


def main() -> int:
    os.chdir(ROOT)
    VAR.mkdir(parents=True, exist_ok=True)
    control = {"pid": os.getpid(), "started_at": run_suite.now(), "state": "running"}
    run_suite.write(VAR / "background-control.json", control)
    error = None
    result = 1
    summary = None
    try:
        result = run_suite.main()
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        try:
            summary = audit.audit()
        except Exception:
            error = (error or "") + traceback.format_exc()
            print(error, flush=True)
        try:
            render_report(summary, error)
        except Exception:
            error = (error or "") + traceback.format_exc()
            print(error, flush=True)
        control.update(
            finished_at=run_suite.now(),
            state="complete" if result == 0 and error is None else "incomplete",
            suite_exit_code=result,
            error=error,
            report=str(REPORT),
        )
        run_suite.write(VAR / "background-control.json", control)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
