#!/usr/bin/env python3
"""把采集的三份派生文件合成**一套可直接评分的「官方判分套件」**。

## 它解决什么

采集目录里现在有三份**互不相识**的产物：

| 文件 | 是什么 |
| --- | --- |
| `official-eval-questions.jsonl` | 10,144 条 search + 题面/选项/题型 + 3 个公开集金标 |
| `official-gold-extra*.jsonl` | 补金标的产物，**一族一份**（`recover_official_gold.py` 等） |
| （采集原文）`official-searches.jsonl` / `official-adds.jsonl` | 题面与语料的一手来源 |

三份的**键各不相同、口径各写在各自的 docstring 里**，谁也没有「这一条该用哪个 pipeline 判、
判的时候 gold 长什么样」这一个字段。本脚本把它合成**一行一题**的套件：

```json
{"seq":27, "dataset":"locomo-refined", "attribution":"public_gold",
 "judge_kind":"llm", "gold_answers":["19 January, 2023"], "gold_rubric":null,
 "gold_judging":"…原文…", "answer_prompt_ref":"…", "judge_ref":"…", "official_score":1.0}
```

## 三件产物

| 文件 | 粒度 | 用途 |
| --- | --- | --- |
| `official-eval-kit.jsonl` | 一题一行（10,144） | 判分套件的**唯一入口**——loader 只读它 |
| `official-attribution.jsonl` | 一 user 一行 | 这个 user 属于哪一族——**对应 pipeline 的答案** |
| `official-prompts.md` | 一数据集一行 | 提示词索引：answer / judge 指回 `文件:符号`，**不抄正文** |

## 归属的判据（按硬度排序，逐行记在 `attribution` 字段里）

1. `tag`——采集里**自带的** `[<dataset>][session_1]…` 标签（132 条 add，最硬）；
2. `official_feedback`——官方回写判分的 `qa_id` 前缀（`locomo1:` / `scriptMem-angry:` …）；
3. `public_gold`——逐字命中公开题库（scriptmem / locomo / personamem）；
4. `gold_extra`——同上，但走 `official-extra` 那 6 个题库；
5. `corpus`——**启发式**（首条 add 的前缀形态），只用于把「无金标」的 user 归类到族，
   **不要当成已证实的归属**；
6. `unknown`——没认出来。

⚠ **`corpus` 那一条是启发式**：实测 `Corpus:` 前缀里有三族语料（MQuAKE 的
`{"subject_id":…}`、文档/表格 JSON、以及零散文本），只靠前缀分不干净。
所以它**只影响 `official-attribution.jsonl` 的族统计**，不影响任何一题的 `dataset`
（题级 `dataset` 只从 1–4 来；`corpus` 归不出题级归属时题级留 `unknown`）。

**用法**：

    uv run python tools/build_official_kit.py

输入输出都在采集目录（默认 `official-dataset-2026-09-29/`，可用 `--dataset-dir` 改）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

__all__ = ["main", "build"]

#: 数据集 → 提示词索引。**只存指针与「采集里实际观察到的指令」**，不复制 prompt 正文
#: （正文的家在那些文件里，抄第二遍就是漂移的开始）。
#:
#: `answer` / `judge` 是 `文件#符号`；`observed` 是**采集原文里出现过的**作答指令
#: （有则给，说明这一批题实际是按哪种包装问的）；`note` 是判分口径的一句话。
PROMPT_REFS: dict[str, dict[str, str]] = {
    "locomo-refined": {
        "answer": "dataset/.upstream/aml/pipeline_locomo-refined.py#render_answer_prompt",
        "judge": "dataset/.upstream/aml/pipeline_locomo-refined.py#render_accuracy_prompt",
        "observed": "（无尾指令——open 题）",
        "note": "二元 CORRECT/WRONG；TIME 块要求相对↔绝对**不得**换算",
    },
    "scriptmem": {
        "answer": "dataset/.upstream/aml/pipeline_scriptmem.py#render_answer_prompt",
        "judge": "dataset/.upstream/aml/pipeline_scriptmem.py#score_item（纯函数，无 LLM）",
        "observed": "「the only correct answer, enclosed in parentheses, e.g., (X).」等三条尾指令"
        "（**逐字在公开题库题面里**，不在 pipeline 里）",
        "note": "选项字母精确匹配：single_choice 集合相等 / ordering 序列相等",
    },
    "personamem-v2": {
        "answer": "dataset/.upstream/aml/pipeline_v2_personamem.py#official_mcq_messages",
        "judge": "dataset/.upstream/aml/pipeline_v2_personamem.py#evaluate_mcq",
        "observed": "（无尾指令——选项在 query 里）",
        "note": "MCQ 字母/文本精确匹配；**不读检索字段**",
    },
    "beam": {
        "answer": "dataset/.upstream/aml/pipeline_beam.py#OPEN_ENDED_ANSWER_TEMPLATE",
        "judge": "dataset/.upstream/aml/pipeline_beam.py#RUBRIC_JUDGE_TEMPLATE",
        "observed": "（无尾指令——open 题）",
        "note": "逐条 rubric 三点制（1.0/0.5/0.0）取均分；**时间规则与 LoCoMo 相反**",
    },
    "clbench": {
        "answer": "dataset/.upstream/aml/clb_pipeline.py#build_answer_prompt",
        "judge": "dataset/.upstream/aml/clb_pipeline.py#rubric_judge_prompt",
        "observed": "（system prompt 由 query 自带，见 doc-pp/doc 族）",
        "note": "严格全有全无；另写 requirement_ratio 作部分分",
    },
    "mquake-remastered": {
        "answer": "eval/harness/extra_pipeline.py#MQUAKE_ANSWER_PROMPT",
        "judge": "eval/harness/extra_pipeline.py#_judge_one（别名表纯函数）",
        "observed": "「Answer using only the supplied paragraphs. If the evidence is "
        "insufficient, respond with INSUFFICIENT_EVIDENCE.」",
        "note": "**没有官方 pipeline**——作答与判分都是我们自写，分数只在仓内比",
    },
    "memtrapbench": {
        "answer": "eval/harness/extra_pipeline.py#MEMTRAP_ANSWER_PROMPT",
        "judge": "eval/harness/extra_pipeline.py#_judge_one（LLM 四维均分）",
        "observed": "（无尾指令——陷阱题）",
        "note": "四维均分 ≥4.0 判过；阈值是我们定的",
    },
    "corporatebench": {
        "answer": "eval/harness/corporatebench_pipeline.py#build_answer_prompt",
        "judge": "eval/harness/extra_pipeline.py#_judge_one（标量/集合纯函数）",
        "observed": "（无尾指令——open 题）",
        "note": "int 精确 / str 集合 F1；`partial` 带标量 EM 或 set-F1",
    },
    "medmemorybench": {
        "answer": "eval/harness/extra_pipeline.py（通用 ANSWER_PROMPT）",
        "judge": "dataset/.upstream/medmemorybench/metrics/（上游发布口径）",
        "observed": "（无尾指令——中文医患）",
        "note": "判分口径照上游 `metrics/`；作答侧是我们写的",
    },
    "tempreason": {
        "answer": "eval/harness/extra_pipeline.py#TEMPR_ANSWER_PROMPT",
        "judge": "eval/harness/extra_pipeline.py#_judge_one（可接受答案串命中）",
        "observed": "（无尾指令；官方问法被改写过，逐字匹配 0 命中）",
        "note": "**没有官方 pipeline**；口径我们定",
    },
    "hybridqa": {
        "answer": "eval/harness/extra_pipeline.py#ANSWER_PROMPT",
        "judge": "eval/harness/extra_pipeline.py#judge_tempreason（别名/串包含，纯函数）",
        "observed": "（无尾指令——open 题）",
        "note": "官方 `evaluate_script.py` 按归一化后的答案串比对",
    },
    "musique": {
        "answer": "eval/harness/extra_pipeline.py#ANSWER_PROMPT",
        "judge": "eval/harness/extra_pipeline.py#judge_tempreason（answer / answer_aliases 命中）",
        "observed": "「Answer using only the supplied paragraphs…」（与 MQuAKE 同一条尾指令）",
        "note": "20 段支持段是重采样变体；判分只用答案串（不回落支持段 F1）",
    },
    "feverous": {
        "answer": "eval/harness/extra_pipeline.py#ANSWER_PROMPT",
        "judge": "eval/harness/official_capture_pipeline.py#_feverous_judge（三分类）",
        "observed": "（无尾指令——claim 核查题）",
        "note": "**只判 label，不判证据 id 的 F1**（我们的 prompt 不要求输出证据）",
    },
    "halumem": {
        "answer": "eval/harness/extra_pipeline.py#ANSWER_PROMPT（上游是逐系统定制的）",
        "judge": "dataset/.upstream/halumem/eval_tools.py#EVALUATION_PROMPT_FOR_QUESTION",
        "observed": "（无尾指令——open 题）",
        "note": "三分类 Correct / Hallucination / Omission；**只有 Correct 算对**",
    },
    "docpp": {
        "answer": "采集的 query 就是官方答案 prompt（+ 我们补的记忆块）",
        "judge": "dataset/.upstream/doc-pp/prompts/judge*.py（上游两条裁判）",
        "observed": "query 自带整段 system prompt（`You are a document-grounded …`）",
        "note": "direct 看 policy_value 在不在；indirect 逐条 checklist 全中才算对",
    },
    "unknown": {
        "answer": "—",
        "judge": "—",
        "observed": "—",
        "note": "语料族未识别或没有公开金标 ⇒ **不评分**",
    },
}

#: 公开金标字段前缀 → 数据集名。
_PUBLIC_GOLD_DATASETS: tuple[tuple[str, str], ...] = (
    ("gold_locomo_", "locomo-refined"),
    ("gold_scriptmem_", "scriptmem"),
    ("gold_personamem_", "personamem-v2"),
)

#: 官方回写判分的 `qa_id` 前缀 → 数据集名。
_QA_ID_DATASETS: tuple[tuple[str, str], ...] = (
    ("locomo", "locomo-refined"),
    ("scriptMem", "scriptmem"),
)


def _norm_tag(text: str) -> str | None:
    """首条 message 的 `[<tag>]` 前缀——**采集自带的标签**（只在一小撮 canary 上）。"""
    match = re.match(r"\[([a-z0-9_]+)\]", text.lstrip()[:40])
    return match.group(1) if match else None


#: 自标标签 → 数据集名（README §4.2.1 那张表）。**只有标签本身即证据的那些**。
TAG_DATASETS: dict[str, str] = {
    "locomo_refined": "locomo-refined",
    "personamem_v2_32k": "personamem-v2",
    "mquake_remastered": "mquake-remastered",
    "memtrapbench": "memtrapbench",
    "beam_100k": "beam",
    "angry": "scriptmem",
    "clbench_0_4k": "clbench",
    "clbench_16_32k": "clbench",
    "tempreason": "tempreason",
    "docpp": "doc-pp",
}


def corpus_hint(text: str) -> str:
    """首条 add 的**形态**归族——启发式，只用于无金标 user 的族统计。

    ⚠ **它不是归属判据**：`Corpus:` 前缀下至少有三族语料（实测 2,674 条里只有 492 条
    带 `subject_id`/`UPDATE`，1,341 条是文档/表格 JSON）。这里分得开的只是"像什么"。
    """
    head = text.lstrip()[:400]
    if head.startswith("Corpus:"):
        if '"subject_id"' in head or head.startswith("Corpus: UPDATE"):
            return "mquake-remastered"
        return "corpus-doc-table"
    if head.startswith("source:"):
        return "source-law"  # ⚠ 中文法条——出处未识别
    if head.startswith("Name:"):
        return "locomo-refined"
    if head.startswith("document: Message-ID"):
        return "corporatebench"
    if head.startswith("Streaming"):
        return "platform-writeback"  # 平台自己回写的判分记录，不是语料
    if re.search(r"[一-鿿]", head[:200]):
        return "medmemorybench?"
    return "persona-interview?"


def _gold_source(row: dict) -> tuple[str, str] | None:
    """一行 `official-eval-questions.jsonl` 里的金标/判分来自谁——返回 (数据集, 依据)。"""
    qa_id = row.get("official_qa_id")
    if isinstance(qa_id, str):
        for prefix, dataset in _QA_ID_DATASETS:
            if qa_id.startswith(prefix):
                return dataset, "official_feedback"
    public = row.get("public_gold")
    if isinstance(public, dict):
        for key, dataset in _PUBLIC_GOLD_DATASETS:
            if any(name.startswith(key) for name in public):
                return dataset, "public_gold"
    return None


def _judge_key(dataset: str) -> str:
    """该数据集用哪一类判分——**套件里唯一的判分语义字段**（见模块 docstring 的枚举）。"""
    return {
        "scriptmem": "letters",
        # ⚠ PersonaMem-v2 在**采集里的那批**是 **open 题 + 长文本答案**（不是它题库里的 MCQ）——
        #   官方的 `evaluate_narrow` 要 `preference` 元数据，而采集的 search 里没有那个字段，
        #   所以退到文本匹配（口径见 `official_capture_pipeline.py` 的 docstring）。
        "personamem-v2": "exact",
        "locomo-refined": "llm",
        "beam": "rubric",
        "memtrapbench": "rubric",
        "clbench": "rubric",
        "mquake-remastered": "exact",
        "corporatebench": "exact",
        "medmemorybench": "exact",
        "tempreason": "exact",
        # Doc-PP：金标是 `policy_value` + `checklist`（**判分要点**，不是答案串）——
        # 上游那两条裁判 prompt 就是按逐条 checklist 打的（见 `extra_pipeline` 的 docpp 分支）。
        "docpp": "rubric",
        # HaluMem：金标是短答串，但**上游只发布一份 LLM 裁判**（Correct/Hallucination/
        # Omission）——`llm` 那一档就是它（见 `official_capture_pipeline._halumem_judge`）。
        "halumem": "llm",
        # 三个 `Corpus:` 族（见 `tools/recover_corpusqa.py`）：HybridQA / MuSiQue 的答案
        # 是短串（别名包含即对）；**FEVEROUS 是三分类的 claim 标签**，判分在
        # `official_capture_pipeline._feverous_judge` 里单独写（别名包含会假阳性）。
        "hybridqa": "exact",
        "musique": "exact",
        "feverous": "exact",
    }.get(dataset, "none")


def _public_gold_payload(public: dict, dataset: str) -> dict:
    """把 `public_gold` 那三套字段名折叠成统一的两个键：`gold_answers` / `gold_rubric`。"""
    if dataset == "locomo-refined":
        return {
            "gold_answers": list(public.get("gold_locomo_answer") or []),
            "gold_rubric": None,
        }
    if dataset == "scriptmem":
        letters = list(public.get("gold_scriptmem_answer_letters") or [])
        answer = public.get("gold_scriptmem_answer")
        return {
            "gold_answers": letters or ([answer] if answer else []),
            "gold_rubric": None,
        }
    if dataset == "personamem-v2":
        answer = public.get("gold_personamem_answer")
        return {"gold_answers": [answer] if answer else [], "gold_rubric": None}
    return {"gold_answers": [], "gold_rubric": None}


def _extra_payload(extra: dict) -> dict:
    """`official-gold-extra.jsonl` 一行的金标载荷（`gold_kind` 是它自己的口径）。

    `gold_native` 是**该数据集裁判要的原生载荷**（CorporateBench 的 `answer_type`、
    MemTrapBench 的 `gold_standard`、MedMemoryBench 的 `query_type`…）——只留
    `gold_answers` 会让裁判走错分支**且不报错**，所以原样搬过去。
    """
    return {
        "gold_answers": list(extra.get("gold_answers") or []),
        "gold_rubric": extra.get("gold_rubric"),
        "gold_native": extra.get("gold_native"),
        "gold_alternative_answers": list(extra.get("gold_alternative_answers") or []),
        "gold_variant": extra.get("gold_variant"),
        "source_locator": extra.get("source_locator"),
    }


def build(dataset_dir: Path) -> dict[str, object]:
    """合成三份产物，返回统计（供 CLI 打印与测试断言）。"""
    questions_path = dataset_dir / "official-eval-questions.jsonl"
    adds_path = dataset_dir / "official-adds.jsonl"
    for path in (questions_path, adds_path):
        if not path.exists():
            raise SystemExit(f"✗ 找不到 {path}（采集目录对吗？）")

    rows = [json.loads(line) for line in questions_path.open(encoding="utf-8") if line.strip()]

    #: 补金标的产物**是按族分文件的**（`official-gold-extra.jsonl` 是本体，
    #: 后面每补一族加一份 `official-gold-extra-<族>.jsonl`）——这里一并收进来，
    #: 免得每加一族都要改一次本脚本。冲突时**先到的赢**（按文件名排序）。
    extra_by_seq: dict[int, dict] = {}
    for path in sorted(dataset_dir.glob("official-gold-extra*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            extra_by_seq.setdefault(record["seq"], {**record, "_source_file": path.name})
    if not extra_by_seq:
        raise SystemExit(f"✗ {dataset_dir} 里没有 official-gold-extra*.jsonl")

    # ── user 级归属：标签最硬，其次金标，再次形态启发式 ──────────────────
    first_add: dict[str, str] = {}
    tag_votes: dict[str, Counter] = defaultdict(Counter)
    for line in adds_path.open(encoding="utf-8"):
        record = json.loads(line)
        user = record["user_id"]
        messages = record.get("messages") or []
        if user not in first_add and messages:
            first_add[user] = str(messages[0].get("content") or "")
        for message in messages:
            tag = _norm_tag(str(message.get("content") or ""))
            if tag and tag in TAG_DATASETS:
                tag_votes[user][TAG_DATASETS[tag]] += 1

    row_dataset: dict[int, tuple[str, str]] = {}
    for row in rows:
        source = _gold_source(row)
        if source:
            row_dataset[row["seq"]] = source
    for seq, extra in extra_by_seq.items():
        row_dataset[seq] = (extra["source_dataset"], "gold_extra")

    user_dataset: dict[str, tuple[str, str]] = {}
    labelled: dict[str, Counter] = defaultdict(Counter)
    seq_to_user = {row["seq"]: row["user_id"] for row in rows}
    for seq, (dataset, _basis) in row_dataset.items():
        labelled[seq_to_user[seq]][dataset] += 1
    for user in first_add:
        if tag_votes[user]:
            user_dataset[user] = (tag_votes[user].most_common(1)[0][0], "tag")
        elif labelled[user]:
            user_dataset[user] = (labelled[user].most_common(1)[0][0], "labelled_rows")
        else:
            user_dataset[user] = (corpus_hint(first_add[user]), "corpus")
    for row in rows:  # 只 search 没 add 的 29 个 user
        user = row["user_id"]
        if user in user_dataset:
            continue
        if labelled[user]:
            user_dataset[user] = (labelled[user].most_common(1)[0][0], "labelled_rows")
        else:
            user_dataset[user] = ("unknown", "unknown")

    # ── 套件：一题一行 ────────────────────────────────────────────────
    kit: list[dict] = []
    for row in rows:
        seq = row["seq"]
        dataset, basis = row_dataset.get(seq, ("unknown", "unknown"))
        payload: dict = {"gold_answers": [], "gold_rubric": None}
        gold_judging = ""
        if seq in extra_by_seq:
            payload = _extra_payload(extra_by_seq[seq])
            gold_judging = str(extra_by_seq[seq].get("gold_judging") or "")
        elif isinstance(row.get("public_gold"), dict):
            payload = _public_gold_payload(row["public_gold"], dataset)
        kit.append(
            {
                "seq": seq,
                "ts": row["ts"],
                "user_id": row["user_id"],
                "task": row.get("task"),
                "query": row.get("query"),
                "question": row.get("question"),
                "options": row.get("options") or [],
                "instruction": row.get("instruction"),
                "prior_adds_for_user": row.get("prior_adds_for_user"),
                "dataset": dataset,
                "attribution": basis,
                "judge_kind": _judge_key(dataset) if dataset != "unknown" else "none",
                "gold_answers": payload.get("gold_answers") or [],
                "gold_rubric": payload.get("gold_rubric"),
                "gold_native": payload.get("gold_native"),
                "gold_alternative_answers": payload.get("gold_alternative_answers") or [],
                "gold_variant": payload.get("gold_variant"),
                "gold_judging": gold_judging,
                "official_score": row.get("official_score"),
                "official_judge_kind": row.get("official_judge_kind"),
                "official_predicted_answer": row.get("official_predicted_answer"),
                "official_accepted_answer": row.get("official_accepted_answer"),
                "answer_prompt_ref": PROMPT_REFS.get(dataset, PROMPT_REFS["unknown"])["answer"],
                "judge_ref": PROMPT_REFS.get(dataset, PROMPT_REFS["unknown"])["judge"],
            }
        )

    _write_jsonl(dataset_dir / "official-eval-kit.jsonl", kit)
    _write_jsonl(
        dataset_dir / "official-attribution.jsonl",
        [
            {"user_id": user, "dataset": dataset, "basis": basis}
            for user, (dataset, basis) in sorted(user_dataset.items())
        ],
    )
    (dataset_dir / "official-prompts.md").write_text(_prompts_markdown(), encoding="utf-8")

    return {
        "rows": len(kit),
        "scorable": sum(1 for r in kit if r["judge_kind"] != "none"),
        "by_dataset": Counter(r["dataset"] for r in kit),
        "by_judge_kind": Counter(r["judge_kind"] for r in kit),
        "users": len(user_dataset),
        "by_attribution": Counter(basis for _d, basis in user_dataset.values()),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompts_markdown() -> str:
    lines = [
        "# 官方采集的提示词索引（生成物，别手改）",
        "",
        "由 [`tools/build_official_kit.py`](../../tools/build_official_kit.py) 生成。",
        "**prompt 正文不住这里**——它住在指到的那个文件里（单数来源）；",
        "本表只回答「这一族用哪份 prompt、判分是什么口径」。",
        "",
        "| 数据集 | answer prompt | judge | 采集里观察到的指令 | 判分口径 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset, entry in PROMPT_REFS.items():
        lines.append(
            f"| `{dataset}` | `{entry['answer']}` | `{entry['judge']}` "
            f"| {entry['observed']} | {entry['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", default="official-dataset-2026-09-29")
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset_dir)

    stats = build(dataset_dir)
    print(f"套件 {stats['rows']} 行，其中**可评分 {stats['scorable']}**")
    print(f"  按数据集：{dict(stats['by_dataset'].most_common())}")
    print(f"  按判分：{dict(stats['by_judge_kind'].most_common())}")
    print(f"  归属：{stats['users']} user，依据分布 {dict(stats['by_attribution'].most_common())}")
    print(f"→ {dataset_dir}/：official-eval-kit.jsonl / official-attribution.jsonl")
    print("   / official-prompts.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
