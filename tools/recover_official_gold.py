#!/usr/bin/env python3
"""给官方采集的 search 补**公开金标**——`official-extra` 那 8 个数据集。

## 它补的是什么口子

[`build_eval_set.py`](../build_eval_set.py)（它生成 `official-eval-questions.jsonl`）只从
**3 个**数据集捞金标：ScriptMem / LoCoMo-Refined / PersonaMem-v2。官方流量里还出现过别的数据集
（[`../official-dataset-2026-09-29/README.md`](../official-dataset-2026-09-29/README.md) §4.2.3），
它们的题库现在都在 `dataset/`（tier = `official-extra`）。

本脚本按**同一套归一化**（NFKC + 小写 + 空白折叠）逐字匹配题面，把它们也接上。

## 输出

`<数据集目录>/official-gold-extra.jsonl`，一行一条被匹配上的 search。**按 `seq` 与
`official-eval-questions.jsonl` join**（两份文件的行序一致，实测过）。

| 字段 | 说明 |
| --- | --- |
| `seq` / `ts` / `user_id` / `question` | 与采集导出逐字相同，用来 join / 复现 |
| `source_dataset` / `source_locator` | 命中的数据集与它在题库里的位置（可回查） |
| `gold_kind` | `exact`（可字符串判定）或 `rubric`（判分要点原文） |
| `gold_answers` | `exact` 型的可接受答案串（含别名） |
| `gold_rubric` | `rubric` 型的判分要点/参考回答原文 |
| `gold_judging` | **该数据集这条标签的判分口径**——引用前必读，见下 |

## ⚠ 口径核验（每个数据集都逐份看过，写进每行的 `gold_judging`）

* **MQuAKE-Remastered**：一份 case 同时有 `answer`（更新**前**）与 `new_answer`（更新**后**）。
  官方流量注入的是 `Corpus: {…}` / `Corpus: UPDATE:` 那类更新，**所以金标取 `new_answer`**——
  实测抽样核对：命中用户的记忆里有 `new_answer` 那一侧的事实。
* **CorporateBench**：`answer` + `answer_type`（int / str），照抄。
* **MedMemoryBench**：`answers` 是 JSON 串，只取 `is_correct: true` 的那些。
* **MemTrapBench**：`gold_standard` 是一段**判分要点**（不是答案串），配 `test_type`
  （red / green）。
* **BEAM**：`probing_questions` 是**字符串化的 dict**（`ast.literal_eval` 解）；
  **金标是每组都有的 `rubric`**（不是 `ideal_response`——10 个组里只有 1 个组用它）。
* **PersonaMem-v2**：`correct_answer` 是长文本，上游 pipeline 做**精确文本匹配**——
  ⚠ 别把它当"任意等价表述都算对"。

**用法**：

    uv run python tools/recover_official_gold.py

题库目录默认取 `TIANXIMEM_BENCHMARK_DIR`（D16：路径不在代码里硬编码）；
数据集目录默认 `official-dataset-2026-09-29`，输出写在它下面。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ⚠ 直接跑脚本时 `sys.path[0]` 是 `tools/`，仓库根不在上面；而 `eval` 不是已安装的包
#   （只有 `src/` 打了包）⇒ 补一条路径（与 `tools/order_probe.py` 同一处置）。
#   也可以 `python -m tools.recover_official_gold` 跑。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets.registry import benchmark_dir  # noqa: E402

__all__ = ["main", "norm", "load_all"]

#: `official-extra` 的归档目录名。逐个查表——**缺哪个就跳过哪个**（可能只 fetch 了 required 档）。
SOURCES = (
    "mquake-remastered",
    "corporatebench",
    "medmemorybench",
    "memtrapbench",
    "beam",
    "personamem-v2",
)

#: BEAM 各 probing 组的答案字段**不统一**（实测 10 组里有 4 种名字，有的组一个都没有）。
_BEAM_ANSWER_KEYS = ("answer", "ideal_answer", "ideal_response", "ideal_summary")

MQUAKE_JUDGING = (
    "MQuAKE-Remastered：同一 case 有 `answer`（更新前）与 `new_answer`（更新后）两套，"
    "**取哪套取决于该 user 的记忆里有没有注入这一 case 的改写**"
    "（`Corpus: UPDATE: replace the prior value …`，按 `edit_triples` 的三元组比对，"
    "且注入时间要早于提问）。官方流量里两种都真实出现过。"
    "`gold_variant` 记的就是这个判断；`gold_alternative_answers` 是另一套，别丢。"
)


def norm(text: str) -> str:
    """题面归一化：NFKC + 转小写 + 空白折叠。两侧都用它，否则匹配率会假低。"""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def _parquet_rows(path: Path, columns: list[str] | None = None, batch_size: int = 512):
    """流式读 parquet——题库是几十 MB，没必要整个读进内存。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover — 依赖里已经有它
        raise SystemExit("读 parquet 需要 pyarrow——`uv sync` 会装上它（dev 依赖）") from exc

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def load_mquake(root: Path) -> dict[str, dict]:
    """多跳知识更新。**答案有两套**（`answer` 更新前 / `new_answer` 更新后）。

    逐条定夺见 `resolve_mquake`。

    ⚠ **同一句题面在四个 parquet 里可能各有一个 case，而且答案不同**（实测抽 200 个重名题面，
    199 个的答案真的不一样）。所以**不能"先到先得"**：这里把每个候选都留着，等 `--adds`
    拿到记忆之后，按"该 case 的改写/原始事实有没有被注入"选定那一个。

    默认（没给 `--adds`）取 `new_answer`——MQuAKE 的题面本来就是"注入改写之后再问"，
    而官方流量里确实注入了 `Corpus: UPDATE:`；`gold_variant` 会记成 `updated(未核)`。
    """
    candidates: dict[str, list[dict]] = {}
    for path in sorted((root / "mquake-remastered" / "data").glob("*.parquet")):
        for row in _parquet_rows(path):
            updated = [a for a in [row["new_answer"], *(row.get("new_answer_alias") or [])] if a]
            original = [a for a in [row["answer"], *(row.get("answer_alias") or [])] if a]
            for variant, question in enumerate(row.get("questions") or []):
                candidates.setdefault(norm(question), []).append(
                    {
                        "source_locator": {
                            "file": path.name,
                            "case_id": row["case_id"],
                            "variant": variant,
                        },
                        "gold_answers": updated,
                        "gold_alternative_answers": original,
                        "gold_edits": [list(e) for e in (row.get("edit_triples") or [])],
                        "gold_fact_subjects": [s for s, _r, _o in row.get("orig_triples") or []],
                    }
                )
    table: dict[str, dict] = {}
    for question, options in candidates.items():
        first = options[0]
        table[question] = {
            "source_dataset": "mquake-remastered",
            "gold_kind": "exact",
            "gold_rubric": None,
            "gold_variant": "updated(未核)",
            **first,
            "gold_candidates": options[1:],  # 0 或 1 个 = 无歧义
            "gold_judging": MQUAKE_JUDGING,
        }
    return table


def load_corporatebench(root: Path) -> dict[str, dict]:
    """邮件线程 + KB 问答。金标就是 `answer`，带 `answer_type`（int / str）。"""
    table: dict[str, dict] = {}
    for sub in ("kb_qa", "topic_qa", "integrated_qa"):
        path = root / "corporatebench" / "data" / sub / "zenith_questions.json"
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")).get("questions", []):
            question = item.get("question")
            if not question:
                continue
            table.setdefault(
                norm(question),
                {
                    "source_dataset": "corporatebench",
                    "source_locator": {
                        "file": f"{sub}/zenith_questions.json",
                        "id": item.get("id"),
                    },
                    "gold_kind": "exact",
                    "gold_answers": [str(item["answer"])],
                    "gold_rubric": None,
                    #: 判分要**整个 dict**（`score_corporatebench` 靠 `answer_type` 决定怎么比），
                    #: 所以原样留一份——套件（`build_official_kit.py`）直接把它交给裁判。
                    "gold_native": {
                        "answer": item["answer"],
                        "answer_type": item.get("answer_type"),
                    },
                    "gold_judging": f"CorporateBench（zenith）：金标是 question 自带的 answer，"
                    f"类型 {item.get('answer_type')}",
                },
            )
    return table


def load_medmemorybench(root: Path) -> dict[str, dict]:
    """中文医患。`answers` 是 JSON 串，只取 `is_correct: true` 的那些。"""
    table: dict[str, dict] = {}
    path = root / "medmemorybench" / "data" / "zh" / "queries.parquet"
    if not path.exists():
        return table
    for row in _parquet_rows(path, columns=["question", "query_id", "query_type", "answers"]):
        try:
            answers = (
                json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
            )
        except json.JSONDecodeError:
            answers = []
        correct = [a["content"] for a in (answers or []) if a.get("is_correct")]
        table.setdefault(
            norm(row["question"]),
            {
                "source_dataset": "medmemorybench",
                "source_locator": {"file": "queries.parquet", "query_id": row["query_id"]},
                "gold_kind": "exact",
                "gold_answers": correct,
                "gold_rubric": None,
                #: 上游裁判**按 `query_type` 分流**（2 条纯函数 + 4 条 LLM）⇒ 必须原样带上，
                #: 只留 `gold_answers` 会让 6 类的题全走错分支（而不会报错）。
                "gold_native": {
                    "query_type": str(row["query_type"]),
                    "answers": answers or [],
                },
                "gold_judging": f"MedMemoryBench（zh）：金标是 answers 里 is_correct=true 的那些"
                f"（query_type = {row['query_type']}）",
            },
        )
    return table


def load_memtrapbench(root: Path) -> dict[str, dict]:
    """记忆陷阱题。`final_trigger` 是题面，`gold_standard` 是**判分要点**（不是答案串）。"""
    table: dict[str, dict] = {}
    base = root / "memtrapbench" / "memtrapbench"
    if not base.is_dir():
        return table
    for path in sorted(base.rglob("*.json")):
        if "_seed" in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            trigger = item.get("final_trigger") if isinstance(item, dict) else None
            if not trigger:
                continue
            table.setdefault(
                norm(trigger),
                {
                    "source_dataset": "memtrapbench",
                    "source_locator": {"file": path.name, "id": item.get("id")},
                    "gold_kind": "rubric",
                    "gold_answers": [],
                    "gold_rubric": item.get("gold_standard"),
                    #: 裁判读 `gold_standard`；`test_type`（red / green）与场景目录只作上下文。
                    "gold_native": {
                        "gold_standard": item.get("gold_standard"),
                        "test_type": item.get("test_type"),
                        "scenario": path.parent.name,
                    },
                    "gold_judging": f"MemTrapBench：{path.parent.name} / "
                    f"test_type={item.get('test_type')}。金标是 gold_standard 那段判分要点——"
                    "判分看的是**有没有抵抗住上下文里的陷阱**，不是答案串相等",
                },
            )
    return table


def load_beam(root: Path) -> dict[str, dict]:
    """长对话记忆。`probing_questions` 是**字符串化的 dict**，得 `ast.literal_eval` 解。

    ⚠ **10 个 probing 组各有各的答案字段**（`answer` / `ideal_answer` / `ideal_response` /
    `ideal_summary`，有的组一个都没有）——**只有 `rubric` 是每组都有**。所以金标以 `rubric` 为准
    （官方 pipeline 也正是"逐条 rubric 三点制"），能顺带取到的答案串放进 `gold_answers`。
    """
    table: dict[str, dict] = {}
    path = root / "beam" / "data" / "100K-00000-of-00001.parquet"
    if not path.exists():
        return table
    for row in _parquet_rows(path, columns=["conversation_id", "probing_questions"]):
        raw = row["probing_questions"]
        try:
            probing = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            continue
        if not isinstance(probing, dict):
            continue
        for group, items in probing.items():
            for item in items if isinstance(items, list) else []:
                question = (item or {}).get("question")
                if not question:
                    continue
                answer_key = next((k for k in _BEAM_ANSWER_KEYS if item.get(k)), None)
                table.setdefault(
                    norm(question),
                    {
                        "source_dataset": "beam",
                        "source_locator": {
                            "file": "100K-00000-of-00001.parquet",
                            "conversation_id": row["conversation_id"],
                            "group": group,
                        },
                        "gold_kind": "rubric",
                        "gold_answers": [item[answer_key]] if answer_key else [],
                        "gold_rubric": item.get("rubric"),
                        #: 官方 `pipeline_beam.py` 是**逐条 rubric 三点制**，逐条打分要的是
                        #: `rubric` 的**整段原文**（可能含多条）。组名与难度只进判分说明。
                        "gold_native": {
                            "rubric": item.get("rubric"),
                            "group": group,
                            "difficulty": item.get("difficulty"),
                        },
                        "gold_judging": f"BEAM（100K）：probing 组 {group}，"
                        f"difficulty={item.get('difficulty')}。"
                        "**金标以 rubric 为准**（官方 pipeline 逐条三点制）"
                        + (
                            f"；答案串取自该组的 `{answer_key}`"
                            if answer_key
                            else "；该组没有答案串字段"
                        )
                        + "。⚠ 组里 abstention 那类是**拒答题**——"
                        "「理想回答」本身就是「记忆里没有」，别把「答不出来」当成错",
                    },
                )
    return table


def load_personamem(root: Path) -> dict[str, dict]:
    """隐式 persona。`user_query` 列是 Python dict 的 repr 串，取里面的 `content`。"""
    table: dict[str, dict] = {}
    path = root / "personamem-v2" / "benchmark" / "text" / "benchmark.csv"
    if not path.exists():
        return table
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("user_query") or ""
            try:
                content = ast.literal_eval(raw).get("content", "") if raw else ""
            except (ValueError, SyntaxError):
                content = ""
            if not content:
                continue
            table.setdefault(
                norm(content),
                {
                    "source_dataset": "personamem-v2",
                    "source_locator": {
                        "file": "text/benchmark.csv",
                        "persona_id": row.get("persona_id"),
                    },
                    "gold_kind": "exact",
                    "gold_answers": [row.get("correct_answer") or ""],
                    "gold_rubric": None,
                    "gold_judging": "PersonaMem-v2：上游 pipeline 对 correct_answer 做"
                    "**精确文本匹配**（另有 incorrect_answers 做干扰项）",
                },
            )
    return table


LOADERS = {
    "mquake-remastered": load_mquake,
    "corporatebench": load_corporatebench,
    "medmemorybench": load_medmemorybench,
    "memtrapbench": load_memtrapbench,
    "beam": load_beam,
    "personamem-v2": load_personamem,
}


def _corpus_index(
    adds_path: Path, users: set[str]
) -> dict[str, dict[str, list[tuple[str, tuple]]]]:
    """把官方 `/add` 里 `Corpus:` 那两族抽成 `user → {"update"/"fact": [(ts, (s, r, o))]}`。

    只留 `users` 里那些人的——采集原文 300+ MB，没必要全留在内存里；坏行一律跳过。
    """
    index: dict[str, dict[str, list[tuple[str, tuple]]]] = {
        u: {"update": [], "fact": []} for u in users
    }
    with adds_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            user = record.get("user_id")
            if user not in index:
                continue
            for message in record.get("messages") or []:
                content = message.get("content")
                if not isinstance(content, str) or not content.startswith("Corpus:"):
                    continue
                kind = "update" if content.startswith("Corpus: UPDATE") else "fact"
                try:
                    body = json.loads(content.split("\n", 1)[-1])
                except (json.JSONDecodeError, IndexError):
                    continue
                triple = (body.get("subject_id"), body.get("relation_id"), body.get("object_id"))
                index[user][kind].append((record["ts"], triple))
    return index


def resolve_mquake(row: dict, index: dict[str, dict[str, list[tuple[str, tuple]]]]) -> None:
    """就地定夺这一条 MQuAKE：**先选定是哪个 case，再定夺答案取新旧哪一侧**。

    1. **选题面撞车的那一个 case**（重名题面的答案实测多不相同）：谁的 `edit_triples`
       在该 user 提问前的 `Corpus: UPDATE` 里出现过，就选谁；没有的话退一步——谁的主体
       出现在提问前的原始事实里，就选谁；再不行才退回第一个候选。
    2. **定答案**：该 case 的改写在提问前注入过 ⇒ 记忆支持 `new_answer`（`updated`）；
       否则记忆里只有原始链 ⇒ 取 `answer`（`original`）。
    """
    cutoff = row["ts"]
    memory = index.get(row["user_id"], {"update": [], "fact": []})
    before = {
        "update": [t for t in memory["update"] if t[0] <= cutoff],
        "fact": [t for t in memory["fact"] if t[0] <= cutoff],
    }

    def updated_hits(candidate: dict) -> int:
        edits = {tuple(e) for e in candidate["gold_edits"]}
        return sum(1 for _ts, triple in before["update"] if triple in edits)

    def fact_hits(candidate: dict) -> int:
        subjects = set(candidate["gold_fact_subjects"])
        return sum(1 for _ts, (_s, _r, _o) in before["fact"] if _s in subjects)

    options = [row, *(row.get("gold_candidates") or [])]
    picked = max(options, key=lambda c: (updated_hits(c), fact_hits(c)))
    injected = updated_hits(picked) > 0
    ambiguous = len(options) > 1 and updated_hits(picked) == 0 and fact_hits(picked) == 0

    row["source_locator"] = picked["source_locator"]
    row["gold_edits"] = picked["gold_edits"]
    if not injected:  # 记忆里只有原始链 ⇒ 金标是更新前那一侧
        picked = {
            **picked,
            "gold_answers": picked["gold_alternative_answers"],
            "gold_alternative_answers": picked["gold_answers"],
        }
    row["gold_answers"] = picked["gold_answers"]
    row["gold_alternative_answers"] = picked["gold_alternative_answers"]
    row["gold_variant"] = "updated" if injected else "original"
    row["gold_judging"] = (
        f"{MQUAKE_JUDGING}⇒ 本条判为 {row['gold_variant']}"
        f"（该 case 的改写在提问前{'已' if injected else '未'}注入"
        + ("；⚠ 题面在题库里有多个 case，**按记忆证据选定的这个**" if len(options) > 1 else "")
        + ("；⚠ 多个候选中没有证据可判，退回第一个" if ambiguous else "")
        + "）"
    )


def load_all(root: Path) -> dict[str, dict]:
    """把 6 张题库合成一张 `{归一化题面: 金标记录}`。**先到先得**，撞了会报出来。"""
    merged: dict[str, dict] = {}
    collisions: list[tuple[str, str, str]] = []
    for slug in SOURCES:
        table = LOADERS[slug](root)
        print(f"  {slug:20s} 题面 {len(table):6d}")
        for question, gold in table.items():
            if question in merged:
                collisions.append((question[:60], merged[question]["source_dataset"], slug))
                continue
            merged[question] = gold
    if collisions:
        print(f"  ⚠ 跨数据集撞题 {len(collisions)} 条（先到先得）：{collisions[:3]}")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir", default="official-dataset-2026-09-29", help="采集导出所在目录"
    )
    parser.add_argument(
        "--benchmark-dir", default=None, help="题库目录（默认取 TIANXIMEM_BENCHMARK_DIR）"
    )
    parser.add_argument(
        "--out", default="official-gold-extra.jsonl", help="输出文件名（写在数据集目录下）"
    )
    parser.add_argument(
        "--adds",
        default=None,
        help="采集的 official-adds.jsonl——给了它才能逐条定夺 MQuAKE 的 gold_variant"
        "（要多扫 300+ MB）；写 auto 表示取数据集目录下那份",
    )
    parser.add_argument("--offline", action="store_true", help="禁止下载缺失的题库材料")
    args = parser.parse_args()

    bench = Path(args.benchmark_dir) if args.benchmark_dir else benchmark_dir()
    dataset_dir = Path(args.dataset_dir)
    searches_path = dataset_dir / "official-searches.jsonl"
    if not searches_path.exists():
        print(f"✗ 找不到 {searches_path}", file=sys.stderr)
        return 1
    from eval.datasets.prepare import ensure_dataset

    for family in SOURCES:
        ensure_dataset(family, bench, offline=args.offline)

    print(f"题库目录 {bench}")
    pool = load_all(bench)
    print(f"合并后题面 {len(pool)}\n")

    rows = []
    per_dataset: Counter[str] = Counter()
    for line in searches_path.open(encoding="utf-8"):
        search = json.loads(line)
        gold = pool.get(norm(search.get("question")))
        if not gold:
            continue
        per_dataset[gold["source_dataset"]] += 1
        rows.append(
            {  # ⚠ 每行一份自己的字典：`resolve_mquake` 会就地换答案，不能改到池子里那份
                "seq": search["seq"],
                "ts": search["ts"],
                "user_id": search["user_id"],
                "question": search["question"],
                **gold,
            }
        )

    # `gold_native` 只是**判分时要原样交给裁判的载荷**；套件（`build_official_kit.py`）
    # 会把它搬进 `official-eval-kit.jsonl`，两份文件之间不需要人手工对齐键名。
    for row in rows:
        row.setdefault("gold_native", None)

    variants: Counter[str] = Counter()
    if args.adds:
        adds_path = Path(args.adds) if args.adds != "auto" else dataset_dir / "official-adds.jsonl"
        mquake_rows = [r for r in rows if r["source_dataset"] == "mquake-remastered"]
        index = _corpus_index(adds_path, {r["user_id"] for r in mquake_rows})
        for row in mquake_rows:
            resolve_mquake(row, index)
            variants[row["gold_variant"]] += 1
        print(f"MQuAKE 逐条定夺（扫过 {adds_path}）：{dict(variants)}")

    out_path = dataset_dir / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # 这两个只服务于"选哪个 case"的过程，不进产物（要回查直接去题库）
            row.pop("gold_candidates", None)
            row.pop("gold_fact_subjects", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"命中 {len(rows)} 条 → {out_path}")
    for slug, count in per_dataset.most_common():
        print(f"  {slug:20s} {count:5d}")
    kinds = Counter(row["gold_kind"] for row in rows)
    print(f"  gold_kind: {dict(kinds)}")
    if not args.adds:
        print("  ⚠ 没给 --adds ⇒ MQuAKE 的 gold_variant 一律是 updated(未核)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
