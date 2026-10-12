"""**只看问题**能不能把链式多跳题拆成有序的单跳子问题？——分解能力探针。

```bash
uv run --env-file .env python tools/coq_decompose_probe.py \
    --manifest eval/coq-musique.jsonl --out eval/reports/runs/coq-decompose
```

## 它回答的是哪一个问题

上一轮（[`../eval/reports/coq-musique-20261011.md`](../eval/reports/coq-musique-20261011.md) §4）
把 CoQ 的失败**定位到"第二轮提不出新查询"**：第 1 轮它改写得完全正确（把中间实体解了出来），
第 2 轮原样重复。当时留下了一个**没有分清**的岔路口：

> ① 第 2 轮的**文档里**确实没有下一跳的线索（**检索**问题）
> ② 提示词没逼它给出不同的查询（**提示词**问题）

本工具从**另一头**回答它：**把检索整个拿掉**，只给模型那句问题，看它能不能自己把依赖链拆出来。

| 结果 | 说明 |
| --- | --- |
| **拆得出来** | ⇒ 失败在**检索侧**（模型知道该问什么，但文档里没有那个线索） |
| **拆不出来** | ⇒ 失败在**提示词/模型侧**，而"第二轮重复"只是它的下游症状 |

⚠ 本工具**只给问题那一句**：不给检索结果、不给金标、不给分解示例。
它测的是**模型自己会不会做多跳分解**，不是"CoQ 那套循环跑得好不好"。

## 怎么打分：客观量 + **人来判**

MuSiQue 的 `question_decomposition` 是**金标分解**（hop N 的问句里写 `#k` 字面引用第 k 跳）。
但模型写的子问题必然是**另一种措辞**（它不知道 `#1` 会被填成什么），所以：

1. ⛔ **不做整串比对**（那只会测出措辞差异，不是结构差异）；
2. ✅ **客观结构量**（零成本、无裁判）：

   | 量 | 定义 |
   | --- | --- |
   | `hop_count_match` | 模型给的步数 == 金标跳数 |
   | `back_ref_steps` | 第 2 步起**回指前面某一步**的步数（靠回指词或与前面步骤共享实词判定） |
   | `final_covers_gold` | **最后一步**是否覆盖金标最后一跳的内容词（"有没有问到点子上"） |

3. ✅ **判定"这两条分解是不是同一条依赖链"这一步由人做**——填
   [`../eval/coq-decompose-labels.jsonl`](../eval/coq-decompose-labels.jsonl)（一题一行，
   只存 `question_id` / `defect` / `note`，⛔ **不存题面**）。
   本工具**只汇总，不代填**；缺标签就只报客观量并直说"未判 N 题"。

## 产物三份（各给一种读者）

| 文件 | 谁看 / 干什么 |
| --- | --- |
| `rows.jsonl` | 机器：每题一条，含**题面与模型原始输出**（⚠ 数据集原文 ⇒ gitignored） |
| `report.json` | 机器：客观结构量 + 标签汇总 |
| **`decompositions.md`** | **人**：原问题 / 模型拆解 / 金标拆解**并排**，逐条复核时直接复制 |

### ⛔ 为什么**不要**让模型来判（2026-10-12 实测，这条是踩过才写的）

原实现是"再叫一次对话模型当裁判"。逐条核对下来它**至少错判 4/25**，而且是**事实性错误**：

| 题 | 模型裁判说 | 实际 |
| --- | --- | --- |
| `4hop1__159467…` | "金标只需要三步" | 金标是**四跳**（`hops=4`），它**没数** |
| `4hop1__105773…` | "候选多了一步 embassy 查找" | 金标 G2 **就是**那一步 |
| `4hop1__43565…` | "候选多了一步 city 查找" | 金标 G2 **就是**那一步 |
| `4hop1__726391…` | "候选跳过了 performer 的 city" | 金标 G2 的答案本来就是**州** |

四个错判同一来源：**裁判看不到每跳的答案**（提示词里刻意不给），于是它只能猜
"哪一步是必要的"——而"必要"恰恰是要拿金标的**图**去比的。
它另外还与被测模型**同源**（同一网关的同一个对话模型）。
⇒ 它给出的 `judge_same = 9/25` **已作废**，不再进任何指标。

> 人的判据（写进 label 的 `defect`，`DEFECTS` 是**封闭词表**）：
> `none` / `bundled` / `restated` / `underspecified` / `oversplit` / `skipped` / `extra`。
> **verdict 由 defect 推出**（`none` ⇒ SAME），不单独存一个字段——
> 少一个可以自相矛盾的字段。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 提示词：给模型的分解指令（⛔ 不许出现金标 / 不许举例用到本题）─────────────
DECOMPOSE_PROMPT = """You are given ONE question. It may require several chained lookup steps \
to answer: a later part of the question can only be asked after an earlier part has been \
resolved to a concrete entity.

Break the question into the MINIMAL ordered list of single-fact sub-questions needed to \
answer it, in dependency order.

Rules:
- Each item must be answerable by ONE lookup.
- Step 1 must not depend on any other step.
- If a step needs the result of an earlier step, refer to that result by description \
(e.g. "the person found in step 1"), because the actual answer is not known yet.
- Do NOT answer any step. Do NOT add steps that are not required by the question.
- Keep every entity that the question already names.

Output ONLY a JSON object with exactly one key "sub_questions", whose value is an array of \
strings in dependency order.

Question: {QUESTION}
"""

#: **人的判据**（⛔ 封闭词表：归一化到这里，否则每次重判都得到另一套说法）。
#: `verdict` 由它推出——`("none",)` ⇒ SAME，其余 ⇒ DIFFERENT。
DEFECTS: Final[tuple[str, ...]] = (
    "none",  # 链条与金标同构
    "bundled",  # 某一步里有 ≥2 次查找（两头都并：把两跳并成一步）
    "restated",  # 末步只是把原问题重复一遍（链没有终止在一次查找上）
    "underspecified",  # 某步丢了题面里的一个**唯一化**限定 ⇒ 该步查到的东西不唯一
    "oversplit",  # 比金标多拆（把一跳拆成两步）
    "skipped",  # 漏掉金标某一跳
    "extra",  # 多出题面不要求的一步
)

#: 标签文件的落点。**它提交进 git**（与 `eval/router-samples.jsonl` 同一条纪律：
#: 只存 id 与判据，⛔ 不存题面）。
DEFAULT_LABELS: Final[Path] = Path("eval/coq-decompose-labels.jsonl")

#: 回指词——第 2 步起用它指代前面某一步的结果，是"这是一条链"的最直接痕迹。
_BACK_REF = re.compile(
    r"\b(that|this|those|these|the same|it|its|his|her|their|the one|"
    r"aforementioned|previously|earlier|mentioned|found in step|from step|the person|"
    r"the place|the book|the event|the company|the country|the city)\b",
    re.IGNORECASE,
)

#: 太常见的词不算"实词"，否则任何两句都会共享它。
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "did",
        "does",
        "do",
        "done",
        "has",
        "have",
        "had",
        "that",
        "this",
        "it",
        "its",
        "their",
        "his",
        "her",
        "they",
        "them",
        "he",
        "she",
        "as",
        "into",
        "during",
        "before",
        "after",
        "before",
        "than",
        "then",
        "name",
        "named",
        "called",
        "one",
        "two",
        "three",
        "not",
        "no",
        "any",
        "all",
        "some",
        "more",
        "most",
        "other",
        "others",
        "same",
    ]
)


#: 只认**全大写**的占位符——于是提示词里那些字面 JSON（`{"verdict": …}`）不会被当成占位符。
_PLACEHOLDER = re.compile(r"\{[A-Z][A-Z_]*\}")


def fill(template: str, **values: str) -> str:
    """把 `{NAME}` 一遍扫描地填上。

    ⛔ **不用 `str.format`**：提示词里写着**字面 JSON**（`{"sub_questions": [...]}`），
    `format` 会把它当成占位符抛 `KeyError`——**实测踩过一次**（当时那句是裁判提示词，
    现在只剩 `DECOMPOSE_PROMPT`，但字面 JSON 还在）。
    ⚠ 也**不用连续 `str.replace`**：填进去的值可能是**模型输出**（不可信），
    里面若恰好有另一个 `{SLOT}` 会在下一轮被替掉。一遍扫描从根上排除这件事。
    """
    slots = {f"{{{name}}}": value for name, value in values.items()}
    return _PLACEHOLDER.sub(lambda match: slots.get(match.group(0), match.group(0)), template)


def _key_words(text: str) -> set[str]:
    """内容词集合（小写、去停用词、只留字母数字串）。"""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", str(text).lower())
    return {word for word in words if word not in _STOP and len(word) > 2}


def parse_sub_questions(text: str) -> list[str] | None:
    """从模型输出里取那个数组。**解析失败返回 `None`**（由调用方记成一次失败）。

    与 CoQ 的解析同一套纪律：先试整串，再取最后一个括号平衡的对象；
    形状不对就**不猜**。⚠ 这里**不**做"从散文里猜步骤"的兜底——
    那会把"模型没按格式答"变成"模型拆错了"，两种失败要分得开。
    """
    candidates: list[str] = []
    stripped = text.strip()
    for blob in (stripped, *_balanced_objects(text)):
        try:
            payload = json.loads(blob)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get("sub_questions")
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            candidates = [item.strip() for item in value if item.strip()]
            break
    return candidates or None


def _balanced_objects(text: str) -> list[str]:
    """扫出所有括号平衡的 `{...}`（同上一次实现：**字符串里的大括号与转义要正确处理**）。"""
    spans, depth, start, in_string, escaped = [], 0, -1, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : index + 1])
    return list(reversed(spans))


def score_shape(candidate: list[str], gold_hops: list[str]) -> dict:
    """**零成本的客观结构量**。⚠ 它们只量"形状"，语义归裁判。"""
    steps = len(candidate)
    back_refs = 0
    for position in range(1, steps):
        own = _key_words(candidate[position])
        earlier: set[str] = set()
        for previous in candidate[:position]:
            earlier |= _key_words(previous)
        if _BACK_REF.search(candidate[position]) or (own & earlier):
            back_refs += 1
    gold_last = _key_words(gold_hops[-1]) if gold_hops else set()
    final_covers = bool(gold_last & _key_words(candidate[-1])) if candidate else False
    return {
        "steps": steps,
        "gold_steps": len(gold_hops),
        "hop_count_match": steps == len(gold_hops),
        "back_ref_steps": back_refs,
        "back_ref_rate": round(back_refs / max(1, steps - 1), 3),
        "final_covers_gold": final_covers,
    }


def build_client(config):
    """一个**通用**的对话客户端。⚠ 本探针不是路由器、也不是执行器，所以不走它们各自的构造器。"""
    from tianximem.llm import OpenAiCompatChat

    return OpenAiCompatChat(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        timeout=config.agentic.timeout_seconds,
    )


# ── 人的判据：读进来、汇总（⛔ 机器不代填）───────────────────────────────


def load_label_rows(path: Path) -> dict[str, dict]:
    """读标签文件 → `{question_id: 整行}`。**形状不对就响亮失败**。

    ⚠ 坏行**不许静默跳过**（少一条 = 分母变小 = 分数悄悄变高，而屏幕上完全正常）。
    一个 id 出现两次同理——那说明两份判读冲突了，挑哪一份都是擅自决定。
    """
    rows: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"✗ {path}:{number} 不是 JSON：{exc}") from exc
        question_id, defect = row.get("question_id"), row.get("defect")
        if not isinstance(question_id, str) or not question_id:
            raise SystemExit(f"✗ {path}:{number} 缺 question_id")
        if defect not in DEFECTS:
            raise SystemExit(f"✗ {path}:{number} 的 defect={defect!r} 不在词表里：{list(DEFECTS)}")
        if question_id in rows:
            raise SystemExit(f"✗ {path}:{number} 的 {question_id} 判了两次")
        rows[question_id] = row
    return rows


def load_labels(path: Path) -> dict[str, str]:
    """`{question_id: defect}`——**判定只用这个词表**（校验在 `load_label_rows`）。"""
    return {qid: row["defect"] for qid, row in load_label_rows(path).items()}


def score_labels(question_ids: list[str], labels: dict[str, str]) -> dict:
    """把标签汇总成比率 + **缺陷分布**。缺标签 ⇒ 报出是哪几道，**不拿未判的当对或当错**。"""
    missing = [qid for qid in question_ids if qid not in labels]
    judged = [qid for qid in question_ids if qid in labels]
    same = [qid for qid in judged if labels[qid] == "none"]
    counts: dict[str, int] = {}
    for qid in judged:
        counts[labels[qid]] = counts.get(labels[qid], 0) + 1
    return {
        "judged": len(judged),
        "unjudged": len(missing),
        "same": len(same),
        "same_rate": round(len(same) / max(1, len(judged)), 3),
        "defect_counts": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "wrong_ids": [qid for qid in question_ids if qid in labels and labels[qid] != "none"],
        "unjudged_ids": missing,
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="只看问题的多跳分解能力探针")
    parser.add_argument("--manifest", default="eval/coq-musique.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bench-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="人的判定（缺了只报客观量）")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="复用 --out/rows.jsonl 里已有的分解，只重新汇总（⛔ 不调模型）",
    )
    args = parser.parse_args(argv)

    import os

    bench_dir = Path(args.bench_dir or os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [
        json.loads(line)["question_id"]
        for line in Path(args.manifest).read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if args.limit is not None:
        wanted = wanted[: args.limit]

    if args.reuse:
        # ⭐ **人在回路里重判不该再摇一次模型**：换一个模型调用就是换一份候选分解，
        # 而标签是逐条读着**那一份**候选填的 ⇒ 重跑会让标签静默对不上它判的东西。
        results = _load_rows(out_dir / "rows.jsonl", wanted)
        print(f"清单 {args.manifest}：{len(wanted)} 题\n复用已有分解：{out_dir / 'rows.jsonl'}")
    else:
        results = _decompose_all(args, bench_dir, out_dir, wanted)

    report = _summarize(results)
    labels_path = Path(args.labels)
    label_rows = load_label_rows(labels_path) if labels_path.exists() else {}
    if label_rows:
        report["labels"] = score_labels(
            [item["question_id"] for item in results], load_labels(labels_path)
        )
    else:
        report["labels"] = {
            "judged": 0,
            "unjudged": len(results),
            "note": f"没有 {labels_path}",
        }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # ⭐ 人读的那一份（原问题 / 模型拆解 / 金标拆解并排）——供复制与逐条复核。
    #    ⚠ 含数据集原文 ⇒ 只落在 runs/（gitignored），**不进 git**。
    dumped = out_dir / "decompositions.md"
    dumped.write_text(
        render_markdown(results, label_rows, _gold_hop_answers(bench_dir, wanted)),
        encoding="utf-8",
    )
    print("\n════ 客观结构量（机器算的）════")
    for key, value in report.items():
        if key not in ("per_question", "labels", "wrong_ids", "unjudged_ids"):
            print(f"  {key}: {value}")
    _print_labels(report["labels"], labels_path)
    print(f"\n产物：{out_dir}（report.json / rows.jsonl —— ⚠ 含数据集原文，不进 Git）")
    print(f"     {dumped}（**人读的那一份**：原问题 / 模型拆解 / 金标拆解）")
    print(f"     {labels_path}（人的判定 —— ⛔ 机器不代填）")
    return 0


def _load_rows(path: Path, wanted: list[str]) -> list[dict]:
    """读回上一次的分解结果。**少一道题就响亮失败**（分母悄悄变小 = 分数悄悄变高）。"""
    if not path.exists():
        raise SystemExit(f"✗ --reuse 要的是已有的分解，但 {path} 不存在")
    results = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    got = {item["question_id"] for item in results}
    missing = [qid for qid in wanted if qid not in got]
    if missing:
        raise SystemExit(
            f"✗ {path} 里缺这些题（先跑一次不带的）：{missing[:3]}（共 {len(missing)} 个）"
        )
    return results


def _decompose_all(
    args: argparse.Namespace, bench_dir: Path, out_dir: Path, wanted: list[str]
) -> list[dict]:
    """逐题叫模型拆一次。⚠ 这是**唯一**花钱的那条路（每题一次调用）。"""
    from eval.datasets.musique import DATA_DIR as MUSIQUE_DIR
    from eval.datasets.musique import JSONL as MUSIQUE_JSONL

    from tianximem.common.config import load_config

    rows: dict[str, dict] = {}
    for line in (bench_dir / MUSIQUE_DIR / MUSIQUE_JSONL).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        # ⚠ **只收可答那一条**：同一个 id 在文件里出现两次（可答 / 不可答变体）。
        if record.get("answerable") and str(record["id"]) in set(wanted):
            rows[str(record["id"])] = record
    missing = [qid for qid in wanted if qid not in rows]
    if missing:
        raise SystemExit(f"✗ 清单里这些 id 不在可答题里：{missing[:3]}")

    client = build_client(load_config())
    print(f"清单 {args.manifest}：{len(wanted)} 题（**只给问题那一句**）\n隔离产物：{out_dir}")

    results: list[dict] = []
    total = len(wanted)
    for index, qid in enumerate(wanted, 1):
        record = rows[qid]
        gold_hops = [str(hop["question"]) for hop in record["question_decomposition"]]
        question = str(record["question"])

        try:
            raw = client.complete(
                fill(DECOMPOSE_PROMPT, QUESTION=question), max_tokens=512, temperature=0.0
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001 — 探针：端点坏了要记成一次失败，不是崩掉
            raw, status = f"{type(exc).__name__}: {exc}", "llm_error"

        steps = parse_sub_questions(raw) if status == "ok" else None
        if status == "ok" and steps is None:
            status = "unparsable"
        entry = {
            "question_id": qid,
            "status": status,
            "question": question,
            "gold_hops": gold_hops,
            "candidate_hops": steps or [],
            "raw": raw,
            "shape": score_shape(steps or [], gold_hops),
        }
        results.append(entry)
        print(
            f"  [{index}/{total}] {qid}  {status}  步数 {entry['shape']['steps']}/"
            f"{entry['shape']['gold_steps']}  回指 {entry['shape']['back_ref_steps']}",
            flush=True,
        )
        (out_dir / "rows.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
            encoding="utf-8",
        )
        time.sleep(0)  # 让 flush 有个落点；网关侧自己有限流
    return results


def _gold_hop_answers(bench_dir: Path, wanted: list[str]) -> dict[str, list[str]]:
    """`qid → 每一跳的金标答案`——**只给"渲染成人读的那一份"用**。

    ⚠ 判定**不看**它：人的判据在 `coq-decompose-labels.jsonl` 里（那里也记了逐条理由）。
    这里只是让人在同一个屏幕上看"模型拆到了哪一步"和"金标那一跳的答案是什么"。
    ⚠ 读不到（数据集不在本机）就返回空表，渲染时省掉那半列——**不报错**：
    它是**显示**，不是判据。
    """
    from eval.datasets.musique import DATA_DIR as MUSIQUE_DIR
    from eval.datasets.musique import JSONL as MUSIQUE_JSONL

    path = bench_dir / MUSIQUE_DIR / MUSIQUE_JSONL
    if not path.exists():
        return {}
    answers: dict[str, list[str]] = {}
    wanted_set = set(wanted)
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        # ⚠ 只收**可答**那一条：同一个 id 在文件里出现两次（可答 / 不可答变体）。
        if record.get("answerable") and str(record["id"]) in wanted_set:
            answers[str(record["id"])] = [
                str(hop.get("answer", "")) for hop in record["question_decomposition"]
            ]
    return answers


def render_markdown(
    results: list[dict], labels: dict[str, dict], gold_answers: dict[str, list[str]]
) -> str:
    """人读的那一份：**原问题 / 模型拆解 / 金标拆解**三块并排（供复制与复核）。

    ⚠ 它含**数据集原文**⇒ 只写进 `runs/`（gitignored），⛔ 绝不进 git。
    """
    parts = [
        "# 多跳分解：原问题 / 模型拆解 / 金标拆解",
        "",
        "> 数据来源 `rows.jsonl`（⚠ 含数据集原文，⛔ 不进 git）",
        "> 人的判定在 `eval/coq-decompose-labels.jsonl`；`defect == none` ⇒ 结构同构",
        "> 模型只拿到**那一句问题**：没有检索结果，也没有金标。",
        "",
    ]
    same = 0
    for index, row in enumerate(results, 1):
        qid = row["question_id"]
        label = labels.get(qid, {})
        defect = label.get("defect", "未判")
        same += defect == "none"
        mark = "✅" if defect == "none" else ("❓" if defect == "未判" else "❌")
        parts += [f"## {index}. `{qid}`　{mark} 判定：`{defect}`", "", "**原问题**", ""]
        parts += [f"> {row['question']}", ""]
        candidate = row["candidate_hops"]
        parts += [f"**模型拆解（{len(candidate)} 步）**", ""]
        parts += [f"{k}. {step}" for k, step in enumerate(candidate, 1)] or ["（没解析出来）"]
        answers = gold_answers.get(qid, [])
        parts += ["", f"**金标拆解（{len(row['gold_hops'])} 跳）**", ""]
        parts += [
            f"{k}. {hop}" + (f"　→　`{answers[k - 1]}`" if k <= len(answers) else "")
            for k, hop in enumerate(row["gold_hops"], 1)
        ]
        if defect not in ("none", "未判") and label.get("note"):
            parts += ["", f"**判据**：{label['note']}"]
        parts += ["", "---", ""]
    parts += [f"## 合计：结构同构 {same}/{len(results)}", ""]
    return "\n".join(parts)


def _print_labels(labels: dict, path: Path) -> None:
    print("\n════ 人的判定（`defect`；verdict 由它推出）════")
    if not labels.get("judged"):
        print(f"  ⚠ 未判：{labels.get('unjudged', 0)} 题（{labels.get('note', path)}）")
        print(f"  ⇒ 逐条读 {Path(path)} 那一行行地填 `question_id` / `defect` / `note`。")
        return
    print(
        f"  判了 {labels['judged']} 题：SAME {labels['same']}（{labels['same_rate']}）"
        f" · DIFFERENT {labels['judged'] - labels['same']}"
    )
    print(f"  缺陷分布：{labels['defect_counts']}")
    if labels["unjudged"]:
        print(f"  ⚠ 还有 {labels['unjudged']} 题未判：{labels['unjudged_ids']}")


def _summarize(results: list[dict]) -> dict:
    """**只汇总机器算得出来的东西**——判定那半边在 `score_labels`（读人的标签）。"""
    total = len(results)
    parsed = [item for item in results if item["status"] == "ok"]
    return {
        "n": total,
        "parsed": len(parsed),
        "unparsable": sum(1 for item in results if item["status"] == "unparsable"),
        "llm_error": sum(1 for item in results if item["status"] == "llm_error"),
        "hop_count_match": sum(1 for item in parsed if item["shape"]["hop_count_match"]),
        "steps_exact_rate": round(
            sum(1 for item in parsed if item["shape"]["hop_count_match"]) / max(1, len(parsed)), 3
        ),
        "back_ref_rate_mean": round(
            sum(item["shape"]["back_ref_rate"] for item in parsed) / max(1, len(parsed)), 3
        ),
        "final_covers_gold": sum(1 for item in parsed if item["shape"]["final_covers_gold"]),
        "per_question": {
            item["question_id"]: {
                "status": item["status"],
                "steps": item["shape"]["steps"],
                "gold_steps": item["shape"]["gold_steps"],
            }
            for item in results
        },
    }


if __name__ == "__main__":
    sys.exit(main())
