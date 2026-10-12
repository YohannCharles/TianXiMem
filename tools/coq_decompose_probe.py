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

## 怎么打分

MuSiQue 的 `question_decomposition` 是**金标分解**（hop N 的问句里写 `#k` 字面引用第 k 跳）。
但模型写的子问题必然是**另一种措辞**（它不知道 `#1` 会被填成什么），所以：

* ⛔ **不做整串比对**（那只会测出措辞差异，不是结构差异）；
* ✅ 先用**客观结构量**（零成本、无裁判）：

  | 量 | 定义 |
  | --- | --- |
  | `hop_count_match` | 模型给的步数 == 金标跳数 |
  | `back_ref_steps` | 第 2 步起**回指前面某一步**的步数（靠回指词或与前面步骤共享实词判定） |
  | `final_covers_gold` | **最后一步**是否覆盖金标最后一跳的内容词（"有没有问到点子上"） |
  | `gold_word_hit` | 金标每一跳的关键词在模型**对应位置**那一步里出现的比例 |

* ✅ 再用**一次 LLM 裁判**判"这两条分解是不是同一条依赖链、顺序对不对"
  （裁判只看到**问题 + 两条分解**，⛔ 看不到检索结果）。

⚠ 两套都给，是因为客观量**便宜但只能看形状**，而裁判能看语义——
**只报其中一个都会被读歪**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

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

#: 裁判看到的：**只有**问题与两条分解（⛔ 不给检索结果、不给别的题）。
JUDGE_PROMPT = """You compare two decompositions of the same multi-hop question.

Each decomposition is an ORDERED list of single-fact sub-questions. The question is \
answerable only by resolving them in order, each step using the previous step's answer.

Judge whether the CANDIDATE captures the SAME dependency chain as the REFERENCE:
1. same number of required steps (a step that is missing, or an extra step that the \
question does not require, is a defect);
2. same ORDER — step k of the candidate must target the same thing as step k of the \
reference;
3. each step must be a single lookup, not a bundle of several lookups.

Content words may be paraphrased. Do NOT reward or penalise wording.

Output ONLY a JSON object: {"verdict": "SAME" | "DIFFERENT", "reason": "<one sentence>"}

Question: {QUESTION}

REFERENCE:
{REFERENCE}

CANDIDATE:
{CANDIDATE}
"""

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
    "a", "an", "the", "of", "in", "on", "at", "to",
    "for", "from", "by", "with", "and", "or", "is", "are",
    "was", "were", "be", "been", "being", "what", "which", "who",
    "whom", "whose", "when", "where", "why", "how", "did", "does",
    "do", "done", "has", "have", "had", "that", "this", "it",
    "its", "their", "his", "her", "they", "them", "he", "she",
    "as", "into", "during", "before", "after", "before", "than", "then",
    "name", "named", "called", "one", "two", "three", "not", "no",
    "any", "all", "some", "more", "most", "other", "others", "same",
    ]
)


#: 只认**全大写**的占位符——于是提示词里那些字面 JSON（`{"verdict": …}`）不会被当成占位符。
_PLACEHOLDER = re.compile(r"\{[A-Z][A-Z_]*\}")


def fill(template: str, **values: str) -> str:
    """把 `{NAME}` 一遍扫描地填上。

    ⛔ **不用 `str.format`**：`JUDGE_PROMPT` 里本身写着字面 JSON
    （`{"verdict": "SAME" | "DIFFERENT"}`），`format` 会把它当成占位符抛
    `KeyError: '"verdict"'`——**实测踩过一次**。
    ⚠ 也**不用连续 `str.replace`**：填进去的 `CANDIDATE` 是**模型输出**（不可信），
    里面若恰好有 `{REFERENCE}` 会在下一轮被替掉。一遍扫描从根上排除这件事。
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


def _render(gold: list[str], candidate: list[str]) -> tuple[str, str]:
    fmt = lambda steps: "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))  # noqa: E731
    return fmt(gold), fmt(candidate)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="只看问题的多跳分解能力探针")
    parser.add_argument("--manifest", default="eval/coq-musique.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bench-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    parser.add_argument("--no-judge", action="store_true", help="只跑分解，不叫裁判")
    args = parser.parse_args(argv)

    import os

    bench_dir = Path(args.bench_dir or os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from eval.datasets.musique import DATA_DIR as MUSIQUE_DIR
    from eval.datasets.musique import JSONL as MUSIQUE_JSONL

    from tianximem.common.config import load_config

    wanted = [
        json.loads(line)["question_id"]
        for line in Path(args.manifest).read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if args.limit is not None:
        wanted = wanted[: args.limit]

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

    config = load_config()
    client = build_client(config)
    print(f"清单 {args.manifest}：{len(wanted)} 题（**只给问题那一句**）\n隔离产物：{out_dir}")

    results: list[dict] = []
    total = len(wanted)
    for index, qid in enumerate(wanted, 1):
        record = rows[qid]
        gold_hops = [
            str(hop["question"]) for hop in record["question_decomposition"]
        ]
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
        if steps is not None and not args.no_judge:
            reference, candidate = _render(gold_hops, steps)
            try:
                verdict_raw = client.complete(
                    fill(
                        JUDGE_PROMPT, QUESTION=question, REFERENCE=reference, CANDIDATE=candidate
                    ),
                    max_tokens=200,
                    temperature=0.0,
                )
                entry["judge"] = _parse_verdict(verdict_raw)
            except Exception as exc:  # noqa: BLE001
                entry["judge"] = {"verdict": None, "reason": f"{type(exc).__name__}: {exc}"}
        results.append(entry)
        print(
            f"  [{index}/{total}] {qid}  {status}  步数 {entry['shape']['steps']}/"
            f"{entry['shape']['gold_steps']}  回指 {entry['shape']['back_ref_steps']}  "
            f"裁判 {entry.get('judge', {}).get('verdict')}",
            flush=True,
        )
        (out_dir / "rows.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
            encoding="utf-8",
        )
        time.sleep(0)  # 让 flush 有个落点；网关侧自己有限流

    report = _summarize(results)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n════ 汇总 ════")
    for key, value in report.items():
        if key != "per_question":
            print(f"  {key}: {value}")
    print(f"\n产物：{out_dir}（report.json / rows.jsonl —— ⚠ 含数据集原文，不进 Git）")
    return 0


def _parse_verdict(text: str) -> dict:
    for blob in (text.strip(), *_balanced_objects(text)):
        try:
            payload = json.loads(blob)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("verdict") in ("SAME", "DIFFERENT"):
            return {
                "verdict": payload["verdict"],
                "reason": str(payload.get("reason", ""))[:300],
            }
    return {"verdict": None, "reason": "裁判输出里没有 SAME/DIFFERENT"}


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    parsed = [item for item in results if item["status"] == "ok"]
    judged = [item for item in results if item.get("judge", {}).get("verdict")]
    same = [item for item in judged if item["judge"]["verdict"] == "SAME"]
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
        "judged": len(judged),
        "judge_same": len(same),
        "judge_same_rate": round(len(same) / max(1, len(judged)), 3),
        "per_question": {
            item["question_id"]: {
                "status": item["status"],
                "steps": item["shape"]["steps"],
                "gold_steps": item["shape"]["gold_steps"],
                "verdict": item.get("judge", {}).get("verdict"),
            }
            for item in results
        },
    }


if __name__ == "__main__":
    sys.exit(main())
