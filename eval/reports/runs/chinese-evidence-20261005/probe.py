"""中文显式事实：手工片段预检 → HTTP Add/Search 定向验证；不改答案链。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import ServiceClient, pipeline_for, run_judge  # noqa: E402

from tianximem.common.config import _read_env_file  # noqa: E402
from tianximem.common.render import render_evidence  # noqa: E402

PEOPLE = ["张岚", "王川", "林悦"]
DATE = "2025-09-01"
STAMP = 1756684800000


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare() -> dict:
    groups = []
    for name, org, role, verb in [
        ("company", "北辰科技", "员工", "工作"),
        ("school", "北辰学校", "学生", "就读"),
        ("clinic", "青禾诊所", "患者", ""),
        ("volunteers", "海港救助站", "志愿者", "做志愿者"),
    ]:
        sources = [f"{person}：我是{org}的{role}。" for person in PEOPLE]
        if verb:
            sources[0] = f"张岚：我目前在{org}{verb}。"
        if name == "company":
            sources[2] = f"林悦: I work for {org}."
        sources += [
            f"王川：我仍然是{org}的{role}。",
            f"陈明：我计划成为{org}的{role}。",
            f"周舟：我不是{org}的{role}。",
            f"赵然：示例说：“陈明是{org}的{role}。”",
        ]
        manual = []
        for slot, person in enumerate(PEOPLE):
            quote = sources[slot].split("：", 1)[-1]
            if slot == 2 and name == "company":
                quote = sources[slot].split(": ", 1)[1]
            manual.append(
                {
                    "source_slot": slot,
                    "quote": quote,
                    "statement": f"{person}是{org}的{role}。",
                    "content": render_evidence(
                        statement=f"{person}是{org}的{role}。",
                        source_date=DATE,
                        source_quote=quote,
                    ),
                }
            )
        questions = [
            ("list", f"{org}有哪些{role}？", "List[str]", PEOPLE),
            ("count", f"{org}的{role}一共有多少人？", "int", 3),
        ]
        if verb:
            questions += [
                ("list-paraphrase", f"请列出在{org}{verb}的人。", "List[str]", PEOPLE),
                ("count-paraphrase", f"在{org}{verb}的人有多少？", "int", 3),
            ]
        else:
            questions += [
                ("list-paraphrase", f"请列出{org}的{role}。", "List[str]", PEOPLE),
                ("count-paraphrase", f"{org}有多少名{role}？", "int", 3),
            ]
        groups.append({"name": name, "sources": sources, "manual": manual, "questions": questions})
    for name, container, items in [
        ("backpack", "蓝色背包", [("笔记本", 2, "本"), ("笔", 3, "支"), ("尺子", 1, "把")]),
        ("crate", "10升灰色箱子", [("杯子", 4, "个"), ("盘子", 2, "只"), ("水壶", 1, "个")]),
    ]:
        numbers = {1: "一", 2: "两", 3: "三", 4: "四"}
        clause = "、".join(f"{numbers[q]}{unit}{item}" for item, q, unit in items)
        claim = f"我的{container}里有{clause}。"
        sources = [
            f"林悦：{claim}",
            f"林悦：{claim}",
            f"王川：我的{container}里有九支笔。",
            f"林悦：我计划往{container}里再放四支笔，还没有买。",
            f"陈明：示例说：“林悦的{container}里有八把尺子。”",
        ]
        manual = [
            {
                "source_slot": 0,
                "quote": claim,
                "statement": f"林悦的{container}里有{q}{unit}{item}。",
                "content": render_evidence(
                    statement=f"林悦的{container}里有{q}{unit}{item}。",
                    source_date=DATE,
                    source_quote=claim,
                    quantitative=True,
                ),
            }
            for item, q, unit in items
        ]
        groups.append(
            {
                "name": name,
                "sources": sources,
                "manual": manual,
                "questions": [
                    (
                        "list",
                        f"林悦的{container}里有哪些物品？",
                        "List[str]",
                        [i[0] for i in items],
                    ),
                    (
                        "count",
                        f"林悦的{container}里总共有多少件物品？",
                        "int",
                        sum(i[1] for i in items),
                    ),
                    (
                        "list-paraphrase",
                        f"请列出林悦的{container}里的物品种类。",
                        "List[str]",
                        [i[0] for i in items],
                    ),
                    (
                        "count-paraphrase",
                        f"林悦的{container}中有几件东西？",
                        "int",
                        sum(i[1] for i in items),
                    ),
                ],
            }
        )
    for group in groups:
        for fragment in group["manual"]:
            assert fragment["quote"] in group["sources"][fragment["source_slot"]]
    return json.loads(
        json.dumps(
            {
                "groups": groups,
                "answer_pipeline_sha256": hashlib.sha256(
                    (ROOT / "eval/harness/corporatebench_pipeline.py").read_bytes()
                ).hexdigest(),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "manual", "baseline", "candidate"))
    parser.add_argument("--base-url")
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("--layout", choices=("original", "object-first"), default="original")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--label", help="新上下文必须使用独立阶段目录，不能复用旧答案")
    args = parser.parse_args()
    for key, value in _read_env_file(ROOT / ".env").items():
        os.environ.setdefault(key, value)
    manifest = prepare()
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    write(path, manifest)
    if args.phase == "prepare":
        return
    phase_dir = args.label or (
        args.phase + ("-" + args.layout if args.layout != "original" else "")
    )
    rows = []
    for group in manifest["groups"]:
        if args.groups and group["name"] not in args.groups:
            continue
        contexts = {}
        if args.phase == "manual":
            fragments = []
            for f in group["manual"]:
                match = re.fullmatch(r"(.+)里有(\d+)([本支把个只])(.+)。", f["statement"])
                if args.layout == "object-first" and match:
                    fragments.append(
                        render_evidence(
                            statement=f"{match[1]}包含{match[4]}，数量为{match[2]}{match[3]}。",
                            source_date=DATE,
                            source_quote=f["quote"],
                            quantitative=True,
                        )
                    )
                else:
                    fragments.append(f["content"])
            context = "\n".join(fragments)
            contexts = {tag: context for tag, *_ in group["questions"]}
            write(OUT / phase_dir / group["name"] / "context.json", fragments)
        else:
            assert args.base_url
            with ServiceClient(args.base_url, timeout=180) as client:
                client.add(
                    request_id=f"chinese-{group['name']}",
                    user_id=f"chinese-{group['name']}",
                    session_id="s",
                    messages=[
                        {"role": "user", "content": t, "timestamp": STAMP} for t in group["sources"]
                    ],
                )
                for tag, query, *_ in group["questions"]:
                    hits = client.search(user_id=f"chinese-{group['name']}", query=query, top_k=100)
                    contexts[tag] = "\n".join(h.content for h in hits)
                    search_path = OUT / phase_dir / group["name"] / tag / "search.json"
                    search_row = {"query": query, "hits": [asdict(h) for h in hits]}
                    if search_path.exists():
                        assert json.loads(search_path.read_text()) == search_row, (
                            "上下文变化须用新 label"
                        )
                    write(search_path, search_row)
        for tag, query, dtype, gold in group["questions"]:
            qid = f"zh-{group['name']}-{tag}"
            item = {
                "id": qid,
                "dataset": "corporatebench",
                "question": query,
                "answer_type": dtype,
                "gold_answer": gold,
                "category": dtype,
                "retrieved_context": contexts[tag],
            }
            row = {
                "qid": qid,
                "query": query,
                "context_sha256": hashlib.sha256(contexts[tag].encode()).hexdigest(),
            }
            if args.phase == "manual" or args.answer:
                result = run_judge(
                    pipeline_for(benchmark_dir(), "corporatebench"),
                    [item],
                    OUT / phase_dir / group["name"] / tag,
                    dataset="corporatebench",
                )[0]
                row.update(asdict(result))
                print(qid, result.is_correct, result.generated_answer, flush=True)
            rows.append(row)
            write(OUT / phase_dir / "results.json", rows)
    if args.phase == "manual":
        assert all(r["is_correct"] for r in rows), "预检失败：保留记录，不实施未通过方案"


if __name__ == "__main__":
    main()
