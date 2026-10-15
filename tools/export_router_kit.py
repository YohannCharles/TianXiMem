"""把**路由的标注样本**打包，交给别人独立测他们自己的路由器（`make export-router-kit`）。

```bash
uv run python tools/export_router_kit.py                 # → var/exports/router-200-<日期>/
uv run python tools/export_router_kit.py --manifest eval/router-samples.jsonl   # 换一份样本
```

## 为什么只需要「问题那一句」

`ROUTER_PROMPT` 的输入是 `{query}`，提示词原文写着
**"Use only the text in `{query}`; do not assume external context or missing metadata"**
⇒ 路由**不吃上下文**（不吃对话、不吃检索结果）。所以这份包**只有问题 + 真值**，
不带 LoCoMo / LongMemEval 的对话正文——包小得多，也少一层数据外泄面。

## 包里的三样

| 文件 | 是什么 |
| --- | --- |
| `questions.jsonl` | 每题一行：`dataset` / `question_id` / `question` / `answer` / |
| | `expected_route` / `question_type` / `boundary` / `label_note`（8 个字段） |
| `README.md` | 判据原文、怎么算分、⚠ 许可与**不得训练** |
| 同名 `.zip` | 上面两个 |

## ⚠ 真值是谁给的（这一条最要紧，README 里也写）

`expected_route` 是**人读着提示词的定义逐条判的**，⛔ **不是数据集自带的类目**。
数据集那套（LoCoMo 的 `category`、LongMemEval 的 `question_type`）描述的是**答案类型**，
不是**检索依赖结构**——本仓实测：LoCoMo 标成 `category=1`（multi-hop）的题里，
按路由判据**大多数其实是 `DEFAULT`**。
⇒ 拿数据集类目当路由真值会得到一份**看不出来的假分数**。

⚠ 标注是**盲标**（标的人没看路由器答什么），且标注模型与路由器**不同源**——
这两条是这份真值仅有的价值所在，README 里必须一起说明。

## 许可（转发数据就带着它走）

* LoCoMo-Refined：**CC BY-NC 4.0**（非商业；⚠ 且本仓口径：不得用于训练）
* LongMemEval：**MIT**
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: 产物落点。⚠ 含数据集原文 ⇒ 只进 `var/`（gitignored），⛔ 绝不进 git。
DEFAULT_OUT = Path("var/exports")


def _gold(base: Path) -> dict[tuple[str, str], str]:
    """`(dataset, question_id) -> 金标答案`。**只读**两份代理评测数据。"""
    gold: dict[tuple[str, str], str] = {}
    with (base / "locomo-refined" / "questions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                answer = row.get("answer")
                # ⚠ LoCoMo 的 `answer` 是 **list**（多个可接受答案）——按行拼，与官方 pipeline 同法。
                gold[("locomo-refined", row["qa_id"])] = (
                    "\n".join(str(a) for a in answer)
                    if isinstance(answer, list)
                    else str(answer or "")
                )
    with (base / "longmemeval-s" / "lme_s_cleaned.json").open(encoding="utf-8") as handle:
        # ⚠ 这份 ~277 MB，**只读顶层字段**（别把 `haystack_sessions` 收进内存）。
        for row in json.load(handle):
            gold[("longmemeval-s", row["question_id"])] = str(row.get("answer") or "")
    return gold


def render_readme(rows: list[dict]) -> str:
    from collections import Counter

    routes = Counter(r["expected_route"] for r in rows)
    kinds = Counter(r.get("question_type", "?") for r in rows)
    boundary = sum(1 for r in rows if r["boundary"])
    # ⚠ 两张表都**列全每一类**（不写 `most_common(1)` 那种偷懒写法）：
    #   分布表少一行，读的人就不知道那一类到底有没有。
    route_rows = "\n".join(
        f"| `{name}` | {count} |" for name, count in routes.most_common()
    ) + f"\n| **合计** | **{len(rows)}** |"
    kind_rows = (
        "\n\n问题类型（**标注时按类型均衡取样**，不是数据集类目）：\n\n"
        "| `question_type` | 数量 |\n| --- | ---: |\n"
        + "\n".join(f"| {name} | {count} |" for name, count in kinds.most_common())
    )
    return f"""# 查询路由的二分类测试集 — {len(rows)} 题

**用途**：测你自己的"用户查询该走哪条检索路"的分类器。**输入只有问题那一句**
（路由不吃上下文——见下），输出二选一。

## 两个类别（判据逐条）

* **`MULTI_HOP`** —— 要先**解出一个未在题面点名的中间实体 / 锚点**，后面的检索才表述得出来
  （例：`after the roadtrip`、`the friend who wrote Deborah ...`）；
* **`DEFAULT`** —— 其余，**含多实体比较与普通时间 / 数值问题**。

⚠ **没有 `COUNT`**（2026-10-14 起）：计数 / 名单类问法改由**共同取证**（先于路由器运行、
确定性）处理，不经过路由器 ⇒ 本套件里**已剔除** `COUNT` 标注的行。

⚠ 这套判据**抄的是原始提示词自己的定义**，不是数据集类目。判据的完整原文见
本仓 `src/tianximem/retrieve/router.py` 的 `ROUTER_PROMPT`（**逐字冻结**）。

## 分布

| `expected_route` | 数量 |
| --- | ---: |
{route_rows}
{kind_rows}

⚠ **`MULTI_HOP` 只占 {routes.get('MULTI_HOP', 0)}/{len(rows)}** ——这一类的置信区间最宽，
**逐类 P/R/F1 里那一列要打折看**。
⚠ **`boundary: true` 的有 {boundary} 条**（`interval` / `currently` 这两族判据本身模糊）
——建议**单列一节**，别混进主矩阵。

## 怎么用

```text
逐题把你自己的路由判定与 expected_route 比 → 混淆矩阵 → 逐类 P/R/F1
⚠ 一定要同时报「退化解」基线：永远答 DEFAULT 的准确率（本仓这批上是 ~0.49）。
  不报它的话，0.7 看起来像不错，其实只比"什么都不做"好一点。
```

⛔ **输入只有 `question`**——不要喂对话或检索结果（原始路由器就只吃这一句，
喂多了会让你的结果与别人不可比）。

## ⚠ 真值是怎么来的（读之前必须知道）

* `expected_route` 是**人读着上面那套判据逐条判的**，⛔ **不是数据集自带的类目**。
  数据集那套（LoCoMo 的 `category`、LongMemEval 的 `question_type`）描述的是**答案类型**，
  不是**检索依赖结构**——实测里 LoCoMo 标成 `category=1`（multi-hop）的题，
  按本判据**大多数其实是 `DEFAULT`**。
* 标注是**盲标**（标的人没看路由器会答什么），标注模型与路由器**不同源**
  ⇒ 真值不是"路由器的自我复述"。
* 每条都带 `label_note`（一句话判据）——**可以逐条反驳它**，欢迎。有争议的标了 `boundary`。

## 许可与使用限制

* **LoCoMo-Refined 部分：CC BY-NC 4.0** ⇒ **非商业**用途；若你的产物要商用，
  这部分结果**引用不了**。
* **LongMemEval 部分：MIT**。
* ⛔ 两部分数据**都不得用于训练模型**（赛事数据条款 + 上游许可）。
* 本题集**只覆盖 LoCoMo-Refined 与 LongMemEval** 两家的问法分布 ⇒
  **这类准确率不能外推**到别的数据集/真实流量。
"""


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="打包路由标注样本（问题 + 真值）")
    parser.add_argument("--manifest", type=Path, default=Path("eval/router-samples-200.jsonl"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bench-dir", default=None)
    args = parser.parse_args(argv)

    from eval.datasets.registry import benchmark_dir
    from tools.router_eval import _questions

    base = Path(args.bench_dir) if args.bench_dir else benchmark_dir()
    questions = _questions(base)
    gold = _gold(base)

    rows = []
    problems: list[str] = []
    legacy = 0
    for line in args.manifest.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        # ⚠ 遗留的 `COUNT` 行**不导出**（2026-10-14）：路由收成二分类，这类问法归
        #   共同取证、到不了路由器 ⇒ 留在套件里会让"正确率"分母无意义。
        if row["expected_route"] == "COUNT":
            legacy += 1
            continue
        key = (row["dataset"], row["question_id"])
        if key not in questions:
            problems.append(f"{key}：题库里没有这道题")
            continue
        answer = gold.get(key, "")
        if not answer.strip():
            problems.append(f"{key}：金标为空（不可答 / 缺字段）")
        rows.append({**row, "question": questions[key], "answer": answer})
    if problems:
        print("✗ 校验没过——**这份包不能发出去**：")
        for item in problems[:20]:
            print("   ", item)
        return 1
    if len({(r["dataset"], r["question_id"]) for r in rows}) != len(rows):
        print("✗ 清单里有重复的题")
        return 1

    out_dir = args.out / f"router-kit-{date.today():%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(render_readme(rows), encoding="utf-8")
    archive = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("questions.jsonl", "README.md"):
            bundle.write(out_dir / name, arcname=f"{out_dir.name}/{name}")

    from collections import Counter

    print(f"✅ 校验通过：{len(rows)} 题（题面 + 金标都解出来了，无重复）")
    if legacy:
        print(f"   ⚠ 已剔除 {legacy} 条遗留 `COUNT` 行（路由二分类后到不了路由器）")
    print(f"   真值分布：{dict(Counter(r['expected_route'] for r in rows))}")
    print(f"   boundary：{sum(1 for r in rows if r['boundary'])}")
    print(f"   目录：{out_dir}")
    print(f"   压缩包：{archive}（{archive.stat().st_size / 1024:.0f} KB）")
    print("   ⚠ 含数据集原文 ⇒ 只在 var/ 下（gitignored）；发出去时 README 的许可是随件的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
