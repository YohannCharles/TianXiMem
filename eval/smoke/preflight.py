"""契约预检——**在花掉任何一次 Smoke 之前，把 [`contract.md`](../../docs/contract.md)
§4 的清单本地跑一遍**。

> **Smoke 次数有限（每轨道 ≤30 次），不要拿它当调试器。**（§2.4）
> **任何能在代理评测上回答的问题，都不该花 Smoke 的额度。**

## 它和单元测试的分工

| | 验什么 |
| --- | --- |
| `tests/` | **函数**（合成输入 → 断言） |
| **本文件** | **真的 HTTP 响应**——打在一个跑着的 Add/Search 服务上 |

两者都要（§13 要求主路径能通过 Smoke 契约校验）。

## 为什么最值得自动化的是这三条

它们**最容易在加了邻域扩展之后悄悄破掉**，而**破掉时不报错**
（[`CLAUDE.md`](./CLAUDE.md)）：

| # | 断言 | 破掉的后果 |
| --- | --- | --- |
| **精确计数** | `len(data) <= top_k`，**含邻域扩展占的名额** | 契约错误——AML **不会**替我们截 |
| **`created_at` 始终存在** | 日粒度或 `""`，**不能缺字段** | CL-Bench 拿不到时间；缺键比空值更糟 |
| **`score` 单调递减** | 且**不是**原始 RRF 分数 | 返回顺序与分数矛盾，模型读到的名次是错的 |

其余可自动化的条目**一并跑**——清单本身是 §4 唯一声明处，本文件不重述它、
只实现它。**清单里剩下的条目要么打不到（见 `NOT_AUTOMATABLE`），要么只能靠 Smoke。**

## ⚠ 它故意**不改**服务状态以外的东西

预检会往服务写数据（用带随机后缀的 `user_id`），也会重复 POST 同一批次来验幂等。
**用一个不与其他 run 冲突的 `user_id` 前缀**，或在跑完后清掉那一份数据。
它**不**会去读 SQLite（那需要 `store/` 的私有 schema，属于另一条边界）。
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from dataclasses import dataclass

import httpx
from eval.datasets import Message
from eval.harness import ServiceClient

__all__ = ["Finding", "NOT_AUTOMATABLE", "check", "main"]

#: §4 清单里**打不到**的条目，以及为什么——列出来是为了别让人以为"跑过了就是全过了"。
NOT_AUTOMATABLE: tuple[tuple[str, str], ...] = (
    ("`Search` 不生成答案、不把答案伪装成记忆记录", "需要语义判断，不是响应形状——只能人工看"),
    (
        "`session_id` 没被当成 Search 的过滤条件",
        "Search 请求里**没有这个字段**；它是服务内部性质，属 `tests/test_isolation.py`",
    ),
    ("前缀截断的安全性（窗口整体连续）", "§11.2 的组内顺序，要等 Step 2 的邻域扩展落地后才有意义"),
    (
        "根级 `limit` 取自请求的 `top_k`",
        "内部实现（Qdrant 查询参数）——`tests/` 覆盖；HTTP 侧只能验结果不超限",
    ),
)

#: 答案窗口的输入上限（§2.2）：128k 扣掉输出与安全余量后的 **117,760**。
ANSWER_WINDOW_INPUT_TOKENS = 117_760


@dataclass(frozen=True, slots=True)
class Finding:
    """一条预检结论。`spec` 一律带 § 号——不带 § 号的断言视为未经验证。"""

    name: str
    ok: bool
    spec: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Probe:
    """预检期间造的那一小份数据。**刻意很小**：它验的是形状，不是检索质量。"""

    user_id: str
    session_id: str
    query: str

    #: 两问两答——**够触发一次配对**即可。内容刻意无害（不含任何数据集原文）。
    messages: tuple[Message, ...] = (
        Message(
            role="user",
            content="Which city did I move to in March?",
            timestamp_ms=1_680_000_000_000,
        ),
        Message(role="assistant", content="You moved to Kyoto.", timestamp_ms=1_680_000_060_000),
        Message(
            role="user", content="And which instrument do I play?", timestamp_ms=1_680_000_120_000
        ),
        Message(role="assistant", content="You play the cello.", timestamp_ms=1_680_000_180_000),
    )


def _fresh_probe() -> _Probe:
    """随机后缀——**别和别的 run 抢同一个 `user_id`**，那会污染彼此的预检结论。"""
    tag = uuid.uuid4().hex[:8]
    return _Probe(
        user_id=f"preflight-{tag}",
        session_id=f"preflight-{tag}-s1",
        query="Which city did I move to?",
    )


def check(base_url: str, *, top_k: int = 10, timeout: float = 1800.0) -> list[Finding]:
    """跑完整个清单，返回全部结论（**不提前退出**——一次看全比反复跑省时间）。"""
    findings: list[Finding] = []
    probe = _fresh_probe()

    with ServiceClient(base_url, timeout=timeout) as client:
        # ── Add ────────────────────────────────────────────────────────
        request_id = f"{probe.session_id}|0"
        response = client.add(
            request_id=request_id,
            user_id=probe.user_id,
            session_id=probe.session_id,
            messages=probe.messages,
        )
        findings.append(
            Finding(
                "Add 响应 `success: true` 且原样回显三个字段",
                response.get("success") is True
                and response.get("request_id") == request_id
                and response.get("user_id") == probe.user_id
                and response.get("session_id") == probe.session_id,
                "§2.1 / §4",
                f"收到 {response!r}",
            )
        )

        # ── Search ─────────────────────────────────────────────────────
        hits = client.search(user_id=probe.user_id, query=probe.query, top_k=top_k)
        findings.append(
            Finding("Search 返回 `data` 数组（空结果是 `[]`）", isinstance(hits, list), "§2.1 / §4")
        )
        findings.append(
            Finding(
                f"精确计数：`len(data) <= top_k`（本次 {len(hits)} <= {top_k}）",
                len(hits) <= top_k,
                "§2.2 / §4",
                "**含邻域扩展占的名额**——Step 2 之后这条最容易悄悄破",
            )
        )

        # 更小的 top_k 也要精确——写死 100 的实现会在这里露出来
        small = client.search(user_id=probe.user_id, query=probe.query, top_k=2)
        findings.append(
            Finding(
                f"`top_k=2` 时也精确（本次 {len(small)} 条）",
                len(small) <= 2,
                "§7.3 / §4",
                "写死 100 会变成返回超限——那是契约错误，不是截断",
            )
        )

        # 形状检查一律吃**原始响应项**——`SearchHit(**item)` 会把"缺字段"变成崩溃，
        # 而缺字段恰恰是最该报成一条结论的东西。
        raw = client.search_raw(user_id=probe.user_id, query=probe.query, top_k=top_k)["data"]
        findings.append(_check_item_shape(raw, "§2.1 / §11.3 / §4"))
        findings.append(_check_no_leading_whitespace(raw, "§4 / D16"))
        findings.append(_check_no_absolute_timestamp(raw, "§11.3 / §4"))
        findings.append(_check_score_monotone(raw, "§11.3 / §4"))
        findings.append(_check_token_budget(raw, "§6.4 / §2.2"))

        # ── 隔离 ───────────────────────────────────────────────────────
        other = client.search(user_id=f"{probe.user_id}-nobody", query=probe.query, top_k=top_k)
        findings.append(Finding("跨 `user_id` 检索返回空（隔离生效）", other == [], "§2.2 / §4"))

        # ── 幂等：同一 `request_id` + 不变 payload 再 POST 一次 ────────
        before = [
            h.id for h in client.search(user_id=probe.user_id, query=probe.query, top_k=top_k)
        ]
        client.add(
            request_id=request_id,
            user_id=probe.user_id,
            session_id=probe.session_id,
            messages=probe.messages,
        )
        after = [h.id for h in client.search(user_id=probe.user_id, query=probe.query, top_k=top_k)]
        findings.append(
            Finding(
                "同 `request_id` 重复 POST → 检索结果不变（无新增行）",
                before == after,
                "§2.2 / §6.5 / §4",
                "HTTP 侧只能这样验；`applied_batches` 守卫本身由 `tests/test_idempotency.py` 覆盖",
            )
        )

    return findings


def _check_item_shape(items: list[dict], spec: str) -> Finding:
    """`id` / `content` 是字符串，**`created_at` 必须存在**（日粒度或 `""`）。

    ⚠ **"缺字段"与"空串"是两件事**：缺 key 会让下游拿不到字段，
    而 `""` 是 §11.3 那条**有定义的降级路径**（渲染成 `- {text}`）。
    """
    problems = []
    for index, item in enumerate(items):
        if not isinstance(item.get("id"), str) or not item.get("id"):
            problems.append(f"[{index}].id 不是非空字符串")
        if not isinstance(item.get("content"), str):
            problems.append(f"[{index}].content 不是字符串")
        if "created_at" not in item:
            problems.append(f"[{index}] **缺 created_at 字段**")
            continue
        # 日粒度（`2026-07-26`）或空串。**裸 Unix 毫秒**渲染出来对模型无意义（§11.3）
        value = item["created_at"]
        if value and not (len(str(value)) == 10 and str(value)[4] == "-"):
            problems.append(f"[{index}].created_at={value!r} 既不是日粒度也不是空串")
    return Finding(
        "每项含 `id` / `content` / **`created_at`（始终存在）**",
        not problems,
        spec,
        "; ".join(problems),
    )


def _check_no_leading_whitespace(items: list[dict], spec: str) -> Finding:
    """AML 只做 `"\\n".join(...)`、**不插分隔符**——首尾空白会直接进模型读到的正文。"""
    bad = [
        i.get("id") for i in items if str(i.get("content", "")) != str(i.get("content", "")).strip()
    ]
    return Finding("`content` 首尾无空白", not bad, spec, f"违规：{bad[:5]}")


def _check_no_absolute_timestamp(items: list[dict], spec: str) -> Finding:
    """**不在 content 里注入绝对时间戳前缀**（§11.3 的两条独立机制都会判负）。

    启发式：正文以 `[<数字>]` / `[<日期>]` 开头就报警。**这是启发式而非证明**——
    真实原因见 §11.3，那两条机制是"粒度变细"与"相对↔绝对互转"。
    """
    pattern = re.compile(r"^\s*[\[(]\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\s*[\[(]\s*\d{9,}")
    bad = [i.get("id") for i in items if pattern.match(str(i.get("content", "")))]
    return Finding("`content` 不含我们自己注入的绝对时间戳前缀", not bad, spec, f"疑似：{bad[:5]}")


def _check_score_monotone(items: list[dict], spec: str) -> Finding:
    """`score` 随名次单调递减，**且不是原始 RRF 分数**（§11.3：后者不是校准量，§8）。

    "不是 RRF"的判据：融合分数会小到 `1/61 ≈ 0.016` 量级且大量并列。
    这里只把**看起来像 RRF**的当作可疑——真正的保证在 `rank/packaging.py`。
    """
    scores = [i.get("score") for i in items]
    if scores != sorted(scores, reverse=True):
        return Finding("`score` 随名次单调递减", False, spec, f"实际：{scores[:10]}")
    if scores and max(scores) < 0.05:
        return Finding(
            "`score` 随名次单调递减",
            False,
            spec,
            f"最大值 {max(scores):.4f} 小得像**原始 RRF 分数**（占位值应是 1/(rank+1) 量级）",
        )
    return Finding("`score` 随名次单调递减且不是原始 RRF 分数", True, spec)


def _check_token_budget(items: list[dict], spec: str) -> Finding:
    """拼接后的总 token ≤ **117,760**——用答案模型自己的分词器（`o200k_base`），
    **不是字符数近似**：单次近似偏差会在 100 个对上放大到几千 token（§6.4）。

    预检的数据量很小，这条**通常必然通过**——留着是为了在数据量大起来之后仍然有人盯着。
    """
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken 是声明过的依赖
        return Finding("答案窗口 token 预算", True, spec, "跳过：tiktoken 未安装")

    encoding = tiktoken.get_encoding("o200k_base")
    total = sum(len(encoding.encode(str(i.get("content", "")))) for i in items)
    return Finding(
        f"答案窗口 token 预算（本次 {total} ≤ {ANSWER_WINDOW_INPUT_TOKENS}）",
        total <= ANSWER_WINDOW_INPUT_TOKENS,
        spec,
        "预检数据量小，这条通常必然通过——它的价值在数据量大起来之后",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="§2 契约合规预检——跑 Smoke 之前先本地过一遍")
    parser.add_argument(
        "--base-url", required=True, help="Add/Search 服务的基址，如 http://127.0.0.1:8000"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="预检用的 top_k（默认 10；AML 线上固定 100）"
    )
    args = parser.parse_args(argv)

    try:
        findings = check(args.base_url, top_k=args.top_k)
    except (httpx.HTTPError, OSError) as exc:
        print(f"连不上服务 {args.base_url}：{exc}", file=sys.stderr)
        return 2

    for finding in findings:
        mark = "✅" if finding.ok else "❌"
        print(f"{mark} [{finding.spec}] {finding.name}")
        if finding.detail:
            print(f"      {finding.detail}")

    print("\n**打不到、需要人工或只能靠 Smoke 的条目**（别以为跑过了就是全过了）：")
    for name, why in NOT_AUTOMATABLE:
        print(f"  ⏭  {name} —— {why}")

    failed = [f for f in findings if not f.ok]
    print(f"\n{len(findings) - len(failed)}/{len(findings)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
