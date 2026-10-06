"""把**官方真跑抓下来的 `POST /add` 请求体**原样回放给服务——S6 的收口验证。

## 它是什么

输入是**请求采集**（`capture.enabled`，见 [`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §0.6）
在服务器上抓到的真实流量导出：`kind=req` 的行里有 `body`（**与官方原文逐字节相同**）
与当时的 `status` / `error` / `retry_count`。

2026-09-29 那次官方 run 的结论是 **24 个 `request_id` / 328 次投递 / 全部失败**
（`ValueError`：`request_id` 形如 `r_<64 hex>`，D25 的正则取不出 chunk 序号）。
D28 把那条解析整个删掉之后，同一批 body 必须**全部写进去**——这就是本工具要证的事。

## 它证三件事

1. **写得进**：每个 body 都拿到 200（D28 之前这里 100% 是 500）
2. **重放干净**：`--repeat 2` 时第二次必须是 `applied=False` 的正常重放，
   **不是 409**（同 `request_id`、同 payload ⇒ 指纹相同）
3. **立即可检索**：写完之后按 `--verify-query` 搜一条，确认记忆真的进了索引（§2.1）

⚠ **它写进的是你指的那个服务/库**（默认本机开发服务）——用的是官方那批 `user_id`，
与本地 harness 的 `user_id` 不重叠，所以不会串到别的实验里。

```bash
uv run python tools/replay_official_adds.py --base-url http://127.0.0.1:8010
```
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import httpx

__all__ = ["load_bodies", "main", "replay"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.datasets.registry import benchmark_dir  # noqa: E402

DEFAULT_FILE = str(benchmark_dir() / ".legacy/unregistered/official-add-bodies-25.jsonl")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def load_bodies(path: str | Path) -> tuple[dict, list[dict]]:
    """读采集导出 ⇒ `(meta, [body, ...])`。`meta` 缺失时回空 dict。

    ⚠ 只取 `kind == "req"` 的行，取的是它们里面的 **`body`**（原始请求体），
    不是整行记录——整行是**记录格式**，body 才是要回放的东西。
    """
    meta: dict = {}
    bodies: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("kind") == "meta":
            meta = entry
        elif entry.get("kind") == "req" and isinstance(entry.get("body"), dict):
            bodies.append(entry["body"])
    return meta, bodies


def replay(
    bodies: list[dict],
    *,
    base_url: str,
    repeat: int = 1,
    timeout: float = 120.0,
) -> tuple[list[tuple[str, int, str]], Counter]:
    """逐个 POST。返回 `[(request_id, status, 摘要), ...]` 与状态码计数。

    ⚠ **按唯一 `request_id` 去重**：采集导出本来就已经一 id 一行，
    但这里再挡一次——不然"同一个 id 打两遍"会被误当成"两批都写进去了"。
    """
    results: list[tuple[str, int, str]] = []
    counts: Counter = Counter()
    seen: set[str] = set()
    with httpx.Client(timeout=timeout) as client:
        for body in bodies:
            request_id = str(body.get("request_id", ""))
            if request_id in seen:
                continue
            seen.add(request_id)
            for attempt in range(1, repeat + 1):
                try:
                    resp = client.post(f"{base_url}/add", json=body)
                except httpx.HTTPError as exc:  # 连不上/超时——也算失败，但要说清是哪种
                    results.append((request_id, 0, f"{type(exc).__name__}: {exc}"))
                    counts[0] += 1
                    continue
                counts[resp.status_code] += 1
                detail = resp.text[:160] if resp.status_code != 200 else "success"
                tag = f"第{attempt}次" if repeat > 1 else ""
                results.append((request_id, resp.status_code, f"{tag} {detail}".strip()))
    return results, counts


def _verify_search(base_url: str, body: dict, *, timeout: float = 60.0) -> str:
    """写完之后搜一发：拿该 body 里第一条消息的几个词当 query（§2.1 的"立即可搜索"）。"""
    user_id = str(body.get("user_id", ""))
    words = str(body["messages"][0].get("content", "")).split()[:6]
    query = " ".join(words) or "a"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url}/search", json={"user_id": user_id, "query": query, "top_k": 3}
        )
    if resp.status_code != 200:
        return f"Search {resp.status_code}：{resp.text[:120]}"
    data = resp.json().get("data", [])
    if not data:
        return "⚠ Search 200 但**一条都没召回**——写进去了但检索不到？"
    top = data[0]
    return f"Search 命中 {len(data)} 段；首段 id={top['id'][:12]}… created_at={top['created_at']!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default=DEFAULT_FILE, help=f"采集导出（默认 {DEFAULT_FILE}）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="每个 body 打几次（默认 2：第 2 次要走**正常重放**，不是 409）",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    meta, bodies = load_bodies(args.file)
    if not bodies:
        print(f"✗ {args.file} 里一条 `kind=req` 都没读到", file=sys.stderr)
        return 2

    if meta:
        print(f"采集来源：{meta.get('source', '?')}")
        print(
            f"  当时：{meta.get('total_add_requests')} 次投递 / "
            f"{meta.get('unique_request_ids')} 个 request_id / "
            f"全失败={meta.get('all_failed')}；clients={meta.get('clients')}"
        )
    print(f"回放：{len(bodies)} 个唯一请求体 → {args.base_url}（每个打 {args.repeat} 次）\n")

    results, counts = replay(
        bodies, base_url=args.base_url, repeat=args.repeat, timeout=args.timeout
    )

    for request_id, status, detail in results:
        mark = "✓" if status == 200 else "✗"
        print(f"  {mark} {status or 'ERR':>4}  {request_id[:24]}…  {detail[:90]}")

    print(f"\n状态码统计：{dict(counts)}")
    ok = counts[200] == len(results) and len(results) == len(bodies) * args.repeat
    print("✓ 全部 200（每个 request_id 都写进去了）" if ok else "✗ 有非 200——见上面逐条")

    print("\n可检索性抽查（§2.1：响应前必须立即可搜索）：")
    print("  " + _verify_search(args.base_url, bodies[0], timeout=args.timeout))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
