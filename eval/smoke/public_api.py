"""通过公网验证 Add/Search（§13 契约校验），仅需 httpx 与目标 URL。

    python eval/smoke/public_api.py
    python eval/smoke/public_api.py --base-url http://127.0.0.1:28088
    python -m eval.smoke.public_api --report eval/reports/runs/public-api/latest.json

复用 preflight 的契约检查，每次生成独立的 test-public-* 用户。
测试数据会持久保留在目标服务中；不会调用 AML 评测任务接口。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

# 同时支持直接运行文件与 python -m；不依赖 cwd 或安装服务包。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.smoke.preflight import (  # noqa: E402
    EXIT_CHECKS_FAILED,
    EXIT_OK,
    EXIT_PRECONDITION_FAILED,
    Preflight,
    Report,
)

DEFAULT_BASE_URL = "https://tianximem.mbgtest.lenovomm.com"


def run_checks(client: httpx.Client, *, nonce: str) -> tuple[Report, int]:
    """先确认目标可达，再运行同一套 HTTP 契约检查。"""
    report = Report()
    try:
        response = client.get("/health")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        report.add("Health · 2xx", False, f"{type(exc).__name__}: {exc}")
        return report, EXIT_PRECONDITION_FAILED
    report.add("Health · 2xx", True, f"HTTP {response.status_code}")

    preflight = Preflight(str(client.base_url), nonce=nonce, client=client)
    try:
        report.results.extend(preflight.run().results)
    finally:
        preflight.close()
    return report, EXIT_CHECKS_FAILED if report.failed else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="目标服务的根 URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="每个 HTTP 请求的超时秒数")
    parser.add_argument("--report", type=Path, help="可选的 JSON 报告路径，自动创建父目录")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须是有限的正数")
    try:
        url = httpx.URL(args.base_url)
    except httpx.InvalidURL as exc:
        parser.error(str(exc))
    if url.scheme not in {"http", "https"} or not url.host:
        parser.error("--base-url 必须是完整的 http:// 或 https:// 地址")

    nonce = f"test-public-{uuid.uuid4().hex[:12]}"
    print(f"目标：{url}\n测试用户前缀：{nonce}", flush=True)
    print("将写入少量持久测试数据；正在检查 Health 与 Add/Search…", flush=True)
    started = time.monotonic()
    with httpx.Client(base_url=url, timeout=args.timeout, trust_env=False) as client:
        report, code = run_checks(client, nonce=nonce)
    elapsed = round(time.monotonic() - started, 3)
    print(report.render())
    print(f"通过 {len(report.results) - len(report.failed)}/{len(report.results)}；耗时 {elapsed}s")

    if args.report:
        payload = {
            "base_url": str(url),
            "test_user_prefix": nonce,
            "elapsed_seconds": elapsed,
            "exit_code": code,
            "passed": code == EXIT_OK,
            "results": [asdict(result) for result in report.results],
        }
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(f"报告写入失败：{exc}", file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        print(f"报告：{args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
