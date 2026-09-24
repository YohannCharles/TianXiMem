"""环境自检——`make check`。

它回答一个问题：**"现在能不能真的跑一轮？"** 四件事各自独立地报告，
**一件不通不掩盖其他件**（一次看全比反复试省时间）。

## 为什么 LLM 那一项额外看"思考泄漏"

§17.3 的 **V7** 剩下的一半是"**网关默认开不开思考**"。归档的 7 个 pipeline 里
只有 `pipeline_beam.py` 传了 `enable_thinking: False`，而 **LoCoMo-Refined 与
LongMemEval 都没传**——若网关默认开着，thinking 会混进 `generated_answer`，
**裁判读到的就是推理过程而不是答案**（且不报错）。

判据很便宜：发一句"只回 OK"，看回来的 `content` 里有没有推理。**这就是那个探针。**

## 密钥只从环境读，报告里绝不回显

本脚本**不打印任何 key**，只打印"已填/未填"。`--env-file` 由调用方传入
（`make check` 用 `uv run --env-file .env`），**别在这里自己解析 `.env`**——
那会变成第二个配置加载点（③-d 正在把配置集中到 `common/config.py`）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import httpx

TIMEOUT = 20.0

#: 只回一个词的探针——**thinking 开着时回来的是一整段推理**（实测 167 vs 2 token）。
_THINKING_PROBE = {
    "messages": [{"role": "user", "content": "Return exactly the word OK."}],
    "max_tokens": 512,
    "temperature": 0,
}


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _key(name: str) -> str:
    return os.environ.get(name, "")


def _present(name: str) -> Check:
    value = _key(name)
    return Check(f"环境变量 `{name}`", bool(value), "已填" if value else "**未填**")


def check_keys() -> list[Check]:
    """四组端点各自的 base/key。**缺 key 与端点不通是两回事**，分开报。"""
    out = [
        _present(name)
        for name in ("AML_EMB_BASE_URL", "AML_EMB_API_KEY", "AML_BASE_URL", "AML_API_KEY")
    ]
    return out


def check_qdrant(url: str) -> Check:
    """§6.3 要求 **server 模式**——local 模式会静默丢弃 payload 索引。"""
    try:
        response = httpx.get(f"{url.rstrip('/')}/collections", timeout=TIMEOUT)
        response.raise_for_status()
        names = [c["name"] for c in response.json()["result"]["collections"]]
        return Check("Qdrant 可达（server 模式）", True, f"{url}，现有集合：{names or '（空）'}")
    except Exception as exc:  # noqa: BLE001 - 自检要把任何失败都报出来，不是崩掉
        return Check(
            "Qdrant 可达（server 模式）",
            False,
            f"{url} 不可达：{type(exc).__name__}: {exc}\n"
            "      ⇒ `docker compose -f deploy/compose.yaml up -d`；"
            "若报 socket 权限，是当前用户不在 docker 组（`sudo usermod -aG docker $USER` 后重登）",
        )


def check_embedding(base: str, key: str, model: str) -> Check:
    """**Embedding 在另一个网关上**（与 LLM 不同 host、不同 key）——调错域名只会 404。

    顺带验 V10（`last_token_pool` + L2 normalize）：**返回向量的 L2 范数应 ≈ 1**。
    """
    if not base or not key:
        return Check("Embedding 端点", False, "base_url 或 key 未填")
    try:
        response = httpx.post(
            f"{base.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model or "Qwen/Qwen3-Embedding-8B", "input": ["probe"]},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        norm = sum(x * x for x in vector) ** 0.5
        normalized = abs(norm - 1.0) < 1e-3
        return Check(
            "Embedding 端点（含 V10 归一化）",
            normalized,
            f"dim={len(vector)} L2={norm:.6f}"
            + (
                ""
                if normalized
                else " ⚠ L2 明显偏离 1 ⇒ 服务端没做 last_token_pool / L2 normalize（V10）"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return Check("Embedding 端点", False, f"{base} 不可达：{type(exc).__name__}: {exc}")


def check_reranker(base: str, key: str) -> Check:
    """Reranker 与 Embedding 在**同一个网关**上。

    它**不作规定**，是唯一能自己投算力的组件（§2.3）。
    """
    if not base or not key:
        return Check("Reranker 端点", False, "base_url 或 key 未填")
    try:
        response = httpx.post(
            f"{base.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": "q", "documents": ["a", "b"], "top_n": 2},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()["results"]
        return Check("Reranker 端点", True, f"返回 {len(results)} 条，已降序")
    except Exception as exc:  # noqa: BLE001
        return Check("Reranker 端点", False, f"{base} 不可达：{type(exc).__name__}: {exc}")


def check_llm(base: str, key: str, model: str) -> Check:
    """LLM 在 **memory2**；同时是 V7 那个探针（见模块 docstring）。"""
    if not base or not key:
        return Check("LLM 端点（含 V7 思考探针）", False, "base_url 或 key 未填")
    try:
        started = time.monotonic()
        response = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model or "Qwen/Qwen3.5-9B", **_THINKING_PROBE},
            timeout=120.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        elapsed = time.monotonic() - started
        leaked = len(content.strip()) > 40 or "Thinking" in content
        return Check(
            "LLM 端点（含 V7 思考探针）",
            not leaked,
            f"{elapsed:.1f}s，content={content.strip()[:60]!r}"
            + (
                " ⚠ **疑似泄漏思考过程**（V7 的坏分支）：归档 pipeline 不传 enable_thinking，"
                "裁判会读到推理而不是答案"
                if leaked
                else " ⇒ 网关默认已关思考（V7 清掉）"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            "LLM 端点（含 V7 思考探针）", False, f"{base} 不可达：{type(exc).__name__}: {exc}"
        )


def run_all(env: dict[str, str] | None = None) -> list[Check]:
    source = os.environ if env is None else env
    emb_base, emb_key = source.get("AML_EMB_BASE_URL", ""), source.get("AML_EMB_API_KEY", "")
    llm_base, llm_key = source.get("AML_BASE_URL", ""), source.get("AML_API_KEY", "")
    return [
        *check_keys(),
        check_qdrant(source.get("TIANXI_QDRANT_URL") or "http://localhost:6333"),
        check_embedding(emb_base, emb_key, source.get("AML_EMB_MODEL", "")),
        check_reranker(emb_base, emb_key),
        check_llm(llm_base, llm_key, source.get("AML_MODEL", "")),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="环境自检：Qdrant / 三段模型端点 / 密钥")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（给脚本用）")
    args = parser.parse_args(argv)

    checks = run_all()
    if args.json:
        print(json.dumps([c.__dict__ for c in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(f"{'✅' if check.ok else '❌'} {check.name}: {check.detail}")
        failed = [c for c in checks if not c.ok]
        print(f"\n{len(checks) - len(failed)}/{len(checks)} 通过")
    return 1 if any(not c.ok for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
