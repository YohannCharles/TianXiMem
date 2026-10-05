"""环境自检——`make check`。

它回答一个问题：**"现在能不能真的跑一轮？"** 四件事各自独立地报告，
**一件不通不掩盖其他件**（一次看全比反复试省时间）。

## 为什么 LLM 那一项额外看"思考泄漏"

§17.3 的 **V7** 问的是"**网关默认开不开思考**"。归档的 7 个 pipeline 里
只有 `pipeline_beam.py` 传了 `enable_thinking: False`，而 **LoCoMo-Refined 与
LongMemEval 都没传**——若网关默认开着，thinking 会混进 `generated_answer`，
**裁判读到的就是推理过程而不是答案**（且不报错）。

判据很便宜：发一句"只回 OK"，看回来的 `content` 里有没有推理。**这就是那个探针。**

## 密钥只从环境读，报告里绝不回显

本脚本**不打印任何 key**，只打印"已填/未填"。`--env-file` 由调用方传入
（`make check` 用 `uv run --env-file .env`），**别在这里自己解析 `.env`**——
那会变成第二个配置加载点（`.env` 现在由
[`common/config.py`](../src/tianximem/common/config.py) 读）。
**本工具刻意不依赖配置层**：配置层自己坏了的时候，这条诊断仍要能跑。
"""

from __future__ import annotations

import argparse
import contextlib
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


def _discover_model(base: str, key: str, *, prefer: str = "") -> str:
    """向网关问"你这里有哪些模型"，挑一个用。

    ⚠ **为什么要问而不是写死**：模型 id 是**服务端**的事，随网关部署而变——
    写死就得到一个**永远红的探针**（而"永远红"和"真的坏了"在屏幕上长得一样，
    最后会被人关掉）。

    ⚠ 也**不读配置层**（本工具的纪律：配置层自己坏了时它还得能跑）。
    `prefer` 给了就用它（例如 `.env` 里的 `TIANXIMEM_RERANKER_MODEL`），否则取 `/v1/models`
    的第一个——**两个 id 都会打进探针结论**，所以"用了哪个"永远是可见的。
    """
    if prefer:
        return prefer
    with contextlib.suppress(Exception):
        resp = httpx.get(
            f"{base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        # ⚠ 只认 embedding：`/v1/models` 可能同时列了别的；名字里带 embed 的优先
        ids = [m.get("id", "") for m in data if isinstance(m, dict)]
        for candidate in ids:
            if "embed" in candidate.lower():
                return candidate
        if ids:
            return ids[0]
    return ""


def check_embedding(base: str, key: str, model: str) -> Check:
    """**Embedding 在另一个网关上**（与 LLM 不同 host、不同 key）——调错域名只会 404。

    顺带验 V10（`last_token_pool` + L2 normalize）：**返回向量的 L2 范数应 ≈ 1**。

    ⚠ 模型 id 由 `_discover_model` 问出来（或由 `AML_EMB_MODEL` 指定）——
    **不写死**，理由见那个函数的 docstring。
    """
    if not base or not key:
        return Check("Embedding 端点", False, "base_url 或 key 未填")
    try:
        chosen = _discover_model(base, key, prefer=model)
        if not chosen:
            return Check("Embedding 端点", False, f"{base} 上没有可用的 embedding 模型 id")
        response = httpx.post(
            f"{base.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": chosen, "input": ["probe"]},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        norm = sum(x * x for x in vector) ** 0.5
        normalized = abs(norm - 1.0) < 1e-3
        return Check(
            "Embedding 端点（含 V10 归一化）",
            normalized,
            f"model={chosen} dim={len(vector)} L2={norm:.6f}"
            + (
                ""
                if normalized
                else " ⚠ L2 明显偏离 1 ⇒ 服务端没做 last_token_pool / L2 normalize（V10）"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return Check("Embedding 端点", False, f"{base} 不可达：{type(exc).__name__}: {exc}")


def check_reranker(base: str, key: str, model: str = "") -> Check:
    """Reranker 与 Embedding 在**同一个网关**上。

    它**不作规定**，是唯一能自己投算力的组件（§2.3）。

    ⚠ **必须带上 `model`**（2026-09-28 加）。不带的话网关会**用它的默认模型**替你跑——
    实测新网关（vllm）在只列了 `qwen3-embedding-8b` 的情况下，**不带 `model` 的
    `/rerank` 仍然返回了 2 条结果**（`"model": "qwen3-reranker-4b"`）。
    ⇒ 那样测出来的是"网关上有**某个** reranker"，而不是"**我们要用的那个 id** 可用"——
    **一次假绿**。而 `RemoteReranker` 是**会带 `model` 的**：id 填错就是 404 + 降级。
    """
    if not base or not key:
        return Check("Reranker 端点", False, "base_url 或 key 未填")
    try:
        # ⚠ 请求形状要**跟着 `RemoteReranker` 走**（见 `rank/reranker.py` 的 `_post`）：
        #   那边**刻意不传 `top_n`**——传了会静默截断，返回的就不再是全部候选，
        #   于是"集合有没有被改动"就验不出来了（那是 §11.2 的一条不变量）。
        #   探针若传了它，测的就不是我们真正会发的那个请求。
        # ⚠ **字段名也要逐字一致**：2026-09-28 网关把 `/v1/rerank` rewrite 到 vLLM 原生
        #   `/v1/score` 之后，请求要 **`queries`（数组）**——写成旧的 `query` 拿到 **400**，
        #   于是本项**永远红**。而"永远红"和"真的坏了"在屏幕上一样，最后会被人关掉。
        payload: dict[str, object] = {"queries": ["q"], "documents": ["a", "b"]}
        if model:
            payload["model"] = model
        response = httpx.post(
            f"{base.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        # ⚠ **容器名收两个**（`results` / `data`）——与 `RemoteReranker._parse` 同一口径：
        #   网关侧的信封并不稳定，而本层**只验连通性**，没有理由比客户端更严格。
        results = body.get("results", body.get("data"))
        if not isinstance(results, list):
            return Check(
                "Reranker 端点",
                False,
                f"响应里既没有 `results` 也不是 `data` 数组：{str(body)[:120]}",
            )
        # ⚠ **不声称"已降序"**：`RemoteReranker` 明确**不依赖**响应的顺序——
        #   它按 `index` 折回输入位置（见 rank/CLAUDE.md 的四条坑）。
        #   探针只报"回了几条"；"顺序与 `index` 对不对得上"由 `make probe-reranker` 验。
        #
        # ⚠ 也**不解析分数键**：见过的是 `score` / `relevance_score`，而这一层只验连通性。
        #   哪个键真的被认，由 `RemoteReranker._parse` 的用例与 `make probe-reranker` 管。
        served = body.get("model", "?")
        asked = model or "（未指定，网关默认）"
        return Check("Reranker 端点", True, f"要 {asked} ⇒ 服务端 {served}，返回 {len(results)} 条")
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
        check_qdrant(source.get("TIANXIMEM_QDRANT_URL") or "http://localhost:6333"),
        check_embedding(emb_base, emb_key, source.get("AML_EMB_MODEL", "")),
        check_reranker(emb_base, emb_key, source.get("TIANXIMEM_RERANKER_MODEL", "")),
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
