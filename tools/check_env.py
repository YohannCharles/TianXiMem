"""环境自检——`make check`。

它回答一个问题：**"现在能不能真的跑一轮？"** 每件事各自独立地报告，
**一件不通不掩盖其他件**（一次看全比反复试省时间）。

## 对话端点有**多个**（D39），每一个都要探

`memory2` 与 `memory3` 提供同一个 `Qwen/Qwen3.5-9B`，但**各有各的 key**（对调都是 401）
⇒ 端点按序号后缀列（`AML_BASE_URL` / `AML_BASE_URL_2` …），**逐个探**。
只探第一个的话，"第二个端点的 key 填错了"要等到判分**轮转到它**才炸——
而那时整轮的 Add / Search 与前面几题的判分都已经付出去了。

## rerank 的字段名跟着对面那个网关走，**两种都试**

`query`（自研封装）与 `queries`（vLLM 原生）**互斥**：写死哪一个都会让本项在另一半部署上
**永远红**——而"永远红"最后会被人关掉（那正是本项存在的理由）。
⇒ 先发 `query`，被 400/422 拒就换 `queries` 再发一次，**并把实际生效的那个写进报告**。

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
from urllib.parse import urlsplit

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


#: 对话端点的序号后缀。**与 [`../eval/harness/api_config.py`](../eval/harness/api_config.py)
#: 的 `_SUFFIXES` 是同一套约定**——那边是"谁在用"，这边是"谁在查"；
#: 本工具**刻意不 import 配置层**（见模块 docstring），所以这套约定在这里写第二遍。
_ENDPOINT_SUFFIXES = ("", "_2", "_3", "_4")


def check_keys() -> list[Check]:
    """各端点各自的 base/key。**缺 key 与端点不通是两回事**，分开报。"""
    names = ["AML_EMB_BASE_URL", "AML_EMB_API_KEY"]
    for suffix in _ENDPOINT_SUFFIXES:
        if suffix and not _key(f"AML_BASE_URL{suffix}"):
            continue  # 没配这个端点就不报它的 key——那不是"没填"，是没有这个端点
        names += [f"AML_BASE_URL{suffix}", f"AML_API_KEY{suffix}"]
    return [_present(name) for name in names]


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


#: **提交口径的嵌入窗口**：`text-embedding-v4` 官方 FAQ 原文
#: ——"Each text can contain at most **8,192 tokens**. Content exceeding this limit is
#: **truncated before embedding**."
#:
#: ⚠ **两件事都是它要紧的地方**：① 窗口是 **8,192**（不是 8B 模型的原生 32,768）；
#: ② 超限时官方是**静默截断**，而本地网关（vLLM）是**响亮 400**
#: ⇒ **本地这个 400 是门禁**：线上不会替我们报这个错，只会把尾巴砍掉。
#: 所以这里盯的是"网关窗口**比提交口径宽**"——那会让本地跑出线上跑不通的东西（V16）。
SUBMIT_WINDOW: int = 8_192


def _model_window(base: str, key: str, model: str) -> int | None:
    """问 `/v1/models` 里那个模型自报的 `max_model_len`（拿不到就返回 `None`）。

    ⚠ **只是念出来，不参与判定**：窗口**小于**提交口径时，本地比线上更严 ⇒ 安全的一侧。
    只有**大于**时才在结论里挂一个 ⚠（见调用处）——那才是"本地跑得通、线上下不来"的方向。
    **这条线画在哪必须写在代码里，不能靠"没写"。**
    """
    with contextlib.suppress(Exception):
        resp = httpx.get(
            f"{base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        for entry in resp.json().get("data") or []:
            if isinstance(entry, dict) and entry.get("id") == model:
                value = entry.get("max_model_len")
                return int(value) if value else None
    return None


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
        window = _model_window(base, key, chosen)
        note = ""
        if window is not None:
            note = f" max_model_len={window}（提交口径 {SUBMIT_WINDOW}）"
            if window > SUBMIT_WINDOW:
                note += (
                    " ⚠ **比提交口径宽**——本地会跑出线上跑不通的块，"
                    "而线上超限是**静默截断**不是报错（V16）"
                )
        return Check(
            "Embedding 端点（含 V10 归一化）",
            normalized,
            f"model={chosen} dim={len(vector)} L2={norm:.6f}"
            + note
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

    ⚠ **请求字段名两种都试**（2026-10-09）：`query`（自研封装，`memory.021130.xyz`）与
    `queries`（vLLM 原生）**互斥**，没有"都对"的写法。写死哪一个都会在另一半部署上**永远红**，
    而"永远红"和"真的坏了"在屏幕上一样，最后会被人关掉（那正是本项存在的理由）。
    两种都被拒才算不通，且报告里写明**实际生效的是哪个**。
    """
    if not base or not key:
        return Check("Reranker 端点", False, "base_url 或 key 未填")
    # 顺序：先试现役主网关的形状（`query`），再退回 vLLM 原生（`queries`）。
    rejected: list[str] = []
    for field in ("query", "queries"):
        # ⚠ 请求形状要**跟着 `RemoteReranker` 走**（见 `rank/reranker.py` 的 `_post`）：
        #   那边**刻意不传 `top_n`**——传了会静默截断，返回的就不再是全部候选，
        #   于是"集合有没有被改动"就验不出来了（那是 §11.2 的一条不变量）。
        #   探针若传了它，测的就不是我们真正会发的那个请求。
        payload: dict[str, object] = {
            field: "q" if field == "query" else ["q"],
            "documents": ["a", "b"],
        }
        if model:
            payload["model"] = model
        try:
            response = httpx.post(
                f"{base.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # 400/422 = **这个字段名不对**，换另一个再试；其余状态码当场报出来。
            if exc.response.status_code in (400, 422) and field == "query":
                rejected.append(f"`query` 被 {exc.response.status_code} 拒")
                continue
            return Check(
                "Reranker 端点",
                False,
                f"{base} 返回 {exc.response.status_code}（字段名 `{field}`）："
                f"{exc.response.text[:120]}",
            )
        except Exception as exc:  # noqa: BLE001
            return Check("Reranker 端点", False, f"{base} 不可达：{type(exc).__name__}: {exc}")
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
        note = f"（{rejected[0]}，已换 `{field}`）" if rejected else ""
        return Check(
            "Reranker 端点",
            True,
            f"要 {asked} ⇒ 服务端 {served}，返回 {len(results)} 条；请求字段 `{field}`{note}",
        )
    # 两个字段名都被 400/422 拒 ⇒ 对面那个网关两种都不收（或 base_url 打错了）。
    tried = "；".join(rejected)
    return Check("Reranker 端点", False, f"{base} 对 `query` / `queries` 都拒绝：{tried}")


def check_llm(base: str, key: str, model: str) -> Check:
    """**一个**对话端点 + V7 那个探针（见模块 docstring）。

    ⚠ **每个端点各调一次**（D39）：`memory2` 与 `memory3` 是同一个模型、**两把 key**，
    只探第一个的话，"第二个端点 key 填错了"要等到判分轮转到它才炸——那时前面几题
    与整轮的 Add / Search 都已经付出去了。名字里带上 host，屏幕上才分得清是哪一台。
    """
    name = f"LLM 端点 {urlsplit(base).netloc or '?'}（含 V7 思考探针）"
    if not base or not key:
        return Check(name, False, "base_url 或 key 未填")
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
            name,
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
        return Check(name, False, f"{base} 不可达：{type(exc).__name__}: {exc}")


def _llm_endpoints(source) -> list[tuple[str, str]]:
    """`((base, key), ...)`——**与 `api_config` 同一套后缀约定**（见上面那个常量）。"""
    out: list[tuple[str, str]] = []
    for suffix in _ENDPOINT_SUFFIXES:
        base = (source.get(f"AML_BASE_URL{suffix}") or "").strip()
        if not base:
            continue
        out.append((base, (source.get(f"AML_API_KEY{suffix}") or "").strip()))
    return out


def run_all(env: dict[str, str] | None = None) -> list[Check]:
    source = os.environ if env is None else env
    emb_base, emb_key = source.get("AML_EMB_BASE_URL", ""), source.get("AML_EMB_API_KEY", "")
    # reranker 有自己的一组变量（`TIANXIMEM_RERANKER_*`），回落 embedding 那台——
    # 二者同网关是**当前部署**的事实，不是规格（`.env.example` 写明它们成对随部署而变）。
    rr_base = source.get("TIANXIMEM_RERANKER_BASE_URL") or emb_base
    rr_key = source.get("TIANXIMEM_RERANKER_API_KEY") or emb_key
    return [
        *check_keys(),
        check_qdrant(source.get("TIANXIMEM_QDRANT_URL") or "http://localhost:6333"),
        check_embedding(emb_base, emb_key, source.get("AML_EMB_MODEL", "")),
        check_reranker(rr_base, rr_key, source.get("TIANXIMEM_RERANKER_MODEL", "")),
        *(
            check_llm(base, key, source.get("AML_MODEL", ""))
            for base, key in _llm_endpoints(source)
        ),
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
