"""喂 `Add`、调 `Search`——**走 HTTP**。

> **本模块不 import `tianximem` 的任何内部模块**，这是全目录最硬的一条边界
> （[`../CLAUDE.md`](../CLAUDE.md)）：§13 要求 B1（ReFind）在我们自己的 harness 里重跑，
> 而 B1 只以"另一个 Add/Search 服务"的形式存在——走进程内调用会让 B1 变成特例、
> 两条基线不可比；而且**只有打 HTTP 才碰得到契约层**。

## 它故意很薄

驱动只做三件事：**发请求、收响应、把响应变成领域对象**。
切批在 [`batching`](./batching.py)、注入与判分在 [`judge`](./judge.py)、
契约全清单在 [`../smoke/preflight.py`](../smoke/preflight.py)——本模块只留**一条**断言
（`len(data) <= top_k`），因为它是**最便宜且后果最重**的那条：返回超限是**契约错误**
而不是截断，**AML 不会替我们截**（§2.2）。
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from eval.datasets import Message, Sample

from .batching import batches, request_id_for

__all__ = ["SearchHit", "ServiceClient", "REQUEST_TIMEOUT_S"]

#: 契约允许**单请求最长 30 分钟**（§2.2）。默认取满：Full run 要连续跑 0.5–2 天，
#: 用一个"看着正常"的短超时会把**慢**伪装成**失败**，直到 Full 才炸（§15）。
REQUEST_TIMEOUT_S = 1800.0


@dataclass(frozen=True, slots=True)
class SearchHit:
    """`data[]` 的一项——**§2.1 的四个字段**，不做任何加工。"""

    id: str
    content: str
    created_at: str
    score: float


class ServiceClient:
    """两个端点的 HTTP 客户端。**对 B1 同样适用**——换 `base_url` 即可。

    `client` 可注入（测试用 `httpx.MockTransport`），生产走默认构造。
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = REQUEST_TIMEOUT_S,
        client: httpx.Client | None = None,
        fallback: ServiceClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        #: **超限兜底**（2026-09-26，只为 B1）：同一个问题再问一次**另一个实例**。
        #:
        #: 动机：ReFind 的 agent 把历次观测累加在对话里、**没有输入 token 预算**，
        #: 而 LongMemEval 里有 2.3 万字符的长消息 ⇒ 一次观测就可能超过 128k ⇒ 网关 400
        #: ⇒ **整轮跑批被打死**。它的 `METHOD_CARD` 声明的提交配置里没有预算这一项。
        #: ⇒ 兜底形态按**项目负责人的指示**做，但**不碰 vendor 代码**：让 harness 换一个
        #: 「同一个 vendor、另一档 `RETRIEVAL_MODE`」的实例去回答那道题。
        #: ⚠ 用了兜底就必须在台账里写明口径——**这一臂不再等于"ReFind 原样"**。
        self._fallback = fallback
        self.fallback_used = 0

    # ── Add ────────────────────────────────────────────────────────────
    def add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: tuple[Message, ...],
    ) -> dict:
        """投一批。**非 200 一律抛**——§15 说非 200 的行为未定义，必须假设 AML 会重试，
        所以 harness 这边**不能把失败当成功继续跑**（那会把"漏跑了"伪装成"分数低"）。"""
        response = self._client.post(
            f"{self.base_url}/add",
            json={
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "messages": [m.to_add_payload() for m in messages],
            },
        )
        response.raise_for_status()
        return response.json()

    # ── Search ─────────────────────────────────────────────────────────
    def search_raw(self, *, user_id: str, query: str, top_k: int) -> dict:
        """原样返回响应体——**预检要看的是这个**，不是我们自己那个 dataclass。

        ⚠ 缺字段（比如 `created_at`）在 `SearchHit(**item)` 里是 `TypeError`，
        而预检必须把它报成**一条结论**而不是崩掉。所以形状检查走这里。
        """
        payload_body = {"user_id": user_id, "query": query, "top_k": top_k}
        try:
            response = self._client.post(f"{self.base_url}/search", json=payload_body)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # 只在**服务端错误**上兜底（4xx 是我们自己的 bug，兜底会把 bug 藏起来）。
            if self._fallback is None or error.response.status_code < 500:
                raise
            self.fallback_used += 1
            print(
                f"  ⚠ {self.base_url} 返回 {error.response.status_code} ⇒ "
                f"**兜底**到 {self._fallback.base_url}（该题）",
                flush=True,
            )
            response = self._fallback._client.post(
                f"{self._fallback.base_url}/search", json=payload_body
            )
            response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            # `null` 会让下游静默变成"没有记忆"——§2.1 要求空结果是 `[]`。
            raise AssertionError(
                f"`data` 必须是数组（空结果是 [] 而不是 null），收到 {type(data).__name__}"
            )
        if len(data) > top_k:
            # §2.2：超限是**契约错误**，AML 不会静默截断。本地抓到死，别等 Smoke。
            raise AssertionError(
                f"返回 {len(data)} 条 > top_k={top_k}——§2.2 契约错误（不会被静默截断）"
            )
        return payload

    def search(self, *, user_id: str, query: str, top_k: int) -> list[SearchHit]:
        return [
            SearchHit(**item)
            for item in self.search_raw(user_id=user_id, query=query, top_k=top_k)["data"]
        ]

    # ── 把一份 Sample 灌进去 ───────────────────────────────────────────
    def ingest(self, sample: Sample) -> int:
        """按 session 逐批投喂，返回批次数。

        **消息顺序不重排**：批**边界**是源序切出来的，`request_id` 里带着那个批序号
        ⇒ 重排会让同一 `request_id` 对应另一批消息。
        """
        count = 0
        for session in sample.sessions:
            for index, batch in enumerate(batches(session.messages)):
                self.add(
                    request_id=request_id_for(sample.user_id, session.session_id, index),
                    user_id=sample.user_id,
                    session_id=session.session_id,
                    messages=batch,
                )
                count += 1
        return count

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
