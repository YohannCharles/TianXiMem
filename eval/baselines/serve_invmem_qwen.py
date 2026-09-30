"""用**我们的 embedding**（Qwen3-Embedding-8B）跑 InvMem 候选仓库的 `/add` `/search`。

```bash
PYTHONPATH=eval/baselines/invmem-candidate/src \\
MEMORY_DB_PATH=/tmp/invmem-qwen/memory.db ALLOW_UNAUTHENTICATED=true \\
AML_EMB_BASE_URL=$AML_EMB_BASE_URL AML_EMB_API_KEY=$AML_EMB_API_KEY \\
PORT=8002 /tmp/invmem-venv/bin/python eval/baselines/serve_invmem_qwen.py
```

## 为什么需要这个文件（2026-09-26）

候选仓库自带的 embedding 是 **`BAAI/bge-small-en-v1.5`（33M 参数）**，而我们是
**Qwen3-Embedding-8B**（开发期；提交期按规定换 `text-embedding-v4`）。
直接比会把"**embedding 强弱**"混进"**管线设计**"里——用户要求改成同一个 embedding 再比。

## 边界：**不改 vendor 代码**（与 B1 同一条纪律）

做法是**在进程内换掉那一个函数**：`vanilla_rag.api.build_embedder` → 走我们网关的实现。
其余（分块 `chunk_document`、混合检索、RRF、`result_window`、打包）**一行不动**，
仍然跑在它的仓库里（[`invmem-candidate/`](./invmem-candidate/)）。

⚠ **这个跑出来的数字不是"候选仓库的分数"，是"它的管线 + 我们的 embedding"**。
两个口径都要在 [`../reports/ledger.md`](../reports/ledger.md) 里写清楚。

⚠ **查询侧不加任何 instruction**（`retrieval.query_instruction` 在 v1 也默认 `""`）——
两边保持一致，否则这一臂又多了第二个变量。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

VENDOR_SRC = Path(__file__).resolve().parent / "invmem-candidate" / "src"


class GatewayEmbedder:
    """走我们网关的 embedding —— **接口与 `SentenceTransformerEmbedder` 逐字相同**。

    ⚠ **必须 L2 归一化**：候选仓库的检索是 `matrix @ query_vectors.T`（内积当余弦用），
    而它自带的 sentence-transformers 那条路是 `normalize_embeddings=True`。
    不归一化不会报错，只会**静默地把检索变成"谁的向量长谁赢"**。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 16,
        timeout: float = 120.0,
        retries: int = 4,
    ) -> None:
        import httpx

        self._client = httpx.Client(timeout=timeout)
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._batch_size = batch_size
        self._retries = retries
        self.identity = f"{model}|query_prefix="

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            payload = {"model": self._model, "input": chunk}
            for attempt in range(self._retries):
                try:
                    response = self._client.post(self._url, headers=self._headers, json=payload)
                    response.raise_for_status()
                    data = response.json()["data"]
                    break
                except Exception:  # noqa: BLE001 —— 重试耗尽后原样抛出，不吞
                    if attempt == self._retries - 1:
                        raise
                    time.sleep(2**attempt)
            vectors.extend(item["embedding"] for item in data)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


def main() -> None:
    sys.path.insert(0, str(VENDOR_SRC))
    import vanilla_rag.api as api  # noqa: PLC0415 —— 必须在 sys.path 之后

    embedder = GatewayEmbedder(
        base_url=os.environ["AML_EMB_BASE_URL"],
        api_key=os.environ["AML_EMB_API_KEY"],
        model=os.environ.get("TIANXIMEM_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B"),
    )
    # ★ 唯一的一处改动：把它的 embedder 工厂换掉。其余全是它的代码。
    api.build_embedder = lambda model_name, device=None, query_prefix=None: embedder  # type: ignore[assignment]
    print(
        f"[shim] embedder = {embedder.identity}\n"
        f"[shim] vendor  =\n{VENDOR_SRC}\n"
        f"[shim] temporal_enrichment = {os.environ.get('MEMORY_TEMPORAL_ENRICHMENT', 'false')}",
        flush=True,
    )

    import uvicorn  # noqa: PLC0415

    uvicorn.run(api.app, host="127.0.0.1", port=int(os.environ.get("PORT", "8002")), workers=1)


if __name__ == "__main__":
    main()
