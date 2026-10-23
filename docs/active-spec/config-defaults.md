# Active Configuration

`configs/default.yaml` is the source of truth for the product profile. Built-in dataclass defaults exist for direct object construction, tests, and compatibility; they are not the product profile.

The mainline uses Add-local role-run composition, per-original-message fact
extraction and original-record projection for validated explicit walks. These are
implemented behaviors rather than experimental profile switches; their owners
and fallback rules are documented in [`../architecture.md`](../architecture.md).

| Key | Product value |
| --- | --- |
| `models.embedder` | `Qwen/Qwen3-Embedding-4B` |
| `retrieval.grounded_evidence` | `true` |
| `retrieval.grounded_planner.enabled` | `true` |
| `packaging.inject_abs_time` | `true` |
| `packaging.annotate_relatives` | `true` |
| `rerank.enabled` | `true` |
| `neighbor.radius` | `2` |
| `router.enabled` | `true` |
| `agentic.enabled` | `true` |
| `agentic.mode` | `plan` |
| `capture.enabled` | `true` |

Profiles override only keys that intentionally differ from the product profile:

- `local`: separate Qdrant collection, reranking disabled.
- `submit`: submission embedding model.
- `v1.0`: v1.0 retrieval compatibility.
- `probe-nocapture`: request capture disabled.
- `probe-react`: `react` executor, request capture disabled.

Change YAML and the corresponding regression test together. Do not duplicate defaults in service assembly or evaluation scripts.
