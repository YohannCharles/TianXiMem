# Configuration

## Ownership

| Layer | Owns |
| --- | --- |
| Environment variables | Credentials, endpoints, model IDs, paths, process shape |
| YAML profiles | Thresholds, weights, collection names, and feature switches |

Only `src/tianximem/common/config/` reads environment variables or YAML. Unknown
YAML keys fail validation. A key belongs in configuration only when runtime code
consumes it.

## Loading

```text
built-in defaults
  -> configs/default.yaml
  -> configs/<profile>.yaml
  -> .env for values not already set
  -> real environment variables
  -> validation
```

`TIANXIMEM_PROFILE` selects the profile and defaults to `default`.
`TIANXIMEM_CONFIG_DIR` changes the directory containing `default.yaml` and the
selected profile. `TIANXIMEM_ENV_FILE` changes the dotenv path.

## Environment Variables

### Runtime

| Variable | Meaning |
| --- | --- |
| `TIANXIMEM_PROFILE` | Active YAML profile. |
| `TIANXIMEM_CONFIG_DIR` | Configuration directory override. |
| `TIANXIMEM_ENV_FILE` | Dotenv file path. |
| `TIANXIMEM_WORKERS` | Must be `1`. |
| `TIANXIMEM_BIND` | Host bind address used by Compose. |
| `TIANXIMEM_PORT` | Host application port used by Compose. |

### Models

| Variable | Meaning |
| --- | --- |
| `AML_EMB_BASE_URL` | OpenAI-compatible embedding endpoint. |
| `AML_EMB_API_KEY` | Embedding API key. |
| `AML_BASE_URL` | OpenAI-compatible chat endpoint. |
| `AML_API_KEY` | Chat API key. |
| `AML_MODEL` | Chat model ID. |
| `TIANXIMEM_RERANKER_BASE_URL` | Optional rerank endpoint. |
| `TIANXIMEM_RERANKER_API_KEY` | Optional rerank API key. |
| `TIANXIMEM_RERANKER_MODEL` | Rerank model ID. |

### Paths and Storage

| Variable | Meaning |
| --- | --- |
| `TIANXIMEM_SQLITE_PATH` | SQLite source-of-truth file. |
| `TIANXIMEM_QDRANT_URL` | Qdrant server URL. |
| `TIANXIMEM_QDRANT_IMAGE` | Optional Compose override for the pinned Qdrant image. |
| `TIANXIMEM_EMBED_CACHE_DIR` | Persistent embedding cache directory. |
| `TIANXIMEM_METRICS_PATH` | Optional metric snapshot path. |
| `TIANXIMEM_CAPTURE_PATH` | Raw request capture path. |

## Product Profile

`configs/default.yaml` is the product source of truth. It enables grounded
evidence, routing, the plan executor, reranking, neighbor expansion, and request
capture.

Profiles override only intentional differences:

| Profile | Purpose |
| --- | --- |
| `local` | Separate development collection; reranking disabled. |
| `submit` | Submission embedding model. |
| `v1.0` | Compatibility retrieval shape; grounded evidence, planner, router, and executor disabled. |
| `probe-nocapture` | Diagnostics without request capture. |
| `probe-react` | React executor comparison; request capture disabled. |

## Important Constants

- `retrieval.rrf.k` must equal `61`.
- `budget.max_tokens` is `117760`.
- `request.top_k` is supplied per request, not by YAML.
- The application must run with one worker.
- Prompt versions and rerank envelopes are identity assertions, not tuning knobs.
