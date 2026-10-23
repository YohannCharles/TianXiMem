# Configuration Reference

## Ownership

Configuration has two non-overlapping layers.

| Layer | Owns |
| --- | --- |
| Environment variables | Credentials, endpoints, model IDs, paths, bind/port settings, worker count |
| YAML profiles | Thresholds, weights, collection names, context budgets, and feature switches |

`src/tianximem/common/config/` is the only package allowed to read either layer. Unknown keys are errors. A key belongs in configuration only when code consumes it.

## Loading

```text
built-in defaults
  -> configs/default.yaml
  -> configs/<profile>.yaml
  -> environment variables
  -> validation
```

- `TIANXIMEM_PROFILE` selects the profile and defaults to `default`.
- `TIANXIMEM_CONFIG_DIR` selects another directory containing `default.yaml` and the selected profile.
- `TIANXIMEM_ENV_FILE` selects the dotenv file.
- `configs/default.yaml` is the product profile.
- `configs/local.yaml` contains development-only overrides.
- `configs/submit.yaml` contains submission-only overrides.

## Value Classes

| Class | Meaning | Rule |
| --- | --- | --- |
| A | External contract constant | Do not change. Configuration presence is for traceability. |
| B | Correctness or identity constant | Invalid or mismatched values must fail validation. |
| C | Tunable threshold or switch | Changes require a comparable evaluation record. |

`top_k` is request input, not a configuration constant. `rrf.k=61` is a B-class correctness constant.

## Active Product Defaults

| Key | Default | Class | Meaning |
| --- | --- | --- | --- |
| `models.embedder` | `Qwen/Qwen3-Embedding-4B` | C | Server-side embedding model ID. |
| `models.embed_dim` | `1024` | B | Expected embedding dimension; the API response remains authoritative. |
| `storage.sqlite.busy_timeout_ms` | `5000` | C | SQLite writer wait limit. |
| `storage.qdrant.collection` | `memories` | C | Qdrant collection namespace. |
| `retrieval.grounded_evidence` | `true` | C | Enable the deterministic grounded-evidence path. |
| `retrieval.evidence_limit` | `12` | C | Maximum evidence and review candidates. |
| `retrieval.evidence_hop_limit` | `4` | C | Maximum relation-walk depth. |
| `retrieval.fact_backfill_limit` | `1024` | C | Maximum source rows backfilled per Search. |
| `retrieval.grounded_planner.enabled` | `true` | C | Use one bounded LLM plan when deterministic compilation fails. |
| `retrieval.grounded_planner.timeout_seconds` | `10.0` | C | Planner request timeout. |
| `retrieval.grounded_planner.max_output_tokens` | `256` | C | Planner output limit. |
| `retrieval.grounded_planner.temperature` | `0.0` | C | Planner sampling temperature. |
| `retrieval.grounded_planner.prompt_version` | `v3` | B | Must match the implemented planner prompt. |
| `retrieval.prefetch_limit` | `200` | C | Candidates entering RRF from each retrieval arm. |
| `retrieval.rrf.k` | `61` | B | Qdrant zero-based equivalent of literature `k=60`. |
| `retrieval.rrf.weights` | `[0.5, 0.5]` | C | BM25 and dense RRF weights. |
| `retrieval.query_instruction` | `""` | C | Optional query-side instruction prefix. |
| `neighbor.expansion_seed_limit` | `1000` | C | Candidates eligible for neighbor expansion. |
| `neighbor.radius` | `2` | C | Expansion distance in blocks; v1.2 expands two blocks on each side, and zero disables expansion. |
| `neighbor.seed_placement` | `keep` | C | Segment-internal order: `keep`, `front`, or `echo`. |
| `budget.max_tokens` | `117760` | A | Answer-model input budget. |
| `budget.tokenizer` | `o200k_base` | A | Submission answer-model tokenizer. |
| `packaging.inject_abs_time` | `true` | C | Put day-granularity dates in indexed renderings. |
| `packaging.annotate_relatives` | `true` | C | Annotate relative dates in response content only. |
| `packaging.annotate_unit_totals` | `false` | C | Add source-stated unit totals for supported transaction facts. |
| `rerank.enabled` | `true` | C | Rerank candidates when an endpoint is configured. |
| `rerank.envelope` | `query` | B | Remote rerank request envelope: `query` or `queries`. |
| `rerank.timeout_seconds` | `30.0` | C | Rerank request timeout. |
| `router.enabled` | `true` | C | Classify incomplete grounded queries as `MULTI_HOP` or `DEFAULT`. |
| `router.timeout_seconds` | `10.0` | C | Router request timeout. |
| `router.max_output_tokens` | `8` | C | Router output limit. |
| `router.temperature` | `0.0` | C | Router sampling temperature. |
| `router.prompt_version` | `v5` | B | Must match the implemented router prompt. |
| `agentic.enabled` | `true` | C | Construct the configured multi-hop executor. |
| `agentic.mode` | `plan` | C | Executor mode: `plan` or `react`. |
| `agentic.force` | `false` | C | Diagnostic switch that forces every query through `MULTI_HOP`. |
| `agentic.max_rounds` | `4` | C | Retrieval rounds, including the first. |
| `agentic.per_round_top_k` | `20` | C | Candidates retained for the model after each round's rerank. |
| `agentic.sufficiency_threshold` | `0.8` | C | React early-stop confidence threshold. |
| `agentic.max_prompt_tokens` | `6000` | C | Per-round agent prompt budget. |
| `agentic.max_output_tokens` | `256` | C | React output limit. |
| `agentic.plan_max_output_tokens` | `768` | C | Plan output limit. |
| `agentic.timeout_seconds` | `30.0` | C | LLM request timeout. |
| `agentic.temperature` | `0.0` | C | LLM sampling temperature. |
| `agentic.round_rerank_enabled` | `true` | C | Rerank candidates inside each multi-hop round. |
| `agentic.final_rerank_query_mode` | `combined` | C | Final rerank query: `combined` or `original`. |
| `agentic.prompt_version` | `coq-original` | B | React prompt version assertion. |
| `agentic.plan_decompose_prompt_version` | `plan-decompose-v2` | B | Plan decomposition prompt version assertion. |
| `agentic.plan_step_prompt_version` | `plan-step-v2` | B | Plan step prompt version assertion. |
| `capture.enabled` | `true` | C | Record raw Add/Search requests. |
| `capture.max_bytes` | `52428800` | C | Request capture file rotation size. |

## Environment Variables

### Runtime

| Variable | Meaning |
| --- | --- |
| `TIANXIMEM_PROFILE` | Active YAML profile. |
| `TIANXIMEM_CONFIG_DIR` | Configuration directory override. |
| `TIANXIMEM_ENV_FILE` | Dotenv file path. |
| `TIANXIMEM_WORKERS` | Must be `1`. |
| `TIANXIMEM_BIND` | Container bind address. |
| `TIANXIMEM_PORT` | Container application port. |

### Models

| Variable | Meaning |
| --- | --- |
| `AML_EMB_BASE_URL` | OpenAI-compatible embedding endpoint. |
| `AML_EMB_API_KEY` | Embedding API key. |
| `AML_BASE_URL` | Primary OpenAI-compatible chat endpoint. |
| `AML_API_KEY` | Primary chat API key. |
| `AML_MODEL` | Primary chat model ID. |
| `AML_BASE_URL_2` | Optional secondary evaluation chat endpoint. |
| `AML_API_KEY_2` | Optional secondary evaluation chat API key. |
| `TIANXIMEM_RERANKER_BASE_URL` | Optional rerank endpoint. |
| `TIANXIMEM_RERANKER_API_KEY` | Optional rerank API key. |
| `TIANXIMEM_RERANKER_MODEL` | Rerank model ID. |

The embedding model ID is `models.embedder` in YAML. The reranker and chat model IDs are endpoint identity and stay with their endpoint variables.

### Paths and Storage

| Variable | Meaning |
| --- | --- |
| `TIANXIMEM_SQLITE_PATH` | SQLite source-of-truth file. |
| `TIANXIMEM_QDRANT_URL` | Qdrant server URL. |
| `TIANXIMEM_QDRANT_IMAGE` | Pinned Qdrant container image. |
| `TIANXIMEM_EMBED_CACHE_DIR` | Persistent embedding cache directory. |
| `TIANXIMEM_METRICS_PATH` | Optional metric snapshot path. |
| `TIANXIMEM_CAPTURE_PATH` | Raw request capture path. |
| `TIANXIMEM_BENCHMARK_DIR` | Evaluation data root. |

## 2. 消融开关

This table is the machine-readable registry of switches backed by a consumed YAML key. The final column must contain `✅ \`key\`` only when the loader accepts that key.

| Switch | Disabled behavior | Preconditions | Must not change | Scope | Wiring |
| --- | --- | --- | --- | --- | --- |
| `retrieval.grounded_evidence` | Use normal hybrid retrieval; Add writes no factual index. | None. | Source rows, idempotency records, Qdrant vectors, and response shape. | Add and Search | ✅ `retrieval.grounded_evidence` |
| `retrieval.grounded_planner.enabled` | Do not use LLM plan generation. | Deterministic plan compilation remains active. | Deterministic plans, evidence review, and fallback behavior. | Search | ✅ `retrieval.grounded_planner.enabled` |
| `rerank.enabled` | Keep RRF order. | A rerank endpoint enables the remote path. | Candidate ID set. | Search | ✅ `rerank.enabled` |
| `neighbor.radius` | Set to zero to disable neighbor expansion. | None. | Seed candidates and segment priority. | Search | ✅ `neighbor.radius` |
| `packaging.inject_abs_time` | Do not add date prefixes to indexed text. | None. | Item count, IDs, `created_at`, and score order. | Add and Search | ✅ `packaging.inject_abs_time` |
| `packaging.annotate_relatives` | Return unannotated relative expressions. | None. | Indexed text, candidate set, IDs, counts, and order. | Search | ✅ `packaging.annotate_relatives` |
| `router.enabled` | Grounded evidence still runs first; incomplete queries use hybrid retrieval. | Enabling requires chat endpoint configuration. | Grounded behavior and hybrid response shape. | Search | ✅ `router.enabled` |
| `agentic.enabled` | `MULTI_HOP` falls back to `DEFAULT`. | Enabling requires chat endpoint configuration. | Hybrid retrieval and packaging. | Search | ✅ `agentic.enabled` |
| `capture.enabled` | Install no capture middleware. | None. | Every request and response. | API | ✅ `capture.enabled` |

`agentic.force=true` requires `agentic.enabled=true`. Enabling the router or executor requires complete chat endpoint configuration.

## Validation Rules

- `rrf.k` must equal `61`.
- `rerank.envelope` must be `query` or `queries` and must match the configured endpoint.
- Router and agent prompt versions must match the implemented constants.
- `agentic.force` cannot be enabled while `agentic.enabled` is false.
- The application worker count must be `1`.
- Unknown YAML or environment keys are errors.
- Missing required embedding or chat endpoint configuration is an error when the corresponding feature is enabled.
- Missing reranker configuration disables reranking with an observable warning; it is not a startup failure.

## Profile Differences

| Profile | Purpose | Differences |
| --- | --- | --- |
| `default` | Product baseline | No overrides. |
| `local` | Development | Separate Qdrant collection; reranking disabled. |
| `submit` | Submission | `models.embedder: text-embedding-v4`. |
| `v1.0` | v1.0 retrieval compatibility | Separate collection; grounded evidence, router, and executor disabled. |
| `probe-nocapture` | Diagnostics | Request capture disabled. |
| `probe-react` | Executor comparison | `react` mode; request capture disabled. |

## Deployment Constants

| Item | Value | Reason |
| --- | --- | --- |
| Qdrant mode | Server | Required payload-index behavior. |
| Qdrant image | `qdrant/qdrant:v1.17.0` | Required for weighted RRF. |
| Shards | `1` | Stable fusion order. |
| Named vectors | `dense`, `bm25` | Hybrid retrieval. |
| Filter indexes | `user_id`, `session_id`, `event_time` | Isolation and temporal filtering. |
| Write wait | `true` | Immediate searchability after Add. |
| Application workers | `1` | Unvalidated multi-process write behavior. |
