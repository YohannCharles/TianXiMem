# deploy

This directory contains the container deployment for the service and Qdrant.

```text
Dockerfile                  Multi-stage application image
compose.yaml                Application plus Qdrant server
compose.two-versions.yaml   Optional two-version comparison stack
.env.example                Deployment environment template
cloudflare.md               Public ingress configuration
```

## Container Properties

- The application runs as one process.
- SQLite and the embedding cache use the persistent `tianxi_data` volume.
- Qdrant uses the persistent `qdrant_storage` volume.
- Qdrant is pinned and runs in server mode.
- The image contains no local model runtime.
- The image preloads the `o200k_base` tokenizer.
- `/health` checks only process liveness.

## Start

```bash
cp deploy/.env.example deploy/.env
uv run python -m tianximem.dev up
uv run python -m tianximem.dev deploy-check
```

Useful commands:

```bash
uv run python -m tianximem.dev image-build
uv run python -m tianximem.dev qdrant-up
uv run python -m tianximem.dev qdrant-down
uv run python -m tianximem.dev down
```

`qdrant-down` stops only the Qdrant service. `down` stops the project stack. Do not use `down -v` unless both source data and derived indexes are intentionally being destroyed.

## Environment

`deploy/.env` supplies credentials and deployment-specific values. The compose file overrides container-specific paths and the internal Qdrant URL.

| Variable | Purpose |
| --- | --- |
| `TIANXIMEM_PROFILE` | Select the YAML profile. |
| `TIANXIMEM_BIND` | Host bind address. Default is loopback. |
| `TIANXIMEM_PORT` | Host application port. |
| `TIANXIMEM_QDRANT_URL` | Qdrant URL as seen from the application container. |
| `AML_EMB_BASE_URL` | Embedding endpoint. |
| `AML_EMB_API_KEY` | Embedding key. |
| `AML_BASE_URL` | Required chat endpoint when router or agent execution is enabled. |
| `AML_API_KEY` | Chat key. |
| `AML_MODEL` | Chat model ID. |
| `TIANXIMEM_RERANKER_*` | Optional rerank endpoint and identity. |

If a model gateway runs on the Docker host, use `host.docker.internal` from inside the application container. `127.0.0.1` refers to the container itself.

## Profiles

- Server product runs use `TIANXIMEM_PROFILE=default` and the `memories` collection.
- Local development uses `local`, the `memories_dev` collection, and reranking disabled.
- The submission profile changes the embedding model to `text-embedding-v4`.

The selected profile and endpoint deployment must match. Never point a submission profile at a development endpoint or reuse the same collection for incompatible vector spaces.

## Runtime Configuration

Configuration files are baked into the image. To change YAML without rebuilding:

1. Copy the complete `configs/` directory to the host.
2. Mount it read-only into the container.
3. Set `TIANXIMEM_CONFIG_DIR` to the mount path.

The mounted directory must contain `default.yaml` and the selected profile. A mounted configuration directory stops following image upgrades; treat it as a deliberate deployment input and keep it versioned.

## Qdrant

- Server mode is required because embedded local mode does not preserve payload-index behavior.
- Keep the client and server versions compatible.
- Collection creation belongs to the application client, not compose.
- Collections use one shard, named `dense` and `bm25` vectors, and payload indexes created before writes.
- Qdrant is rebuildable from SQLite and is not the primary backup target.

## Model Connectivity

Check embedding and chat endpoints with authenticated `/models` requests from inside the container. A `401` means the endpoint is reachable but credentials are wrong.

Check the optional reranker independently:

```bash
uv run python -m tianximem.dev probe-reranker
```

Reranker failure degrades to RRF order. It must be visible in metrics rather than silently treated as a healthy rerank.

## Model or Vector-Space Migration

When changing embedding model, expected dimension, or the server behind the same model:

1. Stop the application.
2. Back up SQLite.
3. Create a new collection with the new vector dimension.
4. Invalidate the embedding cache when the coordinate changes.
5. Rebuild vectors from SQLite.
6. Start the application.
7. Run contract and retrieval checks.
8. Recalibrate thresholds before using the new deployment for comparisons.

If only the endpoint behind an identical model and dimension changes, verify vector compatibility before reusing a collection. Cache coordinates still prevent cross-model cache reuse.

## Backup and Recovery

Back up SQLite first. It is the only non-rebuildable source.

```bash
docker compose -f deploy/compose.yaml stop app
docker compose -f deploy/compose.yaml exec app tar -c -C /data . > tianxi-backup.tar
docker compose -f deploy/compose.yaml start app
```

The archive includes the embedding cache and request capture. They are optional during recovery. Qdrant can be rebuilt from SQLite.

## Request Capture

`capture.enabled` records raw Add and Search requests. The capture path is under `/data/capture/`.

Capture files may contain sensitive evaluation traffic. Do not commit, upload, or share them. Rotate files using `capture.max_bytes`; there is no total disk limit.

## Two-Version Comparison

`compose.two-versions.yaml` runs an additional application container against the same Qdrant server with separate SQLite storage and a separate collection.

Requirements:

- Build each service version from its own checkout.
- Use distinct collection names.
- Use identical answer and judge settings.
- Run versions sequentially unless the experiment explicitly validates concurrent load.
- Give each service its own metric output path.

## Network and Security

- The service has no authentication. `user_id` is an isolation key, not an identity.
- Bind to loopback by default.
- Add TLS and an authenticating reverse proxy before exposing the service externally.
- Keep Qdrant bound to the private Docker network or loopback.
- Do not expose Qdrant without authentication.

Public ingress setup is documented in [`cloudflare.md`](cloudflare.md).
