# TianXiMem Service Package

This package implements the AML `Add` and `Search` endpoints. It returns ranked evidence and never generates or judges a final answer.

## Module Map

| Module | Responsibility |
| --- | --- |
| `service/` | HTTP schemas, orchestration, service assembly, and degradation boundaries |
| `pairing/` | Block composition, Add-scoped adjacency, idempotency, and source writes |
| `facts/` | Deterministic source-backed fact extraction and query-plan compilation |
| `store/` | SQLite source access, factual-index persistence, Qdrant access, and adjacency queries |
| `embed/` | Embedding protocol, model adapters, query adaptation, and vector cache |
| `retrieve/` | Retrieval policy, hybrid fusion, grounded-evidence execution, and query routing |
| `rank/` | Reranking, neighbor expansion, segment merge, and packaging |
| `agent/` | `react` and `plan` multi-hop executors |
| `llm/` | OpenAI-compatible chat client and availability errors |
| `common/` | Rendering, token counting, and configuration loading |
| `observability/` | Per-request metrics and aggregated snapshots |

Dependency direction and invariants are in [`../../docs/architecture.md`](../../docs/architecture.md).

## Package Boundaries

1. `store/` is the only package that accesses the business SQLite database and Qdrant. The embedding cache is a declared rebuildable exception with its own database.
2. `common/render.py` is the only renderer for QA blocks and grounded facts. Indexing, reranking, and response packaging use the same representation.
3. Search may call an LLM only through the bounded grounded planner, query router, or multi-hop executor. Add never calls an LLM.
4. `common/config/` is the only package that reads environment variables or YAML.
5. `facts/` and `retrieve/evidence.py` execute source-backed plans. They do not create answers or synthetic memory.

## Source Layout

The package uses a `src` layout with hatchling packaging. `uv sync` installs the repository as an editable package, so `uvicorn`, `python -m`, and tests do not need a manual `PYTHONPATH`.

Keep each module focused on one architectural responsibility. If a change requires two modules to own the same rule or record shape, move the owner instead of duplicating it.
