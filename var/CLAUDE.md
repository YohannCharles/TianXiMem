# var

`var/` contains local runtime state. It is ignored by Git except for this file.

```text
tianxi.db       SQLite source of truth
embed_cache/    Rebuildable embedding cache
capture/        Raw request capture; sensitive and optional
metrics.json    Optional metric snapshot
*.log           Local diagnostics
```

## Retention

| Path | Recovery |
| --- | --- |
| `tianxi.db` | Not rebuildable; back it up |
| `embed_cache/` | Rebuild by paying embedding cost again |
| `capture/` | Rebuild only by capturing new traffic |
| Metric snapshots and logs | Diagnostic; discard when no longer needed |

Qdrant storage lives in its Docker volume, not this directory. It is derived from SQLite and is not the primary backup target.

## Rules

- Do not commit runtime state.
- Do not treat `var/` as evaluation data.
- Use a new database and collection when switching to an incompatible vector space.
- Stop writes before copying SQLite directly.
- Do not delete `tianxi.db` to reset development state without explicit intent.
