# Configuration Snapshots

`configs/runs/` stores configuration snapshots for isolated comparisons. New snapshot directories are ignored by Git unless the team explicitly decides that a comparison configuration must be shared.

A snapshot must contain the complete profile inputs required by `TIANXIMEM_CONFIG_DIR`, including `default.yaml` and the selected profile.

## Rules

- Freeze configuration before starting a run.
- Record the service revision, configuration directory, data fingerprint, and model identity with the run.
- Use separate Qdrant collections and SQLite databases for independent service versions.
- Do not reuse a snapshot for a different model, data version, or question selection.
- Do not edit a snapshot after its run starts.
- Promote a snapshot into Git only when another person must reproduce the comparison exactly.
