# Development Commands

The canonical command entry point is the Python module:

```bash
uv run python -m tianximem.dev help
```

The same command set works on Linux, macOS, and Windows. `Makefile` targets are compatibility aliases.

## Commands

```text
python -m tianximem.dev sync
python -m tianximem.dev lock
python -m tianximem.dev format
python -m tianximem.dev format-check
python -m tianximem.dev lint
python -m tianximem.dev typecheck
python -m tianximem.dev test
python -m tianximem.dev serve
python -m tianximem.dev config-check
python -m tianximem.dev check
python -m tianximem.dev data-check --dataset <name>
python -m tianximem.dev data-fetch --dataset <name>
python -m tianximem.dev data-patch --dataset <name>
python -m tianximem.dev contract-check
python -m tianximem.dev diagnose
python -m tianximem.dev probe-reranker
python -m tianximem.dev router-eval
python -m tianximem.dev coq-eval
python -m tianximem.dev coq-decompose
python -m tianximem.dev order-probe
python -m tianximem.dev export-multihop
python -m tianximem.dev eval
python -m tianximem.dev baseline
python -m tianximem.dev replay-official
python -m tianximem.dev t1
python -m tianximem.dev t2
python -m tianximem.dev t2-dump
python -m tianximem.dev qdrant-up
python -m tianximem.dev qdrant-down
python -m tianximem.dev image-build
python -m tianximem.dev up
python -m tianximem.dev down
python -m tianximem.dev deploy-check
python -m tianximem.dev smoke
python -m tianximem.dev clean
```

Use `uv run python -m tianximem.dev <command> --help` for arguments and exit codes.

Extra arguments are passed through:

```bash
uv run python -m tianximem.dev eval --limit 3
make eval ARGS='--limit 3'
```

Data preparation accepts repeatable `--dataset`, `--tier all`, `--offline`, and `--dir`.

Processes use argument lists rather than shell strings, so paths containing spaces and Windows paths are preserved.

Docker commands require Docker Desktop or Docker Engine. If the current WSL environment cannot access the Docker socket, run the same command from the Windows side.
