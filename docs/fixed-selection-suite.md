# Shared Fixed Selection Suite

The fixed selection suite evaluates a shared set of native question IDs with existing dataset loaders and scoring pipelines.

- Runner: `uv run python -m eval.experiments.selection_suite`
- Selection: [`eval/experiments/selections/memory600-official10-v1.tsv`](../eval/experiments/selections/memory600-official10-v1.tsv)
- Run output: `eval/reports/runs/<run-id>/` by default

The TSV stores dataset names and native question IDs only. Source content and generated gold data are not stored in Git.

## Verify

```bash
uv run python -m tianximem.dev sync
uv run python -m tianximem.dev data-fetch --dataset <name>
uv run --env-file .env python -m eval.experiments.selection_suite verify
```

Verification loads every selected question and its complete native corpus, records data and scoring fingerprints, and compiles the batch plan. It does not call the service or a model.

Use `--data-root PATH` to override `TIANXIMEM_BENCHMARK_DIR`. Missing selected questions fail verification.

## Run

```bash
uv run --env-file .env python -m eval.experiments.selection_suite run \
  --run-id shared-v1 \
  --service-revision <git-sha> \
  --base-url http://127.0.0.1:8000 \
  --configs-dir configs --profile local --offline
```

The runner preserves each dataset's native corpus and scoring path. It uses HTTP for Add and Search and does not import service internals.

Each run uses a distinct user namespace. Search results are checkpointed before later events are sent. Re-running the same command resumes from successful checkpoints.

Use a new run ID when any of these change:

- service storage was lost;
- question selection changed;
- data or scoring fingerprints changed;
- model endpoint or model identity changed;
- the run must represent an independent sample.

The `--service-revision` value is caller-supplied metadata. Verify it against the running service deployment.

## Summarize

```bash
uv run python -m eval.experiments.selection_suite summarize \
  --run-dir eval/reports/runs/shared-v1
```

The summary reports completed questions, missing questions, correct answers, accuracy, and technical errors. Accuracy is not published when the run is incomplete. Technical errors remain in the denominator.

## Compare

```bash
uv run python -m eval.experiments.selection_suite compare \
  --left eval/reports/runs/shared-v1 \
  --right eval/reports/runs/shared-v2 \
  --output eval/reports/runs/shared-comparison
```

Comparison requires the same selection, answer model, judge model, endpoints, token budget, and scoring source. Service code and product configuration may differ.

## Regenerate Selection

Use `prepare` only when the shared selection itself must change:

```bash
uv run python -m eval.experiments.selection_suite prepare \
  --bundle /path/to/memory600 \
  --capture-dir /path/to/official-capture \
  --selection /path/to/new-selection.tsv \
  --details eval/reports/runs/selection-suite/source-details.json
```

Existing selections are not overwritten. Review any new selection before sharing it.

## Execution Rules

- Run one service version at a time against an isolated collection.
- Reuse the same answer and judge models for both sides of a comparison.
- Do not modify checkpoints by hand.
- Do not mix run IDs across service revisions or data versions.
- Keep captured source data and per-question outputs out of Git.
