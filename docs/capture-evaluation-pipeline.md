# Official Capture Evaluation

Official capture evaluation replays captured Add and Search requests in their original timeline, then answers and scores the selected Search requests.

## Prepare

```bash
uv run python -m eval.experiments.official_dataset prepare \
  --capture-dir <capture-dir> \
  --report-dir eval/reports/runs/official-dataset
```

Preparation joins Search records with available gold data and writes a manifest containing:

- selected Search sequence identifiers;
- user and dataset assignment;
- source file hashes;
- scoring-family assignment;
- whether a record is scorable.

Records without reliable gold data remain available for analysis but are excluded from accuracy denominators. The original capture files are never modified.

The suite may select all scorable questions or use `--per-dataset N` for a deterministic probe.

## Run

```bash
uv run --env-file .env python tools/run_official_eval.py check \
  --base-url http://127.0.0.1:8000

uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-v1 \
  --service-revision <git-sha> \
  --base-url http://127.0.0.1:8000
```

The runner checks the manifest, family dispatch, judge dependencies, endpoints, and `/health` before sending requests.

For every user, Add and Search events are replayed in captured order. Search results are checkpointed after each successful request. A failure blocks later events for that user when the timeline would otherwise expose future memory.

Use `--replay-only` to collect Search results without answer or judge work. Use `--judge-workers N` to overlap answer and judge tasks after each Search checkpoint. Use `--ingest-workers N` to overlap independent users while preserving each user's internal order.

## Failure Handling

| Failure | Behavior |
| --- | --- |
| Add 5xx or network error after bounded retries | Skip the batch, record it in `skipped.jsonl`, and continue. |
| Add 4xx | Fail loudly. This indicates a client or service defect. |
| Search 5xx or network error | Retry with a bound; fail the user timeline if it remains unavailable. |
| Answer or judge subprocess failure | Record the error and continue other queued work. |
| Judge dependency error | Keep the error visible in the technical-error column. |
| Interrupted run | Resume from checkpoints with the same command and run ID. |

## Compare

Run both service versions against the same manifest and identical answer/judge settings:

```bash
uv run python tools/run_official_eval.py compare \
  --left-run-id official-a \
  --right-run-id official-b \
  --left-name a --right-name b
```

Comparison rejects runs with incompatible question selection, scoring source, answer/judge model, endpoints, context budget, or result schema. Service version and product configuration are expected to differ.

## Audit

```bash
uv run python tools/verify_official_gold.py --sample-per-family 3
```

The audit confirms that every scored question has a traceable gold source and that copied gold fields match the source record. It does not prove that the upstream label is semantically correct.

## Output

Run artifacts are written under `eval/reports/runs/<run-id>/`. They include:

- the selected manifest;
- configuration and execution fingerprints;
- per-user request checkpoints;
- answer inputs;
- generated answers;
- judgments;
- aggregate summaries;
- technical errors and skipped requests.

Captured requests and per-question outputs contain evaluation data. Do not commit or share them.
