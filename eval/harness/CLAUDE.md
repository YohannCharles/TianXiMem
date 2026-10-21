# eval/harness

`eval/harness/` drives Add and Search over HTTP and adapts dataset answer/scoring pipelines.

## Main Components

```text
driver.py           HTTP client with bounded retries and checkpoint-safe requests
scheduling.py       bounded user and judging concurrency
plan_driver.py      AML event planning, dispatch, and per-event checkpoints
batching.py         AML batch-size approximation
add_shape.py        AML Add text shape
judge.py            answer and evaluate orchestration
judge_items.py      Search-result to judge-input conversion
judge_results.py    judge-output parsing and resume cleanup
api_config.py       environment adapter for upstream AML pipelines
annotate.py         harness-side relative-time annotation
run_record.py       run fingerprints and result assembly
```

Dataset-specific pipelines remain in files named for their dataset or dataset family.

## HTTP Boundary

The harness must not import `tianximem` internals. It drives the public service as an external client.

Retry rules:

- Add retries only network errors and 5xx responses.
- Search retries only read-safe network errors and 5xx responses.
- 4xx responses fail immediately.
- Requests that write memory are never regenerated with a different payload on retry.

## Add Shape and Batching

`add_shape.py` constructs the text AML is expected to send, including role labels and long-message splitting. `batching.py` approximates AML's message and word limits.

The exact platform adapter is external and unavailable locally. Batch boundaries and synthesized timestamps are therefore recorded as input fingerprints and are not assumed byte-identical to platform traffic.

Changing add shape or batching changes the input contract and requires a new comparison.

## Event Ordering and Resume

- Execute each user's Add and Search events in source order.
- Persist a successful Search before allowing later Add events for that user.
- On resume, reuse successful Search checkpoints.
- Never query an earlier point in the timeline after later memory has been added.
- Use a new run ID when service storage or input fingerprints change.

## Answer and Judge Adapters

Dataset-specific answer construction must:

- use only the question and returned Search evidence;
- never inject gold answers or gold evidence into the answer prompt;
- preserve the dataset's native output format where required;
- record the answer contract, model, endpoint, and output budget;
- reject reuse of answers produced under a different contract or fingerprint.

Judge adapters follow upstream pipeline code for native datasets. Local adapters are valid only for within-dataset comparisons.

## Upstream Pipelines

Upstream AML pipeline scripts are invoked as subprocesses.

- They are not imported as modules.
- `api_config.py` supplies their expected environment names.
- Declared local patches are limited to runnable syntax compatibility.
- Pipeline I/O follows the scripts, not README prose, where they differ.

## Checkpoint Hygiene

- JSONL files are sanitized for a trailing partial record before resume.
- Per-question output uses isolated directories when judging runs concurrently.
- User aggregate files are appended under a lock and deduplicated by question ID.
- Failed technical work is visible in the run record and can be retried.

## Tests

Tests use fake HTTP clients, fake pipelines, and isolated temporary files. They must not call real endpoints.
