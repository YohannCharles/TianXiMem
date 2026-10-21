# eval/reports

This directory stores generated evaluation evidence and run artifacts. It is not a second specification.

## Tracked Content

Only this file and intentionally shared manifests or selections may be tracked. Raw run outputs, per-question data, captured requests, generated answers, and logs remain untracked.

## Run Artifacts

A run directory should contain:

- run and configuration metadata;
- dataset, input-contract, model, and scoring fingerprints;
- checkpoints required for safe resume;
- answer inputs and outputs;
- judgments and aggregate metrics;
- technical errors, missing items, and skipped work.

## Comparison Requirements

A comparison is valid only when both sides use the same:

- dataset and question selection;
- data and scoring fingerprints;
- answer and judge models and endpoints;
- output budgets;
- scheduling settings that affect the result.

Service code and product configuration may differ.

## Data Handling

- Do not commit captured requests or source evaluation text.
- Do not share raw answers or judgments outside the team.
- Do not train or fine-tune on report data.
- Keep one run ID isolated from other service storage and collections.

## Documentation Rule

Do not create narrative design documents here. Current behavior belongs in the owning source, PRD, or `docs/` document. Reports record what a run measured, not how the system should work.
