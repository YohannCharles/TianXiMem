# eval

`eval/` drives local proxy evaluation against the service over HTTP.

## Layout

| Directory | Responsibility |
| --- | --- |
| `datasets/` | Dataset manifest, preparation, loaders, and AML-shaped input planning |
| `harness/` | Add/Search HTTP driving, answer adapters, judging, and run records |
| `experiments/` | Named experiment arms, runners, and shared fixed selections |
| `baselines/` | External reference implementations; never imported by service code |
| `smoke/` | Contract preflight and AML-facing probes |
| `reports/` | Generated measurements and run artifacts |

## Main Boundary

Harness code drives the service over HTTP. It must not import `tianximem` internals.

This boundary verifies the public contract and keeps external baselines as independent services.

## Evaluation Layers

| Layer | Purpose |
| --- | --- |
| Local proxy evaluation | Iteration, comparisons, and threshold selection |
| AML Smoke | Contract validation and locally unobservable platform behavior |
| AML Full | Final frozen submission |

Do not spend AML quota on questions that local evaluation can answer.

## Measurement Rules

- Compare runs only when question selection, data version, add shape, model endpoints, budgets, and scoring code match.
- Record configuration, data, model, and code fingerprints in every run.
- Keep technical errors visible.
- Keep run outputs under `eval/reports/` and out of Git.
- Treat partial-credit metrics as diagnostic only unless a dataset's official score uses them.
- Do not extrapolate one dataset's injection or judge behavior to another dataset.

## Fixed Selection

Use [`../docs/fixed-selection-suite.md`](../docs/fixed-selection-suite.md) for shared question IDs and version comparison.

## Official Capture

Use [`../docs/capture-evaluation-pipeline.md`](../docs/capture-evaluation-pipeline.md) for replay of captured AML requests.
