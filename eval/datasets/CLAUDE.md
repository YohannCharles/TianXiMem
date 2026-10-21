# eval/datasets

This package prepares source data and converts each dataset into service input and evaluation items.

## Responsibilities

```text
manifest.py    URLs, revisions, hashes, and declared local patches
layout.py      manifest name to local path mapping
prepare.py     download, verify, patch, publish, and migration
preprocess.py  schema normalization shared by loaders
registry.py    dataset registry and data fingerprints
aml/           AML-shaped Add/Search event planners
<dataset>.py   dataset-specific loading and gold extraction
```

## Boundaries

- Preserve upstream source bytes; apply only manifest-declared patches.
- Keep dataset file names and paths in this package, not in `src/`.
- Do not place gold answers in Add payloads or answer prompts.
- Keep golden data available only to the scoring side.
- Use `TIANXIMEM_BENCHMARK_DIR` as the only data-root configuration.
- Fail on missing selected questions or malformed source data.
- Record dataset, input-contract, and scoring fingerprints in the run record.

## Input Contracts

`native` uses the dataset's existing loader, answer input, and judge path.

`aml-v1` constructs Add and Search events that follow AML traffic patterns while retaining native questions and scoring. A dataset without a reliable mapping must remain `native`.

The two contracts are not interchangeable. Scores produced under different contracts are not directly comparable.

## Data Shapes

- Long conversations are split into explicit Add batches.
- Session and user identities must remain consistent with the planned dataset scope.
- Timestamps are synthesized only when the source has no message timestamp.
- Gold evidence is used for scoring and diagnostics, not corpus construction unless the declared native contract requires it.

## Validation

- Manifest hashes identify the exact source bytes.
- Local patches are mechanical and covered by tests.
- Parsed source counts and fingerprint inputs are recorded.
- Data preparation is atomic; failed verification publishes nothing.

Operational instructions are in [`../../docs/benchmark-data.md`](../../docs/benchmark-data.md).
