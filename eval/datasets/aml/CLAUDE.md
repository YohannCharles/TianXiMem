# eval/datasets/aml

This package builds AML-shaped Add and Search event plans from native datasets.

## Contract

Each dataset planner returns ordered events with:

- isolated user identity;
- session identity;
- source-order messages;
- explicit event order between Add and Search;
- native question and scoring metadata kept outside Add content.

The planner must not:

- place gold answers or gold evidence in Add payloads;
- let future events affect an earlier Search;
- rewrite native question text for scoring;
- claim AML compatibility for a dataset without a reliable mapping.

## Files

```text
plan.py      event and input-plan types
registry.py  dataset-to-planner registry
common.py    shared conversation, document, and checkpoint conversions
<dataset>.py dataset-specific planners
```

The run record stores the input contract, planner version, and data fingerprint. Inputs built under `aml-v1` are local approximations and are not byte-identical to platform requests.
