# configs

YAML profiles own runtime thresholds, weights, model IDs, collection names, and feature switches.

## Files

```text
default.yaml          Product profile and parent of all profiles
local.yaml            Development overrides
submit.yaml           Submission overrides
v1.0.yaml             v1.0 retrieval compatibility
probe-nocapture.yaml  Diagnostic profile without request capture
probe-react.yaml      Diagnostic react-executor profile
runs/                 Ignored configuration snapshots for isolated evaluations
```

## Rules

- Environment variables own credentials, endpoints, model IDs for endpoints, paths, and process shape.
- YAML owns thresholds, weights, collection names, and feature switches.
- `default.yaml` is the product source of truth.
- A profile contains only keys that intentionally differ from its parent.
- Unknown keys are validation errors.
- A new key requires a runtime consumer, validation, and tests.
- Do not duplicate active defaults in code, command wrappers, or evaluation scripts.

## Value Classes

- A: external contract constant.
- B: correctness or identity constant.
- C: tunable threshold or switch.

`rrf.k=61`, prompt versions, and rerank envelope values are B-class constraints. Changing them requires a deliberate identity change, not threshold tuning.

## Loading

```text
built-in defaults
  -> default.yaml
  -> selected profile
  -> real environment
  -> .env environment for values not already set
  -> validation
```

`TIANXIMEM_PROFILE` selects a profile. `TIANXIMEM_CONFIG_DIR` selects another configuration directory. `TIANXIMEM_ENV_FILE` selects the dotenv file.

The complete key reference is [`../docs/config-reference.md`](../docs/config-reference.md).
