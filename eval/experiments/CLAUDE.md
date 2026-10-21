# eval/experiments

`eval/experiments/` defines reproducible evaluation arms and shared runners.

## Files

```text
run.py               CLI facade for single-dataset evaluation
runner.py            load, Add, Search, answer, and judge orchestration
record.py            result records and metric summaries
recipes.py           frozen per-dataset sampling defaults
arms.py              configuration snapshot, execution, and comparison helpers
selection_suite.py   shared fixed-question runner
official_dataset.py  official capture replay and comparison
replay_official.py   timeline-preserving capture replay
a3_rerank.py         rerank comparison arm
t1_timestamp.py      date-prefix comparison arm
t2_cross_session.py  cross-session failure analysis
```

## Run a Dataset

```bash
uv run python -m eval.experiments.run \
  --dataset locomo-refined \
  --frozen \
  --base-url http://127.0.0.1:8000
```

`--frozen` applies the sampling recipe from `recipes.py`. Do not replace it with copied limits; each dataset counts a different unit such as conversations, questions, personas, or QA subsets.

The service must already be running. Evaluation uses HTTP and never imports service code.

## Profiles and Fingerprints

Every run records:

- selected dataset and question set;
- data and scoring fingerprints;
- Add shape and batching contract;
- configuration profile and snapshot hashes;
- service revision metadata;
- embedding, LLM, and reranker identities;
- scheduling settings;
- per-answer contract and output budget;
- end-to-end score and available dimension scores;
- technical errors, missing items, and truncation warnings.

Runs with different fingerprints are not directly comparable.

## Comparison Rules

- Compare the same question selection and data version.
- Keep answer and judge models, endpoints, budgets, and scoring source identical.
- Product version and product configuration may differ.
- Use isolated service storage and collections.
- Run one expensive chain at a time unless the experiment explicitly validates concurrent load.
- Treat differences smaller than the established noise floor as inconclusive.

## Switch Purity

Each ablation switch must change only the behavior it names. When a switch is disabled, candidate identity, ordering domains, response shape, and unrelated budgets must remain unchanged.

Check switch semantics in [`../../docs/config-reference.md`](../../docs/config-reference.md).

## Fixed Selection

Use [`../../docs/fixed-selection-suite.md`](../../docs/fixed-selection-suite.md) for shared question IDs and cross-version comparison.

## Official Capture

Use [`../../docs/capture-evaluation-pipeline.md`](../../docs/capture-evaluation-pipeline.md) for timeline replay, judging, and report comparison.

## Output

Generated run artifacts belong under `eval/reports/runs/` and remain untracked. They may contain evaluation source text and must not be redistributed.
