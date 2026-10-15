# Common Fact Gold Evaluation

Use `eval/facts/common-facts-gold.jsonl` as the working annotation file. The
template in `common-facts-gold.template.jsonl` shows the required fields:

```json
{"id":"...","text":"...","entities":["..."],"relations":[{"subject":"...","relation":"...","object":"..."}]}
```

Annotate only facts literally asserted by `text`. For questions, negation,
quoted speech, plans, hypotheticals, and examples, keep `relations` empty.
Keep entity and relation strings as exact source spans; the evaluator applies
case and whitespace normalization only for matching.

Run the comparison with:

```bash
uv run python tools/fact_entity_smoke.py \
  --model en_core_web_sm \
  --gold eval/facts/common-facts-gold.jsonl \
  --json
```

The report contains separate exact-triple precision, recall, and F1 for:

- `rules`: the existing deterministic grammar;
- `model_sidecar`: spaCy entity and surface/structured candidates;
- `combined`: the production extraction path with both.

`errors` lists false positives and false negatives per sample. This is the
metric to use before changing the extraction rules or enabling reranking.
