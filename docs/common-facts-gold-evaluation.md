# Grounded Fact Gold Evaluation

The grounded-fact gold file measures deterministic fact extraction against literal source text.

Annotate only facts asserted directly by each source record. Negation, quoted speech, plans, hypotheticals, and examples produce no positive relation.

```json
{
  "id": "sample-id",
  "text": "source text",
  "entities": ["literal entity spans"],
  "relations": [
    {"subject": "literal span", "relation": "literal span", "object": "literal span"}
  ]
}
```

Use exact source spans. Matching applies case and whitespace normalization only.

Run the comparison:

```bash
uv run python tools/fact_entity_smoke.py \
  --model en_core_web_sm \
  --gold eval/facts/common-facts-gold.jsonl \
  --json
```

The output reports exact-triple precision, recall, and F1 for:

- `rules`: deterministic grammar extraction;
- `model_sidecar`: optional spaCy candidates;
- `combined`: the production candidate path.

`errors` lists false positives and false negatives by sample. Use these metrics before changing extraction rules or enabling additional extraction candidates.
