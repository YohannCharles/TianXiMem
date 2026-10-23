# facts

`facts/` derives source-backed facts and compiles supported question forms into deterministic evidence plans. It does not call an LLM, generate answers, or count/list results.

## Files

```text
evidence.py      Fact record, normalized keys, extraction version
grammar.py       Source syntax to atomic fact candidates
sales.py         Explicit completed-transaction facts
query.py         Question syntax to QueryPlan
input.py         Current-input dependency check
paths.py         Supported relation-path parsing
conversation.py  Named utterance and sentence splitting
```

## Fact Record

Every fact carries:

- subject, relation, and object;
- source-stated qualifiers such as quantity, container, or time range;
- source side and verbatim source quote;
- source record date;
- extraction version.

Normalized keys are for matching only. They never replace or mutate source quotes.

The Add owner invokes extraction once per retained original message, preserving
its role and timestamp even when several messages share a merged parent block.
The source position participates in derived fact identity. Original-record
projection is resolved and validated by `service/`; the extractor does not
choose response packaging.

## Extraction Rules

- A positive fact requires a literal source assertion.
- Negation, quotations, plans, hypotheticals, and examples do not create positive facts.
- Non-user roles do not establish first-person facts unless the source explicitly identifies the speaker.
- Source dates are not event dates.
- Quantities keep their unit and are not pre-summed.
- Ambiguous quantities, unit conflicts, unknown recency, and unresolved corrections fall back to source retrieval.
- Explicit corrections retain each observation. The executor removes an older observation only when the replacement and ordering are source-backed.
- The extraction version is part of fact identity and coverage. Change it whenever extraction semantics change.

Named-person binding happens during the Add write path. Within one Add, an explicit self-introduction can bind first-person facts to the speaker's name. The operation never crosses an Add boundary and never mutates source rows.

## Query Plans

| Plan | Meaning |
| --- | --- |
| `filter` | Match a subject, relation, object, and source-stated qualifiers. |
| `walk` | Follow a bounded ordered sequence of source-backed relation edges. |
| `interval` | Compare only time ranges explicitly stated by source text. |
| `current` | Return a legal empty result when the question's complete input is present and history is unnecessary. |

Lists and counts use the same evidence plans. The final list and count are AML responsibilities.

Unsupported syntax, incomplete current input, unresolved branches, conflicts, or exceeded limits return control to the normal retrieval chain.

## Source Coverage

`evidence_coverage` means that an extractor version processed a source. It does not mean the extractor understood every statement in that source.

- Empty extraction still records coverage.
- Only the current extraction version counts as scanned.
- Semantic completeness is checked during evidence execution by reviewing relevant source text for unexplained statements and conflicts.

## Boundaries

- Do not write SQL or access Qdrant.
- Do not implement business-category tables or Search branches.
- Do not import `store/`, `retrieve/`, or `service/`.
- Do not render text outside `common/render.py`.
- Do not call a model in the Add path.

## Validation

- A fact quote must occur verbatim in the declared source side.
- A field quote must occur within the source quote.
- A fact user must match its parent memory user.
- Fact IDs are source-derived and insert-once.
