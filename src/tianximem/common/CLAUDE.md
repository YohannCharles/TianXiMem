# common

`common/` owns shared rendering, token counting, and configuration loading.

```text
render.py      QA and grounded-fact rendering
annotate.py    response-only relative-time annotation
amounts.py     response-only transaction amount annotation
tokens.py      o200k_base counting and token-id helpers
config/        environment and YAML loading, models, validation, and constants
```

## Rendering

`render.py` is the only implementation that builds indexed QA text and grounded-fact text.

- Indexing and reranking use the same rendered QA representation.
- A grounded fact is rendered from its verbatim source quote, source date, statement, and supported fields.
- Do not construct a second rendering path in `embed/`, `rank/`, or `retrieve/`.
- A change to indexed rendering changes the embedding input and requires rebuilding the vector collection.

The QA template is:

```text
Q: {question}
A: {answer}
```

Omit an empty side. Add role labels to every non-user answer segment. Keep content self-delimiting and free of leading or trailing whitespace.

When absolute-date injection is enabled, prefix each indexed block with `[YYYY-MM-DD] `.

Relative-time annotation is presentation-only. It may add a deterministic absolute date after a relative expression in returned `content`. The original words remain, and removing the annotation restores the indexed text exactly. This path must not affect embeddings, reranking, IDs, scores, or segment counts.

The harness may keep a separate annotation implementation because evaluation code is forbidden from importing service internals. Tests must prove the two implementations produce equivalent annotations.

## Token Counting

`tokens.py` uses `o200k_base`, matching the submission answer model.

- Use the real tokenizer; do not approximate with characters or whitespace words.
- The configured answer-context ceiling is `117,760` tokens.
- `tokens.py` reports counts. `rank/packaging.py` decides which segments fit.
- Query truncation uses token-id encoding and decoding to fit the embedding model window. It must not truncate indexed documents.

## Configuration

`config/` is the only package allowed to read environment variables or YAML.

```text
__init__.py      public exports
constants.py     environment names and built-in constants
model.py         configuration data classes and errors
env.py           dotenv parsing and environment merging
yaml_source.py   YAML loading, merging, and section construction
validate.py      semantic validation and single-process assertion
loader.py        load order
```

Rules:

- Environment variables own endpoints, credentials, model IDs, paths, and process shape.
- YAML owns thresholds, weights, collection names, and feature switches.
- Unknown keys are errors.
- Configuration keys require runtime consumers.
- Prompt versions and endpoint envelopes are identity assertions, not tuning knobs.
- `.env` is read only through the configuration package and sits below real environment variables.

## Boundaries

`common/` must not depend on business modules such as `store/`, `retrieve/`, or `service/`. Business modules depend on `common/`, not the reverse.
