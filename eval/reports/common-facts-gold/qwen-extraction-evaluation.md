# Qwen Common Fact Extraction Evaluation

## Run

- Dataset: `eval/facts/common-facts-gold.jsonl` (900 cases)
- Model: `Qwen/Qwen3.5-9B`
- Endpoint: `https://memory.021130.xyz/v1`
- Prompt: `common-facts-qwen-v4`
- Output cache: `qwen3.5-9b-v4.jsonl`
- Temperature: `0`
- Scope: offline evaluation only; Add/Search remain LLM-free

The endpoint was verified with `/v1/models` before the run. This host uses the
`AML_EMB_BASE_URL` / `AML_EMB_API_KEY` pair in the local environment even
though it also serves the Qwen chat model.

## Exact triple scores

| Extractor | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules | 732 | 0 | 118 | 1.0000 | 0.8612 | 0.9254 |
| spaCy sidecar | 91 | 36 | 759 | 0.7165 | 0.1071 | 0.1863 |
| Rules + spaCy | 746 | 22 | 104 | 0.9714 | 0.8776 | 0.9221 |
| Qwen v4 | 788 | 85 | 62 | 0.9026 | 0.9271 | 0.9147 |

The Qwen run adds 42 true positives and removes 42 false negatives compared
with rules + spaCy, but also adds 63 false positives. Its F1 is therefore 0.74
percentage points below the deterministic combination. The useful gain is
recall, especially on sentence forms not covered by the current grammar.

## Main errors

- Long or unusual `position` statements are sometimes mapped to `officeholder`.
- Profile rows can use the whole department string as the organization instead
  of the shorter gold endpoint.
- Long biographies produce shorter endpoints than the current gold annotation;
  these are boundary disagreements rather than clearly hallucinated facts.
- A small number of synthetic `Corpus:` relations are mapped to nearby labels
  such as `creator`, `location of formation`, or `head of government`.
- One conversational filler row is annotated as a fact in the gold file even
  though the prompt correctly returns no relation.

## Decision

Use Qwen as a high-recall candidate generator or adjudication stage, not as a
direct replacement for the current deterministic extractor. A production
experiment should first constrain outputs against the source grammar and
relation vocabulary, then accept an LLM candidate only when its endpoints and
relation agree with a deterministic or source-span check. This keeps the
current precision while testing whether the LLM recall can be recovered.
