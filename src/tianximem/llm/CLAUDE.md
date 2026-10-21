# llm

`llm/` provides a narrow OpenAI-compatible chat interface used by optional Search components.

```text
base.py        LLM protocol and LlmUnavailable
openai_chat.py OpenAI-compatible /chat/completions client
```

## Contract

The protocol exposes one operation:

```python
complete(prompt: str) -> LLMResult
```

The client accepts endpoint, key, and model from the caller. It does not read environment variables or configuration.

`LlmUnavailable` represents an unavailable optional dependency. Callers convert it into a named degraded result instead of returning a service error.

## Consumers

| Consumer | Calls |
| --- | --- |
| Grounded planner | At most one bounded plan-generation request when deterministic compilation fails. |
| Query router | At most one classification request per Search. |
| Multi-hop executor | One request per round, with `plan` mode also using one decomposition request. |

The Add path never calls an LLM.

## Boundaries

- Do not add write-time summarization, entity merging, or fact generation through an LLM.
- Do not train or fine-tune on evaluation data.
- Do not hide model, endpoint, or prompt-version mismatches.
- Optional-path failures must leave a diagnostic reason and must not turn Search into a 500.
