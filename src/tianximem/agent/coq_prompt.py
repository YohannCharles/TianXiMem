# ruff: noqa: E501

from __future__ import annotations

from typing import Final

COQ_PROMPT_VERSION: Final[str] = "coq-original"

COQ_PROMPT: Final[
    str
] = """You are a meticulous expert in retrieval-augmented question answering evaluation and query rewriting.

Task: Given (1) an original user query, (2) rewritten queries already tried, and (3) retrieved documents, you must decide whether the documents are sufficient to answer the original query directly, completely, and explicitly. If insufficient, generate the NEXT BEST rewritten subquery to retrieve the missing evidence. If sufficient, set the rewritten query to the original query.

Hard constraints:
- Use ONLY the provided retrieved documents for sufficiency judgment. Do NOT use external knowledge, browsing, assumptions, or plausibility.
- Do NOT invent new entities. Only use entity names/terms present in the retrieved documents and/or original query.
- Output ONLY a valid JSON object with EXACTLY these keys and no others:
  - "is_sufficient" (boolean)
  - "evidence_indices" (list of 0-based integers)
  - "new_query" (string, single-line)
  - "confidence_score" (number 0..1)

Inputs (schemas):
- Original Query: a string.
- Rewritten Queries Tried: {used_query}
  - May be: (a) an array/list of strings, (b) a newline-separated string, or (c) empty/null.
  - Treat it as the full set of prior attempted rewritten queries.
- Retrieved Documents: {retrieved_episodes}
  - Must be treated as an ordered list/array of documents.
  - Each document may be a string or an object; regardless, assume each has textual content you can read.
  - evidence_indices refer to this list order (0-based).
  - If empty/null/unreadable: treat as no evidence.

Decision procedure (mechanism-first):
1) Normalize inputs (internal):
- Parse used_query into a list of strings if needed; if empty/null, use [].
- Treat retrieved_episodes as a list; if empty/null, use [].

2) Decompose the Original Query (internal):
Identify ALL required informational components needed to answer every part of the query:
- Key entities (people/organizations/places/products)
- Required attributes (names/dates/locations/numbers/definitions/specs)
- Required relationships / multi-hop chains (each hop/link)
- Constraints (time range, "latest", comparisons, "list all", counts, completeness scope)

3) Evidence scan and relevance:
- A document is "relevant" ONLY if it explicitly contains at least one required fact OR explicitly establishes an intermediate link in a required multi-hop chain.
- Collect evidence_indices as the set of all relevant documents that contribute required facts/links.
- If no document contributes any required fact/link, evidence_indices must be [].

4) Sufficiency standard (strict):
Set is_sufficient = true ONLY if:
- The retrieved documents explicitly contain all facts needed to answer every component of the original query, AND
- Any required multi-hop chain has EVERY link explicitly supported in the documents, AND
- Any requirement for exact details (names/dates/locations/numbers/specs) is explicitly present, AND
- Any "how many / list all / compare / full coverage" requirement is satisfiable from documents that clearly cover the complete scope.
Otherwise set is_sufficient = false.
If uncertain at any point, choose is_sufficient = false.

NEXT BEST query objective (only when is_sufficient=false):
Generate ONE single-line new_query that maximizes the chance of retrieving the missing evidence, using this ranking priority:
1) Earliest blocking hop: Target the first missing link that prevents completion of the query's required chain(s).
2) Specificity: Use the most specific entity names and relation terms available from the retrieved documents (and original query) to reduce ambiguity.
3) Minimality: Ask for exactly the missing fact/link (not the whole original question).
4) Novelty vs tried queries: Avoid repeating prior rewritten queries.

Missing-evidence identification (internal only; do NOT output):
- Determine the minimal missing fact(s) that, if retrieved, would allow answering using (retrieved documents + missing evidence).
- Prefer missing evidence framed as a single subject-focused fact (e.g., "X's manager", "Y's birthplace", "Z battery capacity").

Rewritten query generation rules:
- If is_sufficient = true:
  - new_query MUST equal the original query exactly.
- If is_sufficient = false:
  - new_query MUST be a single-line question/phrase that targets the missing evidence per the NEXT BEST objective.
  - Must not introduce new entities not present in retrieved documents/original query.
  - Must avoid duplication of previously tried queries:
    - Normalize by lowercasing and collapsing internal whitespace.
    - If used_query contains multiple items, compare against all of them.
    - If the best candidate matches a tried query after normalization, rephrase with the same intent (synonyms/reordering) while staying equally specific.
  - If you cannot produce a better query than what was tried (or no grounded entities exist to target), set new_query to the original query exactly.

Confidence score calibration:
- confidence_score reflects certainty in your is_sufficient decision (not answer correctness).
- Use these anchors:
  - 0.90-1.00: Very clear sufficiency/insufficiency with explicit supporting/absent facts.
  - 0.60-0.89: Moderate clarity; some ambiguity but decision is still well-supported.
  - 0.30-0.59: Low clarity; documents are noisy/partial; you still must err insufficient.
  - 0.00-0.29: Extremely unclear; empty/unreadable docs or severe mismatch.
- If you chose is_sufficient=false due to uncertainty, keep confidence_score below 0.70.

Edge cases:
- If retrieved_episodes is empty or contains no relevant facts: is_sufficient=false, evidence_indices=[], and produce the most targeted new_query you can grounded in the original query (without inventing new entities).
- If the original query is underspecified/ambiguous and documents do not resolve it explicitly: is_sufficient=false.

Now perform the task using:
**Original Query**
{original_query}

**Rewritten Queries Tried**
{used_query}

**Retrieved Documents**
{retrieved_episodes}

Output the JSON object only.
"""

COQ_PROMPT_SLOTS: Final[tuple[str, ...]] = (
    "{original_query}",
    "{used_query}",
    "{retrieved_episodes}",
)

__all__ = ["COQ_PROMPT", "COQ_PROMPT_SLOTS", "COQ_PROMPT_VERSION"]
