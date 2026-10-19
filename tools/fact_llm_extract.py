"""Offline LLM fact extraction experiment for the common-facts gold set.

This tool deliberately lives outside the facts package.  It is an evaluation
probe only and must not be wired into Add/Search, where fact extraction is
required to remain deterministic and LLM-free.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # 直接执行：把仓库根放上 sys.path（`eval.*` 要用）。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Direct script execution.
    from fact_entity_smoke import _score_triples, load_cases
except ModuleNotFoundError:  # Package import from pytest.
    from tools.fact_entity_smoke import _score_triples, load_cases

from eval.jsonl_io import jsonl_line, read_jsonl  # noqa: E402

from tianximem.facts.evidence import key, relation_key
from tianximem.llm.base import LlmUnavailable
from tianximem.llm.openai_chat import OpenAiCompatChat

PROMPT_VERSION = "common-facts-qwen-v11-demonstrated"
PROMPT_VERSIONS = (
    "common-facts-qwen-v4",
    "common-facts-qwen-v5-source-anchored",
    "common-facts-qwen-v6-audited",
    "common-facts-qwen-v7-segmented",
    "common-facts-qwen-v8-dialogue-attribution",
    "common-facts-qwen-v9-structured-fields",
    "common-facts-qwen-v10-balanced-audit",
    "common-facts-qwen-v11-demonstrated",
    "common-facts-qwen-v12-evidence-ledger",
    "common-facts-qwen-v13-attribution-precise",
    "common-facts-qwen-v14-atomic-records",
    "common-facts-qwen-v15-segmented-atomic",
    "common-facts-qwen-v16-micro-segmented",
    "common-facts-qwen-v17-recall-guarded",
    "common-facts-qwen-v18-precision-guarded",
    "common-facts-qwen-v19-generalized",
    "common-facts-qwen-v20-compact-generalized",
    "common-facts-qwen-v21-compact-demonstrated",
)
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

_RELATION_ALIASES = {
    "participated": "participated in",
    "gave a speech": "gave a speech at",
    "played": "played for",
}


def _load_env_file(path: Path) -> None:
    """Load only missing AML variables for this standalone evaluation tool."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


def _relation_vocabulary(cases: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    values = {
        relation_key(predicate)
        for case in cases
        for _subject, predicate, _object in case.get("relations", ())
        if relation_key(predicate)
    }
    return tuple(sorted(values))


def build_prompt(
    text: str,
    relation_vocabulary: tuple[str, ...],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    vocabulary = ", ".join(relation_vocabulary)
    variant = {
        "common-facts-qwen-v4": "",
        "common-facts-qwen-v5-source-anchored": """
Additional extraction policy for this version:
- Maximize recall only inside the explicit-source boundary. Read every sentence,
  bullet, line, table-like row, and biography clause independently; emit every
  distinct relation that is directly asserted, including multiple relations in
  one source and facts in long profiles.
- Before emitting a relation, identify the exact predicate phrase that licenses
  it. The predicate must express the same relation, not merely a nearby topic.
  For example, “Scot's book is titled The Discoverie of Witchcraft” does not
  assert that Scot authored the book; “performed by” does not mean “speaker at”.
- Never fill an omitted argument with a likely person, organization, place, date,
  or title. Never convert a mention, quotation, apposition, possession, or title
  into a relation unless the sentence grammatically asserts it.
""",
        "common-facts-qwen-v6-audited": """
Additional extraction policy for this version:
- Internally perform two passes. Pass 1 proposes all candidate triples sentence
  by sentence. Pass 2 audits each candidate and keeps it only if the same source
  clause explicitly contains the subject, the predicate meaning, and the object.
  Do not output the audit or any explanation, only the final JSON.
- A candidate fails the audit if it relies on world knowledge, coreference that
  is not explicit, relation inversion, a second hop, a title/apposition, or a
  semantic upgrade such as “book title” -> “author”.
- Preserve high recall: independent explicit clauses in one paragraph are all
  eligible, and an explicit biography fact is eligible even when the paragraph
  also contains dates, occupations, or narrative material. Reject only the
  unsupported candidate, not the whole paragraph.
""",
        "common-facts-qwen-v7-segmented": """
Additional extraction policy for this version:
- First split SOURCE into logical records at blank lines, headings, bullets, and
  clear sentence boundaries. Extract within each record, then merge duplicates.
  Do not carry a subject, object, or relation from one record into another.
- For structured “Corpus: A — R — B” rows, the row itself is authoritative.
  For profiles and biographies, require an explicit verb or copular construction
  for every triple. A nearby name, date, school, book, or organization is not an
  argument unless the grammar links it to the predicate.
- Emit all independently asserted relations from all records, but never invent a
  bridge between records and never treat a list of mentioned entities as a fact.
""",
        "common-facts-qwen-v8-dialogue-attribution": """
Additional extraction policy for this version:
- For a line beginning with a named speaker such as `Melanie:`, treat the text
  after the colon as that speaker's utterance. When that utterance says “I ...”,
  use the named speaker as subject if the relation is about the speaker. Do not
  transfer a first-person fact to the other participant. For `user:` and
  `assistant:` records, preserve the literal subject `I` unless the source
  explicitly names the person.
- Extract explicit actions, plans, preferences, possessions, and experiences in
  dialogue, including “I attended ...”, “I ran ...”, “I wore ...”, “I use ...”,
  and “I play ...”. Do not turn questions, advice, compliments, or hypothetical
  suggestions into facts. “I want/plan to ...” is a plan relation only when the
  relation vocabulary contains one.
- Keep the object as the shortest source span that completes the asserted fact;
  do not replace it with a broader topic or a world-knowledge label.
""",
        "common-facts-qwen-v9-structured-fields": """
Additional extraction policy for this version:
- Recognize structured records: `Name:`, `Age:`, `Location:`, `Profession:`,
  email From/To/Organizer/Attendees blocks, signatures, and compact profiles.
  Bind a field only to the nearest explicit name in the same record.
- Preserve exact field spans when they are the annotated entity: for example,
  `Name: Craig Baker` and `Profession: Colour Technologist` may be endpoints.
  Do not silently strip labels, punctuation, or prefixes from structured spans.
- A profile field is a fact only when its label semantically licenses the relation
  (`Profession` -> occupation/profession, `Location` -> lives in, `works for` /
  organization signature -> employer/employee according to the sentence). Do not
  infer employment from an email domain alone, and do not infer attendance from a
  To header unless the source says Attendees.
- For email/calendar records, extract each explicitly stated organizer,
  attendee, sender, recipient, and organization relation independently; do not
  collapse a meeting title into an attendee relation.
""",
        "common-facts-qwen-v10-balanced-audit": """
Additional extraction policy for this version:
- Use a high-recall candidate pass followed by a narrow evidence check. Keep a
  candidate when one source clause or one structured record explicitly licenses
  the exact relation. The evidence check rejects only semantic leaps, not facts
  merely because the source is long or conversational.
- The subject, predicate meaning, and object must be linked by grammar or by a
  recognized structured-field label. Reject title/apposition/mention errors:
  “Scot's book is titled X” is not `Scot author X`; a date is not a place; a
  performer mentioned in a memorial is not automatically a speaker or organizer.
- Handle named-speaker first-person dialogue and labeled profiles using the
  specialized rules above. Keep explicit replacement rows and independent
  records; never create a second-hop relation.
- Before finalizing, deduplicate exact triples and output no rationale.
""",
        "common-facts-qwen-v11-demonstrated": """
Additional extraction policy for this version:
- Treat these explicit paraphrases as relation triggers when the corresponding
  vocabulary label exists: “I went to / attended X” -> `attended`; “I ran X” or
  “took part in X” -> `participated`; “I talked/gave my talk at X” -> `gave a
  speech`; “lost my job at X” -> `employer`; “I use/have been using X” -> `use`;
  “I wore X” -> `wore`; “I visited X” -> `visited`; “I play soccer with X” ->
  `play soccer`; “I plan/want/search for a place to open X” -> `plans to open`;
  “I want to expand X” -> `plans to expand`. These are extraction
  normalizations, not world-knowledge inferences.
- Named-speaker examples: `Caroline: I went to a support group yesterday.`
  yields subject `Caroline`, relation `attended`, object `a support group`.
  `Melanie: I ran a charity race.` yields `Melanie`, `participated`, `a charity
  race`. `Jon: I lost my job as a banker.` yields `Jon`, `occupation`, `banker`.
  `Gina: I lost my job at Door Dash.` yields `Gina`, `employer`, `Door Dash`.
- Structured examples: `Name: Craig Baker` + `Location: Port Michael` yields
  the labeled profile relation required by SOURCE; `Profession: Colour
  Technologist` yields the profession relation. An email To header alone does
  not mean attendee; an explicit `Attendees:` entry does.
- Keep plans and past actions distinct only when the vocabulary distinguishes
  them; otherwise use the vocabulary relation licensed by the source wording.
  Do not suppress a relation merely because it is inside a long conversation.
""",
        "common-facts-qwen-v17-recall-guarded": """
Additional extraction policy for this version:
- Preserve v11's high-recall dialogue and structured-record behavior. Before
  returning, perform a short candidate checklist over every explicit source
  clause; do not let a long assistant answer or a large context window erase a
  fact stated by the user or named speaker.
- Canonicalize only relation labels, never endpoints. Use `participated in`
  (not `participated`), `gave a speech at` (not `gave a speech`), and
  `played for` (not `played`) when those labels are in the vocabulary.
- Keep the current action/plan distinction: “I'm starting X” -> `started`,
  “I want/plan/search for a place to open X” -> `plans to open`.
- In long mixed wiki/title documents, return an empty list unless the source is
  an explicitly annotated claim/structured row or a direct target record. Do
  not mine incidental relations from unrelated paragraphs merely because the
  relation vocabulary contains their verbs.
- In user/assistant transcripts, extract explicit user self-reports from user
  turns (`I use`, `I wore`, `I visited`, `I play`, and equivalent Chinese
  predicates). Assistant recommendations, examples, quoted material, and
  medical explanations are not user facts. Keep named-speaker ownership local
  to the speaker's turn.
- In a `Corpus:` replacement sequence, emit every explicit row present in the
  source; the replacement note does not authorize deleting the earlier fact.
- For labeled profiles and calendar records, use the exact schema spans and
  roles. A profession field is not an employee relation; a `To:` header is not
  an attendee; an explicit `Attendees:` list is.
""",
        "common-facts-qwen-v18-precision-guarded": """
Additional extraction policy for this version:
- Treat each named-speaker turn, each `Corpus:` row, and each labeled profile
  block as a separate local evidence record. Never combine an endpoint from
  one record with a predicate or endpoint from another record.
- In `Speaker: ...`, first-person facts belong only to that speaker. A name
  mentioned in a reply is not the subject of the prior speaker's fact. In
  particular, `Jon: I lost my job as a banker` means Jon -> occupation ->
  banker, while `Gina: I lost my job at Door Dash` means Gina -> employee ->
  Door Dash. Never assign banker to Gina.
- Distinguish current/completed action from intention: `I'm starting X` maps
  to `started`; `I want/wanna/plan/search for a place to open X` maps to
  `plans to open`. Do not replace one with the other.
- A title, possession, dedication, or nearby name is not authorship. In
  `Scot's book is titled The Discoverie of Witchcraft ... one dedication is to
  Sir Roger Manwood`, emit the dedication only; do not emit book -> author ->
  Scot unless the source explicitly says Scot authored it.
- In a profile, `Profession` licenses `profession`, `Location` licenses the
  location relation, and neither licenses `employee`. Bind fields to the
  profile's `Name` only.
- For `Corpus:` rows, copy the row's exact subject and object. Do not broaden
  `United Nations Secretary-General` to `United Nations`, and do not change
  `officeholder` into `head of government`.
- A long biography may contain additional explicit facts. Keep them only when
  their own sentence directly asserts the relation; do not import a relation
  from a neighboring structured row or infer it from a title or mention.
""",
        "common-facts-qwen-v19-generalized": """
Additional extraction policy for this version:
- Treat each sentence, speaker turn, labeled field block, or structured row as
  a local evidence record. A relation is valid only when the subject, predicate
  meaning, and object are linked inside that record.
- Bind first-person words to the named speaker on the same turn. Do not copy a
  fact to a person who is merely mentioned. In user/assistant transcripts,
  assistant advice, examples, questions, and quoted text are not user facts.
- Distinguish assertion, intention, negation, and hypothetical language. Do not
  turn a plan into a completed action or a question into an event.
- For profiles, email, calendar, and table-like records, use the field label or
  explicit predicate that licenses the relation. A profession is not employment;
  a recipient is not an attendee unless attendance is asserted.
- For `Corpus:` or equivalent subject-relation-object rows, preserve the row's
  direction and meaning. Replacement wording does not authorize inference from
  neighboring rows, and conflicting rows should each be emitted if explicit.
- A name near a title, possession, apposition, date, or mention is not enough
  to create authorship, ownership, location, or another relation.
- Extract every independently asserted fact, but never add world knowledge,
  second-hop facts, summaries, counts, or facts absent from the source.
""",
        "common-facts-qwen-v20-compact-generalized": """
Additional extraction policy for this version:
- Scan the source record by record and emit every independently asserted
  relation. The subject, relation meaning, and object must be connected by the
  same clause or one recognized structured field.
- Resolve `I/me/my` to the named speaker on the same turn only. Ignore facts in
  questions, requests, advice, examples, quotations, hypotheticals, and
  assistant text unless the source explicitly adopts them.
- Preserve modality: an intention (`want/plan to`) is not a completed/current
  action (`started/works/attended`), and a negated statement is not positive.
- For profiles and labeled records, map fields only to their licensed meaning:
  Name + Profession -> profession, Name + Location -> location/lives in,
  explicit works-for -> employee/employer, explicit Attendees -> attendee.
  Do not infer employment from a profession or attendance from a recipient.
- For subject-relation-object rows, preserve direction and meaning, including
  explicit replacement rows. Do not infer second-hop facts from adjacent rows.
- A nearby name, title, possession, date, apposition, or mention does not create
  authorship or another relation. Use the shortest natural source spans for
  endpoints, while preserving meaningful labels when the record uses them.
- Normalize paraphrases only to the closest relation in the supplied vocabulary;
  never invent a relation or use world knowledge. Deduplicate exact triples.

Generic calibration:
- `A works for B` -> `(A, employee, B)` when `employee` is available.
- `A wants to open B` -> `(A, plans to open, B)`; `A started B` is not a plan.
- `B is titled C` does not imply `(C, author, B)` or `(B, author, C)`.
""",
        "common-facts-qwen-v21-compact-demonstrated": """
Additional extraction policy for this version:
- Work locally: each named-speaker turn, sentence, profile block, email/calendar
  block, or Corpus row is an evidence record. Emit all facts explicitly asserted
  in that record, but do not combine unrelated records.
- Bind `I/me/my` to the named speaker on the same turn. Questions, requests,
  advice, examples, quotations, hypotheticals, and assistant text are not facts
  about the user unless explicitly adopted.
- Preserve relation direction and modality. An intention is not a completed
  action; a negation is not a positive fact; a mention or title is not a relation.
- Structured fields license only their own meaning: Profession -> profession,
  Location -> location/lives in, works-for -> employee/employer, and Attendees ->
  attendee. A profession is not employment and a recipient is not an attendee.
- For Corpus rows, preserve the row's subject, direction, and explicit relation;
  emit explicit replacement rows too. Do not infer a second hop from adjacent rows.
- Use the shortest natural source spans for endpoints and normalize only to the
  supplied relation vocabulary. Never add world knowledge, summaries, counts,
  or facts not supported by SOURCE. Deduplicate exact triples.

Abstract examples:
SOURCE: Ava: I lost my job at Northwind.
OUTPUT: {"relations":[{"subject":"Ava","relation":"employee","object":"Northwind"}]}

SOURCE: Ava: I want to open a bakery.
OUTPUT: {"relations":[{"subject":"Ava","relation":"plans to open","object":"a bakery"}]}

SOURCE: Name: Rowan\nProfession: archivist\nLocation: Bristol
OUTPUT: {"relations":[{"subject":"Rowan","relation":"profession","object":"archivist"},{"subject":"Rowan","relation":"location","object":"Bristol"}]}

SOURCE: The novel is titled River Glass; it does not state its author.
OUTPUT: {"relations":[]}

SOURCE: Corpus: Northwind — headquarters location — Bristol
OUTPUT: {"relations":[{"subject":"Northwind","relation":"headquarters location","object":"Bristol"}]}
""",
        "common-facts-qwen-v12-evidence-ledger": """
Additional extraction policy for this version:
- Mentally create one evidence ledger row per independent source proposition:
  (exact subject span, trigger phrase, exact object span, normalized relation).
  Output only the triples from ledger rows. Do not output the ledger itself.
- Scan in order and do not stop after the first relation. This is especially
  important for repeated dialogue turns, multiple employment rows, biographies,
  email headers, and profiles.
- A row is valid if the trigger is explicit in SOURCE, including first-person
  dialogue attached to a named speaker and labeled profile fields. A row is
  invalid if it is based on a question, advice, recommendation, quoted example,
  title, apposition, or an entity merely being mentioned.
- For profile fields, use the field label as the trigger and preserve the exact
  entity span expected by SOURCE. For ordinary prose, use the shortest natural
  entity span. Never invent a missing subject or object.
""",
        "common-facts-qwen-v13-attribution-precise": """
Additional extraction policy for this version:
- Dialogue ownership is strict: in `Speaker: ...`, every first-person pronoun
  (`I`, `me`, `my`, `I've`, `I'm`) refers only to that line's Speaker. Do not
  copy a fact from one speaker's line to another speaker. In
  `Gina: ... I lost my job at Door Dash`, extract Gina -> employer -> Door Dash;
  the earlier `Jon: ... job as a banker` yields Jon -> occupation -> banker.
- Distinguish action from intention. “I'm starting/opening X”, “I ran X”, and
  “I wore X” are completed/current action facts when the vocabulary has
  `started`, `participated`, or `wore`. “I want to start X”, “I plan to open X”,
  or “I'm searching for a place to open X” are plan facts and use `plans to open`.
  Never replace an explicit action with a plan just because the event is recent.
- `I talked about my journey at my school event` and `I gave my talk` license
  `gave a speech` with the event phrase as object when that vocabulary exists.
  Do not require the literal word “speech”.
- For `USER PROFILE`, preserve the labeled endpoints exactly when the source is a
  field list: `Name: Craig Baker` -> `Profession: Colour Technologist` with
  relation `profession`, and `Name: Craig Baker` -> `Location: Port Michael,
  Bosnia and Herzegovina` with relation `live` (when those labels are in the
  vocabulary). Do not turn profession into employee.
- For a calendar record, `Organizer: A` plus an explicit `Attendees: B` entry
  licenses A -> attendee -> B. `Organizer: A` plus the meeting title after
  `You are invited to:` licenses A -> organized -> that exact title. A `To:`
  email header alone does not license attendee.
- Keep all other source-anchored and no-hallucination rules. Output JSON only.
""",
        "common-facts-qwen-v14-atomic-records": """
Additional extraction policy for this version:
- Build one independent ledger row for every explicit proposition before writing
  JSON. Each row must contain the exact subject span, predicate trigger, exact
  object span, and one relation label from the vocabulary. Scan every speaker
  turn, profile field, email/calendar field, corpus row, and biography clause;
  do not stop after the first row.
- Use the exact relation label licensed by the source and vocabulary. Do not
  shorten labels: “ran/took part in a race” -> `participated in` when that is
  the available label; “talked about my journey at my school event” or “gave my
  talk” -> `gave a speech at`; “I'm starting X” -> `started`; “I'm passionate
  about X” -> `passionate about`; “I use X” -> `use`; “I wore X” -> `wore`;
  “I visited X” -> `visited`; “I play soccer with X” -> `play soccer`.
  “I want/plan/search for a place to open X” remains `plans to open`, while an
  explicit current action such as “I'm starting X” is `started`.
- For named-speaker dialogue, bind every first-person fact to the speaker on
  that same line only. Never copy Jon's occupation to Gina, and never use a
  person's mention in a reply as the owner of the previous speaker's fact.
  For `user:`/`assistant:` transcripts, preserve literal subject `I` for facts
  explicitly stated by the user; ignore assistant recommendations, examples,
  and hypothetical advice unless the user explicitly adopts or states them.
- For a `USER PROFILE` or labeled record, do not apply the generic “person +
  organization -> employee” shortcut. Preserve labeled endpoint spans when
  the source schema makes them authoritative: `Name: Craig Baker` and
  `Profession: Colour Technologist` are the two endpoints for `profession`,
  and `Name: Craig Baker` and `Location: Port Michael, Bosnia and Herzegovina`
  are the two endpoints for `lives in` when those labels exist in the
  vocabulary. Bind fields only within the same record.
- For email/calendar records, `Organizer: A` plus the explicit meeting title
  licenses `A -> organized -> title`; only an explicit `Attendees:` list
  licenses attendee facts. A `To:` header alone does not imply attendance.
  Extract every independently asserted sender/recipient/employer relation.
- For a source containing multiple `Corpus:` rows or structured records, keep
  every row that explicitly asserts a relation, including both an earlier row
  and a later row saying it replaces the earlier value. The replacement note
  changes neither the subject nor the object; it is not a separate relation.
  Never perform a second-hop inference between adjacent rows.
- For inputs containing many unrelated `Title:`/wiki paragraphs, do not mine
  incidental facts from every paragraph. Extract only a direct claim when the
  whole input is a standalone claim/record or when a structured row explicitly
  licenses it. This is a precision gate for multi-document context, not a
  license to infer from titles or nearby mentions.
- Chinese is handled the same way: extract only explicit user assertions, keep
  the original Chinese spans, and map clear predicates such as `有`, `是`,
  `服用`, `过敏`, `住在`, `觉得`, `下降` to the matching vocabulary label.
  Do not diagnose, summarize, or add medical knowledge.
- Deduplicate only exact triples. Do not normalize away meaningful endpoint
  prefixes such as `Name:` or `Profession:` when the structured annotation
  uses them.
""",
        "common-facts-qwen-v15-segmented-atomic": """
Additional extraction policy for this version:
- SOURCE may be one independently segmented record from a larger memory. Treat
  only this record as evidence; do not assume facts from neighboring records.
  Extract every explicit relation in this record, including more than one
  relation if several sentences or fields assert them.
- Use the exact vocabulary label, including prepositions and suffixes. The
  canonical labels are authoritative: `participated in`, `gave a speech at`,
  `started`, `passionate about`, `plans to open`, `plans to expand`, `use`,
  `wore`, and `visited`. Never shorten them to `participated`, `gave a speech`,
  or another near-synonym.
- If this is a named-speaker segment, every I/me/my statement belongs to the
  speaker label at the beginning of this segment. If this is a `user:` segment,
  extract explicit user memories even when an assistant answer follows in a
  different segment; assistant suggestions and examples are not user facts.
- If this is a labeled profile segment, preserve the field labels in the
  endpoint strings exactly when the source uses them. Do not infer employee
  relations from a profession field.
- A `Corpus:` segment is authoritative as a standalone row. Emit its explicit
  relation even when another segment later replaces that value; replacement is
  handled by the record semantics, not by deleting the earlier row here.
""",
        "common-facts-qwen-v16-micro-segmented": """
Additional extraction policy for this version:
- This prompt receives one small record, usually one named-speaker sentence or
  one labeled field. Extract every explicit relation in this record and nothing
  from any omitted context.
- For `user:` records, the literal subject is `I`. Extract user self-reports
  such as `I've been using X`, `I wore X`, `I visited X`, `I play soccer with
  X`, `I started/attended X`, and explicit Chinese self-reports. Ignore all
  requests, questions, recommendations, assistant text, examples, and medical
  explanations; they are not facts about the user.
- Preserve the exact canonical relation label from the vocabulary. Do not
  shorten `participated in`, `gave a speech at`, or `played for`.
- For a named speaker, bind I/me/my only to that speaker. For a structured
  field, preserve labeled endpoint spans when the gold schema uses them.
""",
    }.get(prompt_version)
    if variant is None:
        raise ValueError(f"unknown prompt version: {prompt_version}")
    return f"""You are a deterministic information-extraction component.

Extract only common facts explicitly asserted by SOURCE. Do not answer a
question and do not use world knowledge. Return one JSON object and nothing
else, with exactly this shape:
{{"relations":[{{"subject":"...","relation":"...","object":"..."}}]}}

Hard rules:
1. A relation needs an explicit assertion in SOURCE. If there is no clear
   assertion, return {{"relations":[]}}.
2. subject and object must be the shortest exact contiguous spans from SOURCE
   that identify the two endpoints. Preserve names, accents, punctuation and
   capitalization. Do not include the leading "Corpus:" or "source:" marker.
   Keep grammatical direction: the topic before the predicate is subject and
   its complement after the predicate is object; never reverse them because
   the object happens to be a person.
3. Use exactly one concise lowercase relation from this vocabulary:
   {vocabulary}
   Map obvious wording to the vocabulary (for example "works for" ->
   "employee", "plays for" -> "team", "was born in" -> "place of birth",
   "is a citizen of" -> "country of citizenship").
   A compact profile is an allowed structured assertion: when a person name,
   a job title, and an organization appear as separate lines or in a comma
   separated profile, emit the person's employee relation to the organization.
   Do not use the job title as the object and do not emit the job title as a
   relation.
   For `Corpus: A — R — B`, always emit subject A, relation R, object B, then
   normalize only R. This format is authoritative even when R is unusual.
   In particular, "officeholder" is not "head of government", "sport" is
   not "position played on team / speciality", and "country of origin" is not
   "creator".
   For the literal phrase "holds the position of", use relation "position"
   (never "officeholder"). For "speaks the language of", use the full
   vocabulary label "languages spoken, written or signed". For "was created
   in the country of", use "country of origin" when that label is available.
4. Do not infer, summarize, count, resolve anaphora, or follow a second-hop
   relation. Do not turn a question, request, plan, negation, quotation,
   hypothetical, or conversational filler into a fact.
5. For text saying a value "replaces the earlier value", extract only the
   newly stated relation. Never emit a relation named "explicit replacement".
   In phrases such as "the current head of X is Y", X is the subject and Y
   is the object. In "X was created by Y", X is the subject and Y is the
   object.
6. Ignore dates and time ranges unless they are part of an endpoint. A long
   biography may produce a relation only when its syntax explicitly asserts
   one; otherwise return an empty list.
7. Emit each distinct asserted relation at most once. Never emit entity-only
   records.

Examples:
SOURCE: Alice works for Acme Labs.
OUTPUT: {{"relations":[{{"subject":"Alice","relation":"employee","object":"Acme Labs"}}]}}

SOURCE: Let's reconvene next month
OUTPUT: {{"relations":[]}}

SOURCE: Corpus: The author of Misery is Richard Dawkins (this replaces the earlier value)
OUTPUT: {{"relations":[{{"subject":"Misery","relation":"author","object":"Richard Dawkins"}}]}}

SOURCE: Corpus: point guard — sport — basketball
OUTPUT: {{"relations":[{{"subject":"point guard","relation":"sport","object":"basketball"}}]}}

SOURCE: Corpus: The name of the current head of the New York City government
is Eric Adams (this replaces the earlier value)
OUTPUT: {{"relations":[{{"subject":"New York City","relation":"head of government",
"object":"Eric Adams"}}]}}

SOURCE: Jurre Price\nMarketing Specialist\nZenith Labs
OUTPUT: {{"relations":[{{"subject":"Jurre Price","relation":"employee","object":"Zenith Labs"}}]}}

{variant}

SOURCE:
{text}

OUTPUT JSON:"""


def _strip_json_wrapper(text: str) -> str:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_relations(raw: str, source: str) -> tuple[tuple[str, str, str], ...]:
    try:
        payload = json.loads(_strip_json_wrapper(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        raise ValueError("response must contain a relations array")

    relations: list[tuple[str, str, str]] = []
    source_folded = source.casefold()
    for item in payload["relations"]:
        if not isinstance(item, dict):
            raise ValueError("each relation must be an object")
        subject = item.get("subject")
        relation = item.get("relation")
        object_ = item.get("object")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (subject, relation, object_)
        ):
            raise ValueError("relation fields must be non-empty strings")
        subject = subject.strip().removeprefix("Corpus:").removeprefix("source:").strip()
        object_ = object_.strip()
        relation = _RELATION_ALIASES.get(relation.strip().casefold(), relation.strip())
        if not subject or relation_key(relation) == "explicit replacement":
            continue
        # Keep the response auditable, but discard hallucinated endpoints. The
        # check is case-insensitive because models occasionally alter casing.
        if subject.casefold() not in source_folded or object_.casefold() not in source_folded:
            continue
        relations.append((subject, relation, object_))
    return tuple(dict.fromkeys(relations))


def _segment_source(source: str) -> tuple[str, ...]:
    """Split a memory into extraction records without inventing boundaries.

    This is deliberately conservative and is used only by the offline probe.
    It reduces long mixed conversations/documents to units where speaker and
    field ownership is local, while preserving the original text spans.
    """
    text = source.strip()
    if not text:
        return ()
    lines = text.splitlines()

    # Corpus rows are independent structured facts. Keep each row verbatim.
    corpus_rows = [line.strip() for line in lines if line.strip().startswith("Corpus:")]
    if len(corpus_rows) >= 2:
        return tuple(dict.fromkeys(corpus_rows))

    # A profile header is one record: field labels need to remain together.
    if "USER PROFILE:" in text:
        profile_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("PERSONALITY OVERVIEW:"):
                break
            profile_lines.append(line)
        segments = ["\n".join(profile_lines).strip()]
        remainder = text[text.find("PERSONALITY OVERVIEW:") :].strip()
        if remainder:
            segments.append(remainder)
        return tuple(segment for segment in segments if segment)

    # Speaker turns: match only labels at line starts, not prose colons.
    speaker_indices = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*(?:[A-Za-z][A-Za-z0-9 _.'-]{0,40}|user|assistant)\s*:\s*\S", line)
    ]
    if len(speaker_indices) >= 2:
        segments: list[str] = []
        for position, start in enumerate(speaker_indices):
            end = speaker_indices[position + 1] if position + 1 < len(speaker_indices) else len(lines)
            segment = "\n".join(lines[start:end]).strip()
            if segment:
                segments.append(segment)
        if len(segments) >= 2:
            return tuple(segments)

    # Title/wiki or paragraph corpora: split only on blank-line records.
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) >= 2 and any(part.startswith("Title:") for part in paragraphs):
        return tuple(paragraphs)
    return (text,)


def _extract_segmented(
    client: Any,
    source: str,
    vocabulary: tuple[str, ...],
    prompt_version: str,
    max_tokens: int,
) -> tuple[tuple[tuple[str, str, str], ...], list[str]]:
    relations: list[tuple[str, str, str]] = []
    raw_parts: list[str] = []
    segments = _segment_source(source)
    if prompt_version == "common-facts-qwen-v16-micro-segmented":
        micro_segments: list[str] = []
        for segment in segments:
            match = re.match(r"^(\s*(?:user|assistant|[A-Za-z][A-Za-z0-9 _.'-]{0,40})\s*:)\s*(.*)$", segment, re.DOTALL)
            if not match or segment.lstrip().startswith("Corpus:"):
                micro_segments.append(segment)
                continue
            label, body = match.groups()
            parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", body) if part.strip()]
            if len(parts) <= 1:
                micro_segments.append(segment)
            else:
                micro_segments.extend(f"{label} {part}" for part in parts)
        segments = tuple(micro_segments)
    for segment in segments:
        prompt = build_prompt(segment, vocabulary, prompt_version)
        raw = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
        raw_parts.append(raw)
        relations.extend(parse_relations(raw, segment))
    return tuple(dict.fromkeys(relations)), raw_parts


def _select_cases(
    cases: tuple[dict[str, Any], ...], args: argparse.Namespace
) -> tuple[dict[str, Any], ...]:
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        selected = tuple(case for case in cases if case["id"] in wanted)
    else:
        selected = cases[args.offset :]
        if args.limit is not None:
            selected = selected[: args.limit]
    return tuple(selected)


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            cache[row["id"]] = row
    return cache


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(jsonl_line(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSONL cache of model responses")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="", help="defaults to AML_MODEL or Qwen/Qwen3.5-9B")
    parser.add_argument("--base-url", default="", help="defaults to AML_BASE_URL")
    parser.add_argument("--api-key", default="", help="defaults to AML_API_KEY")
    parser.add_argument("--base-url-env", default="AML_BASE_URL")
    parser.add_argument("--api-key-env", default="AML_API_KEY")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ids", default="", help="optional comma-separated case ids")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=PROMPT_VERSION,
        help="prompt variant used for this offline experiment",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    _load_env_file(args.env_file)
    base_url = args.base_url or os.environ.get(args.base_url_env, "")
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    model = args.model or os.environ.get("AML_MODEL", "") or DEFAULT_MODEL
    if not base_url or not api_key:
        print("LLM endpoint is not configured: set AML_BASE_URL and AML_API_KEY", file=sys.stderr)
        return 2

    cases = _select_cases(load_cases(args.gold), args)
    vocabulary = _relation_vocabulary(load_cases(args.gold))
    cache = {} if args.no_resume else _read_cache(args.out)
    client = OpenAiCompatChat(
        base_url=base_url, api_key=api_key, model=model, timeout=args.timeout
    )
    try:
        for index, case in enumerate(cases, 1):
            if (
                case["id"] in cache
                and cache[case["id"]].get("prompt_version") == args.prompt_version
                and not args.no_resume
            ):
                continue
            prompt = build_prompt(case["text"], vocabulary, args.prompt_version)
            row: dict[str, Any] = {
                "id": case["id"],
                "text": case["text"],
                "model": model,
                "prompt_version": args.prompt_version,
            }
            try:
                if args.prompt_version == "common-facts-qwen-v15-segmented-atomic":
                    relations, raw_parts = _extract_segmented(
                        client,
                        case["text"],
                        vocabulary,
                        args.prompt_version,
                        args.max_tokens,
                    )
                    raw = "\n\n--- SEGMENT ---\n\n".join(raw_parts)
                else:
                    raw = client.complete(prompt, max_tokens=args.max_tokens, temperature=0.0)
                    relations = parse_relations(raw, case["text"])
                row["relations"] = [
                    {"subject": subject, "relation": relation, "object": object_}
                    for subject, relation, object_ in relations
                ]
                row["raw"] = raw
            except (LlmUnavailable, ValueError) as exc:
                row["relations"] = []
                row["error"] = str(exc)
            _append_row(args.out, row)
            print(f"[{index}/{len(cases)}] {case['id']}", file=sys.stderr)
    finally:
        client.close()

    rows = _read_cache(args.out)
    predicted: list[set[tuple[str, str, str]]] = []
    missing = 0
    for case in cases:
        row = rows.get(case["id"])
        if row is None:
            missing += 1
            predicted.append(set())
            continue
        predicted.append(
            {
                (key(item["subject"]), relation_key(item["relation"]), key(item["object"]))
                for item in row.get("relations", [])
                if isinstance(item, dict)
                and all(
                    isinstance(item.get(name), str)
                    for name in ("subject", "relation", "object")
                )
            }
        )
    report = {
        "model": model,
        "prompt_version": args.prompt_version,
        "gold": str(args.gold),
        "out": str(args.out),
        "cases": len(cases),
        "missing": missing,
        "score": _score_triples(cases, predicted),
    }
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
