"""Model-first entity and relation candidates for fact extraction experiments.

This module is deliberately a sidecar to :mod:`facts.grammar`.  It does not
emit ``EvidenceFact`` instances; grammar applies a deterministic validator
before persisting the conservative candidates during Add/Search indexing.
The output keeps source spans so the validator can preserve exact evidence
quotes.

spaCy is an optional dependency.  The extractor accepts an already-loaded
pipeline, which keeps the core package importable without a local model and
makes the component easy to benchmark with a frozen pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tianximem.facts.assertions import interrogative
from tianximem.facts.evidence import key, relation_key

__all__ = [
    "EntityCandidate",
    "EntityExtractor",
    "ExtractionCandidates",
    "RelationCandidate",
    "SpacyEntityExtractor",
    "load_spacy_extractor",
]


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    """A model or fallback entity span copied verbatim from the source."""

    text: str
    label: str
    start: int
    end: int
    normalized: str
    source: str


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """A conservative subject--predicate--object candidate.

    ``object`` may be empty when the parser found a subject and predicate but
    no reliable object.  Such candidates are useful for diagnostics, but must
    not be persisted as facts without a later validator.
    """

    subject: EntityCandidate
    relation: str
    object: EntityCandidate | None
    sentence: str
    source: str
    assertion: str = "asserted"


@dataclass(frozen=True, slots=True)
class ExtractionCandidates:
    entities: tuple[EntityCandidate, ...]
    relations: tuple[RelationCandidate, ...]
    backend: str
    model_name: str


class EntityExtractor(Protocol):
    """Protocol used by the future fact validator and offline benchmark."""

    def extract(self, text: str) -> ExtractionCandidates: ...


_CAPITALIZED = re.compile(r"\b(?:[A-Z][A-Za-z'’-]*)(?:[ \t]+[A-Z][A-Za-z'’-]*){0,4}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}(?:[-/][A-Z0-9]+)*\b")
_SUBJECT_LINE = re.compile(r"(?im)^Subject:\s*(?P<title>[^\r\n]+)")
_CORPUS_PREFIX = re.compile(r"(?i)^Corpus:\s*")
_NUMBER = re.compile(r"(?<!\w)(?:\d{4}-\d{2}-\d{2}|\$\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?%?)(?!\w)")
_UNSAFE_SENTENCE = re.compile(
    r"[?？]|\b(?:not|never|no|if|would|might|fictional|imagining|said|says|quoted|"
    r"plan|plans|planned|hope|hopes|want|wants|will|going to)\b|"
    r"不是|并非|没有|不在|不再|如果|假设|虚构|示例|据说|听说|计划|打算|希望|想要|将会",
    re.I,
)
_SURFACE_RELATIONS = (
    ("is employed by", "employee"),
    ("works for", "employee"),
    ("work for", "employee"),
    ("works at", "employee"),
    ("work at", "employee"),
    ("studies at", "student"),
    ("study at", "student"),
    ("studied at", "student"),
    ("is enrolled at", "student"),
    ("volunteers at", "volunteer"),
    ("volunteer at", "volunteer"),
    ("plays for", "team"),
    ("play for", "team"),
    ("lives in", "lives in"),
    ("live in", "lives in"),
    ("is married to", "spouse"),
    ("was created by", "creator"),
    ("was born in", "place of birth"),
)
_ROLE_LINE = re.compile(
    r"\b(?:chief|director|engineer|manager|marketing|member|officer|"
    r"president|researcher|specialist|staff|team|vice|lead|head|department)\b",
    re.I,
)
_ORG_LINE = re.compile(
    r"\b(?:labs?|inc|corp(?:oration)?|ltd|llc|company|technologies|research|"
    r"university|foundation|institute)\b",
    re.I,
)
_PERSON_LINE = re.compile(r"^[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,4}$")
_INVALID_RELATION_OBJECTS = frozenset(
    {
        "today",
        "tomorrow",
        "yesterday",
        "tonight",
        "now",
        "then",
        "here",
        "there",
        "i'm",
        "im",
        "i am",
        "we're",
        "were",
        "we are",
        "you're",
        "you are",
    }
)
_ENTITY_CONNECTOR = re.compile(r"^(?:\s+(?:of|and|the|de|la|van|von)\s*|[,;:/&-]\s*)$", re.I)
_ENTITY_LABELS = frozenset(
    {
        "GPE",
        "LOC",
        "FAC",
        "ORG",
        "NORP",
        "LANGUAGE",
        "PERSON",
        "PER",
        "ENTITY",
        "PROPN",
        "RULE_PROPN",
    }
)


def _span_key(candidate: EntityCandidate) -> tuple[int, int, str]:
    return candidate.start, candidate.end, candidate.label


def _entity_for_token(token: Any, entities: Sequence[EntityCandidate]) -> EntityCandidate | None:
    start = int(token.idx)
    containing = [entity for entity in entities if entity.start <= start < entity.end]
    if not containing:
        return None
    return min(containing, key=lambda entity: (entity.end - entity.start, entity.start))


def _entity_in_span(
    entities: Sequence[EntityCandidate], start: int, end: int, *, from_right: bool = False
) -> EntityCandidate | None:
    candidates = [entity for entity in entities if entity.start >= start and entity.end <= end]
    if not candidates:
        return None
    if from_right:
        return sorted(candidates, key=lambda entity: (entity.start, entity.end), reverse=True)[0]
    return max(candidates, key=lambda entity: (entity.end - entity.start, entity.end))


def _merge_composite_entities(
    text: str, entities: Sequence[EntityCandidate]
) -> list[EntityCandidate]:
    """Join model spans split by harmless name connectors.

    Small NER models commonly split ``United Kingdom of Great Britain`` or
    ``Charles, Prince of Wales`` into several entities.  The merged span is
    still an exact source substring and is only used by relation validation.
    """
    result = list(entities)
    changed = True
    while changed:
        changed = False
        result.sort(key=lambda entity: (entity.start, entity.end))
        for index, left in enumerate(result[:-1]):
            right = result[index + 1]
            if (
                left.label.upper() not in _ENTITY_LABELS
                or right.label.upper() not in _ENTITY_LABELS
            ):
                continue
            gap = text[left.end : right.start]
            if not gap or "\n" in gap or "." in gap or "?" in gap or "!" in gap:
                continue
            if not _ENTITY_CONNECTOR.fullmatch(gap) and not re.search(
                r"(?:^|\s)(?:of|and|the|de|la|van|von)\s*$", left.text, re.I
            ):
                continue
            start, end = left.start, right.end
            merged_text = text[start:end]
            merged_label = left.label if left.label.upper() == right.label.upper() else "ENTITY"
            merged = EntityCandidate(
                text=merged_text,
                label=merged_label,
                start=start,
                end=end,
                normalized=key(merged_text),
                source="model" if left.source == right.source == "model" else "rule",
            )
            result[index : index + 2] = [merged]
            changed = True
            break
    return result


def _organization_entity(
    text: str, line_start: int, line_end: int, entities: Sequence[EntityCandidate]
) -> EntityCandidate | None:
    """Select the organization fragment from a profile line.

    A small model may label ``Marketing Specialist, Zenith Labs`` as one
    entity.  Profile extraction needs the organization suffix, while keeping
    the original span for all other relation types.
    """
    line = text[line_start:line_end]
    parts = [part.strip() for part in re.split(r"\s*[|,;]\s*|\s+-\s+", line) if part.strip()]
    for part in reversed(parts):
        if not _ORG_LINE.search(part) or _ROLE_LINE.fullmatch(part):
            continue
        relative = line.rfind(part)
        start, end = line_start + relative, line_start + relative + len(part)
        for entity in entities:
            if entity.start == start and entity.end == end:
                return entity
        return EntityCandidate(
            text=part,
            label="RULE_ORG",
            start=start,
            end=end,
            normalized=key(part),
            source="rule",
        )
    return None


def _surface_relations(
    sentence: str, entities: Sequence[EntityCandidate], offset: int
) -> list[RelationCandidate]:
    """Recover common prepositional relations that dependency parses miss."""
    if interrogative(sentence) or _UNSAFE_SENTENCE.search(sentence):
        return []
    output: list[RelationCandidate] = []
    for phrase, normalized_relation in _SURFACE_RELATIONS:
        pattern = re.compile(
            rf"(?P<subject>[^,;.!?\n]+?)\s+{re.escape(phrase)}\s+"
            rf"(?P<object>[^,;.!?\n]+)",
            re.I,
        )
        for match in pattern.finditer(sentence):
            subject_start = offset + match.start("subject")
            subject_end = offset + match.end("subject")
            object_start = offset + match.start("object")
            object_end = offset + match.end("object")
            subject = _entity_in_span(entities, subject_start, subject_end, from_right=True)
            object_entity = _entity_in_span(entities, object_start, object_end)
            if (
                subject is None
                or object_entity is None
                or subject == object_entity
                or key(object_entity.text) in _INVALID_RELATION_OBJECTS
            ):
                continue
            output.append(
                RelationCandidate(
                    subject=subject,
                    relation=relation_key(normalized_relation),
                    object=object_entity,
                    sentence=sentence,
                    source="surface",
                )
            )
    return output


def _structured_relations(
    text: str, entities: Sequence[EntityCandidate], offset: int
) -> list[RelationCandidate]:
    """Recover conservative relations from common profile-style line blocks."""
    if interrogative(text) or _UNSAFE_SENTENCE.search(text):
        return []
    lines = [line for line in re.finditer(r"[^\r\n]+", text) if line.group(0).strip()]
    output: list[RelationCandidate] = []
    for start, first in enumerate(lines):
        if (
            ":" in first.group(0)
            or _ROLE_LINE.search(first.group(0))
            or not _PERSON_LINE.fullmatch(first.group(0).strip())
        ):
            continue
        subject = _entity_in_span(entities, offset + first.start(), offset + first.end())
        if subject is None or subject.label.upper() not in {
            "PERSON",
            "PER",
            "PROPN",
            "ENTITY",
            "RULE_PROPN",
        }:
            continue
        found_relation = False
        # Compact directory cards often put title and organization on the
        # same line: ``Name`` followed by ``Marketing Specialist, Zenith Labs``.
        if start + 1 < len(lines):
            last = lines[start + 1]
            if (
                ":" not in last.group(0)
                and _ROLE_LINE.search(last.group(0))
                and _ORG_LINE.search(last.group(0))
            ):
                object_entity = _organization_entity(
                    text,
                    offset + last.start(),
                    offset + last.end(),
                    entities,
                )
                if object_entity is not None and object_entity != subject:
                    output.append(
                        RelationCandidate(
                            subject=subject,
                            relation="employee",
                            object=object_entity,
                            sentence="\n".join(line.group(0) for line in lines[start : start + 2]),
                            source="structured-rule",
                        )
                    )
                    found_relation = True
        for end in range(start + 2, min(len(lines), start + 6)):
            middle = lines[start + 1 : end]
            last = lines[end]
            if any(":" in line.group(0) for line in (*middle, last)):
                break
            if not any(_ROLE_LINE.search(line.group(0)) for line in middle):
                continue
            if not _ORG_LINE.search(last.group(0)):
                continue
            object_entity = _organization_entity(
                text,
                offset + last.start(),
                offset + last.end(),
                entities,
            ) or _entity_in_span(
                entities, offset + last.start(), offset + last.end(), from_right=True
            )
            if object_entity is None or object_entity == subject:
                continue
            if object_entity.label.upper() not in {
                "ORG",
                "GPE",
                "FAC",
                "PRODUCT",
                "ENTITY",
                "PROPN",
                "RULE_ORG",
                "RULE_PROPN",
                "PERSON",
                "PER",
            }:
                continue
            output.append(
                RelationCandidate(
                    subject=subject,
                    relation="employee",
                    object=object_entity,
                    sentence="\n".join(line.group(0) for line in lines[start : end + 1]),
                    source="structured-rule",
                )
            )
            found_relation = True
        if found_relation:
            continue
    return output


def _rule_entities(text: str, existing: Iterable[EntityCandidate]) -> list[EntityCandidate]:
    """Add only high-precision surface candidates missing from the model."""

    result = list(existing)
    occupied = [(candidate.start, candidate.end) for candidate in result]
    patterns = (
        (_SUBJECT_LINE, "RULE_TITLE"),
        (_CAPITALIZED, "RULE_PROPN"),
        (_ACRONYM, "RULE_ORG"),
        (_NUMBER, "RULE_NUMBER"),
    )
    for pattern, label in patterns:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            value = match.groupdict().get("title") or match.group(0)
            start = match.start("title") if "title" in match.groupdict() else match.start()
            end = match.end("title") if "title" in match.groupdict() else match.end()
            containing = [
                candidate
                for candidate in result
                if candidate.start <= start and end <= candidate.end
            ]
            nested = [
                candidate
                for candidate in result
                if start <= candidate.start and candidate.end <= end and candidate.text != value
            ]
            if containing and label == "RULE_TITLE":
                # A model may include the ``Subject:`` label in its span;
                # the title rule has the exact source boundary required by
                # document-title facts.
                result = [
                    candidate
                    for candidate in result
                    if not (candidate.start <= start and end <= candidate.end)
                ]
                containing = []
            # Prefer a complete surface number over a parser's nested token,
            # e.g. ``$3.50`` over ``3.50``.  Other model spans remain primary.
            if containing or (nested and label != "RULE_NUMBER"):
                continue
            # A sentence-initial stopword is not an entity merely because it is
            # capitalized.  Keep names and organizations with at least one
            # non-stopword token.
            if label == "RULE_PROPN" and key(value) in {"i", "the", "a", "an"}:
                continue
            if nested:
                result = [candidate for candidate in result if candidate not in nested]
            result.append(
                EntityCandidate(
                    text=value,
                    label=label,
                    start=start,
                    end=end,
                    normalized=key(value),
                    source="rule",
                )
            )
            occupied.append(span)
    # Dataset records prefix declarative triples with ``Corpus:``.  Remove
    # that marker from the entity stream so the first subject keeps its full
    # span (for example ``Father Brown`` rather than ``Brown``).
    prefix = _CORPUS_PREFIX.match(text)
    if prefix:
        prefix_end = prefix.end()
        result = [entity for entity in result if entity.start >= prefix_end]
        subject_start = prefix_end
        subject_match = re.match(r"(.+?)\s+(?:is|was|were|has|have)\s+", text[subject_start:], re.I)
        if subject_match:
            value = subject_match.group(1).strip()
            end = subject_start + len(value)
            result.append(
                EntityCandidate(
                    text=value,
                    label="RULE_PROPN",
                    start=subject_start,
                    end=end,
                    normalized=key(value),
                    source="rule",
                )
            )
    return result


class SpacyEntityExtractor:
    """Use a spaCy pipeline first, with deterministic span fallback.

    The fallback is intentionally entity-only.  It never invents a relation,
    changes a model label, or turns a span into a persisted fact.
    """

    def __init__(self, nlp: Any, *, model_name: str = "unknown") -> None:
        if not callable(nlp):
            raise TypeError("nlp 必须是可调用的 spaCy pipeline")
        self._nlp = nlp
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract(self, text: str) -> ExtractionCandidates:
        doc = self._nlp(text)
        entities: list[EntityCandidate] = []
        for entity in getattr(doc, "ents", ()):
            value = entity.text
            if not value.strip() or "\n" in value or "\r" in value:
                # Small spaCy models occasionally merge adjacent profile lines
                # into one entity.  Such a span cannot support a safe relation;
                # the line-aware fallback below will recover its components.
                continue
            entities.append(
                EntityCandidate(
                    text=value,
                    label=str(entity.label_ or "ENTITY"),
                    start=int(entity.start_char),
                    end=int(entity.end_char),
                    normalized=key(value),
                    source="model",
                )
            )
        entities = _rule_entities(text, entities)
        entities = _merge_composite_entities(text, entities)
        entities.sort(key=lambda entity: (entity.start, entity.end, entity.source != "model"))

        relations: list[RelationCandidate] = []
        whole_document = False
        try:
            sentences = tuple(doc.sents)
        except ValueError:
            # A tokenizer-only pipeline (for example ``spacy.blank``) has no
            # sentence boundary component.  Entity extraction remains useful;
            # treat the complete document as one sentence and emit no parser
            # relations unless the model supplies them.
            sentences = (doc,)
            whole_document = True
        for sentence in sentences:
            sentence_start = 0 if whole_document else int(sentence.start_char)
            sentence_end = len(text) if whole_document else int(sentence.end_char)
            sent_entities = [
                entity
                for entity in entities
                if entity.start >= sentence_start and entity.end <= sentence_end
            ]
            if not sent_entities:
                continue
            safe_sentence = not interrogative(sentence.text) and not _UNSAFE_SENTENCE.search(
                sentence.text
            )
            for token in sentence:
                if not safe_sentence or getattr(token, "pos_", "") not in {"VERB", "AUX"}:
                    continue
                subject = next(
                    (
                        _entity_for_token(child, sent_entities)
                        for child in token.children
                        if getattr(child, "dep_", "") in {"nsubj", "nsubjpass", "csubj"}
                    ),
                    None,
                )
                if subject is None:
                    continue
                object_entity = next(
                    (
                        _entity_for_token(child, sent_entities)
                        for child in token.children
                        if getattr(child, "dep_", "")
                        in {"obj", "dobj", "pobj", "attr", "oprd", "dative"}
                    ),
                    None,
                )
                relations.append(
                    RelationCandidate(
                        subject=subject,
                        relation=str(token.lemma_ or token.text),
                        object=object_entity,
                        sentence=sentence.text,
                        source="model",
                    )
                )
            relations.extend(_surface_relations(sentence.text, sent_entities, sentence_start))
            relations.extend(_structured_relations(sentence.text, sent_entities, sentence_start))

        # Stable de-duplication makes benchmark diffs reproducible when a
        # pipeline emits overlapping entities or repeated dependency arcs.
        seen_entities: set[tuple[int, int, str]] = set()
        unique_entities_list: list[EntityCandidate] = []
        for entity in entities:
            marker = _span_key(entity)
            if marker not in seen_entities:
                seen_entities.add(marker)
                unique_entities_list.append(entity)
        unique_entities = tuple(unique_entities_list)

        seen_relations: set[tuple[int, int, str, int, int]] = set()
        unique_relations_list: list[RelationCandidate] = []
        for relation in relations:
            relation_marker = (
                relation.subject.start,
                relation.subject.end,
                relation.relation,
                relation.object.start if relation.object else -1,
                relation.object.end if relation.object else -1,
            )
            if relation_marker not in seen_relations:
                seen_relations.add(relation_marker)
                unique_relations_list.append(relation)
        unique_relations = tuple(unique_relations_list)
        backend = "spacy+rules" if any(e.source == "model" for e in unique_entities) else "rules"
        return ExtractionCandidates(
            entities=unique_entities,
            relations=unique_relations,
            backend=backend,
            model_name=self._model_name,
        )


def load_spacy_extractor(model_name: str = "en_core_web_sm") -> SpacyEntityExtractor:
    """Load a named spaCy model lazily, keeping spaCy optional for core users."""

    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("事实抽取模型需要安装 spaCy；核心服务不默认依赖它") from exc
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:  # pragma: no cover - model availability is environment-specific
        raise RuntimeError(f"无法加载 spaCy 模型 {model_name!r}") from exc
    return SpacyEntityExtractor(nlp, model_name=model_name)
