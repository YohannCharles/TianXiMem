"""Offline guards used by the experimental LLM fact verifier."""

from tools.fact_llm_verify import _candidate_scope, _modality_conflict


def test_candidate_scope_keeps_named_speaker_turns_separate():
    source = (
        "Jon: I lost my job as a banker yesterday.\n"
        "Gina: I also lost my job at Door Dash this month."
    )

    assert _candidate_scope(source, "Jon", "banker") is not None
    assert _candidate_scope(source, "Gina", "banker") is None


def test_modality_conflict_rejects_plan_label_for_current_action():
    source = "Jon: I'm starting a dance studio because I love dancing."

    assert _modality_conflict(source, ("Jon", "plans to open", "dance studio"))
    assert not _modality_conflict(source, ("Jon", "plans to open", "restaurant"))


def test_modality_conflict_allows_explicit_intention():
    source = "Jon: I want to open a dance studio next year."

    assert not _modality_conflict(source, ("Jon", "plans to open", "dance studio"))

