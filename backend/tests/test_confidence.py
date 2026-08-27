"""Tests for backend.services.confidence.

Every expected value below is hand-computed from the documented
formula C = w1*S_rag + w2*S_web + w3*A (w1=0.4, w2=0.3, w3=0.3),
S_web weights {"high": 1.0, "moderate": 0.6, "low": 0.3}, and
A = max(0, 1 - 0.25 * len(conflicts)) -- not derived by calling the
function under test and asserting it matches itself.
"""

import pytest

from backend.services.confidence import (
    THRESHOLD,
    W1_RAG,
    W2_WEB,
    W3_AGREEMENT,
    _compute_agreement,
    _compute_s_rag,
    _compute_s_web,
    compute_confidence,
)
from backend.models.schemas import EvidenceItem, FusionResult, LiveEvidence, LiveEvidenceSource, RAGEvidence


def _rag(scores: list[float]) -> RAGEvidence:
    return RAGEvidence(evidence=[EvidenceItem(text="t", source="s", score=s) for s in scores])


def _live(levels: list[str]) -> LiveEvidence:
    return LiveEvidence(
        sources=[
            LiveEvidenceSource(
                title="t", source="s", url="u", summary="sum", evidence_level=level
            )
            for level in levels
        ]
    )


def _fusion(conflicts: list[str]) -> FusionResult:
    return FusionResult(diagnoses=[], overall_confidence=0.5, conflicts=conflicts)


def test_weights_sum_to_one():
    assert W1_RAG + W2_WEB + W3_AGREEMENT == pytest.approx(1.0)


def test_threshold_is_065():
    assert THRESHOLD == 0.65


# --------------------------------------------------------------------------
# Component functions, hand-computed
# --------------------------------------------------------------------------


def test_s_rag_mean_of_scores():
    # mean(0.9, 0.7, 0.5) = 2.1 / 3 = 0.7
    assert _compute_s_rag(_rag([0.9, 0.7, 0.5])) == pytest.approx(0.7)


def test_s_rag_empty_is_zero():
    assert _compute_s_rag(_rag([])) == 0.0


def test_s_rag_top_k_caps_at_five():
    # 7 scores given, only the top 5 (1.0, 0.9, 0.8, 0.7, 0.6) should be
    # averaged: mean = 4.0 / 5 = 0.8, NOT mean of all 7.
    scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.1, 0.05]
    assert _compute_s_rag(_rag(scores)) == pytest.approx(0.8)


def test_s_web_mixed_levels():
    # (1.0 + 0.6) / 2 = 0.8
    assert _compute_s_web(_live(["high", "moderate"])) == pytest.approx(0.8)


def test_s_web_no_live_sources_is_zero():
    assert _compute_s_web(_live([])) == 0.0


def test_s_web_unknown_level_weighted_zero():
    # (1.0 + 0.0) / 2 = 0.5
    assert _compute_s_web(_live(["high", "unknown_level"])) == pytest.approx(0.5)


def test_agreement_no_conflicts_is_one():
    assert _compute_agreement(_fusion([])) == 1.0


def test_agreement_one_conflict():
    # 1 - 0.25*1 = 0.75
    assert _compute_agreement(_fusion(["conflict A"])) == pytest.approx(0.75)


def test_agreement_floors_at_zero_with_many_conflicts():
    # 1 - 0.25*5 = -0.25 -> floored to 0.0
    conflicts = [f"conflict {i}" for i in range(5)]
    assert _compute_agreement(_fusion(conflicts)) == 0.0


# --------------------------------------------------------------------------
# Full formula, hand-computed
# --------------------------------------------------------------------------


def test_compute_confidence_basic_combo():
    rag = _rag([0.9, 0.7, 0.5])  # S_rag = 0.7
    live = _live(["high", "moderate"])  # S_web = 0.8
    fusion = _fusion([])  # A = 1.0
    # C = 0.4*0.7 + 0.3*0.8 + 0.3*1.0 = 0.28 + 0.24 + 0.3 = 0.82
    assert compute_confidence(rag, live, fusion) == pytest.approx(0.82)


def test_compute_confidence_no_live_evidence():
    rag = _rag([0.6, 0.4])  # S_rag = 0.5
    live = _live([])  # S_web = 0.0
    fusion = _fusion(["one conflict"])  # A = 0.75
    # C = 0.4*0.5 + 0.3*0.0 + 0.3*0.75 = 0.2 + 0 + 0.225 = 0.425
    assert compute_confidence(rag, live, fusion) == pytest.approx(0.425)


def test_compute_confidence_no_conflicts_full_agreement_credit():
    rag = _rag([0.5])  # S_rag = 0.5
    live = _live(["low"])  # S_web = 0.3
    fusion = _fusion([])  # A = 1.0
    # C = 0.4*0.5 + 0.3*0.3 + 0.3*1.0 = 0.2 + 0.09 + 0.3 = 0.59
    assert compute_confidence(rag, live, fusion) == pytest.approx(0.59)


def test_compute_confidence_all_conflicts_zero_agreement():
    rag = _rag([0.5])  # S_rag = 0.5
    live = _live(["moderate"])  # S_web = 0.6
    fusion = _fusion([f"conflict {i}" for i in range(6)])  # A floors to 0.0
    # C = 0.4*0.5 + 0.3*0.6 + 0.3*0.0 = 0.2 + 0.18 + 0 = 0.38
    assert compute_confidence(rag, live, fusion) == pytest.approx(0.38)


def test_compute_confidence_everything_empty_is_zero():
    rag = _rag([])
    live = _live([])
    fusion = _fusion([])  # A = 1.0, but S_rag = S_web = 0
    # C = 0.4*0 + 0.3*0 + 0.3*1.0 = 0.3
    assert compute_confidence(rag, live, fusion) == pytest.approx(0.3)


def test_compute_confidence_result_bounded_between_zero_and_one():
    rag = _rag([1.0, 1.0, 1.0])
    live = _live(["high", "high"])
    fusion = _fusion([])
    result = compute_confidence(rag, live, fusion)
    assert 0.0 <= result <= 1.0
