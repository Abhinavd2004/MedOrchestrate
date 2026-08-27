"""Tests for backend.agents.optimizer_agent.

rag_agent.run_rag, evidence_agent.gather_live_evidence, fusion_agent.fuse,
and the Claude-backed _select_refinement_strategy are all mocked -- no
network access. Covers: (a) confidence clears threshold on iteration 1,
(b) never clears threshold after 3 iterations -> review_required True,
best-of-all-iterations returned, and (c) one iteration raises an
exception but the loop still terminates safely within the 3-iteration cap.
"""

from unittest.mock import AsyncMock

import pytest

import backend.agents.optimizer_agent as optimizer_agent
from backend.agents.optimizer_agent import MAX_ITERATIONS, RefinementError, optimize
from backend.models.schemas import (
    CaseInput,
    ClinicalFindings,
    EvidenceItem,
    FusionResult,
    LiveEvidence,
    Patient,
    RAGEvidence,
)
from backend.services.confidence import THRESHOLD

CASE = CaseInput(case_id="CASE001", patient=Patient(age=45, sex="male"))
CLINICAL = ClinicalFindings(
    demographics={}, symptoms=["headache", "dizziness"], history=["hypertension"], labs={}, imaging_text_findings=[]
)


def _fusion(confidence_hint: float, conflicts: list[str] | None = None) -> FusionResult:
    return FusionResult(
        diagnoses=[{"name": "Some diagnosis", "confidence": confidence_hint, "supporting_evidence": [], "rank": 1}],
        overall_confidence=confidence_hint,
        conflicts=conflicts or [],
    )


def _rag(score: float = 0.5) -> RAGEvidence:
    return RAGEvidence(evidence=[EvidenceItem(text="t", source="s", score=score)])


def _refinement(strategy: str, query: str) -> dict:
    return {"strategy": strategy, "query": query, "rationale": "test rationale"}


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Patches run_rag/gather_live_evidence/fuse/_select_refinement_strategy
    with AsyncMocks the test configures per-scenario."""
    run_rag_mock = AsyncMock()
    gather_live_evidence_mock = AsyncMock(return_value=LiveEvidence(sources=[]))
    fuse_mock = AsyncMock()
    select_strategy_mock = AsyncMock()

    monkeypatch.setattr(optimizer_agent, "run_rag", run_rag_mock)
    monkeypatch.setattr(optimizer_agent, "gather_live_evidence", gather_live_evidence_mock)
    monkeypatch.setattr(optimizer_agent, "fuse", fuse_mock)
    monkeypatch.setattr(optimizer_agent, "_select_refinement_strategy", select_strategy_mock)

    return {
        "run_rag": run_rag_mock,
        "gather_live_evidence": gather_live_evidence_mock,
        "fuse": fuse_mock,
        "select_strategy": select_strategy_mock,
    }


# --------------------------------------------------------------------------
# Already above threshold: short-circuit, no loop at all
# --------------------------------------------------------------------------


async def test_already_above_threshold_short_circuits(mock_pipeline):
    initial_fusion = _fusion(0.9)

    result = await optimize(CASE, CLINICAL, None, initial_fusion, initial_confidence=0.9)

    assert result["final_fusion"] == initial_fusion
    assert result["final_confidence"] == 0.9
    assert result["iterations"] == 0
    assert result["review_required"] is False
    assert result["iteration_log"] == []
    mock_pipeline["select_strategy"].assert_not_called()
    mock_pipeline["run_rag"].assert_not_called()


# --------------------------------------------------------------------------
# (a) confidence clears threshold on iteration 1
# --------------------------------------------------------------------------


async def test_clears_threshold_on_first_iteration(mock_pipeline, monkeypatch):
    mock_pipeline["select_strategy"].return_value = _refinement("expand", "headache dizziness hypertension neuro")
    mock_pipeline["run_rag"].return_value = _rag(0.9)
    final_fusion = _fusion(0.8)
    mock_pipeline["fuse"].return_value = final_fusion
    monkeypatch.setattr(optimizer_agent, "compute_confidence", lambda rag, live, fusion: 0.8)

    result = await optimize(CASE, CLINICAL, None, _fusion(0.4), initial_confidence=0.4)

    assert result["iterations"] == 1
    assert result["final_confidence"] == 0.8
    assert result["final_fusion"] == final_fusion
    assert result["review_required"] is False
    assert len(result["iteration_log"]) == 1
    assert result["iteration_log"][0] == {
        "iteration": 1,
        "strategy": "expand",
        "query": "headache dizziness hypertension neuro",
        "confidence": 0.8,
    }
    mock_pipeline["select_strategy"].assert_awaited_once()


# --------------------------------------------------------------------------
# (b) never clears threshold after 3 iterations -> review_required True,
# best-of-all-iterations (not just the last) is returned
# --------------------------------------------------------------------------


async def test_never_clears_threshold_after_max_iterations(mock_pipeline, monkeypatch):
    strategies = [
        _refinement("expand", "query v1"),
        _refinement("narrow", "query v2"),
        _refinement("paraphrase", "query v3"),
    ]
    mock_pipeline["select_strategy"].side_effect = strategies
    mock_pipeline["run_rag"].return_value = _rag(0.5)

    fusions = [_fusion(0.55), _fusion(0.5), _fusion(0.45)]  # iteration 1 is the best
    mock_pipeline["fuse"].side_effect = fusions

    confidences = [0.55, 0.5, 0.45]
    confidence_iter = iter(confidences)
    monkeypatch.setattr(optimizer_agent, "compute_confidence", lambda rag, live, fusion: next(confidence_iter))

    result = await optimize(CASE, CLINICAL, None, _fusion(0.4), initial_confidence=0.4)

    assert result["iterations"] == MAX_ITERATIONS == 3
    assert result["review_required"] is True
    # best across ALL iterations (0.55 from iteration 1), not the last (0.45)
    assert result["final_confidence"] == 0.55
    assert result["final_fusion"] == fusions[0]
    assert len(result["iteration_log"]) == 3
    assert [entry["iteration"] for entry in result["iteration_log"]] == [1, 2, 3]
    assert [entry["strategy"] for entry in result["iteration_log"]] == ["expand", "narrow", "paraphrase"]
    assert [entry["query"] for entry in result["iteration_log"]] == ["query v1", "query v2", "query v3"]
    assert mock_pipeline["select_strategy"].await_count == 3


async def test_best_result_can_beat_initial_but_still_below_threshold(mock_pipeline, monkeypatch):
    # initial_confidence (0.3) is worse than what iteration 1 achieves (0.5),
    # neither clears THRESHOLD -- best_fusion should reflect iteration 1.
    mock_pipeline["select_strategy"].return_value = _refinement("expand", "q")
    mock_pipeline["run_rag"].return_value = _rag(0.5)
    iter1_fusion = _fusion(0.5)
    mock_pipeline["fuse"].return_value = iter1_fusion
    monkeypatch.setattr(optimizer_agent, "compute_confidence", lambda rag, live, fusion: 0.5)

    result = await optimize(CASE, CLINICAL, None, _fusion(0.3), initial_confidence=0.3)

    assert result["final_confidence"] == 0.5
    assert result["final_fusion"] == iter1_fusion
    assert result["review_required"] is True


# --------------------------------------------------------------------------
# (c) one iteration raises -> loop terminates safely, still bounded by 3
# --------------------------------------------------------------------------


async def test_one_iteration_raises_exception_loop_continues_safely(mock_pipeline, monkeypatch):
    mock_pipeline["select_strategy"].side_effect = [
        _refinement("expand", "query v1"),
        RuntimeError("simulated retrieval outage"),
        _refinement("paraphrase", "query v3"),
    ]
    mock_pipeline["run_rag"].return_value = _rag(0.5)

    fusion_v1 = _fusion(0.5)
    fusion_v3 = _fusion(0.55)
    mock_pipeline["fuse"].side_effect = [fusion_v1, fusion_v3]

    confidences = iter([0.5, 0.55])
    monkeypatch.setattr(optimizer_agent, "compute_confidence", lambda rag, live, fusion: next(confidences))

    result = await optimize(CASE, CLINICAL, None, _fusion(0.3), initial_confidence=0.3)

    assert result["iterations"] == 3  # still ran exactly 3 attempts, no crash
    assert len(result["iteration_log"]) == 3

    assert result["iteration_log"][0]["confidence"] == 0.5
    assert "error" not in result["iteration_log"][0]

    assert result["iteration_log"][1]["confidence"] is None
    assert "error" in result["iteration_log"][1]
    assert "simulated retrieval outage" in result["iteration_log"][1]["error"]

    assert result["iteration_log"][2]["confidence"] == 0.55
    assert "error" not in result["iteration_log"][2]

    # best across all non-failed iterations
    assert result["final_confidence"] == 0.55
    assert result["review_required"] is True


async def test_refinement_error_from_strategy_selection_is_caught(mock_pipeline):
    mock_pipeline["select_strategy"].side_effect = RefinementError("model never produced valid JSON")

    result = await optimize(CASE, CLINICAL, None, _fusion(0.3), initial_confidence=0.3)

    assert result["iterations"] == MAX_ITERATIONS
    assert result["review_required"] is True
    assert all("error" in entry for entry in result["iteration_log"])
    assert all(entry["confidence"] is None for entry in result["iteration_log"])
    # run_rag/fuse should never be reached if strategy selection itself failed
    mock_pipeline["run_rag"].assert_not_called()


async def test_hard_cap_never_exceeds_three_even_with_all_failures(mock_pipeline):
    mock_pipeline["select_strategy"].side_effect = RuntimeError("persistent failure")

    result = await optimize(CASE, CLINICAL, None, _fusion(0.1), initial_confidence=0.1)

    assert result["iterations"] <= MAX_ITERATIONS
    assert len(result["iteration_log"]) <= MAX_ITERATIONS
    assert mock_pipeline["select_strategy"].await_count == MAX_ITERATIONS
