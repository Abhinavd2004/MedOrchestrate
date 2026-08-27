"""Tests for backend.services.orchestrator.

All external calls are mocked at the Phase 1-7 function boundary
(extract_clinical_findings, analyze_image, run_rag, gather_live_evidence,
fuse, optimize -- as imported into orchestrator.py's own namespace), and
orchestrator._execute is monkeypatched to call each CrewAI tool directly
instead of going through a real Crew.kickoff() (which would require a
live Claude call) -- see orchestrator._execute's docstring for why this
is a safe substitution: the tools ignore whatever the LLM would pass
them and read the shared _CaseContext instead. No network access, no
API key required, offline and fast.
"""

from unittest.mock import AsyncMock

import pytest

import backend.services.orchestrator as orchestrator
from backend.models.schemas import (
    CaseInput,
    ClinicalFindings,
    EvidenceItem,
    FusionResult,
    ImagingFindings,
    LiveEvidence,
    LiveEvidenceSource,
    Patient,
    RAGEvidence,
)
from backend.services import storage
from backend.services.confidence import THRESHOLD
from backend.services.orchestrator import run_case


def _direct_execute(agent, task):
    """Bypass CrewAI's real Crew.kickoff()/LLM call -- run the agent's
    one tool directly, exactly as orchestrator._execute's own docstring
    says is a safe substitution for testing."""
    return agent.tools[0]._run()


@pytest.fixture(autouse=True)
def _bypass_crewai(monkeypatch):
    monkeypatch.setattr(orchestrator, "_execute", _direct_execute)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    # orchestrator.py now persists every agent call via
    # storage.log_agent_event() -- point at an isolated temp file per
    # test rather than the real dev database.
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_orchestrator.db"))
    storage._init_tables()


HIGH_CONFIDENCE_FUSION = FusionResult(
    diagnoses=[{"name": "Tension-type headache", "confidence": 0.8, "supporting_evidence": ["[RAG1] ..."], "rank": 1}],
    overall_confidence=0.8,
    conflicts=[],
)
LOW_CONFIDENCE_FUSION = FusionResult(
    diagnoses=[{"name": "Uncertain", "confidence": 0.2, "supporting_evidence": [], "rank": 1}],
    overall_confidence=0.2,
    conflicts=["No corroborating evidence found."],
)


@pytest.fixture
def mock_agents(monkeypatch):
    mocks = {
        "extract_clinical_findings": AsyncMock(
            return_value=ClinicalFindings(
                demographics={}, symptoms=["headache", "dizziness"], history=["hypertension"], labs={},
                imaging_text_findings=[],
            )
        ),
        "analyze_image": AsyncMock(
            return_value=ImagingFindings(modality="MRI brain scan", findings=["white matter hyperintensity"], confidence=0.9)
        ),
        "run_rag": AsyncMock(
            return_value=RAGEvidence(evidence=[EvidenceItem(text="offline evidence", source="synthetic", score=0.9)])
        ),
        "gather_live_evidence": AsyncMock(
            return_value=LiveEvidence(
                sources=[
                    LiveEvidenceSource(
                        title="live evidence", source="PubMed", url="u", summary="s",
                        evidence_level="high", publication_date="2024",
                    )
                ]
            )
        ),
        "fuse": AsyncMock(return_value=HIGH_CONFIDENCE_FUSION),
        "optimize": AsyncMock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(orchestrator, name, mock)

    # compute_confidence() is real math tested separately in
    # test_confidence.py -- here we want direct control over the
    # confidence value to deterministically exercise the optimizer
    # branch, not whatever the formula derives from the mocked
    # rag/live evidence above. Defaults to a value >= THRESHOLD so
    # tests that don't care about the optimizer path skip it by
    # default; individual tests override this to go below THRESHOLD.
    confidence_holder = {"value": 0.8}

    def _fake_compute_confidence(rag, live, fusion):
        return confidence_holder["value"]

    monkeypatch.setattr(orchestrator, "compute_confidence", _fake_compute_confidence)
    mocks["confidence_holder"] = confidence_holder
    return mocks


CASE_TEXT_ONLY = CaseInput(
    case_id="CASE001",
    patient=Patient(age=45, sex="male"),
    clinical_text="headache and dizziness, history of hypertension",
    history=["hypertension"],
)
CASE_TEXT_AND_IMAGE = CaseInput(
    case_id="CASE001",
    patient=Patient(age=45, sex="male"),
    clinical_text="headache and dizziness, history of hypertension",
    image_path="data/images/synthetic_demo_brain_mri_001.png",
    history=["hypertension"],
)
CASE_IMAGE_ONLY = CaseInput(
    case_id="CASE002",
    patient=Patient(age=60, sex="female"),
    image_path="data/images/synthetic_demo_brain_mri_001.png",
)


# --------------------------------------------------------------------------
# Text-only and text+image full runs
# --------------------------------------------------------------------------


def test_text_only_full_run(mock_agents):
    report = run_case(CASE_TEXT_ONLY)

    assert report.case_id == "CASE001"
    assert report.confidence == 0.8
    assert report.review_required is False
    assert report.diagnoses == HIGH_CONFIDENCE_FUSION.diagnoses
    assert {e["type"] for e in report.evidence} == {"rag", "live"}

    mock_agents["extract_clinical_findings"].assert_awaited_once()
    mock_agents["analyze_image"].assert_not_called()  # no image_path
    mock_agents["run_rag"].assert_awaited_once()
    mock_agents["gather_live_evidence"].assert_awaited_once()
    mock_agents["fuse"].assert_awaited_once()
    mock_agents["optimize"].assert_not_called()  # confidence already >= THRESHOLD


def test_text_and_image_full_run(mock_agents):
    report = run_case(CASE_TEXT_AND_IMAGE)

    assert report.case_id == "CASE001"
    assert report.confidence == 0.8
    mock_agents["analyze_image"].assert_awaited_once()
    call_args = mock_agents["analyze_image"].call_args
    assert call_args.args[0] == CASE_TEXT_AND_IMAGE.image_path

    _, fuse_kwargs = mock_agents["fuse"].call_args
    # fuse(clinical, imaging, rag, live) -- imaging must be the real
    # ImagingFindings produced by analyze_image, not None
    fuse_call_args = mock_agents["fuse"].call_args.args
    assert fuse_call_args[1] is not None
    assert fuse_call_args[1].modality == "MRI brain scan"


def test_image_only_full_run_no_clinical_text_or_report(monkeypatch, mock_agents):
    # Explicitly the case called out in the spec: clinical_text and
    # medical_report are both None, only image_path is set.
    report = run_case(CASE_IMAGE_ONLY)

    assert report.case_id == "CASE002"
    mock_agents["extract_clinical_findings"].assert_awaited_once_with(None, None)
    mock_agents["analyze_image"].assert_awaited_once()


# --------------------------------------------------------------------------
# Optimizer wiring: only runs below threshold, and its result flows through
# --------------------------------------------------------------------------


def test_high_confidence_skips_optimizer(mock_agents):
    run_case(CASE_TEXT_ONLY)
    mock_agents["optimize"].assert_not_called()


def test_low_confidence_triggers_optimizer_and_uses_its_result(mock_agents):
    mock_agents["confidence_holder"]["value"] = 0.2
    mock_agents["fuse"].return_value = LOW_CONFIDENCE_FUSION
    optimized_fusion = FusionResult(
        diagnoses=[{"name": "Refined diagnosis", "confidence": 0.7, "supporting_evidence": [], "rank": 1}],
        overall_confidence=0.7,
        conflicts=[],
    )
    mock_agents["optimize"].return_value = {
        "final_fusion": optimized_fusion,
        "final_confidence": 0.7,
        "iterations": 2,
        "review_required": False,
        "iteration_log": [
            {"iteration": 1, "strategy": "expand", "query": "q1", "confidence": 0.5},
            {"iteration": 2, "strategy": "narrow", "query": "q2", "confidence": 0.7},
        ],
    }

    report = run_case(CASE_TEXT_ONLY)

    mock_agents["optimize"].assert_awaited_once()
    assert report.confidence == 0.7
    assert report.diagnoses == optimized_fusion.diagnoses
    assert report.review_required is False


def test_optimizer_review_required_propagates(mock_agents):
    mock_agents["confidence_holder"]["value"] = 0.2
    mock_agents["fuse"].return_value = LOW_CONFIDENCE_FUSION
    mock_agents["optimize"].return_value = {
        "final_fusion": LOW_CONFIDENCE_FUSION,
        "final_confidence": 0.2,
        "iterations": 3,
        "review_required": True,
        "iteration_log": [{"iteration": i, "strategy": "expand", "query": f"q{i}", "confidence": 0.2} for i in range(1, 4)],
    }

    report = run_case(CASE_TEXT_ONLY)

    assert report.review_required is True
    assert report.confidence == 0.2


# --------------------------------------------------------------------------
# Graceful degradation: a single failed call must not crash run_case()
# --------------------------------------------------------------------------


def test_clinical_failure_degrades_gracefully(mock_agents):
    mock_agents["extract_clinical_findings"].side_effect = RuntimeError("Claude API down")

    report = run_case(CASE_TEXT_ONLY)  # must not raise

    assert isinstance(report.confidence, float)
    mock_agents["fuse"].assert_awaited_once()
    fuse_call_args = mock_agents["fuse"].call_args.args
    assert fuse_call_args[0].symptoms == []  # degraded to empty ClinicalFindings


def test_rag_failure_degrades_gracefully(mock_agents):
    mock_agents["run_rag"].side_effect = RuntimeError("Qdrant connection refused")

    report = run_case(CASE_TEXT_ONLY)  # must not raise

    assert isinstance(report.confidence, float)
    fuse_call_args = mock_agents["fuse"].call_args.args
    assert fuse_call_args[2].evidence == []  # degraded to empty RAGEvidence


def test_imaging_failure_degrades_gracefully(mock_agents):
    mock_agents["analyze_image"].side_effect = RuntimeError("BiomedCLIP load failure")

    report = run_case(CASE_TEXT_AND_IMAGE)  # must not raise

    fuse_call_args = mock_agents["fuse"].call_args.args
    assert fuse_call_args[1] is None  # degraded to no imaging findings


def test_fusion_failure_degrades_to_low_confidence_and_review_required(mock_agents):
    mock_agents["fuse"].side_effect = RuntimeError("Claude API down")
    mock_agents["optimize"].return_value = {
        "final_fusion": FusionResult(diagnoses=[], overall_confidence=0.0, conflicts=["Fusion reasoning failed"]),
        "final_confidence": 0.0,
        "iterations": 3,
        "review_required": True,
        "iteration_log": [],
    }

    report = run_case(CASE_TEXT_ONLY)  # must not raise

    assert report.confidence == 0.0
    assert report.diagnoses == []
    # confidence 0.0 < THRESHOLD -> optimizer should have been attempted
    mock_agents["optimize"].assert_awaited_once()


# --------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------


def test_logs_agent_calls_for_each_step(monkeypatch, mock_agents):
    logged_events = []
    monkeypatch.setattr(orchestrator, "log_event", lambda event, **fields: logged_events.append(fields))

    run_case(CASE_TEXT_ONLY)

    agents_logged = {entry["agent_name"] for entry in logged_events}
    assert agents_logged == {"clinical", "rag", "evidence", "fusion"}  # no imaging, no optimizer
    assert all(entry["case_id"] == CASE_TEXT_ONLY.case_id for entry in logged_events)
    assert all(entry["iteration"] == 0 for entry in logged_events)
    assert all(entry["status"] == "success" for entry in logged_events)
    assert all("latency_ms" in entry for entry in logged_events)
