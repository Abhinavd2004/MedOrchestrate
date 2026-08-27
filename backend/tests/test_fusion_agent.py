"""Tests for backend.agents.fusion_agent.

The Anthropic API call is mocked via the mock_anthropic_client fixture
(same pattern as test_clinical_agent.py), so this suite runs without
network access. Covers: text-only input, text+image input, a case with
real evidence conflicts, empty-evidence input, citation grounding
(fabricated citations get dropped), and the retry-on-invalid-JSON path.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.agents.fusion_agent as fusion_agent
from backend.agents.fusion_agent import FusionError, fuse
from backend.models.schemas import (
    ClinicalFindings,
    EvidenceItem,
    FusionResult,
    ImagingFindings,
    LiveEvidence,
    LiveEvidenceSource,
    RAGEvidence,
)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock()
    monkeypatch.setattr(fusion_agent, "_get_client", lambda: fake_client)
    return fake_client


def _canned(payload: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload))


CLINICAL = ClinicalFindings(
    demographics={},
    symptoms=["headache", "dizziness"],
    history=["hypertension"],
    labs={},
    imaging_text_findings=[],
)
IMAGING = ImagingFindings(
    modality="MRI brain scan", findings=["white matter hyperintensity"], confidence=0.95
)
RAG = RAGEvidence(
    evidence=[
        EvidenceItem(text="White matter hyperintensities are linked to chronic hypertension.", source="synthetic", score=0.91),
        EvidenceItem(text="Hypertensive emergency can present with severe headache and dizziness.", source="synthetic", score=0.4),
    ]
)
LIVE = LiveEvidence(
    sources=[
        LiveEvidenceSource(
            title="Vestibular migraine overview",
            source="PubMed",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            summary="Migraine is a common cause of episodic dizziness and headache.",
            evidence_level="moderate",
            publication_date="2020",
        )
    ]
)
EMPTY_RAG = RAGEvidence(evidence=[])
EMPTY_LIVE = LiveEvidence(sources=[])


async def test_text_only_input(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {
                "name": "Chronic small vessel disease related to hypertension",
                "confidence": 0.6,
                "supporting_evidence": ["[RAG1] White matter hyperintensities are linked to chronic hypertension."],
                "rank": 1,
            }
        ],
        "overall_confidence": 0.55,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, None, RAG, EMPTY_LIVE)

    assert isinstance(result, FusionResult)
    assert len(result.diagnoses) == 1
    assert result.diagnoses[0]["name"] == "Chronic small vessel disease related to hypertension"
    assert result.diagnoses[0]["supporting_evidence"] == [
        "[RAG1] White matter hyperintensities are linked to chronic hypertension."
    ]
    assert result.overall_confidence == 0.55
    assert result.conflicts == []

    # imaging section should say "no imaging" since imaging=None
    _, kwargs = mock_anthropic_client.messages.create.call_args
    sent = kwargs["messages"][0]["content"]
    assert "no imaging performed" in sent


async def test_text_and_image_input(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {
                "name": "Hypertensive cerebral small vessel disease",
                "confidence": 0.7,
                "supporting_evidence": [
                    "[RAG1] White matter hyperintensities are linked to chronic hypertension.",
                ],
                "rank": 1,
            },
            {
                "name": "Vestibular migraine",
                "confidence": 0.3,
                "supporting_evidence": ["[LIVE1] Migraine is a common cause of episodic dizziness and headache."],
                "rank": 2,
            },
        ],
        "overall_confidence": 0.6,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, IMAGING, RAG, LIVE)

    assert len(result.diagnoses) == 2
    assert result.diagnoses[0]["rank"] == 1
    assert result.diagnoses[1]["supporting_evidence"][0].startswith("[LIVE1]")

    _, kwargs = mock_anthropic_client.messages.create.call_args
    sent = kwargs["messages"][0]["content"]
    assert "MRI brain scan" in sent
    assert "white matter hyperintensity" in sent


async def test_conflict_case_propagates_conflicts(mock_anthropic_client):
    # RAG evidence points toward a vascular/hypertensive cause; live
    # evidence points toward migraine -- a real disagreement the model
    # should flag.
    canned = {
        "diagnoses": [
            {
                "name": "Hypertensive small vessel disease",
                "confidence": 0.5,
                "supporting_evidence": ["[RAG1] White matter hyperintensities are linked to chronic hypertension."],
                "rank": 1,
            },
            {
                "name": "Vestibular migraine",
                "confidence": 0.45,
                "supporting_evidence": ["[LIVE1] Migraine is a common cause of episodic dizziness and headache."],
                "rank": 2,
            },
        ],
        "overall_confidence": 0.5,
        "conflicts": [
            "Offline evidence points toward a vascular/hypertensive cause (RAG1) while live evidence "
            "emphasizes a migrainous cause (LIVE1) for the same symptoms."
        ],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, IMAGING, RAG, LIVE)

    assert len(result.conflicts) == 1
    assert "vascular" in result.conflicts[0].lower() or "migrain" in result.conflicts[0].lower()


async def test_empty_evidence_input_short_circuits_without_calling_model(mock_anthropic_client):
    empty_clinical = ClinicalFindings(
        demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[]
    )

    result = await fuse(empty_clinical, None, EMPTY_RAG, EMPTY_LIVE)

    assert result == FusionResult(diagnoses=[], overall_confidence=0.0, conflicts=[])
    mock_anthropic_client.messages.create.assert_not_called()


async def test_no_rag_or_live_evidence_but_clinical_present_still_calls_model(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {"name": "Tension-type headache", "confidence": 0.3, "supporting_evidence": [], "rank": 1}
        ],
        "overall_confidence": 0.25,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, None, EMPTY_RAG, EMPTY_LIVE)

    mock_anthropic_client.messages.create.assert_awaited_once()
    assert result.diagnoses[0]["supporting_evidence"] == []
    assert result.overall_confidence < 0.5  # low confidence, no evidence backing


async def test_fabricated_citation_is_dropped(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {
                "name": "Some diagnosis",
                "confidence": 0.5,
                "supporting_evidence": [
                    "[RAG1] White matter hyperintensities are linked to chronic hypertension.",
                    "[RAG99] This citation ID does not exist in the input.",
                    "No tag at all, just made up text.",
                ],
                "rank": 1,
            }
        ],
        "overall_confidence": 0.5,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, None, RAG, EMPTY_LIVE)

    # only the valid [RAG1] citation survives; the fabricated ones are dropped
    assert result.diagnoses[0]["supporting_evidence"] == [
        "[RAG1] White matter hyperintensities are linked to chronic hypertension."
    ]


async def test_retries_once_on_invalid_json_then_succeeds(mock_anthropic_client):
    canned = {
        "diagnoses": [{"name": "Tension-type headache", "confidence": 0.4, "supporting_evidence": [], "rank": 1}],
        "overall_confidence": 0.4,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.side_effect = [
        _FakeResponse("Sure, here's the JSON: {not valid"),
        _canned(canned),
    ]

    result = await fuse(CLINICAL, None, EMPTY_RAG, EMPTY_LIVE)

    assert result.diagnoses[0]["name"] == "Tension-type headache"
    assert mock_anthropic_client.messages.create.await_count == 2


async def test_raises_after_two_invalid_responses(mock_anthropic_client):
    mock_anthropic_client.messages.create.side_effect = [
        _FakeResponse("not json"),
        _FakeResponse("still not json"),
    ]

    with pytest.raises(FusionError):
        await fuse(CLINICAL, None, EMPTY_RAG, EMPTY_LIVE)

    assert mock_anthropic_client.messages.create.await_count == 2
