"""Tests for backend.agents.clinical_agent.

The Anthropic API call is mocked via the mock_anthropic_client fixture,
so this suite runs without network access.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.agents.clinical_agent as clinical_agent
from backend.agents.clinical_agent import (
    ClinicalExtractionError,
    extract_clinical_findings,
)
from backend.models.schemas import ClinicalFindings


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Patch _get_client to return a fake client with a mocked messages.create."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock()
    monkeypatch.setattr(clinical_agent, "_get_client", lambda: fake_client)
    return fake_client


def _canned(payload: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload))


async def test_text_only(mock_anthropic_client):
    canned = {
        "demographics": {"age": 45, "sex": "male"},
        "symptoms": ["headache", "dizziness"],
        "history": ["hypertension"],
        "labs": {},
        "imaging_text_findings": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await extract_clinical_findings(
        clinical_text="45-year-old male with HTN presents with headache and dizziness",
        medical_report=None,
    )

    assert result == ClinicalFindings(**canned)
    mock_anthropic_client.messages.create.assert_awaited_once()
    _, kwargs = mock_anthropic_client.messages.create.call_args
    assert "headache" in kwargs["messages"][0]["content"]


async def test_report_only(mock_anthropic_client):
    canned = {
        "demographics": {},
        "symptoms": ["dizziness"],
        "history": [],
        "labs": {},
        "imaging_text_findings": ["no acute intracranial abnormality"],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await extract_clinical_findings(
        clinical_text=None,
        medical_report=(
            "MRI brain report: no acute intracranial abnormality. "
            "Patient reports occasional dizziness."
        ),
    )

    assert result == ClinicalFindings(**canned)
    mock_anthropic_client.messages.create.assert_awaited_once()
    _, kwargs = mock_anthropic_client.messages.create.call_args
    assert "MRI brain report" in kwargs["messages"][0]["content"]


async def test_both_text_and_report(mock_anthropic_client):
    canned = {
        "demographics": {"age": 45, "sex": "male"},
        "symptoms": ["headache", "dizziness"],
        "history": ["hypertension"],
        "labs": {},
        "imaging_text_findings": ["no acute intracranial abnormality"],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await extract_clinical_findings(
        clinical_text="45-year-old male with HTN presents with headache and dizziness",
        medical_report="MRI brain report: no acute intracranial abnormality.",
    )

    assert result == ClinicalFindings(**canned)
    _, kwargs = mock_anthropic_client.messages.create.call_args
    sent_content = kwargs["messages"][0]["content"]
    assert "headache" in sent_content
    assert "MRI brain report" in sent_content


async def test_neither_returns_empty_findings_without_api_call(mock_anthropic_client):
    result = await extract_clinical_findings(clinical_text=None, medical_report=None)

    assert result == ClinicalFindings(
        demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[]
    )
    mock_anthropic_client.messages.create.assert_not_called()


async def test_labs_extracted_from_text(mock_anthropic_client):
    canned = {
        "demographics": {},
        "symptoms": ["fatigue"],
        "history": ["anemia"],
        "labs": {"hemoglobin": "12.1 g/dL"},
        "imaging_text_findings": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await extract_clinical_findings(
        clinical_text="Patient reports fatigue, hemoglobin 12.1, history of anemia",
        medical_report=None,
    )

    assert result == ClinicalFindings(**canned)
    assert result.labs["hemoglobin"] == "12.1 g/dL"


async def test_retries_once_on_invalid_json_then_succeeds(mock_anthropic_client):
    canned = {
        "demographics": {},
        "symptoms": ["cough"],
        "history": [],
        "labs": {},
        "imaging_text_findings": [],
    }
    mock_anthropic_client.messages.create.side_effect = [
        _FakeResponse("Sure! Here's the JSON: {not valid json"),
        _canned(canned),
    ]

    result = await extract_clinical_findings(
        clinical_text="Patient has a cough", medical_report=None
    )

    assert result == ClinicalFindings(**canned)
    assert mock_anthropic_client.messages.create.await_count == 2
    # second call should include a correction message
    _, kwargs = mock_anthropic_client.messages.create.call_args
    assert kwargs["messages"][-1]["content"] == clinical_agent.CORRECTION_MESSAGE


async def test_raises_after_two_invalid_responses(mock_anthropic_client):
    mock_anthropic_client.messages.create.side_effect = [
        _FakeResponse("not json at all"),
        _FakeResponse("still not json"),
    ]

    with pytest.raises(ClinicalExtractionError):
        await extract_clinical_findings(
            clinical_text="Patient has a cough", medical_report=None
        )

    assert mock_anthropic_client.messages.create.await_count == 2
