"""Tests for the FastAPI routes (backend/api/routes.py, backend/main.py).

orchestrator.run_case() is mocked at the backend.api.routes module
boundary (where it's imported), so these tests never touch Claude,
Qdrant, PubMed, or BiomedCLIP -- fully offline and fast. The in-memory
storage stub is cleared between tests via an autouse fixture, since its
module-level dicts would otherwise leak state across tests.
"""

import io
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import backend.api.routes as routes
from backend.main import app
from backend.models.schemas import FinalReport
from backend.services import storage

client = TestClient(app)

SAMPLE_REPORT = FinalReport(
    case_id="CASE001",
    diagnoses=[{"name": "Tension-type headache", "confidence": 0.8, "supporting_evidence": [], "rank": 1}],
    confidence=0.8,
    evidence=[{"type": "rag", "text": "evidence text", "source": "synthetic", "score": 0.9}],
    review_required=False,
)


@pytest.fixture(autouse=True)
def _clear_storage(tmp_path, monkeypatch):
    # Cases, reports, reviews, and agent_logs are all real SQLite now --
    # point at an isolated temp file per test rather than the real dev
    # database, and (re)create the tables there.
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_api.db"))
    storage._init_tables()


@pytest.fixture
def mock_run_case(monkeypatch):
    mock = MagicMock(return_value=SAMPLE_REPORT)
    monkeypatch.setattr(routes, "run_case", mock)
    return mock


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# POST /diagnose
# --------------------------------------------------------------------------


def test_diagnose_text_only(mock_run_case):
    response = client.post(
        "/diagnose",
        data={
            "case_id": "CASE001",
            "age": 45,
            "sex": "male",
            "clinical_text": "headache and dizziness, history of hypertension",
            "history": json.dumps(["hypertension"]),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "CASE001"
    assert body["confidence"] == 0.8
    assert body["review_required"] is False

    mock_run_case.assert_called_once()
    submitted_case = mock_run_case.call_args.args[0]
    assert submitted_case.case_id == "CASE001"
    assert submitted_case.clinical_text == "headache and dizziness, history of hypertension"
    assert submitted_case.history == ["hypertension"]
    assert submitted_case.image_path is None


def test_diagnose_generates_case_id_when_not_provided(mock_run_case):
    response = client.post("/diagnose", data={"age": 30, "sex": "female"})

    assert response.status_code == 200
    submitted_case = mock_run_case.call_args.args[0]
    assert submitted_case.case_id  # non-empty, auto-generated
    assert len(submitted_case.case_id) > 0


def test_diagnose_with_image_upload(mock_run_case):
    fake_image = io.BytesIO(b"not a real image, just bytes for the test")

    response = client.post(
        "/diagnose",
        data={"case_id": "CASE_IMG", "age": 60, "sex": "female"},
        files={"image": ("scan.png", fake_image, "image/png")},
    )

    assert response.status_code == 200
    submitted_case = mock_run_case.call_args.args[0]
    assert submitted_case.image_path is not None
    assert submitted_case.image_path.endswith(".png")


def test_diagnose_missing_required_fields_returns_422(mock_run_case):
    # missing both "age" and "sex"
    response = client.post("/diagnose", data={"case_id": "CASE001"})

    assert response.status_code == 422
    mock_run_case.assert_not_called()


def test_diagnose_invalid_labs_json_returns_422(mock_run_case):
    response = client.post(
        "/diagnose",
        data={"age": 45, "sex": "male", "labs": "{not valid json"},
    )

    assert response.status_code == 422
    assert "labs" in response.json()["detail"]
    mock_run_case.assert_not_called()


def test_diagnose_internal_error_returns_500_without_stack_trace(monkeypatch):
    monkeypatch.setattr(routes, "run_case", MagicMock(side_effect=RuntimeError("secret internal detail")))

    response = client.post("/diagnose", data={"age": 45, "sex": "male"})

    assert response.status_code == 500
    body = response.json()
    assert "secret internal detail" not in json.dumps(body)
    assert body["detail"] == "Internal server error while processing the case"


# --------------------------------------------------------------------------
# GET /diagnose/{case_id}
# --------------------------------------------------------------------------


def test_get_diagnosis_not_found():
    response = client.get("/diagnose/does-not-exist")
    assert response.status_code == 404


def test_get_diagnosis_found_after_diagnose(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.get("/diagnose/CASE001")

    assert response.status_code == 200
    assert response.json()["case_id"] == "CASE001"


# --------------------------------------------------------------------------
# POST /review/{case_id}
# --------------------------------------------------------------------------


def test_review_not_found():
    response = client.post(
        "/review/does-not-exist",
        json={"decision": "accept", "reviewer": "Dr. Smith"},
    )
    assert response.status_code == 404


def test_review_accept(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.post(
        "/review/CASE001",
        json={"decision": "accept", "reviewer": "Dr. Smith"},
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.pop("timestamp"), str) and body["case_id"]  # server-assigned, just check it's there
    assert body == {
        "case_id": "CASE001",
        "decision": "accept",
        "reviewer": "Dr. Smith",
        "annotation": None,
        "recorded": True,
    }


def test_review_annotate_without_annotation_returns_422(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.post(
        "/review/CASE001",
        json={"decision": "annotate", "reviewer": "Dr. Smith"},
    )

    assert response.status_code == 422


def test_review_annotate_with_annotation_succeeds(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.post(
        "/review/CASE001",
        json={"decision": "annotate", "annotation": "Consider MRI follow-up.", "reviewer": "Dr. Smith"},
    )

    assert response.status_code == 200
    assert response.json()["annotation"] == "Consider MRI follow-up."
