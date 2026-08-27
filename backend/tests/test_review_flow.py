"""End-to-end HITL review flow: submit a review, then read it back and
confirm it round-trips.

Covers both the API layer (POST then GET /review/{case_id}, mocking
orchestrator.run_case exactly like test_api.py so this stays offline)
and the storage layer directly (services/storage.py's real SQLite
persistence), each pointed at an isolated temp database file so this
suite never touches the real dev database and stays repeatable.
"""

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
    diagnoses=[
        {"name": "Chronic hypertensive cerebral small vessel disease", "confidence": 0.58, "supporting_evidence": [], "rank": 1},
        {"name": "Vestibular migraine", "confidence": 0.3, "supporting_evidence": [], "rank": 2},
    ],
    confidence=0.58,
    evidence=[{"type": "rag", "text": "evidence text", "source": "synthetic", "score": 0.9}],
    review_required=True,
    conflicts=["Offline and live evidence point toward different underlying causes."],
)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_review_flow.db"))
    storage._init_tables()


@pytest.fixture
def diagnosed_case(monkeypatch):
    """Puts a report in storage the way a real /diagnose call would,
    since /review requires the case to have been diagnosed first."""
    monkeypatch.setattr(routes, "run_case", MagicMock(return_value=SAMPLE_REPORT))
    response = client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})
    assert response.status_code == 200
    return "CASE001"


# --------------------------------------------------------------------------
# API round-trip: POST /review/{case_id} then GET /review/{case_id}
# --------------------------------------------------------------------------


def test_submit_then_get_review_round_trips_via_api(diagnosed_case):
    post_response = client.post(
        "/review/CASE001",
        json={"decision": "accept", "annotation": None, "reviewer": "Dr. Smith"},
    )
    assert post_response.status_code == 200
    posted = post_response.json()

    get_response = client.get("/review/CASE001")
    assert get_response.status_code == 200
    fetched = get_response.json()

    assert fetched["case_id"] == "CASE001"
    assert fetched["decision"] == "accept"
    assert fetched["reviewer"] == "Dr. Smith"
    assert fetched["annotation"] is None
    assert fetched["timestamp"] == posted["timestamp"]  # same row, not just same shape


def test_override_round_trips_with_annotation_carrying_the_new_diagnosis(diagnosed_case):
    override_text = "Override diagnosis: Vestibular migraine (reviewer judgment based on episodic pattern)"

    client.post(
        "/review/CASE001",
        json={"decision": "override", "annotation": override_text, "reviewer": "Dr. Lee"},
    )

    response = client.get("/review/CASE001")

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "override"
    assert body["annotation"] == override_text
    assert body["reviewer"] == "Dr. Lee"


def test_annotate_round_trips(diagnosed_case):
    client.post(
        "/review/CASE001",
        json={"decision": "annotate", "annotation": "Recommend follow-up MRI in 3 months.", "reviewer": "Dr. Patel"},
    )

    response = client.get("/review/CASE001")

    assert response.status_code == 200
    assert response.json() == {
        "case_id": "CASE001",
        "decision": "annotate",
        "reviewer": "Dr. Patel",
        "annotation": "Recommend follow-up MRI in 3 months.",
        "timestamp": response.json()["timestamp"],
        "recorded": True,
    }


def test_get_review_not_found_for_case_never_reviewed(diagnosed_case):
    response = client.get("/review/CASE001")
    assert response.status_code == 404


def test_re_review_returns_the_most_recent_one(diagnosed_case):
    client.post("/review/CASE001", json={"decision": "annotate", "annotation": "First pass note.", "reviewer": "Dr. Smith"})
    client.post("/review/CASE001", json={"decision": "accept", "reviewer": "Dr. Lee"})

    response = client.get("/review/CASE001")

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "accept"
    assert body["reviewer"] == "Dr. Lee"


# --------------------------------------------------------------------------
# Storage layer round-trip, direct (no HTTP)
# --------------------------------------------------------------------------


def test_storage_save_review_then_get_reviews_round_trips():
    stored = storage.save_review(
        "CASE002", {"decision": "accept", "annotation": None, "reviewer": "Dr. Nguyen"}
    )

    fetched = storage.get_reviews("CASE002")

    assert len(fetched) == 1
    assert fetched[0]["case_id"] == "CASE002"
    assert fetched[0]["decision"] == "accept"
    assert fetched[0]["reviewer"] == "Dr. Nguyen"
    assert fetched[0]["annotation"] is None
    assert fetched[0]["timestamp"] == stored["timestamp"]
    assert fetched[0]["id"] == stored["id"]


def test_storage_get_reviews_returns_full_history_most_recent_first():
    storage.save_review("CASE003", {"decision": "annotate", "annotation": "note 1", "reviewer": "Dr. A"})
    storage.save_review("CASE003", {"decision": "override", "annotation": "note 2", "reviewer": "Dr. B"})

    history = storage.get_reviews("CASE003")

    assert len(history) == 2
    assert history[0]["reviewer"] == "Dr. B"  # most recent first
    assert history[1]["reviewer"] == "Dr. A"


def test_storage_get_reviews_returns_empty_list_when_never_reviewed():
    assert storage.get_reviews("never-reviewed-case") == []
