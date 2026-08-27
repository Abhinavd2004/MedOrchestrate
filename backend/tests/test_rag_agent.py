"""Tests for backend.agents.rag_agent (+ retriever.py / reranker.py).

Qdrant and the cross-encoder reranker model are both mocked -- no real
network/Qdrant server and no real model download/inference. Retrieval
and reranking still run their real code paths against fakes, so
query-building, the retrieve -> rerank pipeline, and the low-score
fallback logic in rag_agent.py are all genuinely exercised.
"""

import pytest

import backend.rag.reranker as reranker_module
import backend.rag.retriever as retriever_module
from backend.agents.rag_agent import (
    LOW_SCORE_THRESHOLD,
    _build_fallback_query,
    _build_query,
    run_rag,
)
from backend.models.schemas import ClinicalFindings, ImagingFindings, RAGEvidence

CASE001_FINDINGS = ClinicalFindings(
    demographics={},
    symptoms=["headache", "dizziness"],
    history=["hypertension"],
    labs={},
    imaging_text_findings=[],
)
CASE001_IMAGING = ImagingFindings(
    modality="MRI brain scan",
    findings=["white matter hyperintensity"],
    confidence=0.98,
)


class _FakeHit:
    def __init__(self, score: float, payload: dict):
        self.score = score
        self.payload = payload


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    """Returns the same canned candidate set regardless of query text --
    realistic enough for exercising the pipeline/fallback logic, since
    what actually changes between the primary and fallback query in
    these tests is the reranker's relevance scoring, not retrieval."""

    def __init__(self, payloads: list[dict]):
        self._payloads = payloads
        self.query_calls: list[list[float]] = []

    def query_points(self, collection_name, query, limit):
        self.query_calls.append(query)
        hits = [_FakeHit(score=0.5, payload=p) for p in self._payloads[:limit]]
        return _FakeQueryResult(hits)


class _FakeCrossEncoder:
    """Returns a preconfigured sequence of raw-logit lists, one per call
    to .predict() -- lets tests script "low scores first call, high
    scores second call" to drive the fallback path deterministically."""

    def __init__(self, score_sequence: list[list[float]]):
        self._score_sequence = score_sequence
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        idx = min(len(self.calls) - 1, len(self._score_sequence) - 1)
        scores = self._score_sequence[idx]
        return scores[: len(pairs)]


CANDIDATE_PAYLOADS = [
    {"text": "Brain MRI findings overview...", "source": "synthetic", "doc_id": "brain_mri_findings_overview", "page": 1, "chunk_length": 163},
    {"text": "Stroke and TIA red flags...", "source": "synthetic", "doc_id": "stroke_tia_red_flags", "page": 1, "chunk_length": 168},
    {"text": "Dizziness and vertigo overview...", "source": "synthetic", "doc_id": "dizziness_vertigo_overview", "page": 1, "chunk_length": 181},
]


@pytest.fixture
def fake_qdrant(monkeypatch):
    client = _FakeQdrantClient(CANDIDATE_PAYLOADS)
    monkeypatch.setattr(retriever_module, "_get_client", lambda: client)
    monkeypatch.setattr(retriever_module, "embed_text", lambda text: [0.0] * 768)
    return client


def _patch_cross_encoder(monkeypatch, score_sequence: list[list[float]]) -> _FakeCrossEncoder:
    fake_model = _FakeCrossEncoder(score_sequence)
    monkeypatch.setattr(reranker_module, "_get_model", lambda: fake_model)
    return fake_model


# --------------------------------------------------------------------------
# Query-building logic
# --------------------------------------------------------------------------


def test_build_query_includes_symptoms_history_and_imaging():
    query = _build_query(CASE001_FINDINGS, CASE001_IMAGING)

    assert "headache" in query
    assert "dizziness" in query
    assert "hypertension" in query
    assert "white matter hyperintensity" in query


def test_build_query_without_imaging_omits_imaging_section():
    query = _build_query(CASE001_FINDINGS, None)

    assert "headache" in query
    assert "imaging findings" not in query


def test_build_query_empty_findings_returns_empty_string():
    empty = ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])

    assert _build_query(empty, None) == ""


def test_build_fallback_query_uses_top_symptoms_only():
    fallback = _build_fallback_query(CASE001_FINDINGS)

    assert fallback == "headache, dizziness"
    assert "hypertension" not in fallback


# --------------------------------------------------------------------------
# run_rag: happy path (no fallback needed)
# --------------------------------------------------------------------------


async def test_run_rag_happy_path_returns_ragevidence(fake_qdrant, monkeypatch):
    # raw logit 4.0 -> sigmoid ~0.982, comfortably above LOW_SCORE_THRESHOLD
    fake_model = _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING)

    assert isinstance(result, RAGEvidence)
    assert len(result.evidence) == 3
    assert all(0.0 <= item.score <= 1.0 for item in result.evidence)
    # sorted descending by (reranked) score
    scores = [item.score for item in result.evidence]
    assert scores == sorted(scores, reverse=True)
    assert all(item.source == "synthetic" for item in result.evidence)
    # only one retrieve + one rerank call -- fallback should not trigger
    assert len(fake_qdrant.query_calls) == 1
    assert len(fake_model.calls) == 1


async def test_run_rag_empty_findings_short_circuits_without_calling_pipeline(monkeypatch):
    client = _FakeQdrantClient(CANDIDATE_PAYLOADS)
    monkeypatch.setattr(retriever_module, "_get_client", lambda: client)
    fake_model = _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])

    empty = ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])
    result = await run_rag(empty, None)

    assert result == RAGEvidence(evidence=[])
    assert client.query_calls == []
    assert fake_model.calls == []


# --------------------------------------------------------------------------
# run_rag: fallback path
# --------------------------------------------------------------------------


async def test_run_rag_retries_with_fallback_query_when_all_scores_low(fake_qdrant, monkeypatch):
    # First call: all raw logits very negative -> sigmoid scores near 0,
    # all below LOW_SCORE_THRESHOLD. Second call (fallback): high scores.
    fake_model = _patch_cross_encoder(
        monkeypatch,
        [[-6.0, -6.0, -6.0], [4.0, 3.0, 2.0]],
    )

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING)

    assert len(fake_model.calls) == 2
    # second rerank call should have used the simplified fallback query
    first_call_query = fake_model.calls[0][0][0]
    second_call_query = fake_model.calls[1][0][0]
    assert first_call_query != second_call_query
    assert second_call_query == _build_fallback_query(CASE001_FINDINGS)

    # final result reflects the fallback (high-score) attempt
    assert all(item.score >= LOW_SCORE_THRESHOLD for item in result.evidence)


async def test_run_rag_gives_up_after_one_retry_if_still_low(fake_qdrant, monkeypatch):
    # Both attempts score low -- should NOT retry a third time.
    fake_model = _patch_cross_encoder(
        monkeypatch,
        [[-6.0, -6.0, -6.0], [-5.0, -5.0, -5.0]],
    )

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING)

    assert len(fake_model.calls) == 2  # exactly one retry, not more
    assert all(item.score < LOW_SCORE_THRESHOLD for item in result.evidence)
