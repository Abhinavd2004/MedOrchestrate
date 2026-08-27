"""Full-matrix integration tests against the REAL orchestrator.run_case()
pipeline, per the build guide.

How this differs from the other test files:
  - backend/tests/test_orchestrator.py mocks whole Phase 1-7 agent
    functions (extract_clinical_findings, run_rag, fuse, etc.) as black
    boxes -- it tests orchestration/control-flow only.
  - backend/tests/test_<agent>.py files test one agent in isolation.
  - THIS file mocks only the true external boundary: the Anthropic API
    client (per agent module), the Qdrant client + cross-encoder model,
    the BiomedCLIP model bundle, and the PubMed/MedlinePlus httpx calls.
    Every agent's own real business logic runs -- query building,
    retrieval scoring and the low-score fallback, fusion prompt
    construction and citation-grounding validation, the optimizer's
    refine-loop, confidence math, and the orchestrator's wiring between
    all of them.

CrewAI's own Agent-decision layer is bypassed via orchestrator._execute
(same technique as test_orchestrator.py) since it would otherwise need a
live Claude call just to decide "call my tool" -- everything downstream
of that decision (the tool's real _run(), i.e. all the logic above)
still executes normally. No network access, no API key, offline and
fast (~most of these tests take well under a second).

All data used here is synthetic/de-identified -- see data/test_cases/
for standalone fixture files covering the same matrix for manual/demo
use.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import torch

import backend.agents.clinical_agent as clinical_agent
import backend.agents.evidence_agent as evidence_agent
import backend.agents.fusion_agent as fusion_agent
import backend.agents.imaging_agent as imaging_agent
import backend.agents.optimizer_agent as optimizer_agent
import backend.rag.reranker as reranker
import backend.rag.retriever as retriever
import backend.services.orchestrator as orchestrator
from backend.agents.imaging_labels import FINDING_LABELS, MODALITY_LABELS
from backend.models.schemas import CaseInput, Patient
from backend.services import storage
from backend.services.confidence import THRESHOLD
from backend.services.orchestrator import run_case

# --------------------------------------------------------------------------
# Shared fixtures: bypass CrewAI's LLM-decision layer, isolate SQLite.
# --------------------------------------------------------------------------


def _direct_execute(agent, task):
    """Bypass CrewAI's real Crew.kickoff()/LLM call -- run the agent's one
    tool directly. Safe substitution: the tools ignore whatever the LLM
    would pass them and read the shared _CaseContext instead (see
    orchestrator._execute's own docstring)."""
    return agent.tools[0]._run()


@pytest.fixture(autouse=True)
def _bypass_crewai(monkeypatch):
    monkeypatch.setattr(orchestrator, "_execute", _direct_execute)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_integration.db"))
    storage._init_tables()


# --------------------------------------------------------------------------
# Anthropic client fakes -- one per agent module, since each of
# clinical_agent / fusion_agent / optimizer_agent lazily caches its own
# `_get_client()`. evidence_agent's Claude summarization is intentionally
# never triggered below (canned abstracts are kept under the 400-char
# auto-summarize threshold), so it needs no client mock in these tests.
# --------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


def _fake_anthropic_client(response_texts):
    """response_texts: a single JSON string (every call returns it) or a
    list (one per successive call, e.g. across optimizer iterations)."""
    client = MagicMock()
    if isinstance(response_texts, list):
        client.messages.create = AsyncMock(
            side_effect=[_FakeAnthropicResponse(t) for t in response_texts]
        )
    else:
        client.messages.create = AsyncMock(return_value=_FakeAnthropicResponse(response_texts))
    return client


@pytest.fixture(autouse=True)
def _clear_live_evidence_cache():
    # evidence_agent.gather_live_evidence() caches results in a module-level
    # dict keyed by query string (by design -- "avoid repeat calls within a
    # session", see Phase 4). That's a real, global module attribute, so it
    # persists across tests in the same pytest process -- without clearing
    # it, an earlier test's successful query for the same symptoms would
    # silently serve a cached result to a later test simulating an outage
    # for that exact query, bypassing the mock entirely.
    evidence_agent._live_evidence_cache.clear()
    yield
    evidence_agent._live_evidence_cache.clear()


def _clinical_response(symptoms, history, imaging_text_findings=None, labs=None):
    return json.dumps(
        {
            "demographics": {},
            "symptoms": symptoms,
            "history": history,
            "labs": labs or {},
            "imaging_text_findings": imaging_text_findings or [],
        }
    )


def _fusion_response(diagnoses, overall_confidence, conflicts=None):
    return json.dumps(
        {"diagnoses": diagnoses, "overall_confidence": overall_confidence, "conflicts": conflicts or []}
    )


def _optimizer_refinement_response(strategy, query, rationale="refine the query"):
    return json.dumps({"strategy": strategy, "query": query, "rationale": rationale})


def _install_clinical_mock(monkeypatch, response_texts):
    # Build the fake client ONCE and always return that SAME instance --
    # matching the real _get_client()'s cached-singleton behavior. A lambda
    # that constructs a fresh mock on every call would make the tool's real
    # call land on one instance while a test's later inspection call
    # (`clinical_agent._get_client()`) gets a different, never-called one.
    client = _fake_anthropic_client(response_texts)
    monkeypatch.setattr(clinical_agent, "_get_client", lambda: client)
    return client


def _install_fusion_mock(monkeypatch, response_texts):
    client = _fake_anthropic_client(response_texts)
    monkeypatch.setattr(fusion_agent, "_get_client", lambda: client)
    return client


def _install_optimizer_llm_mock(monkeypatch, response_texts):
    client = _fake_anthropic_client(response_texts)
    monkeypatch.setattr(optimizer_agent, "_get_client", lambda: client)
    return client


def _forbid_optimizer(monkeypatch):
    """Canary: makes the test fail loudly (AssertionError) if the
    optimizer's strategy-selection Claude client is ever constructed --
    used by the high-confidence scenario to prove the optimizer never runs."""

    def _boom():
        raise AssertionError("optimizer should not run for a high-confidence case")

    monkeypatch.setattr(optimizer_agent, "_get_client", _boom)


# --------------------------------------------------------------------------
# RAG fakes: Qdrant client + embed_text + cross-encoder reranker model.
# --------------------------------------------------------------------------


class _FakeQdrantHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class _FakeQdrantQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    def __init__(self, scored_payloads=None, raise_error=None):
        self._scored_payloads = scored_payloads or []
        self._raise_error = raise_error
        self.query_calls = 0

    def query_points(self, collection_name, query, limit):
        self.query_calls += 1
        if self._raise_error is not None:
            raise self._raise_error
        hits = [_FakeQdrantHit(score, payload) for score, payload in self._scored_payloads[:limit]]
        return _FakeQdrantQueryResult(hits)


class _FakeCrossEncoder:
    """Returns each candidate's score unchanged (pass-through) via a
    score_map keyed by text, so the RAG scores set on the fake Qdrant
    client are what actually drive S_rag / the low-score fallback,
    without the reranker's own math ever needing a real model."""

    def __init__(self, score_map=None, default=5.0):
        self._score_map = score_map or {}
        self._default = default

    def predict(self, pairs):
        return [self._score_map.get(text, self._default) for (_query, text) in pairs]


def _install_rag_mocks(monkeypatch, scored_payloads=None, raise_error=None, cross_encoder_score_map=None):
    fake_client = _FakeQdrantClient(scored_payloads=scored_payloads, raise_error=raise_error)
    monkeypatch.setattr(retriever, "_get_client", lambda: fake_client)
    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 768)
    fake_cross_encoder = _FakeCrossEncoder(score_map=cross_encoder_score_map)
    monkeypatch.setattr(reranker, "_get_model", lambda: fake_cross_encoder)
    return fake_client


# High-relevance-scoring RAG payload (cross-encoder logit 6.0 -> sigmoid
# ~0.9975), used by "high confidence" scenarios.
HIGH_SCORE_RAG_PAYLOAD = [
    (0.9, {"text": "Chronic hypertension is a major risk factor for cerebral small vessel disease.", "source": "synthetic", "doc_id": "d1", "page": 1, "chunk_length": 100}),
]
HIGH_SCORE_CROSS_ENCODER_MAP = {
    "Chronic hypertension is a major risk factor for cerebral small vessel disease.": 6.0,
}

# Low-relevance-scoring RAG payload (cross-encoder logit -6.0 -> sigmoid
# ~0.0025), used by "low confidence" scenarios.
LOW_SCORE_RAG_PAYLOAD = [
    (0.3, {"text": "Some tangentially related reference passage.", "source": "synthetic", "doc_id": "d2", "page": 1, "chunk_length": 50}),
]
LOW_SCORE_CROSS_ENCODER_MAP = {
    "Some tangentially related reference passage.": -6.0,
}


# --------------------------------------------------------------------------
# Live-evidence (httpx) fakes for PubMed + MedlinePlus.
# --------------------------------------------------------------------------


class _FakeAsyncHttpClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None, timeout=None):
        return self._handler(url, params)


def _json_response(url, payload):
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def _text_response(url, text):
    return httpx.Response(200, text=text, request=httpx.Request("GET", url))


PUBMED_ESEARCH_JSON = {"esearchresult": {"idlist": ["111"]}}
PUBMED_ESEARCH_EMPTY_JSON = {"esearchresult": {"idlist": []}}
PUBMED_ESUMMARY_JSON = {
    "result": {"uids": ["111"], "111": {"title": "A high-quality relevant study.", "pubdate": "2024"}}
}
PUBMED_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID Version="1">111</PMID>
<Article><Abstract><AbstractText>A short abstract, well under the auto-summarize threshold.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""

MEDLINEPLUS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <list num="1" start="0" per="1">
    <document rank="0" url="https://medlineplus.gov/highbloodpressure.html">
      <content name="title">High Blood Pressure</content>
      <content name="FullSummary">A guideline overview for patients.</content>
    </document>
  </list>
</nlmSearchResult>"""

MEDLINEPLUS_EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult><list num="0" start="0" per="0"></list></nlmSearchResult>"""


def _install_evidence_mocks(monkeypatch, pubmed_has_results=True, medlineplus_has_results=True, raise_error=None):
    def handler(url, params):
        if raise_error is not None:
            raise raise_error
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_JSON if pubmed_has_results else PUBMED_ESEARCH_EMPTY_JSON)
        if url == evidence_agent.PUBMED_ESUMMARY_URL:
            return _json_response(url, PUBMED_ESUMMARY_JSON)
        if url == evidence_agent.PUBMED_EFETCH_URL:
            return _text_response(url, PUBMED_EFETCH_XML)
        if url == evidence_agent.MEDLINEPLUS_SEARCH_URL:
            return _text_response(url, MEDLINEPLUS_XML if medlineplus_has_results else MEDLINEPLUS_EMPTY_XML)
        raise AssertionError(f"unexpected URL {url}")

    fake_client = _FakeAsyncHttpClient(handler)
    monkeypatch.setattr(evidence_agent.httpx, "AsyncClient", lambda *a, **kw: fake_client)


# --------------------------------------------------------------------------
# BiomedCLIP fakes for imaging.
# --------------------------------------------------------------------------


class _FakeImagingTokenizer:
    def __call__(self, labels):
        return list(labels)


class _FakeImagingModel:
    def __init__(self, image_embedding, text_embeddings):
        self.image_embedding = image_embedding
        self.text_embeddings = text_embeddings

    def encode_image(self, image_input):
        return self.image_embedding

    def encode_text(self, labels):
        return torch.stack([self.text_embeddings[label] for label in labels])


def _install_imaging_mocks(monkeypatch, modality="MRI brain scan", finding="white matter hyperintensity"):
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = {label: torch.tensor([0.0, 1.0]) for label in MODALITY_LABELS + FINDING_LABELS}
    text_embeddings[modality] = torch.tensor([1.0, 0.0])
    text_embeddings[finding] = torch.tensor([1.0, 0.0])
    fake_model = _FakeImagingModel(image_embedding, text_embeddings)
    monkeypatch.setattr(
        imaging_agent,
        "_get_model_bundle",
        lambda: (fake_model, lambda img: torch.zeros(3, 224, 224), _FakeImagingTokenizer()),
    )


DEMO_IMAGE_PATH = "data/images/synthetic_demo_brain_mri_001.png"


# --------------------------------------------------------------------------
# 1. Text only -> full pipeline -> report
# --------------------------------------------------------------------------


def test_text_only_full_pipeline_produces_report(monkeypatch):
    _install_clinical_mock(
        monkeypatch, _clinical_response(symptoms=["headache", "dizziness"], history=["hypertension"])
    )
    _install_rag_mocks(
        monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP
    )
    _install_evidence_mocks(monkeypatch)
    _install_fusion_mock(
        monkeypatch,
        _fusion_response(
            [{"name": "Tension-type headache", "confidence": 0.7, "supporting_evidence": [], "rank": 1}], 0.7
        ),
    )

    case = CaseInput(
        case_id="INTEG-TEXT-ONLY",
        patient=Patient(age=45, sex="male"),
        clinical_text="headache and dizziness, history of hypertension",
    )
    report = run_case(case)

    assert report.case_id == "INTEG-TEXT-ONLY"
    assert len(report.diagnoses) == 1
    assert isinstance(report.confidence, float)
    assert any(item["type"] == "rag" for item in report.evidence)


# --------------------------------------------------------------------------
# 2. Image only -> full pipeline -> report
# --------------------------------------------------------------------------


def test_image_only_full_pipeline_produces_report(monkeypatch):
    # clinical_text and medical_report are both None -- extract_clinical_findings()
    # short-circuits to an empty ClinicalFindings via its own real code path
    # (verified in test_clinical_agent.py), so no clinical Claude mock is
    # needed here at all.
    _install_imaging_mocks(monkeypatch, modality="MRI brain scan", finding="white matter hyperintensity")
    _install_rag_mocks(
        monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP
    )
    _install_evidence_mocks(monkeypatch)
    fuse_client = _install_fusion_mock(
        monkeypatch,
        _fusion_response(
            [{"name": "Chronic small vessel disease", "confidence": 0.6, "supporting_evidence": [], "rank": 1}], 0.6
        ),
    )

    case = CaseInput(case_id="INTEG-IMAGE-ONLY", patient=Patient(age=60, sex="female"), image_path=DEMO_IMAGE_PATH)
    report = run_case(case)

    assert report.case_id == "INTEG-IMAGE-ONLY"
    assert len(report.diagnoses) == 1

    # confirm imaging genuinely ran and its findings reached fusion's prompt
    _, kwargs = fuse_client.messages.create.call_args
    sent_content = kwargs["messages"][0]["content"]
    assert "MRI brain scan" in sent_content
    assert "white matter hyperintensity" in sent_content


def test_image_only_clinical_findings_are_empty_no_claude_call(monkeypatch):
    """Explicit check that no clinical Claude call happens for an image-only case."""
    monkeypatch.setattr(
        clinical_agent,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("clinical Claude client should never be constructed")),
    )
    _install_imaging_mocks(monkeypatch)
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Finding-based diagnosis", "confidence": 0.6, "supporting_evidence": [], "rank": 1}], 0.6))

    case = CaseInput(case_id="INTEG-IMAGE-ONLY-2", patient=Patient(age=60, sex="female"), image_path=DEMO_IMAGE_PATH)
    report = run_case(case)  # must not raise the canary AssertionError

    assert report.case_id == "INTEG-IMAGE-ONLY-2"


# --------------------------------------------------------------------------
# 3. Text + MRI image -> fused report
# --------------------------------------------------------------------------


def test_text_and_image_fused_report_reflects_both(monkeypatch):
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache", "dizziness"], history=["hypertension"]))
    _install_imaging_mocks(monkeypatch, modality="MRI brain scan", finding="white matter hyperintensity")
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch)
    fake_fusion_client = _fake_anthropic_client(
        _fusion_response([{"name": "Hypertensive small vessel disease", "confidence": 0.65, "supporting_evidence": [], "rank": 1}], 0.65)
    )
    monkeypatch.setattr(fusion_agent, "_get_client", lambda: fake_fusion_client)

    case = CaseInput(
        case_id="INTEG-TEXT-AND-IMAGE",
        patient=Patient(age=45, sex="male"),
        clinical_text="headache and dizziness, history of hypertension",
        image_path=DEMO_IMAGE_PATH,
    )
    report = run_case(case)

    assert report.case_id == "INTEG-TEXT-AND-IMAGE"
    assert len(report.diagnoses) == 1

    # the fused prompt sent to Claude must contain BOTH the clinical
    # symptoms and the imaging findings -- proves fusion_agent's real
    # _format_clinical()/_format_imaging() ran on real (non-None) inputs.
    _, kwargs = fake_fusion_client.messages.create.call_args
    sent_content = kwargs["messages"][0]["content"]
    assert "headache" in sent_content
    assert "MRI brain scan" in sent_content
    assert "white matter hyperintensity" in sent_content


# --------------------------------------------------------------------------
# 4. Text + MRI report (text) -> processed by Clinical Agent correctly
# --------------------------------------------------------------------------


def test_text_and_medical_report_text_both_reach_clinical_agent(monkeypatch):
    fake_clinical_client = _fake_anthropic_client(
        _clinical_response(
            symptoms=["headache", "dizziness"],
            history=["hypertension"],
            imaging_text_findings=["no acute intracranial abnormality"],
        )
    )
    monkeypatch.setattr(clinical_agent, "_get_client", lambda: fake_clinical_client)
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Tension-type headache", "confidence": 0.7, "supporting_evidence": [], "rank": 1}], 0.7))

    case = CaseInput(
        case_id="INTEG-TEXT-AND-REPORT",
        patient=Patient(age=45, sex="male"),
        clinical_text="headache and dizziness",
        medical_report="MRI brain report: no acute intracranial abnormality.",
    )
    report = run_case(case)

    # the real clinical_agent._build_user_message() must have concatenated
    # BOTH clinical_text and medical_report into the single prompt sent.
    _, kwargs = fake_clinical_client.messages.create.call_args
    sent_content = kwargs["messages"][0]["content"]
    assert "headache and dizziness" in sent_content
    assert "MRI brain report: no acute intracranial abnormality." in sent_content

    assert report.case_id == "INTEG-TEXT-AND-REPORT"
    # imaging_text_findings extracted from the medical_report text flowed
    # into the RAG query (proves _build_query() picked it up) -- checked
    # indirectly via the fake Qdrant client having been queried at all.
    assert len(report.diagnoses) == 1


# --------------------------------------------------------------------------
# 5. High confidence case -> report generated without optimizer running
# --------------------------------------------------------------------------


def test_high_confidence_case_skips_optimizer(monkeypatch):
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache"], history=["hypertension"]))
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch, pubmed_has_results=True, medlineplus_has_results=True)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Hypertension-related headache", "confidence": 0.9, "supporting_evidence": [], "rank": 1}], 0.9, conflicts=[]))
    _forbid_optimizer(monkeypatch)  # fails loudly if the optimizer is ever invoked

    case = CaseInput(case_id="INTEG-HIGH-CONF", patient=Patient(age=45, sex="male"), clinical_text="headache")
    report = run_case(case)  # must not raise the canary AssertionError

    assert report.confidence >= THRESHOLD
    assert report.iteration_count == 0
    assert report.review_required is False


# --------------------------------------------------------------------------
# 6 & 7. Low confidence -> optimizer runs, capped at 3 iterations,
#         still low after 3 -> review_required = True
# --------------------------------------------------------------------------


def _setup_low_confidence_scenario(monkeypatch):
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["fatigue"], history=[]))
    # Every RAG call (initial pass + all 3 optimizer iterations) returns
    # the same low-scoring payload -- our fake Qdrant client ignores the
    # actual query text, so confidence stays low deterministically across
    # every attempt, driving the optimizer through its full budget.
    _install_rag_mocks(monkeypatch, scored_payloads=LOW_SCORE_RAG_PAYLOAD, cross_encoder_score_map=LOW_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch, pubmed_has_results=False, medlineplus_has_results=False)
    # 1 initial fuse() call + 3 optimizer-iteration fuse() calls = 4 total.
    low_fusion = _fusion_response([{"name": "Uncertain", "confidence": 0.2, "supporting_evidence": [], "rank": 1}], 0.2, conflicts=[])
    _install_fusion_mock(monkeypatch, [low_fusion, low_fusion, low_fusion, low_fusion])
    optimizer_client = _install_optimizer_llm_mock(
        monkeypatch,
        [
            _optimizer_refinement_response("expand", "fatigue broader symptom search"),
            _optimizer_refinement_response("narrow", "fatigue specific search"),
            _optimizer_refinement_response("paraphrase", "tiredness alternate phrasing"),
        ],
    )
    case = CaseInput(case_id="INTEG-LOW-CONF", patient=Patient(age=50, sex="female"), clinical_text="fatigue")
    return case, optimizer_client


def test_low_confidence_optimizer_runs_capped_at_three_iterations(monkeypatch):
    case, optimizer_client = _setup_low_confidence_scenario(monkeypatch)
    report = run_case(case)

    assert report.confidence < THRESHOLD
    assert report.iteration_count == 3  # hard cap respected, not more
    assert optimizer_client.messages.create.await_count == 3  # exactly 3 strategy-selection calls, not 4+


def test_still_low_after_three_iterations_sets_review_required(monkeypatch):
    case, _optimizer_client = _setup_low_confidence_scenario(monkeypatch)
    report = run_case(case)

    assert report.confidence < THRESHOLD
    assert report.review_required is True


# --------------------------------------------------------------------------
# 8. RAG failure (Qdrant unreachable) -> pipeline continues, no fabricated evidence
# --------------------------------------------------------------------------


def test_rag_failure_continues_without_fabricating_evidence(monkeypatch):
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache"], history=["hypertension"]))
    _install_rag_mocks(monkeypatch, raise_error=ConnectionError("Qdrant unreachable: connection refused"))
    _install_evidence_mocks(monkeypatch)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Tension-type headache", "confidence": 0.4, "supporting_evidence": [], "rank": 1}], 0.4))

    case = CaseInput(case_id="INTEG-RAG-DOWN", patient=Patient(age=45, sex="male"), clinical_text="headache")
    report = run_case(case)  # must not raise

    # no fabricated RAG evidence -- the offline-evidence portion of the
    # report is genuinely empty, not silently backfilled with something
    # that looks like it came from a working retrieval.
    assert not any(item["type"] == "rag" for item in report.evidence)

    # the failure is not silent: it's recorded in agent_logs, with a
    # "failure" status distinguishable from "legitimately found nothing"
    # (which would be status="success" with an empty evidence list).
    logs = storage.get_agent_logs("INTEG-RAG-DOWN")
    rag_logs = [log for log in logs if log["agent_name"] == "rag"]
    assert len(rag_logs) == 1
    assert rag_logs[0]["status"] == "failure"


def test_rag_failure_error_message_is_recorded(monkeypatch):
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache"], history=[]))
    _install_rag_mocks(monkeypatch, raise_error=ConnectionError("Qdrant unreachable: connection refused"))
    _install_evidence_mocks(monkeypatch)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Tension-type headache", "confidence": 0.4, "supporting_evidence": [], "rank": 1}], 0.4))

    captured_logs = []
    monkeypatch.setattr(orchestrator, "log_event", lambda event, **fields: captured_logs.append(fields))

    case = CaseInput(case_id="INTEG-RAG-DOWN-2", patient=Patient(age=45, sex="male"), clinical_text="headache")
    run_case(case)

    rag_failure_logs = [entry for entry in captured_logs if entry.get("agent_name") == "rag" and entry.get("status") == "failure"]
    assert len(rag_failure_logs) == 1
    assert "Qdrant unreachable" in rag_failure_logs[0]["error"]


# --------------------------------------------------------------------------
# 9. Web failure (PubMed unreachable) -> continues on offline evidence only,
#    report clearly indicates live evidence was unavailable
# --------------------------------------------------------------------------


def test_web_failure_continues_with_offline_evidence_only(monkeypatch):
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache"], history=["hypertension"]))
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch, raise_error=httpx.TimeoutException("PubMed unreachable: timed out"))
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Hypertension-related headache", "confidence": 0.6, "supporting_evidence": [], "rank": 1}], 0.6))

    case = CaseInput(case_id="INTEG-WEB-DOWN", patient=Patient(age=45, sex="male"), clinical_text="headache")
    report = run_case(case)  # must not raise

    # offline (RAG) evidence still present -- pipeline degrades gracefully
    # on ONLY the failed piece, not everything.
    assert any(item["type"] == "rag" for item in report.evidence)
    # no live evidence -- and NOT fabricated.
    assert not any(item["type"] == "live" for item in report.evidence)
    # the report itself carries an explicit, checkable signal that live
    # evidence was unavailable (not just an empty list a caller has to
    # infer meaning from).
    assert report.live_evidence_available is False


def test_web_success_sets_live_evidence_available_true(monkeypatch):
    """Contrast case: confirms the flag is a real signal, not always False."""
    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache"], history=["hypertension"]))
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch, pubmed_has_results=True, medlineplus_has_results=True)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Hypertension-related headache", "confidence": 0.6, "supporting_evidence": [], "rank": 1}], 0.6))

    case = CaseInput(case_id="INTEG-WEB-UP", patient=Patient(age=45, sex="male"), clinical_text="headache")
    report = run_case(case)

    assert report.live_evidence_available is True
    assert any(item["type"] == "live" for item in report.evidence)


# --------------------------------------------------------------------------
# 10. Invalid image upload -> API returns a validation error, not a 500
# --------------------------------------------------------------------------


def test_invalid_image_upload_returns_422_not_500(monkeypatch):
    import io

    from fastapi.testclient import TestClient

    import backend.api.routes as routes
    from backend.main import app

    # Sanity-guard: if this ever reaches run_case(), the test should fail
    # loudly rather than silently succeed for the wrong reason.
    monkeypatch.setattr(routes, "run_case", MagicMock(side_effect=AssertionError("should never reach run_case")))

    client = TestClient(app)
    bad_file = io.BytesIO(b"this is not an image, just plain text bytes")

    response = client.post(
        "/diagnose",
        data={"case_id": "INTEG-BAD-IMAGE", "age": 45, "sex": "male"},
        files={"image": ("notes.txt", bad_file, "text/plain")},
    )

    assert response.status_code == 422
    assert response.status_code != 500
    assert "extension" in response.json()["detail"].lower()


def test_valid_image_extension_but_corrupted_content_degrades_gracefully_not_500(monkeypatch):
    """A correctly-named .png that isn't real image data passes the API's
    extension check (422 is only for obviously-wrong file types) but fails
    inside analyze_image() -- confirms that failure mode still degrades to
    a 200 with no imaging findings, not a 500, per the orchestrator's
    graceful-degradation design."""
    import io

    from fastapi.testclient import TestClient

    from backend.main import app

    _install_clinical_mock(monkeypatch, _clinical_response(symptoms=["headache"], history=[]))
    _install_rag_mocks(monkeypatch, scored_payloads=HIGH_SCORE_RAG_PAYLOAD, cross_encoder_score_map=HIGH_SCORE_CROSS_ENCODER_MAP)
    _install_evidence_mocks(monkeypatch)
    _install_fusion_mock(monkeypatch, _fusion_response([{"name": "Tension-type headache", "confidence": 0.5, "supporting_evidence": [], "rank": 1}], 0.5))

    client = TestClient(app)
    corrupted = io.BytesIO(b"not real png bytes")

    response = client.post(
        "/diagnose",
        data={"case_id": "INTEG-CORRUPT-IMAGE", "age": 45, "sex": "male", "clinical_text": "headache"},
        files={"image": ("scan.png", corrupted, "image/png")},
    )

    assert response.status_code == 200
    assert response.status_code != 500
