"""Tests for backend.agents.evidence_agent.

All httpx calls are mocked via a fake httpx.AsyncClient (no real
network access, no real Anthropic calls) so the suite is fully offline.
Covers: PubMed success, timeout-with-retry, empty-results; MedlinePlus
success; the evidence-level heuristic; abstract summarization
(short/long/Claude-failure fallback); and gather_live_evidence's
merge + in-memory cache behavior.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import backend.agents.evidence_agent as evidence_agent
from backend.agents.evidence_agent import (
    _assess_evidence_level,
    _summarize_abstract,
    gather_live_evidence,
    search_medlineplus_guidelines,
    search_pubmed,
)
from backend.models.schemas import LiveEvidence

PUBMED_ESEARCH_JSON = {"esearchresult": {"idlist": ["111", "222"]}}
PUBMED_ESEARCH_EMPTY_JSON = {"esearchresult": {"idlist": []}}
PUBMED_ESUMMARY_JSON = {
    "result": {
        "uids": ["111", "222"],
        "111": {"title": "A Systematic Review of Headache in Hypertensive Patients", "pubdate": "2024 Jan"},
        "222": {"title": "A Small Case Report on Dizziness", "pubdate": "2023 Jun"},
    }
}
PUBMED_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID Version="1">111</PMID>
<Article><Abstract><AbstractText>This systematic review examines headache in hypertensive patients.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">222</PMID>
<Article><Abstract><AbstractText>A brief case report describing a single patient with dizziness.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""

MEDLINEPLUS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <list num="1" start="0" per="1">
    <document rank="0" url="https://medlineplus.gov/highbloodpressure.html">
      <content name="title">&lt;span class="qt0"&gt;High Blood Pressure&lt;/span&gt;</content>
      <content name="organizationName">National Library of Medicine</content>
      <content name="FullSummary">&lt;p&gt;Blood pressure guideline overview for patients.&lt;/p&gt;</content>
    </document>
  </list>
</nlmSearchResult>"""


class _FakeAsyncClient:
    """Mimics `async with httpx.AsyncClient() as client: await client.get(...)`."""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._handler(url, params)


def _json_response(url: str, payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def _text_response(url: str, text: str) -> httpx.Response:
    return httpx.Response(200, text=text, request=httpx.Request("GET", url))


def _patch_client(monkeypatch, handler):
    fake_client = _FakeAsyncClient(handler)
    monkeypatch.setattr(evidence_agent.httpx, "AsyncClient", lambda *a, **kw: fake_client)
    return fake_client


# --------------------------------------------------------------------------
# search_pubmed
# --------------------------------------------------------------------------


async def test_search_pubmed_success(monkeypatch):
    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_JSON)
        if url == evidence_agent.PUBMED_ESUMMARY_URL:
            return _json_response(url, PUBMED_ESUMMARY_JSON)
        if url == evidence_agent.PUBMED_EFETCH_URL:
            return _text_response(url, PUBMED_EFETCH_XML)
        raise AssertionError(f"unexpected URL {url}")

    _patch_client(monkeypatch, handler)

    results = await search_pubmed("headache dizziness hypertension", max_results=2)

    assert len(results) == 2
    first = results[0]
    assert first["title"] == "A Systematic Review of Headache in Hypertensive Patients"
    assert first["source"] == "PubMed"
    assert first["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert "systematic review" in first["summary"].lower()
    assert first["evidence_level"] == "high"  # "systematic review" keyword
    assert first["publication_date"] == "2024 Jan"

    second = results[1]
    assert second["evidence_level"] == "low"  # case report, no keyword match


async def test_search_pubmed_empty_results_short_circuits(monkeypatch):
    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_EMPTY_JSON)
        raise AssertionError("esummary/efetch should not be called when idlist is empty")

    fake_client = _patch_client(monkeypatch, handler)

    results = await search_pubmed("a query with no hits")

    assert results == []
    assert len(fake_client.calls) == 1  # only esearch was called


async def test_search_pubmed_timeout_retries_then_returns_empty(monkeypatch):
    call_count = 0

    def handler(url, params):
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("simulated timeout")

    _patch_client(monkeypatch, handler)

    results = await search_pubmed("headache dizziness hypertension")

    assert results == []
    assert call_count == evidence_agent.MAX_ATTEMPTS  # 1 initial + 1 retry, then give up


async def test_search_pubmed_efetch_failure_degrades_gracefully(monkeypatch):
    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_JSON)
        if url == evidence_agent.PUBMED_ESUMMARY_URL:
            return _json_response(url, PUBMED_ESUMMARY_JSON)
        if url == evidence_agent.PUBMED_EFETCH_URL:
            raise httpx.TimeoutException("efetch down")
        raise AssertionError(f"unexpected URL {url}")

    _patch_client(monkeypatch, handler)

    results = await search_pubmed("headache dizziness hypertension", max_results=2)

    # titles still come through even though abstracts failed
    assert len(results) == 2
    assert results[0]["title"] == "A Systematic Review of Headache in Hypertensive Patients"
    assert results[0]["summary"] == "(no abstract available)"


# --------------------------------------------------------------------------
# search_medlineplus_guidelines
# --------------------------------------------------------------------------


async def test_search_medlineplus_success(monkeypatch):
    def handler(url, params):
        assert url == evidence_agent.MEDLINEPLUS_SEARCH_URL
        return _text_response(url, MEDLINEPLUS_XML)

    _patch_client(monkeypatch, handler)

    results = await search_medlineplus_guidelines("hypertension", max_results=3)

    assert len(results) == 1
    assert results[0]["title"] == "High Blood Pressure"
    assert results[0]["source"] == "MedlinePlus"
    assert results[0]["url"] == "https://medlineplus.gov/highbloodpressure.html"
    assert "guideline overview" in results[0]["summary"]
    assert "<p>" not in results[0]["summary"]  # HTML stripped


async def test_search_medlineplus_timeout_returns_empty(monkeypatch):
    def handler(url, params):
        raise httpx.TimeoutException("simulated timeout")

    _patch_client(monkeypatch, handler)

    results = await search_medlineplus_guidelines("hypertension")

    assert results == []


# --------------------------------------------------------------------------
# Evidence-level heuristic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A systematic review of migraine treatment", "high"),
        ("Results from a randomized controlled trial of triptans", "high"),
        ("A prospective cohort study of dizziness in older adults", "moderate"),
        ("A narrative review of vestibular disorders", "moderate"),
        ("A single case report of an unusual presentation", "low"),
        ("", "low"),
    ],
)
def test_assess_evidence_level(text, expected):
    assert _assess_evidence_level(text) == expected


# --------------------------------------------------------------------------
# Abstract summarization
# --------------------------------------------------------------------------


async def test_summarize_abstract_short_returned_as_is():
    short_abstract = "A short abstract under the threshold."
    result = await _summarize_abstract(short_abstract)
    assert result == short_abstract


async def test_summarize_abstract_empty():
    result = await _summarize_abstract("")
    assert result == "(no abstract available)"


async def test_summarize_abstract_long_uses_claude(monkeypatch):
    long_abstract = "word " * 200  # well over ABSTRACT_SUMMARIZE_THRESHOLD_CHARS

    fake_client = MagicMock()
    fake_block = MagicMock()
    fake_block.text = "Concise Claude-generated summary."
    fake_response = MagicMock()
    fake_response.content = [fake_block]
    fake_client.messages.create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(evidence_agent, "_get_anthropic_client", lambda: fake_client)

    result = await _summarize_abstract(long_abstract)

    assert result == "Concise Claude-generated summary."
    fake_client.messages.create.assert_awaited_once()


async def test_summarize_abstract_claude_failure_falls_back_to_truncation(monkeypatch):
    long_abstract = "word " * 200

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
    monkeypatch.setattr(evidence_agent, "_get_anthropic_client", lambda: fake_client)

    result = await _summarize_abstract(long_abstract)

    assert result.endswith("...")
    assert len(result) <= 310  # truncated, not the full 1000-char abstract


# --------------------------------------------------------------------------
# gather_live_evidence: merge + cache
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    evidence_agent._live_evidence_cache.clear()
    yield
    evidence_agent._live_evidence_cache.clear()


async def test_gather_live_evidence_merges_both_sources(monkeypatch):
    pubmed_item = {
        "title": "PubMed Title",
        "source": "PubMed",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "summary": "summary",
        "evidence_level": "high",
        "publication_date": "2024",
    }
    guideline_item = {
        "title": "Guideline Title",
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/x.html",
        "summary": "summary",
        "evidence_level": "moderate",
        "publication_date": None,
    }
    monkeypatch.setattr(evidence_agent, "search_pubmed", AsyncMock(return_value=[pubmed_item]))
    monkeypatch.setattr(
        evidence_agent, "search_medlineplus_guidelines", AsyncMock(return_value=[guideline_item])
    )

    result = await gather_live_evidence("headache dizziness hypertension")

    assert isinstance(result, LiveEvidence)
    assert len(result.sources) == 2
    assert {s.source for s in result.sources} == {"PubMed", "MedlinePlus"}


async def test_gather_live_evidence_caches_by_query(monkeypatch):
    pubmed_mock = AsyncMock(return_value=[])
    guideline_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(evidence_agent, "search_pubmed", pubmed_mock)
    monkeypatch.setattr(evidence_agent, "search_medlineplus_guidelines", guideline_mock)

    query = "headache dizziness hypertension"
    first = await gather_live_evidence(query)
    second = await gather_live_evidence(query)

    assert first == second
    pubmed_mock.assert_awaited_once()
    guideline_mock.assert_awaited_once()


async def test_gather_live_evidence_both_sources_fail_returns_empty(monkeypatch):
    monkeypatch.setattr(evidence_agent, "search_pubmed", AsyncMock(return_value=[]))
    monkeypatch.setattr(evidence_agent, "search_medlineplus_guidelines", AsyncMock(return_value=[]))

    result = await gather_live_evidence("an unreachable query")

    assert result == LiveEvidence(sources=[])
