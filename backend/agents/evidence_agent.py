"""Evidence agent: fetches CURRENT (live) evidence, distinct from the
offline knowledge base in rag_agent.py, and normalizes it into
LiveEvidence.

Sources:
  - PubMed, via NCBI E-utilities (esearch -> esummary -> efetch),
    no API key required at this call volume.
  - MedlinePlus health topics search (wsearch.nlm.nih.gov), chosen as
    the "one guideline source" over WHO/NICE because it is a keyless,
    reliable public API in the same NLM/NIH family as PubMed, with a
    predictable XML response -- WHO and NICE do not offer an equivalent
    simple, keyless search endpoint. It surfaces consumer/clinician
    health-topic overviews rather than primary guideline documents, but
    is the closest practical "current guideline-adjacent" source
    reachable without scraping or an API key.

Evidence level: `evidence_level` is assigned by a crude keyword-match
heuristic over each result's title+abstract/summary text (see
_assess_evidence_level below). This is a PROTOTYPE SIMPLIFICATION, not
a validated evidence-grading system (e.g. GRADE) -- real grading
requires structured study-design metadata and expert appraisal.

Schema note: LiveEvidenceSource in backend/models/schemas.py gained an
additive, backward-compatible `publication_date: str | None = None`
field so PubMed's publication date (explicitly requested) has somewhere
to go, without overloading the `summary` field for it.

Resilience: every network call is wrapped in a timeout + one retry;
on final failure a source function logs a warning (this is how "live
evidence unavailable" is signaled upstream -- via logs, since the
shared LiveEvidence schema has no error/status field to encode it in)
and returns an empty list rather than raising, so gather_live_evidence
always returns a valid LiveEvidence -- worst case, sources=[] -- and the
rest of the pipeline can proceed on offline evidence alone.
"""

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET

import httpx
from anthropic import AsyncAnthropic

from backend.models.schemas import LiveEvidence, LiveEvidenceSource

logger = logging.getLogger(__name__)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"

HTTP_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 2  # 1 initial attempt + 1 retry

# Abstracts at or under this length are used as-is for `summary`;
# longer ones are auto-summarized via Claude (falling back to a
# truncated snippet if that call fails for any reason).
ABSTRACT_SUMMARIZE_THRESHOLD_CHARS = 400
CLAUDE_MODEL = "claude-sonnet-4-6"

# --------------------------------------------------------------------------
# Evidence-level heuristic -- PROTOTYPE SIMPLIFICATION.
#
# This is crude keyword matching over title+abstract text, not a
# validated evidence hierarchy. It exists so the UI has *something* to
# sort/filter on until real structured evidence grading is built.
# --------------------------------------------------------------------------
_HIGH_EVIDENCE_KEYWORDS = (
    "guideline",
    "meta-analysis",
    "meta analysis",
    "systematic review",
    "randomized controlled trial",
    "randomised controlled trial",
    " rct ",
    " rct.",
)
_MODERATE_EVIDENCE_KEYWORDS = (
    "cohort",
    "case-control",
    "case control",
    "observational study",
    "review",
)


def _assess_evidence_level(text: str) -> str:
    lowered = f" {text.lower()} "
    if any(kw in lowered for kw in _HIGH_EVIDENCE_KEYWORDS):
        return "high"
    if any(kw in lowered for kw in _MODERATE_EVIDENCE_KEYWORDS):
        return "moderate"
    return "low"


class EvidenceSourceUnavailableError(RuntimeError):
    """Raised internally when a source's HTTP calls fail after retrying."""


_live_evidence_cache: dict[str, LiveEvidence] = {}

_anthropic_client: AsyncAnthropic | None = None


def _get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic()
    return _anthropic_client


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _truncate(text: str, limit: int = 300) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


async def _get_with_retries(
    client: httpx.AsyncClient, url: str, params: dict, source_name: str
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_exc = exc
            logger.warning(
                "%s request failed (attempt %d/%d): %s", source_name, attempt, MAX_ATTEMPTS, exc
            )
    raise EvidenceSourceUnavailableError(
        f"{source_name} unavailable after {MAX_ATTEMPTS} attempts: {last_exc}"
    )


# --------------------------------------------------------------------------
# PubMed
# --------------------------------------------------------------------------


async def _esearch(client: httpx.AsyncClient, query: str, max_results: int) -> list[str]:
    response = await _get_with_retries(
        client,
        PUBMED_ESEARCH_URL,
        {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        },
        "PubMed esearch",
    )
    return response.json().get("esearchresult", {}).get("idlist", [])


async def _esummary(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, dict]:
    response = await _get_with_retries(
        client,
        PUBMED_ESUMMARY_URL,
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        "PubMed esummary",
    )
    result = response.json().get("result", {})
    return {pmid: result[pmid] for pmid in pmids if pmid in result}


def _parse_abstracts_xml(xml_text: str) -> dict[str, str]:
    abstracts: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return abstracts

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pieces = [el.text.strip() for el in article.findall(".//AbstractText") if el.text]
        if pieces:
            abstracts[pmid_el.text] = " ".join(pieces)
    return abstracts


async def _efetch_abstracts(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, str]:
    response = await _get_with_retries(
        client,
        PUBMED_EFETCH_URL,
        {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        "PubMed efetch",
    )
    return _parse_abstracts_xml(response.text)


async def _summarize_abstract(abstract: str) -> str:
    if not abstract:
        return "(no abstract available)"
    if len(abstract) <= ABSTRACT_SUMMARIZE_THRESHOLD_CHARS:
        return abstract

    try:
        client = _get_anthropic_client()
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            system=(
                "Summarize the following PubMed abstract in 1-2 sentences "
                "for a clinician quickly scanning search results. Return "
                "only the summary text, no preamble."
            ),
            messages=[{"role": "user", "content": abstract}],
        )
        summary = response.content[0].text.strip()
        return summary or _truncate(abstract)
    except Exception as exc:  # any failure here must not crash the pipeline
        logger.warning("Claude abstract summarization failed, falling back to truncation: %s", exc)
        return _truncate(abstract)


async def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            pmids = await _esearch(client, query, max_results)
        except EvidenceSourceUnavailableError as exc:
            logger.warning("PubMed live evidence unavailable for query %r: %s", query, exc)
            return []

        if not pmids:
            return []

        try:
            summaries = await _esummary(client, pmids)
        except EvidenceSourceUnavailableError as exc:
            logger.warning("PubMed esummary unavailable for query %r: %s", query, exc)
            summaries = {}

        try:
            abstracts = await _efetch_abstracts(client, pmids)
        except EvidenceSourceUnavailableError as exc:
            logger.warning("PubMed efetch unavailable for query %r: %s", query, exc)
            abstracts = {}

    results = []
    for pmid in pmids:
        summary_meta = summaries.get(pmid, {})
        title = (summary_meta.get("title") or "").strip() or "(title unavailable)"
        pubdate = summary_meta.get("pubdate") or None
        abstract = abstracts.get(pmid, "")

        evidence_summary = await _summarize_abstract(abstract)
        results.append(
            {
                "title": title,
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "summary": evidence_summary,
                "evidence_level": _assess_evidence_level(f"{title} {abstract}"),
                "publication_date": pubdate,
            }
        )
    return results


# --------------------------------------------------------------------------
# MedlinePlus (guideline-adjacent source)
# --------------------------------------------------------------------------


def _parse_medlineplus_xml(xml_text: str, max_results: int) -> list[dict]:
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for document in root.findall(".//document")[:max_results]:
        url = document.get("url", "")
        title_el = document.find("content[@name='title']")
        summary_el = document.find("content[@name='FullSummary']")

        title = _strip_html(title_el.text if title_el is not None else "")
        summary = _strip_html(summary_el.text if summary_el is not None else "")

        results.append(
            {
                "title": title or "(title unavailable)",
                "source": "MedlinePlus",
                "url": url,
                "summary": _truncate(summary, limit=400) if summary else "(no summary available)",
                "evidence_level": _assess_evidence_level(f"{title} {summary}"),
                "publication_date": None,
            }
        )
    return results


async def search_medlineplus_guidelines(query: str, max_results: int = 3) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            response = await _get_with_retries(
                client,
                MEDLINEPLUS_SEARCH_URL,
                {"db": "healthTopics", "term": query, "retmax": max_results},
                "MedlinePlus",
            )
        except EvidenceSourceUnavailableError as exc:
            logger.warning("MedlinePlus live evidence unavailable for query %r: %s", query, exc)
            return []

    return _parse_medlineplus_xml(response.text, max_results)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


async def gather_live_evidence(query: str) -> LiveEvidence:
    if query in _live_evidence_cache:
        return _live_evidence_cache[query]

    pubmed_results, guideline_results = await asyncio.gather(
        search_pubmed(query),
        search_medlineplus_guidelines(query),
    )

    sources = [LiveEvidenceSource(**item) for item in pubmed_results + guideline_results]
    result = LiveEvidence(sources=sources)

    _live_evidence_cache[query] = result
    return result
