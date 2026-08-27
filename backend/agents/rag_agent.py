"""RAG agent: turns ClinicalFindings (+ ImagingFindings) into a query,
retrieves and reranks evidence from the local knowledge base, and
returns RAGEvidence.

Retrieval (retriever.py) and reranking (reranker.py) are both
synchronous, CPU/IO-bound calls, so the pipeline runs inside asyncio's
default executor -- same pattern as imaging_agent.py -- keeping this a
proper async function without blocking the event loop.
"""

import asyncio

from backend.models.schemas import ClinicalFindings, EvidenceItem, ImagingFindings, RAGEvidence
from backend.rag.reranker import rerank
from backend.rag.retriever import retrieve

RETRIEVE_TOP_K = 10
RERANK_TOP_K = 5

# If every reranked result scores below this, the first query is
# treated as having failed to find relevant evidence and a broadened
# fallback query is tried once before giving up.
LOW_SCORE_THRESHOLD = 0.3

# Fallback query strategy: broaden to just the top N symptoms, dropping
# history/imaging detail that may have over-specified the first query.
FALLBACK_MAX_SYMPTOMS = 2


def _build_query(findings: ClinicalFindings, imaging: ImagingFindings | None) -> str:
    """Join symptoms + history + imaging findings into a natural clinical query."""
    parts = []
    if findings.symptoms:
        parts.append(", ".join(findings.symptoms))
    if findings.history:
        parts.append("history of " + ", ".join(findings.history))
    if imaging is not None and imaging.findings:
        parts.append("imaging findings: " + ", ".join(imaging.findings))
    return "; ".join(parts)


def _build_fallback_query(findings: ClinicalFindings) -> str:
    """Broaden to just the top symptoms, dropping history/imaging terms."""
    return ", ".join(findings.symptoms[:FALLBACK_MAX_SYMPTOMS])


def _run_pipeline(query: str) -> list[dict]:
    candidates = retrieve(query, top_k=RETRIEVE_TOP_K)
    if not candidates:
        return []
    return rerank(query, candidates, top_k=RERANK_TOP_K)


def _all_below_threshold(results: list[dict]) -> bool:
    return not results or all(r["score"] < LOW_SCORE_THRESHOLD for r in results)


async def run_rag(
    findings: ClinicalFindings,
    imaging: ImagingFindings | None,
    query_override: str | None = None,
) -> RAGEvidence:
    """query_override lets a caller (e.g. optimizer_agent.py, re-running
    retrieval with a Claude-refined query) skip the normal
    findings/imaging-derived query and supply one directly. When given,
    the low-score fallback below still applies -- a refined query that
    still scores poorly falls back to just the top symptoms, same as
    the default path.
    """
    query = query_override or _build_query(findings, imaging)
    if not query:
        return RAGEvidence(evidence=[])

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run_pipeline, query)

    if _all_below_threshold(results):
        fallback_query = _build_fallback_query(findings)
        if fallback_query and fallback_query != query:
            fallback_results = await loop.run_in_executor(None, _run_pipeline, fallback_query)
            if fallback_results:
                results = fallback_results

    evidence = [
        EvidenceItem(text=r["text"], source=r["source"], score=r["score"]) for r in results
    ]
    return RAGEvidence(evidence=evidence)
