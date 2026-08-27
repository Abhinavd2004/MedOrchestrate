"""Cross-encoder reranking of retrieved evidence for answer relevance.

Design choice: (a) a cross-encoder model (cross-encoder/ms-marco-MiniLM-
L-6-v2), not (b) an Anthropic Claude scoring call.

Why: reranking runs on every retrieval -- including the fallback retry
in rag_agent.py -- so it needs to be fast, free of per-call API
latency/cost, and available offline, consistent with the rest of this
RAG layer being a local, deterministic retrieval pipeline (BioBERT
embeddings + Qdrant). A cross-encoder jointly encodes (query, passage)
pairs, so it scores actual query-passage relevance rather than just
re-sorting by the same embedding similarity used for retrieval -- this
is a real second-stage relevance signal, not a no-op.

Caveat: MS MARCO cross-encoders are general-domain, not medically
fine-tuned. Swapping in a biomedical cross-encoder (once one is
selected and evaluated) is a natural future upgrade; the rerank()
interface would not need to change.
"""

import math

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazily load and cache the cross-encoder model at module level."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank `candidates` by cross-encoder relevance to `query`.

    Each candidate's "score" is overwritten with the cross-encoder's
    relevance score (its raw logit passed through a sigmoid, so the
    result is an interpretable value in (0, 1) rather than an
    unbounded logit) -- this is what callers compare against a
    relevance threshold. Candidates are re-sorted by this new score,
    not by whatever score they arrived with (e.g. cosine similarity
    from retrieval).
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    raw_scores = model.predict(pairs)

    reranked = [
        {**candidate, "score": _sigmoid(float(raw_score))}
        for candidate, raw_score in zip(candidates, raw_scores)
    ]
    reranked.sort(key=lambda c: c["score"], reverse=True)

    return reranked[:top_k]
