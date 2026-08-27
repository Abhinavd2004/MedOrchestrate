"""Vector retrieval against the "medorchestrate_kb" Qdrant collection.

Reuses the Qdrant client factory and collection name from ingest.py
(single source of truth for how we connect to Qdrant / what the
collection is called) rather than re-implementing connection setup.
"""

from qdrant_client import QdrantClient

from backend.rag.embeddings import embed_text
from backend.rag.ingest import COLLECTION_NAME, get_qdrant_client

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Lazily create and cache a single Qdrant client for this process.

    Cached (rather than constructed fresh per call) because the local
    on-disk Qdrant fallback (see ingest.py) takes an exclusive file
    lock -- opening a second client against the same path while one is
    still open would fail. A single reused client also avoids the
    overhead of reconnecting on every retrieve() call.
    """
    global _client
    if _client is None:
        _client = get_qdrant_client()
    return _client


def retrieve(query: str, top_k: int = 10) -> list[dict]:
    """Embed `query` and return the top_k nearest chunks from Qdrant.

    Each result dict carries the chunk's payload fields (text, source,
    doc_id, page, chunk_length) plus a "score" -- the raw cosine
    similarity from Qdrant, later replaced by the reranker's
    relevance score.
    """
    client = _get_client()
    query_vector = embed_text(query)

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "text": hit.payload.get("text"),
            "source": hit.payload.get("source"),
            "doc_id": hit.payload.get("doc_id"),
            "page": hit.payload.get("page"),
            "chunk_length": hit.payload.get("chunk_length"),
            "score": hit.score,
        }
        for hit in hits
    ]
