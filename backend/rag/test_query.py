"""Manual retrieval smoke test (not a pytest test).

Embeds a fixed query and prints the top-5 Qdrant hits with scores and
metadata, so a human can eyeball whether retrieval looks relevant.
Requires the corpus to already be ingested (see ingest.py / `python -m
backend.rag.ingest`).

Run with:
    python -m backend.rag.test_query
"""

from backend.rag.embeddings import embed_text
from backend.rag.ingest import COLLECTION_NAME, get_qdrant_client

QUERY = "headache dizziness MRI lesion"
TOP_K = 5


def run_query(query: str = QUERY, top_k: int = TOP_K):
    client = get_qdrant_client()
    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return results


def main():
    print(f"Query: {QUERY!r}\n")
    results = run_query()

    if not results:
        print("No results. Have you run `python -m backend.rag.ingest` yet?")
        return

    for rank, hit in enumerate(results, start=1):
        payload = hit.payload or {}
        print(f"#{rank}  score={hit.score:.4f}")
        print(f"    doc_id: {payload.get('doc_id')}")
        print(f"    source: {payload.get('source')}")
        print(f"    page: {payload.get('page')}  chunk_length: {payload.get('chunk_length')}")
        print(f"    text: {payload.get('text', '')[:200]}...")
        print()


if __name__ == "__main__":
    main()
