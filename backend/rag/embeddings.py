"""BioBERT sentence embedding utilities.

Model: pritamdeka/S-BioBert-snli-multinli-stsb
  - BioBERT (BERT-base architecture, hidden size 768) fine-tuned for
    sentence similarity on SNLI/MultiNLI/STSB, wrapped as a
    sentence-transformers model (mean-pooled, no separate projection
    head) -- output embedding dimension is 768. Confirmed at runtime
    via model.get_sentence_embedding_dimension() in
    backend/tests/test_embeddings.py.
  - Public on huggingface.co: https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
EMBEDDING_DIM = 768

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load and cache the sentence-transformer model at module level.

    Loaded once on first use (not at import time), so importing this
    module doesn't force a model download/load until embeddings are
    actually needed. Cached thereafter so repeated calls reuse it.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single string into a 768-dim BioBERT sentence embedding."""
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
