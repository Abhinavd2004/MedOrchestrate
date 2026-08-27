"""Tests for backend.rag.embeddings.

The sentence-transformers model load is mocked so this suite runs
without downloading real weights or network access. The real 768-dim
output was separately confirmed by actually loading
pritamdeka/S-BioBert-snli-multinli-stsb once and checking
model.get_embedding_dimension() == 768 (see embeddings.py docstring).
"""

from unittest.mock import MagicMock

import numpy as np

import backend.rag.embeddings as embeddings_module
from backend.rag.embeddings import EMBEDDING_DIM, embed_text


def test_embed_text_returns_list_of_floats_with_documented_dim(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    monkeypatch.setattr(embeddings_module, "_get_model", lambda: fake_model)

    result = embed_text("headache and dizziness")

    assert isinstance(result, list)
    assert len(result) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in result)
    fake_model.encode.assert_called_once()


def test_model_loaded_once_and_cached(monkeypatch):
    monkeypatch.setattr(embeddings_module, "_model", None)
    load_calls = []

    class _FakeModel:
        def encode(self, text, convert_to_numpy=True):
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def _fake_constructor(model_name):
        load_calls.append(model_name)
        return _FakeModel()

    monkeypatch.setattr(embeddings_module, "SentenceTransformer", _fake_constructor)

    embed_text("first call")
    embed_text("second call")

    assert len(load_calls) == 1
