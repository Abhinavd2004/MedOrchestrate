"""Tests for backend.agents.imaging_agent.

The BiomedCLIP model load and inference are fully mocked via the
fake_model_bundle fixture below — no weights are downloaded and no real
inference runs. Fakes return real (small, hand-picked) torch tensors so
the actual cosine-similarity/softmax scoring logic in imaging_agent.py
is still exercised end to end, just against canned embeddings.
"""

import math
from pathlib import Path

import pytest
import torch

import backend.agents.imaging_agent as imaging_agent
from backend.agents.imaging_agent import (
    ImageNotFoundError,
    UnsupportedImageFormatError,
    analyze_image,
)
from backend.agents.imaging_labels import FINDING_LABELS, MODALITY_LABELS
from backend.models.schemas import ImagingFindings

DEMO_IMAGE_1 = Path("data/images/synthetic_demo_brain_mri_001.png")
DEMO_IMAGE_2 = Path("data/images/synthetic_demo_brain_mri_002.jpg")


class _RecordingTokenizer:
    """Fake tokenizer: passes label strings straight through, unchanged."""

    def __call__(self, labels):
        return list(labels)


class _RecordingModel:
    """Fake BiomedCLIP model.

    encode_image always returns a fixed canned image embedding.
    encode_text looks up each label in a canned {label: embedding} map
    and stacks the results, so the real cosine-similarity/softmax math
    in imaging_agent.py runs against deterministic, known vectors.
    """

    def __init__(self, image_embedding: torch.Tensor, text_embeddings: dict[str, torch.Tensor]):
        self.image_embedding = image_embedding
        self.text_embeddings = text_embeddings
        self.encode_text_calls: list[list[str]] = []

    def encode_image(self, image_input):
        return self.image_embedding

    def encode_text(self, labels):
        self.encode_text_calls.append(list(labels))
        return torch.stack([self.text_embeddings[label] for label in labels])


def _fake_preprocess(image):
    return torch.zeros(3, 224, 224)


def _make_bundle(image_embedding: torch.Tensor, text_embeddings: dict[str, torch.Tensor]):
    model = _RecordingModel(image_embedding, text_embeddings)
    return model, _fake_preprocess, _RecordingTokenizer()


def _uniform_text_embeddings(vector: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every modality + finding label maps to the same embedding vector."""
    return {label: vector.clone() for label in MODALITY_LABELS + FINDING_LABELS}


@pytest.fixture
def dominant_finding_bundle(monkeypatch):
    """One finding/modality is a near-perfect match; everything else is orthogonal."""
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))
    text_embeddings["MRI brain scan"] = torch.tensor([1.0, 0.0])
    text_embeddings["normal brain MRI, no acute findings"] = torch.tensor([1.0, 0.0])

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))
    return model


async def test_valid_demo_image_returns_valid_schema(dominant_finding_bundle):
    result = await analyze_image(str(DEMO_IMAGE_1))

    assert isinstance(result, ImagingFindings)
    assert result.modality == "MRI brain scan"
    assert result.findings == ["normal brain MRI, no acute findings"]
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.9  # near-perfect match should score high


async def test_valid_demo_image_jpg(dominant_finding_bundle):
    result = await analyze_image(str(DEMO_IMAGE_2))

    assert isinstance(result, ImagingFindings)
    assert result.findings  # never empty


async def test_modality_hint_skips_modality_classification(dominant_finding_bundle):
    result = await analyze_image(str(DEMO_IMAGE_1), modality_hint="CT scan")

    assert result.modality == "CT scan"
    # only the findings classification call should have happened
    assert dominant_finding_bundle.encode_text_calls == [FINDING_LABELS]


async def test_no_hint_classifies_modality_then_findings(dominant_finding_bundle):
    await analyze_image(str(DEMO_IMAGE_1))

    assert dominant_finding_bundle.encode_text_calls == [MODALITY_LABELS, FINDING_LABELS]


async def test_missing_file_raises_typed_error(monkeypatch):
    called = False

    def _should_not_load():
        nonlocal called
        called = True
        raise AssertionError("model should not load when the file is missing")

    monkeypatch.setattr(imaging_agent, "_get_model_bundle", _should_not_load)

    with pytest.raises(ImageNotFoundError):
        await analyze_image("data/images/does_not_exist.png")

    assert called is False


async def test_unsupported_extension_raises_typed_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        imaging_agent,
        "_get_model_bundle",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not an image")

    with pytest.raises(UnsupportedImageFormatError):
        await analyze_image(str(bad_file))


async def test_undecodable_content_with_allowed_extension_raises_typed_error(tmp_path):
    fake_png = tmp_path / "broken.png"
    fake_png.write_bytes(b"this is not real png data")

    with pytest.raises(UnsupportedImageFormatError):
        await analyze_image(str(fake_png))


async def test_findings_never_empty_when_all_scores_below_threshold(monkeypatch):
    # All labels equally (orthogonally) similar -> uniform softmax over 13
    # finding labels is ~1/13 ≈ 0.077, below FINDING_SCORE_THRESHOLD (0.15).
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.findings == [FINDING_LABELS[0]]
    expected_score = 1.0 / len(FINDING_LABELS)
    assert math.isclose(result.confidence, expected_score, rel_tol=1e-4)


async def test_multiple_findings_above_threshold_sorted_descending(monkeypatch):
    # Two labels close to the image vector (both should clear the 0.15
    # threshold), everything else far away (should not).
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([-1.0, 0.0]))
    text_embeddings["white matter hyperintensity"] = torch.tensor([1.0, 0.0])
    text_embeddings["mass lesion"] = torch.tensor([0.99, math.sqrt(1 - 0.99**2)])

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.findings[0] == "white matter hyperintensity"
    assert "mass lesion" in result.findings
    assert all(result.findings[i] != result.findings[i + 1] for i in range(len(result.findings) - 1))
    # descending order
    assert result.findings.index("white matter hyperintensity") < result.findings.index("mass lesion")


async def test_empty_image_path_raises_value_error():
    with pytest.raises(ValueError):
        await analyze_image("")
