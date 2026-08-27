"""Medical Imaging Agent: extracts ImagingFindings from an uploaded image.

IMPORTANT — scope and limitations:
This agent is a ZERO-SHOT CLASSIFIER built on BiomedCLIP, a pretrained
biomedical vision-language embedding model. It is bounded to a small,
hand-picked, prototype-scoped label set (see imaging_labels.py) — it is
NOT a comprehensive radiology ontology and NOT a clinically validated
diagnostic imaging model. Its output is presented to clinicians as
decision support only, never as a diagnosis. No LLM API is called in
this file; BiomedCLIP is a fixed, non-fine-tuned embedding model used
purely for image/text similarity scoring.

Named "Medical Imaging Agent" (not "MRI Agent") because MRI is only the
modality demoed first — the same zero-shot approach extends to CT,
X-ray, etc. by adding labels to imaging_labels.py, no code changes here.
"""

import asyncio
from pathlib import Path

import open_clip
import torch
from PIL import Image, UnidentifiedImageError

from backend.agents.imaging_labels import FINDING_LABELS, MODALITY_LABELS
from backend.models.schemas import ImagingFindings

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# File extensions this agent will attempt to open. PNG/JPEG are decoded
# directly by Pillow. DICOM (.dcm/.dicom) is only "if-convertible": we
# attempt to open it the same way, and if the runtime's Pillow build
# can't decode it, that surfaces as a clear UnsupportedImageFormatError
# rather than a crash (this prototype does not bundle a DICOM codec).
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}

# Zero-shot softmax score threshold for including a finding label in the
# result. Chosen empirically as a low bar so mildly-plausible findings
# surface for clinician review rather than being silently dropped; the
# single top-scoring label is always included regardless (see
# analyze_image), so `findings` is never empty.
FINDING_SCORE_THRESHOLD = 0.15

# CLIP-style logit scale applied to cosine similarities before softmax,
# matching BiomedCLIP's training configuration.
LOGIT_SCALE = 100.0


class ImagingAgentError(RuntimeError):
    """Base class for imaging agent errors."""


class ImageNotFoundError(ImagingAgentError):
    """Raised when the given image_path does not point to an existing file."""


class UnsupportedImageFormatError(ImagingAgentError):
    """Raised when the file extension or content is not a supported image."""


_model = None
_preprocess = None
_tokenizer = None


def _get_model_bundle():
    """Lazily load and cache the BiomedCLIP model, preprocessor, and tokenizer.

    Loaded once on first use (not at import time) so importing this
    module — e.g. for /health or unrelated tests — never triggers a
    network fetch or the multi-second model load. Cached in module
    globals thereafter so repeated calls to analyze_image reuse it.
    """
    global _model, _preprocess, _tokenizer
    if _model is None:
        model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
        tokenizer = open_clip.get_tokenizer(MODEL_ID)
        model.eval()
        _model, _preprocess, _tokenizer = model, preprocess, tokenizer
    return _model, _preprocess, _tokenizer


def _validate_image_path(image_path: str) -> Path:
    if not image_path:
        raise ValueError(
            "analyze_image requires a non-empty image_path; callers must "
            "confirm an image is present before calling this function."
        )

    path = Path(image_path)
    if not path.is_file():
        raise ImageNotFoundError(f"Image file not found: {image_path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedImageFormatError(
            f"Unsupported image file extension '{path.suffix}' for {image_path}. "
            f"Supported extensions: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return path


def _zero_shot_scores(
    model, tokenizer, image_features: torch.Tensor, labels: list[str]
) -> list[tuple[str, float]]:
    """Cosine-similarity zero-shot scores between the image and each label, softmaxed."""
    text_tokens = tokenizer(labels)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = (LOGIT_SCALE * image_features @ text_features.T).softmax(dim=-1)
    scores = logits.squeeze(0).tolist()

    return sorted(zip(labels, scores), key=lambda pair: pair[1], reverse=True)


def _run_zero_shot_classification(
    path: Path, modality_hint: str | None
) -> tuple[str, list[str], float]:
    """Synchronous, CPU/GPU-bound BiomedCLIP inference. Run via an executor."""
    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormatError(
            f"Could not decode image content at {path}: {exc}"
        ) from exc

    model, preprocess, tokenizer = _get_model_bundle()
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if modality_hint is None:
            modality_scores = _zero_shot_scores(model, tokenizer, image_features, MODALITY_LABELS)
            modality = modality_scores[0][0]
        else:
            modality = modality_hint

        finding_scores = _zero_shot_scores(model, tokenizer, image_features, FINDING_LABELS)

    top_label, top_score = finding_scores[0]
    findings = [label for label, score in finding_scores if score > FINDING_SCORE_THRESHOLD]
    if not findings:
        findings = [top_label]

    return modality, findings, float(top_score)


async def analyze_image(image_path: str, modality_hint: str | None = None) -> ImagingFindings:
    """Classify an image into ImagingFindings via BiomedCLIP zero-shot scoring.

    Only call this when image_path is actually present — CaseInput.image_path
    is optional, and callers upstream (e.g. the orchestrator) are responsible
    for skipping this agent entirely when no image was provided, rather than
    passing an empty/None path in here.
    """
    path = _validate_image_path(image_path)

    loop = asyncio.get_running_loop()
    modality, findings, confidence = await loop.run_in_executor(
        None, _run_zero_shot_classification, path, modality_hint
    )

    return ImagingFindings(modality=modality, findings=findings, confidence=confidence)
