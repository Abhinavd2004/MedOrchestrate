"""Fusion agent: combines ClinicalFindings, optional ImagingFindings,
RAGEvidence, and LiveEvidence into a ranked differential diagnosis
(FusionResult), via a single Claude reasoning call.

Any subset of inputs may be present -- imaging can be None, and
rag.evidence / live.sources can both be empty lists. The prompt and
grounding logic below are written to degrade gracefully in every case,
never assuming all four inputs are populated.

Evidence grounding: every piece of RAG/live evidence handed to the
model is tagged with a bracketed ID ([RAG1], [LIVE2], ...). The model
is instructed to cite only those IDs in supporting_evidence, and the
response is post-validated: any supporting_evidence string that doesn't
start with a real, known ID is dropped (logged, not silently trusted).
This is how "don't invent citations" is enforced beyond just asking
nicely in the prompt.

overall_confidence: this is a RAW fusion-model confidence signal only
(how confident the fusion reasoning itself is, given what evidence it
had). It is NOT the final clinical confidence score -- Phase 6's
backend/services/confidence.py is responsible for blending this with
other signals (e.g. imaging confidence, evidence quality/quantity) via
a weighted formula (w1/w2/w3). Nothing here should be read as a final
answer.
"""

import json
import logging
import re

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from backend.models.schemas import ClinicalFindings, FusionResult, ImagingFindings, LiveEvidence, RAGEvidence

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a clinical evidence-fusion reasoning system. You are given \
structured clinical findings, optional imaging findings, offline \
knowledge-base evidence, and live evidence (from PubMed/guideline \
sources), and must synthesize them into a ranked differential \
diagnosis. This output is CLINICAL DECISION SUPPORT ONLY, presented to \
a clinician for review -- it is never a diagnosis.

Each piece of RAG/live evidence is labeled with a bracketed ID like \
[RAG1] or [LIVE2]. When you cite evidence in supporting_evidence, you \
MUST use one of exactly those IDs, at the very start of the string \
(e.g. "[RAG1] white matter hyperintensities are associated with..."). \
NEVER invent a citation ID that was not given to you, and NEVER cite \
evidence that isn't actually present in the input -- every \
supporting_evidence string must trace back to something in the given \
RAG or live evidence. If no RAG or live evidence was provided (or none \
of it is relevant to a candidate diagnosis), supporting_evidence for \
that diagnosis must be an empty list -- reason from clinical/imaging \
findings alone in that case, and reflect the lack of evidence backing \
with a lower confidence score rather than fabricating support.

Explicitly compare the different evidence sources against each other. \
Whenever imaging findings, offline evidence, and live evidence point in \
different directions (e.g. imaging suggests X but live evidence \
emphasizes Y), add a short, human-readable string describing that \
disagreement to the "conflicts" list. If sources agree, or there is not \
enough evidence to compare, conflicts should be an empty list.

Return ONLY a single JSON object -- no prose, no markdown code fences, \
no explanation -- matching exactly this shape:

{
  "diagnoses": [
    {"name": "<diagnosis name>", "confidence": <number 0-1>, "supporting_evidence": ["[RAG1] ...", "[LIVE2] ..."], "rank": 1},
    ...
  ],
  "overall_confidence": <number 0-1>,
  "conflicts": ["<short description of a disagreement between sources>", ...]
}

Rank diagnoses from most to least likely (rank 1 = most likely). \
"overall_confidence" is your own raw confidence in this differential \
given the available evidence -- a separate downstream step combines it \
with other signals into a final score, so just be honest about how \
much the available evidence actually supports your reasoning."""

CORRECTION_MESSAGE = (
    "That response was not valid JSON matching the required schema, or "
    "cited evidence IDs that don't exist. Return valid JSON only -- no "
    "prose, no markdown code fences -- a single JSON object with exactly "
    "the keys diagnoses, overall_confidence, and conflicts, where every "
    "supporting_evidence string starts with one of the [RAGn]/[LIVEn] IDs "
    "actually given to you."
)

_TAG_RE = re.compile(r"^\[(RAG\d+|LIVE\d+)\]")


class FusionError(RuntimeError):
    """Raised when the model fails to produce a valid FusionResult JSON after retry."""


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


def _format_clinical(clinical: ClinicalFindings) -> str:
    parts = []
    if clinical.demographics:
        parts.append(f"Demographics: {clinical.demographics}")
    if clinical.symptoms:
        parts.append(f"Symptoms: {', '.join(clinical.symptoms)}")
    if clinical.history:
        parts.append(f"History: {', '.join(clinical.history)}")
    if clinical.labs:
        parts.append(f"Labs: {clinical.labs}")
    if clinical.imaging_text_findings:
        parts.append(
            f"Imaging findings mentioned in text/report: {', '.join(clinical.imaging_text_findings)}"
        )
    return "\n".join(parts) if parts else "(no clinical findings extracted)"


def _format_imaging(imaging: ImagingFindings | None) -> str:
    if imaging is None:
        return "(no imaging performed/provided for this case)"
    return (
        f"Modality: {imaging.modality}\n"
        f"Findings: {', '.join(imaging.findings)}\n"
        f"Imaging model confidence: {imaging.confidence:.2f}"
    )


def _format_rag_evidence(rag: RAGEvidence) -> tuple[str, set[str]]:
    ids: set[str] = set()
    lines = []
    for i, item in enumerate(rag.evidence, start=1):
        eid = f"RAG{i}"
        ids.add(eid)
        lines.append(f"[{eid}] (source: {item.source}, score: {item.score:.2f}) {item.text}")
    block = "\n".join(lines) if lines else "(no offline knowledge-base evidence retrieved)"
    return block, ids


def _format_live_evidence(live: LiveEvidence) -> tuple[str, set[str]]:
    ids: set[str] = set()
    lines = []
    for i, item in enumerate(live.sources, start=1):
        eid = f"LIVE{i}"
        ids.add(eid)
        pub = f", published {item.publication_date}" if item.publication_date else ""
        lines.append(
            f"[{eid}] (source: {item.source}, evidence_level: {item.evidence_level}{pub}) "
            f"{item.title}: {item.summary}"
        )
    block = "\n".join(lines) if lines else "(no live evidence retrieved)"
    return block, ids


def _is_completely_empty(
    clinical: ClinicalFindings, imaging: ImagingFindings | None, rag: RAGEvidence, live: LiveEvidence
) -> bool:
    clinical_empty = (
        not clinical.demographics
        and not clinical.symptoms
        and not clinical.history
        and not clinical.labs
        and not clinical.imaging_text_findings
    )
    return clinical_empty and imaging is None and not rag.evidence and not live.sources


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _validate_diagnosis(entry, valid_ids: set[str]) -> dict:
    if not isinstance(entry, dict):
        raise ValueError("each diagnosis must be a JSON object")

    name = entry.get("name")
    confidence = entry.get("confidence")
    supporting_evidence = entry.get("supporting_evidence")
    rank = entry.get("rank")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("diagnosis missing a valid 'name'")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (
        0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("diagnosis 'confidence' must be a number in [0, 1]")
    if not isinstance(supporting_evidence, list) or not all(
        isinstance(s, str) for s in supporting_evidence
    ):
        raise ValueError("diagnosis 'supporting_evidence' must be a list of strings")
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError("diagnosis 'rank' must be an int")

    grounded = []
    for citation in supporting_evidence:
        match = _TAG_RE.match(citation.strip())
        if match and match.group(1) in valid_ids:
            grounded.append(citation.strip())
        else:
            logger.warning(
                "Dropping ungrounded supporting_evidence citation for %r: %r", name, citation
            )

    return {
        "name": name.strip(),
        "confidence": float(confidence),
        "supporting_evidence": grounded,
        "rank": rank,
    }


def _parse_response(text: str, valid_ids: set[str]) -> FusionResult:
    payload = json.loads(_strip_code_fences(text))
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")

    diagnoses_raw = payload.get("diagnoses")
    if not isinstance(diagnoses_raw, list):
        raise ValueError("'diagnoses' must be a list")
    diagnoses = [_validate_diagnosis(d, valid_ids) for d in diagnoses_raw]

    conflicts = payload.get("conflicts")
    if not isinstance(conflicts, list) or not all(isinstance(c, str) for c in conflicts):
        raise ValueError("'conflicts' must be a list of strings")

    overall_confidence = payload.get("overall_confidence")
    if (
        not isinstance(overall_confidence, (int, float))
        or isinstance(overall_confidence, bool)
        or not (0.0 <= float(overall_confidence) <= 1.0)
    ):
        raise ValueError("'overall_confidence' must be a number in [0, 1]")

    # NOTE: overall_confidence here is a raw fusion-model signal, not a
    # final score -- see module docstring. Phase 6's confidence.py owns
    # blending this with other signals via the w1/w2/w3 formula.
    return FusionResult.model_validate(
        {
            "diagnoses": diagnoses,
            "overall_confidence": float(overall_confidence),
            "conflicts": conflicts,
        }
    )


async def _call_model(client: AsyncAnthropic, messages: list[dict]) -> str:
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


async def fuse(
    clinical: ClinicalFindings,
    imaging: ImagingFindings | None,
    rag: RAGEvidence,
    live: LiveEvidence,
) -> FusionResult:
    if _is_completely_empty(clinical, imaging, rag, live):
        return FusionResult(diagnoses=[], overall_confidence=0.0, conflicts=[])

    rag_block, rag_ids = _format_rag_evidence(rag)
    live_block, live_ids = _format_live_evidence(live)
    valid_ids = rag_ids | live_ids

    user_message = (
        f"Clinical findings:\n{_format_clinical(clinical)}\n\n"
        f"Imaging findings:\n{_format_imaging(imaging)}\n\n"
        f"Offline knowledge-base evidence:\n{rag_block}\n\n"
        f"Live evidence:\n{live_block}"
    )

    client = _get_client()
    messages = [{"role": "user", "content": user_message}]

    raw_response = await _call_model(client, messages)
    try:
        return _parse_response(raw_response, valid_ids)
    except (json.JSONDecodeError, ValueError, ValidationError) as first_error:
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": CORRECTION_MESSAGE})
        retry_response = await _call_model(client, messages)
        try:
            return _parse_response(retry_response, valid_ids)
        except (json.JSONDecodeError, ValueError, ValidationError) as second_error:
            raise FusionError(
                f"Failed to produce a valid FusionResult JSON after retry: {second_error}"
            ) from second_error
