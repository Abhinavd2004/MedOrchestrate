"""Clinical agent: extracts ClinicalFindings from case text via Claude.

Schema-constrained LLM extraction only — no custom NER model. The model
is instructed to return raw JSON matching ClinicalFindings; the response
is validated with Pydantic and, on failure, retried once with an
explicit correction message before raising.
"""

import json

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from backend.models.schemas import ClinicalFindings

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a clinical information extraction system.

Given a patient's free-text clinical note and/or medical report, extract \
structured findings and return ONLY a single JSON object — no prose, no \
explanation, no markdown code fences — matching exactly this shape:

{
  "demographics": {},
  "symptoms": [],
  "history": [],
  "labs": {},
  "imaging_text_findings": []
}

Field rules:
- "demographics": any age/sex/other demographic details mentioned in the \
text itself (not inferred from outside context). Empty object if none.
- "symptoms": each distinct symptom or complaint as a short string.
- "history": past medical history, conditions, or relevant background as \
short strings.
- "labs": lab test names mapped to their reported value (as a string, \
including units if given), e.g. {"hemoglobin": "12.1 g/dL"}.
- "imaging_text_findings": any imaging findings mentioned in the TEXT \
(e.g. from a prior radiology report) — not an actual image analysis.

Normalize clinical terminology and abbreviations to standard full terms \
(for example "HTN" -> "hypertension", "DM" -> "diabetes mellitus", "SOB" \
-> "shortness of breath", "CP" -> "chest pain"). Use empty objects/arrays \
for any field with no supporting information. Return valid JSON only."""

CORRECTION_MESSAGE = (
    "That response was not valid JSON matching the required schema. "
    "Return valid JSON only — no prose, no markdown code fences, no "
    "explanation — a single JSON object with exactly the keys "
    "demographics, symptoms, history, labs, imaging_text_findings."
)

_EMPTY_FINDINGS_KWARGS = {
    "demographics": {},
    "symptoms": [],
    "history": [],
    "labs": {},
    "imaging_text_findings": [],
}


class ClinicalExtractionError(RuntimeError):
    """Raised when the model fails to produce valid ClinicalFindings JSON."""


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


def _build_user_message(clinical_text: str | None, medical_report: str | None) -> str:
    parts = []
    if clinical_text:
        parts.append(f"Clinical text:\n{clinical_text}")
    if medical_report:
        parts.append(f"Medical report:\n{medical_report}")
    return "\n\n".join(parts)


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


def _parse_response(text: str) -> ClinicalFindings:
    payload = json.loads(_strip_code_fences(text))
    return ClinicalFindings.model_validate(payload)


async def _call_model(client: AsyncAnthropic, messages: list[dict]) -> str:
    response = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


async def extract_clinical_findings(
    clinical_text: str | None, medical_report: str | None
) -> ClinicalFindings:
    if not clinical_text and not medical_report:
        return ClinicalFindings(**_EMPTY_FINDINGS_KWARGS)

    client = _get_client()
    messages = [{"role": "user", "content": _build_user_message(clinical_text, medical_report)}]

    raw_response = await _call_model(client, messages)
    try:
        return _parse_response(raw_response)
    except (json.JSONDecodeError, ValidationError) as first_error:
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": CORRECTION_MESSAGE})
        retry_response = await _call_model(client, messages)
        try:
            return _parse_response(retry_response)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise ClinicalExtractionError(
                "Failed to extract valid ClinicalFindings JSON after retry: "
                f"{second_error}"
            ) from second_error
