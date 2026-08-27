"""Optimizer agent: when the confidence from Phase 5's fusion is below
THRESHOLD, iteratively refines the retrieval query and re-runs
RAG + live evidence + fusion, up to MAX_ITERATIONS times, before
handing off to human-in-the-loop (HITL) review.

Architecture note: optimize() receives only `initial_fusion` and
`initial_confidence`, not the original RAGEvidence/LiveEvidence objects
that produced them (see the function signature required by this
phase's spec) -- so the very first refinement decision reasons from
the fusion result alone (its conflicts and how well each diagnosis was
evidence-grounded). From iteration 2 onward, the optimizer has its own
freshly-fetched rag/live results from the previous iteration and uses
those too, for a richer signal.

Caller responsibility: the loop-entry check in optimize() (stop
immediately if initial_confidence >= THRESHOLD) is the correctness
guarantee, but the caller (the future orchestrator) should ALSO check
confidence before invoking optimize() at all -- calling this function
only to have it immediately return is a wasted call/setup on the hot
path once an orchestrator exists.
"""

import json
import logging

from anthropic import AsyncAnthropic

from backend.agents.evidence_agent import gather_live_evidence
from backend.agents.fusion_agent import fuse
from backend.agents.rag_agent import _build_query as _build_base_query
from backend.agents.rag_agent import run_rag
from backend.models.schemas import (
    CaseInput,
    ClinicalFindings,
    FusionResult,
    ImagingFindings,
    LiveEvidence,
    RAGEvidence,
)
from backend.services.confidence import THRESHOLD, compute_confidence

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 512

# Hard cap. The for-loop below is bounded by this constant -- there is
# no code path (including error handling) that can iterate a 4th time.
MAX_ITERATIONS = 3

VALID_STRATEGIES = ("expand", "narrow", "paraphrase")

STRATEGY_SYSTEM_PROMPT = """You are a query-refinement assistant for a clinical evidence retrieval \
pipeline. Given the current search query, the clinical findings, \
optional imaging findings, and the most recent differential-diagnosis \
fusion result (including any flagged conflicts and how well each \
candidate diagnosis was grounded in evidence), choose ONE refinement \
strategy and produce a new query.

Strategies:
- "expand": the query is too narrow/specific and evidence retrieval \
  came up thin (e.g. diagnoses have little or no supporting_evidence, \
  or few evidence items were found) -- add related clinical terms \
  (synonyms, broader category terms, associated conditions) to surface \
  more evidence.
- "narrow": the query is too broad/ambiguous and pulled in irrelevant \
  or conflicting evidence (e.g. multiple conflicts were flagged) -- \
  drop vague or overly general terms and focus on the most specific or \
  distinctive symptoms/findings.
- "paraphrase": the query's scope looks about right, but its wording \
  may not match how evidence sources are phrased -- keep roughly the \
  same scope but rephrase using different, still-accurate clinical \
  terminology.

Return ONLY a single JSON object -- no prose, no markdown code fences, \
no explanation -- matching exactly this shape:

{"strategy": "expand" | "narrow" | "paraphrase", "query": "<new query string>", "rationale": "<one short sentence explaining the choice>"}"""

STRATEGY_CORRECTION_MESSAGE = (
    "That response was not valid JSON matching the required schema. "
    "Return valid JSON only -- no prose, no markdown code fences -- a "
    "single JSON object with exactly the keys strategy (one of "
    '"expand", "narrow", "paraphrase"), query, and rationale.'
)


class RefinementError(RuntimeError):
    """Raised when the model fails to produce a valid refinement JSON after retry."""


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


def _format_fusion_for_prompt(fusion: FusionResult) -> str:
    lines = []
    for d in fusion.diagnoses:
        name = d.get("name", "?")
        confidence = d.get("confidence", "?")
        n_evidence = len(d.get("supporting_evidence", []) or [])
        lines.append(f"- {name} (confidence={confidence}, grounded evidence citations={n_evidence})")
    diagnoses_block = "\n".join(lines) if lines else "(no diagnoses produced)"

    conflicts_block = (
        "\n".join(f"- {c}" for c in fusion.conflicts) if fusion.conflicts else "(none flagged)"
    )

    return f"Diagnoses from the most recent fusion result:\n{diagnoses_block}\n\nConflicts flagged:\n{conflicts_block}"


def _format_evidence_summary(rag: RAGEvidence | None, live: LiveEvidence | None) -> str:
    if rag is None and live is None:
        return "(no retrieval has run yet in this optimization session -- this is the first refinement)"

    rag_count = len(rag.evidence) if rag else 0
    rag_top = f"{max((e.score for e in rag.evidence), default=0.0):.2f}" if rag and rag.evidence else "n/a"
    live_count = len(live.sources) if live else 0

    return (
        f"Most recent retrieval: {rag_count} offline knowledge-base evidence item(s) "
        f"(top score {rag_top}), {live_count} live evidence source(s)."
    )


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


def _parse_refinement(text: str) -> dict:
    payload = json.loads(_strip_code_fences(text))
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")

    strategy = payload.get("strategy")
    query = payload.get("query")
    rationale = payload.get("rationale", "")

    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"'strategy' must be one of {VALID_STRATEGIES}, got {strategy!r}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")

    return {"strategy": strategy, "query": query.strip(), "rationale": rationale}


async def _call_model(client: AsyncAnthropic, messages: list[dict]) -> str:
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=STRATEGY_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


async def _select_refinement_strategy(
    clinical: ClinicalFindings,
    imaging: ImagingFindings | None,
    current_query: str,
    fusion: FusionResult,
    rag: RAGEvidence | None,
    live: LiveEvidence | None,
) -> dict:
    """Ask Claude to pick a refinement strategy and produce a new query.

    Returns {"strategy": ..., "query": ..., "rationale": ...}. Raises
    RefinementError if the model can't produce valid JSON after one
    retry -- the caller (optimize()) catches this per-iteration.
    """
    user_message = (
        f"Current query: {current_query!r}\n\n"
        f"Clinical symptoms: {', '.join(clinical.symptoms) or '(none extracted)'}\n"
        f"Clinical history: {', '.join(clinical.history) or '(none extracted)'}\n"
        f"Imaging: {', '.join(imaging.findings) if imaging else '(no imaging)'}\n\n"
        f"{_format_evidence_summary(rag, live)}\n\n"
        f"{_format_fusion_for_prompt(fusion)}"
    )

    client = _get_client()
    messages = [{"role": "user", "content": user_message}]

    raw_response = await _call_model(client, messages)
    try:
        return _parse_refinement(raw_response)
    except (json.JSONDecodeError, ValueError) as first_error:
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": STRATEGY_CORRECTION_MESSAGE})
        retry_response = await _call_model(client, messages)
        try:
            return _parse_refinement(retry_response)
        except (json.JSONDecodeError, ValueError) as second_error:
            raise RefinementError(
                f"Failed to produce a valid refinement JSON after retry: {second_error}"
            ) from second_error


async def optimize(
    case: CaseInput,
    clinical: ClinicalFindings,
    imaging: ImagingFindings | None,
    initial_fusion: FusionResult,
    initial_confidence: float,
) -> dict:
    # Step 1: stop immediately if Phase 5's fusion already cleared the
    # bar -- this must hold regardless of what the caller already
    # checked (see module docstring).
    if initial_confidence >= THRESHOLD:
        return {
            "final_fusion": initial_fusion,
            "final_confidence": initial_confidence,
            "iterations": 0,
            "review_required": False,
            "iteration_log": [],
        }

    iteration_log: list[dict] = []
    best_fusion = initial_fusion
    best_confidence = initial_confidence

    current_query = _build_base_query(clinical, imaging)
    current_fusion = initial_fusion
    current_rag: RAGEvidence | None = None
    current_live: LiveEvidence | None = None
    iterations_run = 0

    for i in range(1, MAX_ITERATIONS + 1):
        iterations_run = i
        strategy = None
        refined_query = None
        try:
            refinement = await _select_refinement_strategy(
                clinical, imaging, current_query, current_fusion, current_rag, current_live
            )
            strategy = refinement["strategy"]
            refined_query = refinement["query"]

            new_rag = await run_rag(clinical, imaging, query_override=refined_query)
            new_live = await gather_live_evidence(refined_query)
            new_fusion = await fuse(clinical, imaging, new_rag, new_live)
            new_confidence = compute_confidence(new_rag, new_live, new_fusion)

            iteration_log.append(
                {"iteration": i, "strategy": strategy, "query": refined_query, "confidence": new_confidence}
            )

            current_query = refined_query
            current_fusion = new_fusion
            current_rag = new_rag
            current_live = new_live

            if new_confidence > best_confidence:
                best_confidence = new_confidence
                best_fusion = new_fusion

            if new_confidence >= THRESHOLD:
                return {
                    "final_fusion": new_fusion,
                    "final_confidence": new_confidence,
                    "iterations": i,
                    "review_required": False,
                    "iteration_log": iteration_log,
                }
        except Exception as exc:  # noqa: BLE001 -- any failure in this iteration's steps
            # (strategy selection via RefinementError/ValidationError,
            # retrieval, fusion, or confidence scoring) must not crash
            # the whole optimizer -- log it, record it, and move on to
            # the next iteration attempt (still bounded by MAX_ITERATIONS).
            logger.warning("Optimizer iteration %d failed: %s", i, exc)
            iteration_log.append(
                {
                    "iteration": i,
                    "strategy": strategy,
                    "query": refined_query,
                    "confidence": None,
                    "error": str(exc),
                }
            )
            continue

    return {
        "final_fusion": best_fusion,
        "final_confidence": best_confidence,
        "iterations": iterations_run,
        "review_required": True,
        "iteration_log": iteration_log,
    }
