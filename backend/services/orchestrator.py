"""Orchestrator: wires the Phase 1-7 agent functions together into a
single CrewAI-driven pipeline, exposed via the single entry point
run_case(case: CaseInput) -> FinalReport.

Design -- CrewAI as a thin orchestration layer, not a reasoning layer:

Every unit of actual work here (clinical extraction, imaging
classification, retrieval, live evidence, fusion, optimization) is
already implemented and independently tested in backend/agents/*.py and
backend/services/confidence.py. This module does not re-implement any
of that logic. Each CrewAI Agent below is given exactly one Tool, and
that Tool's _run() does nothing but call the corresponding real,
already-existing async function and store its result -- the Agent's
role/goal/backstory instruct it to call its tool once and return the
result verbatim, with no independent analysis.

Data does NOT flow between agents through CrewAI's LLM-mediated task
text/context. It flows through a shared, in-process `_CaseContext`
object that each Tool reads from and writes to directly. This is a
deliberate choice: routing structured, citation-grounded medical output
(e.g. fusion_agent's supporting_evidence, which must trace back exactly
to given evidence IDs) through a second LLM pass -- which is what
happens if you let a CrewAI agent "produce" the next task's structured
input from the previous task's text output -- risks silently corrupting
that grounding. CrewAI still genuinely orchestrates each step (a real
Claude call, per the Agent's own LLM, decides to invoke its tool); the
difference is that this module, not the LLM, is the one wiring the
*data* between steps, same as the rest of this codebase's control flow.

Conditional branching (imaging only if case.image_path is set;
optimizer only if confidence < THRESHOLD) is plain Python in run_case()
for the same reason it is everywhere else in this pipeline: CrewAI's
sequential Process has no clean way to skip a task conditionally, and
that branching is exactly the kind of decision this orchestrator -- not
an LLM -- should make deterministically.
"""

import asyncio
import time
from dataclasses import dataclass, field

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import PrivateAttr

from backend.agents.clinical_agent import extract_clinical_findings
from backend.agents.evidence_agent import gather_live_evidence
from backend.agents.fusion_agent import fuse
from backend.agents.imaging_agent import analyze_image
from backend.agents.optimizer_agent import optimize
from backend.agents.rag_agent import _build_query, run_rag
from backend.models.schemas import (
    CaseInput,
    ClinicalFindings,
    FinalReport,
    FusionResult,
    ImagingFindings,
    LiveEvidence,
    RAGEvidence,
)
from backend.services import storage
from backend.services.confidence import THRESHOLD, compute_confidence
from backend.services.logging import log_event

# litellm (CrewAI's LLM backend) expects a provider-prefixed model id --
# unlike the other agents in this codebase, which call the Anthropic SDK
# directly with the bare "claude-sonnet-4-6" string.
CREWAI_MODEL = "anthropic/claude-sonnet-4-6"

TOOL_PASSTHROUGH_EXPECTED_OUTPUT = (
    "The exact string returned by your tool call. Do not add commentary, "
    "analysis, or reformat it."
)

_crewai_llm: LLM | None = None


def _get_crewai_llm() -> LLM:
    global _crewai_llm
    if _crewai_llm is None:
        _crewai_llm = LLM(model=CREWAI_MODEL)
    return _crewai_llm


def _run_sync(coro):
    """Bridge an async Phase 1-7 call into CrewAI's synchronous tool
    execution. Safe here because CrewAI's (synchronous) Crew.kickoff()
    does not itself run inside an already-active event loop.
    """
    return asyncio.run(coro)


@dataclass
class _CaseContext:
    """Shared, in-process state each Tool reads from / writes to. Never
    serialized through CrewAI's LLM layer -- see module docstring."""

    case: CaseInput
    clinical: ClinicalFindings | None = None
    imaging: ImagingFindings | None = None
    rag: RAGEvidence | None = None
    live: LiveEvidence | None = None
    fusion: FusionResult | None = None
    confidence: float | None = None
    review_required: bool = False
    iterations: int = 0
    call_log: list[dict] = field(default_factory=list)


def _log_agent_call(
    context: _CaseContext,
    agent: str,
    iteration: int,
    status: str,
    latency_ms: float | None = None,
    confidence: float | None = None,
    error: str | None = None,
    **extra,
) -> None:
    """Logs one agent-call event to both stdout (services/logging.py's
    structured JSON lines) and SQLite (storage.log_agent_event) -- the
    former for real-time observability, the latter so agent call
    history survives a restart and can be queried per case.

    `confidence` defaults to context.confidence (the pipeline's current
    confidence at the time of the call) but can be overridden -- used
    by the optimizer's per-iteration logging below, where each
    iteration's own confidence is known but hasn't yet been written
    back to context.confidence (that only happens once optimize()
    returns).

    `**extra` (e.g. the optimizer's strategy/query per iteration) is
    included in the stdout log for richer observability, but not
    passed to storage.log_agent_event -- the agent_logs table's columns
    are fixed to the spec's schema (case_id, agent_name, iteration,
    status, latency, confidence).
    """
    resolved_confidence = confidence if confidence is not None else context.confidence
    entry = {
        "case_id": context.case.case_id,
        "agent_name": agent,
        "iteration": iteration,
        "status": status,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "confidence": resolved_confidence,
        **extra,
    }
    if error:
        entry["error"] = error
    context.call_log.append(entry)
    log_event("agent_call", **entry)
    storage.log_agent_event(
        case_id=context.case.case_id,
        agent_name=agent,
        iteration=iteration,
        status=status,
        latency=latency_ms,
        confidence=resolved_confidence,
    )


# --------------------------------------------------------------------------
# Tools -- each wraps exactly one Phase 1-7 function. All external calls
# (Claude, Qdrant, PubMed) reachable from these functions are caught here
# so a single failure degrades gracefully rather than crashing run_case().
# --------------------------------------------------------------------------


class ClinicalTool(BaseTool):
    name: str = "clinical_extraction"
    description: str = (
        "Extracts structured clinical findings from the case's clinical "
        "text and medical report. Takes no meaningful input -- reads the "
        "case directly from the shared orchestration context."
    )
    _context: _CaseContext = PrivateAttr()

    def __init__(self, context: _CaseContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def _run(self) -> str:
        ctx = self._context
        start = time.perf_counter()
        try:
            ctx.clinical = _run_sync(
                extract_clinical_findings(ctx.case.clinical_text, ctx.case.medical_report)
            )
            _log_agent_call(ctx, "clinical", 0, "success", (time.perf_counter() - start) * 1000)
            return f"clinical extraction succeeded: {len(ctx.clinical.symptoms)} symptom(s) found"
        except Exception as exc:  # Claude API failure, etc. -- degrade, don't crash
            ctx.clinical = ClinicalFindings(
                demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[]
            )
            _log_agent_call(
                ctx, "clinical", 0, "failure", (time.perf_counter() - start) * 1000, error=str(exc)
            )
            return f"clinical extraction failed, degraded to empty findings: {exc}"


class ImagingTool(BaseTool):
    name: str = "imaging_analysis"
    description: str = (
        "Analyzes the case's image via BiomedCLIP zero-shot classification "
        "into modality + findings. Reads the case from the shared context."
    )
    _context: _CaseContext = PrivateAttr()

    def __init__(self, context: _CaseContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def _run(self) -> str:
        ctx = self._context
        start = time.perf_counter()
        try:
            ctx.imaging = _run_sync(analyze_image(ctx.case.image_path))
            _log_agent_call(ctx, "imaging", 0, "success", (time.perf_counter() - start) * 1000)
            return f"imaging analysis succeeded: modality={ctx.imaging.modality}"
        except Exception as exc:  # missing/unsupported file, model load failure, etc.
            ctx.imaging = None
            _log_agent_call(
                ctx, "imaging", 0, "failure", (time.perf_counter() - start) * 1000, error=str(exc)
            )
            return f"imaging analysis failed, degraded to no imaging findings: {exc}"


class RAGTool(BaseTool):
    name: str = "rag_retrieval"
    description: str = (
        "Retrieves and reranks offline knowledge-base evidence for the "
        "case's clinical + imaging findings, from the shared context."
    )
    _context: _CaseContext = PrivateAttr()

    def __init__(self, context: _CaseContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def _run(self) -> str:
        ctx = self._context
        start = time.perf_counter()
        try:
            ctx.rag = _run_sync(run_rag(ctx.clinical, ctx.imaging))
            _log_agent_call(ctx, "rag", 0, "success", (time.perf_counter() - start) * 1000)
            return f"retrieved {len(ctx.rag.evidence)} offline evidence item(s)"
        except Exception as exc:  # Qdrant connection failure, etc.
            ctx.rag = RAGEvidence(evidence=[])
            _log_agent_call(
                ctx, "rag", 0, "failure", (time.perf_counter() - start) * 1000, error=str(exc)
            )
            return f"RAG retrieval failed, degraded to no offline evidence: {exc}"


class EvidenceTool(BaseTool):
    name: str = "live_evidence_gathering"
    description: str = (
        "Gathers live evidence (PubMed + MedlinePlus) for the case's "
        "clinical + imaging findings, from the shared context."
    )
    _context: _CaseContext = PrivateAttr()

    def __init__(self, context: _CaseContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def _run(self) -> str:
        ctx = self._context
        start = time.perf_counter()
        try:
            query = _build_query(ctx.clinical, ctx.imaging)
            ctx.live = _run_sync(gather_live_evidence(query)) if query else LiveEvidence(sources=[])
            _log_agent_call(ctx, "evidence", 0, "success", (time.perf_counter() - start) * 1000)
            return f"gathered {len(ctx.live.sources)} live evidence source(s)"
        except Exception as exc:  # PubMed/MedlinePlus down, etc. (gather_live_evidence is
            # already internally resilient, so this mainly guards against
            # something unexpected, e.g. a bug in query building)
            ctx.live = LiveEvidence(sources=[])
            _log_agent_call(
                ctx, "evidence", 0, "failure", (time.perf_counter() - start) * 1000, error=str(exc)
            )
            return f"live evidence gathering failed, degraded to no live evidence: {exc}"


class FusionTool(BaseTool):
    name: str = "evidence_fusion"
    description: str = (
        "Fuses clinical, imaging, RAG, and live evidence (from the shared "
        "context) into a ranked differential diagnosis, and computes the "
        "resulting confidence score."
    )
    _context: _CaseContext = PrivateAttr()

    def __init__(self, context: _CaseContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def _run(self) -> str:
        ctx = self._context
        start = time.perf_counter()
        try:
            ctx.fusion = _run_sync(fuse(ctx.clinical, ctx.imaging, ctx.rag, ctx.live))
            ctx.confidence = compute_confidence(ctx.rag, ctx.live, ctx.fusion)
            _log_agent_call(ctx, "fusion", 0, "success", (time.perf_counter() - start) * 1000)
            return (
                f"fusion produced {len(ctx.fusion.diagnoses)} diagnos(es), "
                f"confidence={ctx.confidence:.3f}"
            )
        except Exception as exc:  # Claude API failure, invalid JSON after retry, etc.
            ctx.fusion = FusionResult(
                diagnoses=[], overall_confidence=0.0, conflicts=[f"Fusion reasoning failed: {exc}"]
            )
            ctx.confidence = 0.0
            _log_agent_call(
                ctx, "fusion", 0, "failure", (time.perf_counter() - start) * 1000, error=str(exc)
            )
            return f"fusion failed, degraded to an empty differential: {exc}"


class OptimizerTool(BaseTool):
    name: str = "confidence_optimizer"
    description: str = (
        "When confidence is below threshold, iteratively refines the "
        "query and re-runs RAG + live evidence + fusion (up to 3 "
        "iterations), reading the current case state from the shared "
        "context."
    )
    _context: _CaseContext = PrivateAttr()

    def __init__(self, context: _CaseContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    def _run(self) -> str:
        ctx = self._context
        start = time.perf_counter()
        try:
            result = _run_sync(optimize(ctx.case, ctx.clinical, ctx.imaging, ctx.fusion, ctx.confidence))
            ctx.fusion = result["final_fusion"]
            ctx.confidence = result["final_confidence"]
            ctx.review_required = result["review_required"]
            ctx.iterations = result["iterations"]

            # optimize() already ran (and internally logged nothing of its
            # own -- it's a pure function), so surface its per-iteration
            # detail as additional log entries here, using the real
            # iteration numbers (1..3) rather than the outer
            # orchestrator's iteration=0 first-pass convention. No
            # per-iteration latency is available from optimize()'s
            # return value, so latency_ms is left None for these.
            for entry in result["iteration_log"]:
                _log_agent_call(
                    ctx,
                    "optimizer",
                    entry["iteration"],
                    "success" if entry.get("confidence") is not None else "failure",
                    confidence=entry.get("confidence"),
                    error=entry.get("error"),
                    strategy=entry.get("strategy"),
                    query=entry.get("query"),
                )

            _log_agent_call(ctx, "optimizer", 0, "success", (time.perf_counter() - start) * 1000)
            return (
                f"optimizer ran {result['iterations']} iteration(s), "
                f"final confidence={ctx.confidence:.3f}, review_required={ctx.review_required}"
            )
        except Exception as exc:  # optimize() is already internally exception-safe, so this
            # mainly guards against something unexpected upstream of it
            ctx.review_required = True
            _log_agent_call(
                ctx, "optimizer", 0, "failure", (time.perf_counter() - start) * 1000, error=str(exc)
            )
            return f"optimizer failed to run, forcing review_required=True: {exc}"


# --------------------------------------------------------------------------
# Agent role/goal/backstory -- deliberately generic and instruction-only.
# No business logic lives here; see module docstring.
# --------------------------------------------------------------------------

_AGENT_SPECS = {
    "clinical": (
        "Clinical Findings Agent",
        "Call your clinical_extraction tool exactly once and return its output verbatim.",
        "A thin wrapper agent. You do not analyze clinical text yourself -- your "
        "one tool already does that. Your only job is to invoke it and relay its result.",
    ),
    "imaging": (
        "Medical Imaging Agent",
        "Call your imaging_analysis tool exactly once and return its output verbatim.",
        "A thin wrapper agent. You do not interpret images yourself -- your one "
        "tool already does that. Your only job is to invoke it and relay its result.",
    ),
    "rag": (
        "Knowledge Base Retrieval Agent",
        "Call your rag_retrieval tool exactly once and return its output verbatim.",
        "A thin wrapper agent. You do not search the knowledge base yourself -- "
        "your one tool already does that. Your only job is to invoke it and relay its result.",
    ),
    "evidence": (
        "Live Evidence Agent",
        "Call your live_evidence_gathering tool exactly once and return its output verbatim.",
        "A thin wrapper agent. You do not query PubMed or guidelines yourself -- "
        "your one tool already does that. Your only job is to invoke it and relay its result.",
    ),
    "fusion": (
        "Evidence Fusion Agent",
        "Call your evidence_fusion tool exactly once and return its output verbatim.",
        "A thin wrapper agent. You do not reason about diagnoses yourself -- your "
        "one tool already does that. Your only job is to invoke it and relay its result.",
    ),
    "optimizer": (
        "Confidence Optimizer Agent",
        "Call your confidence_optimizer tool exactly once and return its output verbatim.",
        "A thin wrapper agent. You do not refine queries or re-run retrieval "
        "yourself -- your one tool already does that. Your only job is to invoke "
        "it and relay its result.",
    ),
}


def _build_agent(tool: BaseTool, spec_key: str) -> Agent:
    role, goal, backstory = _AGENT_SPECS[spec_key]
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[tool],
        llm=_get_crewai_llm(),
        allow_delegation=False,
        verbose=False,
    )


def _build_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        description="Call your tool now and return exactly what it returns.",
        expected_output=TOOL_PASSTHROUGH_EXPECTED_OUTPUT,
    )


def _execute(agent: Agent, task: Task) -> str:
    """Runs one Agent+Task as its own single-task Crew.

    This is the seam between CrewAI's LLM-driven layer and this
    module's Python-controlled branching -- see module docstring for
    why each step is its own tiny Crew rather than one big static one.

    Tests monkeypatch this function to call the task's tool directly,
    skipping the real Claude call CrewAI would otherwise make to
    "decide" to invoke its one tool. Since these tools ignore whatever
    the LLM passes them anyway (they read the shared _CaseContext, not
    their arguments), skipping that decision step in tests changes
    nothing about the business logic under test -- only whether a real
    Claude call happens.
    """
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


def _combine_evidence(rag: RAGEvidence | None, live: LiveEvidence | None) -> list[dict]:
    """Merge RAG + live evidence into FinalReport.evidence (list[dict] --
    intentionally untyped in the shared schema; see schemas.py). Each
    entry is tagged with "type" so a consumer can tell offline
    knowledge-base evidence apart from live PubMed/MedlinePlus evidence.
    """
    combined: list[dict] = []
    if rag:
        combined.extend({"type": "rag", **item.model_dump()} for item in rag.evidence)
    if live:
        combined.extend({"type": "live", **item.model_dump()} for item in live.sources)
    return combined


def run_case(case: CaseInput) -> FinalReport:
    context = _CaseContext(case=case)

    # Step 1: Clinical (always) + Imaging (only if image_path is set).
    # extract_clinical_findings already handles clinical_text=None,
    # medical_report=None gracefully (empty findings, no API call), so
    # running it unconditionally supports text-only, image-only, and
    # text+image cases uniformly -- imaging is the only conditional step.
    clinical_agent = _build_agent(ClinicalTool(context), "clinical")
    _execute(clinical_agent, _build_task(clinical_agent))

    if case.image_path:
        imaging_agent = _build_agent(ImagingTool(context), "imaging")
        _execute(imaging_agent, _build_task(imaging_agent))
    # else: context.imaging stays None -- handled uniformly downstream.

    # Step 2: the structured multimodal case at this point is exactly
    # (context.clinical, context.imaging) -- nothing further to build.

    # Step 3: RAG + Evidence.
    rag_agent_instance = _build_agent(RAGTool(context), "rag")
    _execute(rag_agent_instance, _build_task(rag_agent_instance))

    evidence_agent_instance = _build_agent(EvidenceTool(context), "evidence")
    _execute(evidence_agent_instance, _build_task(evidence_agent_instance))

    # Step 4: Fusion, then confidence (computed inside FusionTool, right
    # after fuse(), from exactly the rag/live/fusion triple just produced).
    fusion_agent_instance = _build_agent(FusionTool(context), "fusion")
    _execute(fusion_agent_instance, _build_task(fusion_agent_instance))

    # Step 5: optimizer only if confidence is below threshold.
    if context.confidence is not None and context.confidence < THRESHOLD:
        optimizer_agent_instance = _build_agent(OptimizerTool(context), "optimizer")
        _execute(optimizer_agent_instance, _build_task(optimizer_agent_instance))

    # Step 6: assemble FinalReport.
    fusion = context.fusion or FusionResult(diagnoses=[], overall_confidence=0.0, conflicts=[])
    confidence = context.confidence if context.confidence is not None else 0.0

    return FinalReport(
        case_id=case.case_id,
        diagnoses=fusion.diagnoses,
        confidence=confidence,
        evidence=_combine_evidence(context.rag, context.live),
        review_required=context.review_required,
        conflicts=fusion.conflicts,
        iteration_count=context.iterations,
    )
