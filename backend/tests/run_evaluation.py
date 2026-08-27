"""Evaluation harness for MedOrchestrate.

NOT a pytest test -- deliberately named without a test_ prefix so pytest
never auto-collects it (it does real, slow, non-deterministic work:
real RAG/Qdrant retrieval, real PubMed/MedlinePlus network calls, and
optionally real BiomedCLIP inference). Run it directly:

    python -m backend.tests.run_evaluation
    python -m backend.tests.run_evaluation --ablation   # stretch goal, see below

Runs every synthetic case in data/test_cases/ through the real
orchestrator.run_case() and reports the metrics the task asked for, to
stdout as a table and to a markdown + CSV report file.

Mocking: no ANTHROPIC_API_KEY is configured in this dev environment, so
Claude calls (clinical extraction, fusion reasoning, optimizer strategy
selection) and CrewAI's own Agent-decision layer are mocked with a
GENERIC, case-agnostic mock (see _install_mocks_for_case) -- not hand-
authored per case. Everything else is real: RAG retrieval (real local
Qdrant + BioBERT), live evidence (real PubMed/MedlinePlus network
calls), imaging (real BiomedCLIP, since the model is already cached
locally), the optimizer's control flow, and the confidence formula
(compute_confidence() ignores the mocked fusion's own overall_confidence
entirely -- it recomputes from real RAG scores, real live evidence
levels, and the mocked fusion's conflicts list -- so confidence still
varies meaningfully per case based on real retrieval quality, not an
arbitrary mocked number).

To run this against a real, fully live pipeline once a real
ANTHROPIC_API_KEY is configured: pass --no-mock-claude, which skips
_install_mocks_for_case() entirely and lets every agent call the real
Anthropic API through the real CrewAI Agent-decision layer.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import backend.agents.clinical_agent as clinical_agent
import backend.agents.fusion_agent as fusion_agent
import backend.agents.optimizer_agent as optimizer_agent
import backend.services.orchestrator as orchestrator
from backend.agents.clinical_agent import extract_clinical_findings
from backend.agents.evidence_agent import gather_live_evidence
from backend.agents.fusion_agent import fuse
from backend.agents.imaging_agent import analyze_image
from backend.agents.rag_agent import _build_query, run_rag
from backend.models.schemas import CaseInput, LiveEvidence, RAGEvidence
from backend.services.confidence import compute_confidence
from backend.services.orchestrator import run_case

TEST_CASES_DIR = Path("data/test_cases")
DEFAULT_MD_REPORT = Path("backend/tests/evaluation_report.md")
DEFAULT_CSV_REPORT = Path("backend/tests/evaluation_report.csv")


# --------------------------------------------------------------------------
# Generic, case-agnostic mocks for Claude + CrewAI's Agent-decision layer.
# --------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


def _fake_client(response_texts):
    """response_texts: a single JSON string (every call gets it), or a
    list that's cycled indefinitely (so an unpredictable number of calls
    -- e.g. across optimizer iterations plus an ablation run -- never
    exhausts it)."""
    client = MagicMock()
    if isinstance(response_texts, list):
        cycle = itertools.cycle(response_texts)

        async def _create(*args, **kwargs):
            return _FakeAnthropicResponse(next(cycle))

        client.messages.create = AsyncMock(side_effect=_create)
    else:
        client.messages.create = AsyncMock(return_value=_FakeAnthropicResponse(response_texts))
    return client


_SYMPTOM_VOCAB = [
    "headache", "dizziness", "vertigo", "fatigue", "nausea", "lightheadedness",
    "gait unsteadiness", "unsteadiness", "vision", "weakness", "confusion",
    "palpitations", "hyperpigmentation",
]


def _derive_terms(text: str | None) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return [term for term in _SYMPTOM_VOCAB if term in lowered]


def _clinical_mock_response(case: CaseInput) -> str:
    """Generic clinical-extraction mock: echoes case.history directly and
    scans clinical_text/medical_report against a small symptom vocabulary
    -- not real medical NLP, just a case-agnostic stand-in so every case
    gets a plausible-shaped ClinicalFindings without hand-authoring 17
    bespoke responses."""
    symptoms = _derive_terms(case.clinical_text) or (["nonspecific symptoms"] if case.clinical_text else [])
    imaging_text_findings = _derive_terms(case.medical_report)
    if case.medical_report and "no acute intracranial abnormality" in case.medical_report.lower():
        imaging_text_findings.append("no acute intracranial abnormality")
    return json.dumps(
        {
            "demographics": {},
            "symptoms": symptoms,
            "history": case.history,
            "labs": case.labs,
            "imaging_text_findings": imaging_text_findings,
        }
    )


def _fusion_mock_response(label: str) -> str:
    """Generic single-diagnosis fusion mock. Always attempts to cite
    [RAG1] -- fusion_agent's real grounding validator silently drops that
    citation if the case's real RAG evidence didn't actually produce an
    item with that ID, so evidence_coverage still reflects genuine
    retrieval outcomes, not a fabricated citation."""
    return json.dumps(
        {
            "diagnoses": [
                {
                    "name": f"Working differential ({label})",
                    "confidence": 0.5,
                    "supporting_evidence": ["[RAG1] Most relevant offline evidence retrieved for this case."],
                    "rank": 1,
                }
            ],
            "overall_confidence": 0.5,
            "conflicts": [],
        }
    )


def _optimizer_mock_responses(case: CaseInput) -> list[str]:
    base = ", ".join(case.history) or (case.clinical_text or "symptoms")
    return [
        json.dumps({"strategy": "expand", "query": f"{base} broader search", "rationale": "expand"}),
        json.dumps({"strategy": "narrow", "query": f"{base} specific search", "rationale": "narrow"}),
        json.dumps({"strategy": "paraphrase", "query": f"{base} alternate phrasing", "rationale": "paraphrase"}),
    ]


def _direct_execute(agent, task):
    """Bypass CrewAI's real Crew.kickoff()/LLM call -- run the agent's
    one tool directly (same technique used throughout this project's test
    suite)."""
    return agent.tools[0]._run()


# Captured at import time, before anything ever overwrites
# orchestrator._execute -- lets --no-mock-claude restore the real,
# Claude-calling CrewAI execution path.
_REAL_EXECUTE = orchestrator._execute


def _install_mocks_for_case(case: CaseInput) -> None:
    orchestrator._execute = _direct_execute
    clinical_agent._get_client = lambda: _fake_client(_clinical_mock_response(case))
    fusion_agent._get_client = lambda: _fake_client([_fusion_mock_response("run")])
    optimizer_agent._get_client = lambda: _fake_client(_optimizer_mock_responses(case))
    # Imaging (BiomedCLIP) and RAG (Qdrant/BioBERT) and live evidence
    # (PubMed/MedlinePlus) are all left real/unmocked.


# --------------------------------------------------------------------------
# rag_retrieval_success heuristic -- DOCUMENTED CHOICE.
#
# The task offered two options: (a) a manual/human-scored relevance flag
# stored alongside each case, or (b) a simple keyword-overlap heuristic.
# This script uses (b): a case's RAG retrieval is judged "relevant" if its
# best-scoring retrieved passage shares at least 2 meaningful words
# (length > 3, common stopwords excluded) with the case's own
# clinical_text/medical_report/history. This is a prototype-level PROXY
# for relevance, not a validated judgment -- it can't distinguish a
# topically-adjacent passage from a truly clinically apt one, and it's
# vulnerable to synonyms/paraphrasing it can't see (e.g. "vertigo" vs
# "dizziness"). It was chosen over hand-scoring because it's reproducible
# on every run without a human re-labeling 17 cases whenever the corpus
# or retrieval logic changes -- it re-derives its answer from the run's
# own output every time.
# --------------------------------------------------------------------------

_STOPWORDS = {
    "and", "the", "with", "for", "over", "from", "this", "that", "has", "have",
    "history", "patient", "presenting", "presents", "recent", "recently",
}


def _extract_query_terms(case: CaseInput) -> set[str]:
    text = " ".join(filter(None, [case.clinical_text, case.medical_report, " ".join(case.history)]))
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def rag_retrieval_looks_relevant(case: CaseInput, report) -> bool:
    rag_items = [e for e in report.evidence if e.get("type") == "rag"]
    if not rag_items:
        return False
    query_terms = _extract_query_terms(case)
    if not query_terms:
        return False
    best = max(rag_items, key=lambda e: e.get("score", 0))
    passage_words = set(re.findall(r"[a-zA-Z]+", best.get("text", "").lower()))
    return len(query_terms & passage_words) >= 2


def has_usable_citation(report) -> bool:
    """evidence_coverage: does at least one diagnosis cite real, grounded
    evidence (not just "some evidence was retrieved somewhere")?"""
    return any(d.get("supporting_evidence") for d in report.diagnoses)


# --------------------------------------------------------------------------
# Case loading + evaluation loop.
# --------------------------------------------------------------------------


def load_cases(cases_dir: Path = TEST_CASES_DIR) -> list[tuple[str, CaseInput]]:
    results = []
    for path in sorted(cases_dir.glob("*.json")):
        try:
            case = CaseInput(**json.loads(path.read_text()))
        except Exception:
            continue  # not a CaseInput fixture -- skip non-case JSON in the dir
        results.append((path.name, case))
    return results


@dataclass
class CaseResult:
    case_id: str
    filename: str
    success: bool
    error: str | None = None
    confidence: float | None = None
    review_required: bool | None = None
    iteration_count: int | None = None
    latency_seconds: float | None = None
    rag_relevant: bool | None = None
    evidence_coverage: bool | None = None
    diagnosis_count: int | None = None
    live_evidence_available: bool | None = None


def run_evaluation(cases: list[tuple[str, CaseInput]], mock_claude: bool = True) -> list[CaseResult]:
    results = []
    for filename, case in cases:
        if mock_claude:
            _install_mocks_for_case(case)
        else:
            orchestrator._execute = _REAL_EXECUTE

        start = time.perf_counter()
        try:
            report = run_case(case)
        except Exception as exc:  # the evaluation harness itself must survive a bad case
            elapsed = time.perf_counter() - start
            results.append(
                CaseResult(case_id=case.case_id, filename=filename, success=False, error=str(exc), latency_seconds=elapsed)
            )
            continue
        elapsed = time.perf_counter() - start

        results.append(
            CaseResult(
                case_id=case.case_id,
                filename=filename,
                success=True,
                confidence=report.confidence,
                review_required=report.review_required,
                iteration_count=report.iteration_count,
                latency_seconds=elapsed,
                rag_relevant=rag_retrieval_looks_relevant(case, report),
                evidence_coverage=has_usable_citation(report),
                diagnosis_count=len(report.diagnoses),
                live_evidence_available=report.live_evidence_available,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> dict:
    total = len(results)
    successful = [r for r in results if r.success]
    n_success = len(successful)

    def pct(n: int) -> float:
        return round(100 * n / total, 1) if total else 0.0

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    iteration_counts = [r.iteration_count or 0 for r in successful]

    return {
        "cases_tested": total,
        "error_rate_pct": pct(total - n_success),
        "rag_retrieval_success_pct": pct(sum(1 for r in successful if r.rag_relevant)),
        "average_confidence": avg([r.confidence for r in successful if r.confidence is not None]),
        "optimizer_activation_rate_pct": pct(sum(1 for r in successful if (r.iteration_count or 0) > 0)),
        "hitl_rate_pct": pct(sum(1 for r in successful if r.review_required)),
        "average_latency_seconds": avg([r.latency_seconds for r in successful if r.latency_seconds is not None]),
        "average_iterations": avg(iteration_counts),
        "max_iterations": max(iteration_counts, default=0),
        "evidence_coverage_pct": pct(sum(1 for r in successful if r.evidence_coverage)),
    }


SUMMARY_LABELS = {
    "cases_tested": "Cases tested",
    "error_rate_pct": "Error rate (%)",
    "rag_retrieval_success_pct": "RAG retrieval success (%, keyword-overlap heuristic)",
    "average_confidence": "Average confidence",
    "optimizer_activation_rate_pct": "Optimizer activation rate (%)",
    "hitl_rate_pct": "HITL rate -- review_required (%)",
    "average_latency_seconds": "Average latency (s)",
    "average_iterations": "Average optimizer iterations",
    "max_iterations": "Max optimizer iterations",
    "evidence_coverage_pct": "Evidence coverage (%, >=1 grounded citation)",
}


def print_summary_table(summary: dict) -> None:
    width = max(len(v) for v in SUMMARY_LABELS.values())
    print("\n=== MedOrchestrate Evaluation Summary ===")
    for key, label in SUMMARY_LABELS.items():
        print(f"  {label:<{width}} : {summary[key]}")
    print()


def write_markdown_report(results: list[CaseResult], summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MedOrchestrate Evaluation Report", "", "## Summary", "", "| Metric | Value |", "|---|---|"]
    for key, label in SUMMARY_LABELS.items():
        lines.append(f"| {label} | {summary[key]} |")
    lines += ["", "## Per-case results", "", "| Case | Success | Confidence | Review required | Iterations | Latency (s) | RAG relevant | Evidence coverage | Live evidence | Error |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.case_id} | {r.success} | {r.confidence} | {r.review_required} | {r.iteration_count} | "
            f"{round(r.latency_seconds, 2) if r.latency_seconds is not None else ''} | {r.rag_relevant} | "
            f"{r.evidence_coverage} | {r.live_evidence_available} | {r.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_report(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


# --------------------------------------------------------------------------
# STRETCH / OPTIONAL: ablation comparison. NOT run by default -- pass
# --ablation. Runs a small subset of cases through 4 pipeline tiers.
#
# Interpretation note: the source "guide" document wasn't available to
# consult directly for the exact intended tier boundaries, so this is a
# documented, best-effort reading of "baseline LLM-only vs RAG-only vs
# RAG+live vs +fusion vs full system" as 4 (not 5) meaningfully distinct
# tiers -- fusion is what RECONCILES multiple evidence sources, so it
# only becomes a meaningfully distinct step once a tier actually HAS more
# than one source to reconcile:
#
#   1. baseline_llm_only    -- fuse() with empty RAG + empty live evidence
#                               (pure LLM reasoning over clinical/imaging
#                               findings alone, no external evidence).
#   2. rag_only              -- fuse() with real RAG evidence, empty live.
#   3. rag_plus_live_fusion  -- fuse() with real RAG AND real live evidence
#                               together -- this is also where "+fusion"
#                               is satisfied, since tiers 1-2 give fusion
#                               at most one source to work with.
#   4. full_system           -- the complete run_case() pipeline,
#                               including the iterative optimizer if
#                               confidence is low.
#
# Confidence for each tier is computed via the REAL compute_confidence()
# formula against that tier's real/empty rag+live+fusion objects, so the
# comparison reflects the actual confidence formula's behavior, not an
# invented ablation-specific metric.
# --------------------------------------------------------------------------


async def _run_ablation_tiers_for_case(case: CaseInput) -> dict:
    _install_mocks_for_case(case)

    clinical = await extract_clinical_findings(case.clinical_text, case.medical_report)
    imaging = await analyze_image(case.image_path) if case.image_path else None

    query = _build_query(clinical, imaging)
    real_rag = await run_rag(clinical, imaging)
    real_live = await gather_live_evidence(query) if query else LiveEvidence(sources=[])

    empty_rag = RAGEvidence(evidence=[])
    empty_live = LiveEvidence(sources=[])

    fusion_baseline = await fuse(clinical, imaging, empty_rag, empty_live)
    baseline_confidence = compute_confidence(empty_rag, empty_live, fusion_baseline)

    fusion_rag_only = await fuse(clinical, imaging, real_rag, empty_live)
    rag_only_confidence = compute_confidence(real_rag, empty_live, fusion_rag_only)

    fusion_rag_live = await fuse(clinical, imaging, real_rag, real_live)
    rag_live_confidence = compute_confidence(real_rag, real_live, fusion_rag_live)

    return {
        "baseline_llm_only": round(baseline_confidence, 4),
        "rag_only": round(rag_only_confidence, 4),
        "rag_plus_live_fusion": round(rag_live_confidence, 4),
    }


def run_ablation(cases: list[tuple[str, CaseInput]]) -> list[dict]:
    rows = []
    for filename, case in cases:
        tiers = asyncio.run(_run_ablation_tiers_for_case(case))
        _install_mocks_for_case(case)
        full_report = run_case(case)
        rows.append(
            {
                "case_id": case.case_id,
                **tiers,
                "full_system": round(full_report.confidence, 4),
            }
        )
    return rows


def print_ablation_table(rows: list[dict]) -> None:
    print("\n=== STRETCH: Ablation comparison (subset) ===")
    print(f"{'Case':<35} {'baseline':>10} {'rag_only':>10} {'rag+live':>10} {'full':>10}")
    for row in rows:
        print(
            f"{row['case_id']:<35} {row['baseline_llm_only']:>10} {row['rag_only']:>10} "
            f"{row['rag_plus_live_fusion']:>10} {row['full_system']:>10}"
        )
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MedOrchestrate evaluation harness")
    parser.add_argument("--cases-dir", type=Path, default=TEST_CASES_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_REPORT)
    parser.add_argument(
        "--no-mock-claude",
        action="store_true",
        help="Skip the generic Claude mock and run the real CrewAI/Claude pipeline (requires a working ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="STRETCH/OPTIONAL: also run the 4-tier ablation comparison on a small subset of cases.",
    )
    parser.add_argument("--ablation-subset-size", type=int, default=3)
    args = parser.parse_args()

    cases = load_cases(args.cases_dir)
    print(f"Loaded {len(cases)} case(s) from {args.cases_dir}")

    results = run_evaluation(cases, mock_claude=not args.no_mock_claude)
    summary = summarize(results)

    print_summary_table(summary)
    write_markdown_report(results, summary, args.output_md)
    write_csv_report(results, args.output_csv)
    print(f"Markdown report written to {args.output_md}")
    print(f"CSV report written to {args.output_csv}")

    if args.ablation:
        subset = cases[: args.ablation_subset_size]
        ablation_rows = run_ablation(subset)
        print_ablation_table(ablation_rows)


if __name__ == "__main__":
    main()
