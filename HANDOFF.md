# MedOrchestrate — Session Handoff

**Read this file first if you're picking up this project in a new Claude
session.** It's a complete account of what's been built, verified, and
pushed to GitHub, plus what's known to still be missing. Written
2026-08-28 because the previous session was approaching its context
limit with ~2 phases still remaining on the user's build guide.

## What this project is

A clinician-facing, multimodal (text + medical report + MRI image)
clinical decision-support prototype. Not a real medical device — a
prototype. Backend: Python 3.11+/FastAPI. Agent orchestration: CrewAI,
LLM = Anthropic Claude. RAG: BioBERT + Qdrant. Imaging: BiomedCLIP
zero-shot. Frontend: React + Vite + Tailwind. DB: SQLite.

## Repo state right now

- **GitHub**: `https://github.com/Abhinavd2004/MedOrchestrate` — `main`
  branch, 3 commits, fully pushed and in sync with local as of this
  writing.
- **Tests**: `154 passed` (backend, `pytest` from repo root, fully
  offline/mocked, ~15s). Also a separate `backend/tests/run_evaluation.py`
  (not a pytest test — real-pipeline evaluation harness, see below).
- **`.env`**: exists locally with `ANTHROPIC_API_KEY=` **empty**. This
  is the single biggest thing to know: nothing Claude-dependent has ever
  been run for real in this project. Every "it works" claim below for
  Claude-dependent steps was verified with the Claude API call itself
  mocked (documented, consistent methodology throughout) — the
  surrounding code (parsing, validation, retry logic, orchestration) is
  real and tested; the actual LLM call has not been.

## What's built, phase by phase (all done, all tested)

| # | What | Key files | Status |
|---|---|---|---|
| 0 | Repo scaffold, shared Pydantic schemas, `/health` | `backend/models/schemas.py`, `backend/main.py` | ✅ |
| 1 | Clinical extraction agent (Claude) | `backend/agents/clinical_agent.py` | ✅ 7 tests |
| 2 | Imaging agent (BiomedCLIP zero-shot, NOT fine-tuned) | `backend/agents/imaging_agent.py`, `imaging_labels.py` | ✅ 10 tests; **real model verified working** (downloaded + ran once, see below) |
| 3 | RAG: embeddings, ingest, retrieve, rerank | `backend/rag/{embeddings,ingest,retriever,reranker}.py` | ✅; real local Qdrant store populated (`data/qdrant_local/`, git-ignored — **must re-run `python -m backend.rag.ingest` on a fresh clone**, see Known Gaps) |
| 4 | Live evidence: PubMed + MedlinePlus (keyless) | `backend/agents/evidence_agent.py` | ✅; verified live, real network calls |
| 5 | Fusion agent (citation-grounded differential) | `backend/agents/fusion_agent.py` | ✅ 8 tests |
| 6 | Confidence formula (`C = w1*S_rag + w2*S_web + w3*A`, THRESHOLD=0.65) | `backend/services/confidence.py` | ✅ 17 tests, hand-verified math. Weights explicitly marked "NOT CLINICALLY VALIDATED — tuning targets for Phase 15 evaluation" (per the user's own build guide numbering) |
| 7 | Optimizer (iterative query refinement, 3-iteration cap, HITL handoff) | `backend/agents/optimizer_agent.py` | ✅ 7 tests |
| 8 | CrewAI orchestrator wiring everything | `backend/services/orchestrator.py` | ✅ 11 tests. See "CrewAI design note" below — important |
| 9 | FastAPI routes (`/diagnose`, `/review`) | `backend/api/{routes,schemas}.py` | ✅ 13 tests |
| 10 | React frontend (Submission/Processing/Report screens) | `frontend/src/pages/{CaseSubmission,Processing,Report}.jsx` | ✅ built + browser-verified |
| 11 | HITL review flow (full UI: Accept/Override/Annotate) | `frontend/src/pages/Review.jsx`, `/review/{case_id}` GET+POST | ✅ browser-verified round-trip |
| 12 | SQLite storage (cases/reports/reviews/agent_logs, real persistence) | `backend/services/storage.py` | ✅ 16 tests, real file inspected |
| 13 | Structured/redacted logging + optional Langfuse (off by default) | `backend/services/logging.py` | ✅ 14 tests; PII-leak audit done, documented in the module docstring |
| — | Full-matrix integration tests (real `run_case()`, only external boundary mocked) | `backend/tests/test_integration.py` | ✅ 14 tests, covers text-only/image-only/combined/high-low-confidence/optimizer-cap/RAG-failure/web-failure/invalid-upload |
| — | 17 synthetic demo cases | `data/test_cases/CASE001–016*.json` + `README.md` | ✅ all validate against real `CaseInput` schema |
| — | Evaluation harness | `backend/tests/run_evaluation.py` | ✅ real 17-case run completed, 0 errors — see numbers below |

### Last real evaluation run's numbers (for reference — re-run to regenerate)

```
Cases tested: 17          Error rate: 0.0%
RAG retrieval success: 82.4%  (keyword-overlap heuristic, documented in the script)
Average confidence: 0.7378    Optimizer activation rate: 70.6%
HITL rate: 29.4%              Average latency: 5.26s/case
Avg optimizer iterations: 1.29   Max: 3 (cap respected)
Evidence coverage: 100.0%
```

## Architecture notes worth knowing before touching anything

1. **CrewAI is a thin orchestration layer, not a reasoning layer.**
   Every CrewAI Agent has exactly one Tool; the Tool's `_run()` just
   calls the real, already-tested agent function
   (`extract_clinical_findings`, `run_rag`, `fuse`, etc.) and stores the
   result. Data flows between steps through a shared in-process
   `_CaseContext` object, **not** through CrewAI's LLM-mediated task
   text — deliberate, to avoid a second LLM pass silently corrupting
   citation-grounded output. Full rationale in
   `backend/services/orchestrator.py`'s module docstring.

2. **CrewAI's own Agent-decision layer needs a live Claude call for
   every step**, even RAG/imaging which don't otherwise need Claude.
   Without a real `ANTHROPIC_API_KEY`, a genuine `POST /diagnose` call
   returns a clean `500` (graceful, not a crash — confirmed real
   behavior, not a guess). Every "it works" demo in this project bypassed
   this one specific layer via `orchestrator._execute` monkeypatching
   (documented, consistent pattern across every test file).

3. **Shared schema is the single source of truth**:
   `backend/models/schemas.py`. It grew additively across phases
   (`conflicts`, `iteration_count`, `live_evidence_available` were all
   added later, backward-compatibly, when a real downstream need
   surfaced) — always check this file before assuming a field doesn't
   exist.

4. **Reranker/RAG models cache in HuggingFace's local cache
   (`~/.cache/huggingface/hub/`)**, not in the repo. BioBERT and the
   cross-encoder are confirmed cached on this machine. **BiomedCLIP was
   also confirmed downloaded and run for real once** (not just mocked)
   — see the "BiomedCLIP imaging" conversation; it works.

## Known gaps / honest caveats (told to the user directly, not hidden)

- **No real Claude API call has ever completed in this project.**
  Everything Claude-dependent is verified via mocking, not a live call.
  First real end-to-end test needs a valid `ANTHROPIC_API_KEY` in
  `.env`.
- **Docker Compose is incomplete.** `docker-compose.yml` exists from
  Phase 0 (references `build: ./backend` and `./frontend`) but **no
  Dockerfiles were ever created**, and Docker itself isn't installed in
  this dev environment, so it's never been tested. If a future phase is
  "containerize/deploy", start here.
- **`/diagnose` is fully synchronous.** `storage.py`'s `cases.status`
  column (`pending`/`completed`/`failed`) already exists for a future
  async/polling version but isn't wired into `GET /diagnose/{case_id}`'s
  response logic yet.
- **The optimizer's later iterations' rag/live evidence aren't exposed
  back to the orchestrator** — `optimize()`'s return dict has
  `final_fusion`/`final_confidence`/`iterations`/`review_required`/
  `iteration_log`, but not the actual RAGEvidence/LiveEvidence objects
  used internally during refinement. `FinalReport.evidence` and
  `live_evidence_available` currently reflect only the *first-pass*
  retrieval, pre-optimizer.
- **`data/qdrant_local/` (the populated RAG vector store) is
  git-ignored** — a fresh clone needs `python -m backend.rag.ingest` run
  once before RAG retrieval will find anything. Same for
  `~/.cache/huggingface/` — models re-download on a new machine.
- **CASE001.json and CASE001_with_image.json share the same
  `case_id`** ("CASE001") — a pre-existing quirk from Phase 0, harmless
  for most purposes but means they'd collide in the `cases`/`reports`
  SQLite tables if both were submitted via the real API.

## Quick orientation for a new session

1. Read `README.md` for the actual setup steps (venv, `pip install -r
   requirements.txt`, `.env`, `python -m backend.rag.ingest`, `uvicorn
   backend.main:app --reload`, `cd frontend && npm install && npm run
   dev`).
2. Run `pytest` from repo root to confirm the 154 tests still pass —
   that's the fastest sanity check that nothing's broken.
3. `git log --oneline` to see the 3 commits so far; `git status` should
   be clean.

## What's left

**The user said ~2 more phases remain on their own build guide** — a
document this session never had full visibility into (only individual
phase instructions were given one at a time as the user pasted them).
**This session does not know the exact content of those remaining
phases.** If you're a new Claude session picking this up:

- **Ask the user directly** for the next phase's exact instructions
  (they'll paste it, same pattern as every phase so far).
- Do not guess or invent phase content.
- Known candidates based on gaps above (the user may or may not have
  these in mind): Docker Compose completion / deployment, async
  `/diagnose` with polling, or a final demo-prep/documentation pass. But
  treat these as *guesses*, not confirmed scope, until the user says so.

## Working style notes for whoever continues this

- Every phase in this project followed the same rhythm: implement →
  write tests → run tests for real → for anything Claude-dependent,
  mock at the `_get_client()` boundary and say so explicitly → for
  anything else external (Qdrant, PubMed, BiomedCLIP), prefer running it
  for real since none of those need an API key.
- The user asks "is all ok/good to go" frequently between phases —
  respond with an actual fresh `pytest` run, not a cached claim.
- The user cares about git hygiene: review `git status`/`git add -A`
  output before committing, keep `.env` and generated artifacts
  (`*.db`, `logs/`, `data/qdrant_local/`, evaluation reports) out of
  git, confirm push actually happened rather than assuming.
- Be honest about what's mocked vs. real — this has been a consistent,
  explicit theme every single phase, not a one-off.
