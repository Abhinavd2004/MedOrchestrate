# MedOrchestrate

Clinician-facing, multimodal (text + medical report + MRI image) clinical
decision-support prototype.

- **Backend**: Python 3.11+ (developed against 3.12) + FastAPI
- **Agent orchestration**: CrewAI (Claude as the CrewAI LLM)
- **RAG**: BioBERT sentence embeddings + Qdrant (local embedded fallback if no Qdrant server configured)
- **Imaging**: BiomedCLIP zero-shot classification
- **Live evidence**: PubMed + MedlinePlus (no API key required)
- **Frontend**: React + Vite + Tailwind
- **DB**: SQLite
- **LLM provider**: Anthropic Claude API

## Status

The full pipeline is implemented end to end: clinical extraction →
imaging classification → offline RAG retrieval → live evidence →
evidence fusion → confidence scoring → iterative optimization →
CrewAI orchestration → FastAPI HTTP layer → SQLite persistence →
structured/redacted logging → a 4-screen React frontend (submission,
processing, report, human-in-the-loop review).

**Running the pipeline for real requires a valid `ANTHROPIC_API_KEY`**
(see below) — several steps (clinical extraction, fusion reasoning, the
optimizer, and CrewAI's own agent-decision layer) call Claude directly.
Without a key, `/diagnose` will return a clean `500` rather than hang
or crash (the backend degrades gracefully), but it won't produce a real
diagnosis.

---

## Setup — step by step (VS Code)

### 1. Prerequisites

- [VS Code](https://code.visualstudio.com/)
- Python 3.11+ ([python.org](https://www.python.org/downloads/))
- Node.js 18+ ([nodejs.org](https://nodejs.org/)) — includes npm
- Git
- Recommended VS Code extensions: **Python** (ms-python.python), **Pylance**, and (for the frontend) **ES7+ React/Redux/JS snippets** or just the built-in JS/TS support — none are required, just convenient.

### 2. Open the project

Open the `MedOrchestrate` folder in VS Code (`File > Open Folder…`), or
from a terminal:

```bash
code D:\MedOrchestra
```

### 3. Create and activate the backend virtual environment

Open a terminal in VS Code (`` Ctrl+` ``) at the project root:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell / VS Code integrated terminal default)
venv\Scripts\Activate.ps1

# Windows (Git Bash)
source venv/Scripts/activate

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` in the terminal prompt. In VS Code, also select
this interpreter so Pylance/IntelliSense and the built-in test runner
use it: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one
under `.\venv\Scripts\python.exe` (Windows) or `./venv/bin/python`.

### 4. Install backend dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, CrewAI, the Anthropic SDK, sentence-transformers
(BioBERT), open_clip (BiomedCLIP), torch, Qdrant client, etc. The first
install takes a few minutes (torch and the model libraries are large).

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Required? | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes**, for real runs | Get one from console.anthropic.com. Without it, Claude-dependent steps fail gracefully but the pipeline won't produce real output. |
| `QDRANT_URL` / `QDRANT_API_KEY` | No | Leave blank to use an embedded local Qdrant store at `data/qdrant_local/` — no server needed. Set these only if you have a real Qdrant Cloud/self-hosted instance. |
| `PUBMED_API_KEY` | No | PubMed/MedlinePlus work without a key at this call volume. |
| `DATABASE_URL` | No | Defaults to `sqlite:///./medorchestrate.db`; created automatically on first run. |
| `LANGFUSE_ENABLED` | No | Defaults to `false`. Leave off unless you have Langfuse credentials. |

**`.env` is git-ignored — never commit it.**

### 6. Ingest the offline knowledge base (one-time, required for RAG)

The RAG agent queries a local Qdrant collection that starts empty. Populate it once:

```bash
python -m backend.rag.ingest
```

This downloads the BioBERT embedding model (first run only, a few
minutes) and embeds ~18 reference passages into the local Qdrant store.
You only need to do this once (the local store persists on disk).

### 7. Run the backend

```bash
uvicorn backend.main:app --reload
```

- API base: `http://localhost:8000`
- Health check: `http://localhost:8000/health` → `{"status": "ok"}`
- Interactive API docs: `http://localhost:8000/docs`

Keep this terminal running. Open a **second** VS Code terminal
(`` Ctrl+Shift+` `` or the `+` in the terminal panel) for the frontend.

### 8. Run the frontend

In the new terminal:

```bash
cd frontend
npm install
npm run dev
```

- App: `http://localhost:5173`
- The frontend is pre-configured to call the backend at
  `http://localhost:8000` (see `frontend/src/services/api.js`) — no
  extra config needed as long as both are running locally on their
  default ports.

Open `http://localhost:5173` in a browser to use the app: submit a
case (age/sex + clinical text and/or an image), watch it process, and
view the report. If confidence is below threshold, you'll be routed
automatically to the review screen (also reachable manually via the
"Review Case" button on any report).

### 9. Run the tests (optional but recommended)

Backend, from the project root with the venv active:

```bash
pytest
```

All ~140 tests run offline (no API key or network access required —
external calls are mocked). Should finish in under 20 seconds.

Frontend build check:

```bash
cd frontend
npm run build
```

---

## Quick reference: what needs to run, and in what order

1. `venv` activated, `pip install -r requirements.txt` done — once.
2. `python -m backend.rag.ingest` — once (populates the local Qdrant store).
3. `uvicorn backend.main:app --reload` — every session, terminal 1.
4. `npm run dev` (from `frontend/`) — every session, terminal 2.
5. Browse to `http://localhost:5173`.

## Project structure

```
MedOrchestrate/
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── api/                   # Routes (/diagnose, /review) + API-layer schemas
│   ├── agents/                # clinical, imaging, rag, evidence, fusion, optimizer
│   ├── rag/                   # BioBERT embeddings + Qdrant ingest/retrieve/rerank
│   ├── services/              # orchestrator (CrewAI), confidence, storage (SQLite), logging
│   ├── models/schemas.py      # Shared Pydantic schemas -- single source of truth
│   └── tests/                 # ~140 offline tests
├── frontend/
│   └── src/
│       ├── pages/             # CaseSubmission, Processing, Report, Review
│       ├── components/        # ConfidenceBar, ReportSummary
│       └── services/api.js    # fetch wrapper for the backend
├── data/
│   ├── knowledge_base/        # (reference only; actual corpus lives in backend/rag/ingest.py)
│   ├── images/                # synthetic demo images + user uploads
│   ├── test_cases/            # CASE001.json etc.
│   └── qdrant_local/          # local embedded Qdrant store (git-ignored, created by ingest.py)
├── logs/                      # rotating structured log file (git-ignored)
├── medorchestrate.db          # SQLite dev database (git-ignored, auto-created)
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Shared contracts

All agents and API routes import their data shapes from
[`backend/models/schemas.py`](backend/models/schemas.py) rather than
redefining them, so `CaseInput`, `ClinicalFindings`, `ImagingFindings`,
`RAGEvidence`, `LiveEvidence`, `FusionResult`, and `FinalReport` stay
consistent across the whole pipeline.
