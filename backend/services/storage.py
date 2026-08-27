"""Case, report, review, and agent-log persistence.

Backend: stdlib `sqlite3`, not SQLAlchemy. Documented choice: this
module has always used the standard library sqlite3 module (Phase 11's
reviews table introduced it), the schema here is small and fixed (four
tables, no relationships beyond simple FKs, "no migration framework
needed yet" per the spec), and pulling in SQLAlchemy would mean an ORM
layer, a new dependency, and a session/engine lifecycle to manage for
no real benefit at this size. If the schema grows substantially or a
migration tool becomes necessary, SQLAlchemy + Alembic is the natural
next step -- nothing here should make that harder to introduce later,
since every table access is already funneled through this module's
functions.

Tables (see the CREATE TABLE statements below for the authoritative
definitions):
  - cases: one row per submitted case (case_id PK).
  - reports: one row per case's current FinalReport (case_id PK, so
    saving a new report for the same case_id replaces it -- matches
    the Phase 9 in-memory dict's overwrite semantics).
  - reviews: one row per review event (a case can be reviewed more than
    once -- e.g. re-reviewed after a correction -- so this is a proper
    one-to-many table with its own `id`, not one row per case).
  - agent_logs: one row per orchestrator agent-call event (per Phase 8;
    also written to stdout via services/logging.py -- this table is
    the persisted, queryable copy of the same events).

`REFERENCES cases(case_id)` in reports/reviews/agent_logs documents the
intended relationship but is NOT enforced (SQLite has foreign keys off
by default, and this module deliberately leaves them off) -- agents can
be logged, and reports/reviews queried, independently of exactly when
save_case() ran, which matters for unit-testing individual pieces (e.g.
backend/tests/test_orchestrator.py) without needing to seed a `cases`
row first.
"""

import os
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv

from backend.models.schemas import CaseInput, FinalReport

load_dotenv()


def _resolve_sqlite_path() -> str:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./medorchestrate.db")
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return database_url[len(prefix) :]
    # Not a sqlite:// URL (e.g. a real Postgres URL from a future
    # storage upgrade) -- this module only speaks SQLite, so fall back
    # to a local file rather than trying to connect to it.
    return "medorchestrate.db"


# Module-level so tests can monkeypatch it to an isolated temp file and
# call _init_tables() again, without touching the real dev database.
DB_PATH = _resolve_sqlite_path()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_tables() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                case_id TEXT PRIMARY KEY REFERENCES cases (case_id),
                report_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                iteration_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL REFERENCES cases (case_id),
                decision TEXT NOT NULL,
                annotation TEXT,
                reviewer TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_case_id ON reviews (case_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL REFERENCES cases (case_id),
                agent_name TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                status TEXT NOT NULL,
                latency REAL,
                confidence REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_logs_case_id ON agent_logs (case_id)")


_init_tables()


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

# Conventional status values (not DB-enforced via CHECK, to keep the
# schema easy to extend without a migration): "pending" while a case is
# being processed, "completed" once a report is saved, "failed" if
# run_case() raised. GET /diagnose/{case_id} does not yet branch on
# this (it still just checks whether a report exists), but the column
# is here for that distinction once an async version needs it.


def save_case(case: CaseInput, status: str = "pending") -> None:
    """Insert a case, or update its input_json/status if case_id already
    exists -- created_at is set once, on first insert, and preserved on
    later calls (e.g. status transitioning pending -> completed/failed
    for the same case_id).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, created_at, input_json, status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                input_json = excluded.input_json,
                status = excluded.status
            """,
            (case.case_id, created_at, case.model_dump_json(), status),
        )


def get_case(case_id: str) -> CaseInput | None:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    return CaseInput.model_validate_json(row["input_json"])


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def save_report(case_id: str, report: FinalReport) -> None:
    """Upserts the report for case_id -- a case has at most one current
    report; a new save_report() call replaces the previous one, same
    as the old in-memory dict's `_reports[case_id] = report`."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO reports (case_id, report_json, confidence, iteration_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                report_json = excluded.report_json,
                confidence = excluded.confidence,
                iteration_count = excluded.iteration_count
            """,
            (case_id, report.model_dump_json(), report.confidence, report.iteration_count),
        )


def get_report(case_id: str) -> FinalReport | None:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM reports WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    return FinalReport.model_validate_json(row["report_json"])


# --------------------------------------------------------------------------
# reviews
# --------------------------------------------------------------------------


def save_review(case_id: str, review: dict) -> dict:
    """Insert a review row for case_id (a case can be reviewed more than
    once, so this appends rather than overwrites). Returns the stored
    row, including the generated id and server-assigned timestamp, so
    callers can echo back exactly what was persisted.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO reviews (case_id, decision, annotation, reviewer, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, review["decision"], review.get("annotation"), review["reviewer"], timestamp),
        )
        review_id = cursor.lastrowid

    return {
        "id": review_id,
        "case_id": case_id,
        "decision": review["decision"],
        "annotation": review.get("annotation"),
        "reviewer": review["reviewer"],
        "timestamp": timestamp,
    }


def get_reviews(case_id: str) -> list[dict]:
    """Every review recorded for case_id, most recent first."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE case_id = ? ORDER BY timestamp DESC, id DESC",
            (case_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# agent_logs
# --------------------------------------------------------------------------


def log_agent_event(
    case_id: str,
    agent_name: str,
    iteration: int,
    status: str,
    latency: float | None = None,
    confidence: float | None = None,
) -> None:
    """Persists one orchestrator agent-call event (see
    backend/services/orchestrator.py's _log_agent_call, which calls
    this alongside the stdout structured log from services/logging.py).
    """
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO agent_logs (case_id, agent_name, iteration, status, latency, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, agent_name, iteration, status, latency, confidence),
        )


def get_agent_logs(case_id: str) -> list[dict]:
    """Every agent_logs row for case_id, in the order they were recorded."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_logs WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        ).fetchall()
    return [dict(row) for row in rows]
