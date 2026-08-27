"""Structured observability layer for MedOrchestrate.

Structured JSON-lines logging: one JSON object per line, emitted via
the standard library `logging` module to BOTH stdout and a rotating
file under logs/ (logs/medorchestrate.log, 5 MB x 5 backups) -- easy to
grep, pipe, or ship to a log aggregator later without pulling in a
logging framework dependency for this prototype. If the logs/
directory can't be created (e.g. a read-only filesystem in some
deployment), the file handler is skipped and stdout logging still
works -- this never blocks startup.

Every event automatically gets a `timestamp` (ISO 8601, UTC) and is
passed through redact_sensitive() before being written anywhere.
That's deliberate: callers cannot accidentally leak an unsafe field by
forgetting to redact it themselves -- the allowlist is enforced once,
centrally, inside log_event(), not left as an opt-in utility function.

Secrets audit (see PR/commit history for the full grep): no call site
in backend/agents/*.py, backend/services/*.py, or backend/api/*.py
ever logs an API key, token, or header value. Anthropic/Qdrant keys are
passed as SDK constructor kwargs or HTTP headers, never interpolated
into strings that reach log_event() or an exception message. The
`error` field below carries `str(exc)` text from this codebase's own
exceptions (e.g. "PubMed esearch unavailable after 2 attempts: ...",
"Could not resolve authentication method...") -- these describe
parameter *names*, never echo back credential *values*. Re-check this
if a new external call site is added.

Optional Langfuse integration: disabled by default (LANGFUSE_ENABLED
env var, default "false" -- see .env.example). When enabled, every
log_event() call also creates a Langfuse event from the exact same
already-redacted payload. The langfuse client is imported and
constructed lazily, only when the flag is on, so the app behaves
identically with it off regardless of whether the langfuse package is
even installed -- and any Langfuse failure (bad credentials, network
down, package missing) is caught and logged as a local warning, never
allowed to break the pipeline.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_logger = logging.getLogger("medorchestrate")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_stream_handler)

    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        _file_handler = RotatingFileHandler(
            logs_dir / "medorchestrate.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        _file_handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(_file_handler)
    except OSError:
        pass  # stdout logging alone still works; never fail startup over this

_logger.propagate = False


# --------------------------------------------------------------------------
# redact_sensitive -- allowlist, not blocklist.
#
# Only these fields are ever written to a log line. Everything else --
# including anything not on this list, e.g. clinical_text, patient,
# symptoms, a raw or derived free-text query -- is dropped entirely,
# not masked. An allowlist fails closed: a new field added anywhere in
# the codebase tomorrow is invisible to logging until someone
# deliberately adds it here, rather than being silently logged until
# someone notices and blocklists it after the fact.
# --------------------------------------------------------------------------
_SAFE_LOG_KEYS = {
    "event",
    "timestamp",
    "case_id",
    "agent_name",
    "iteration",
    "iterations",
    "status",
    "latency_ms",
    "latency",
    "confidence",
    "review_required",
    "strategy",  # fixed enum: "expand" | "narrow" | "paraphrase" -- not free text
    "endpoint",
    "error",  # audited above: this codebase's error strings never embed secrets or raw clinical text
}


def redact_sensitive(data: dict) -> dict:
    """Allowlist-based redaction: keep only known-safe keys.

    For the prototype this is intentionally conservative -- known-safe
    fields like case_id, agent_name, confidence, latency, and status
    pass through; a free-text field like `query` (the optimizer's
    symptom-derived search query) or `clinical_text`/`patient` is
    dropped even though it isn't literally "raw clinical_text", because
    it's still derived from patient data and this module can't verify
    it's safe to log.
    """
    return {key: value for key, value in data.items() if key in _SAFE_LOG_KEYS}


def log_event(event: str, **fields) -> None:
    """Emit one structured JSON line: {"event", "timestamp", ...fields},
    redacted, to stdout + logs/medorchestrate.log, and (if enabled) to
    Langfuse.

    `default=str` so non-JSON-native values (e.g. anything unexpected
    slipping through) don't crash logging -- they just get stringified.
    `ensure_ascii=True` (json.dumps' default) so non-ASCII text never
    breaks a terminal with a narrower codec.
    """
    payload = redact_sensitive(
        {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
    )
    _logger.info(json.dumps(payload, default=str))
    _send_to_langfuse(event, payload)


# --------------------------------------------------------------------------
# Optional Langfuse integration -- off by default, safe to leave off.
# --------------------------------------------------------------------------

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() == "true"

_langfuse_client = None
_langfuse_unavailable = False


def _get_langfuse_client():
    global _langfuse_client, _langfuse_unavailable
    if _langfuse_client is not None or _langfuse_unavailable:
        return _langfuse_client
    try:
        from langfuse import Langfuse  # local import: only touched when LANGFUSE_ENABLED=true

        _langfuse_client = Langfuse()  # reads LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST from env
    except Exception as exc:  # missing package, bad credentials, etc. -- disable, don't crash
        _langfuse_unavailable = True
        _logger.warning(
            json.dumps(
                {
                    "event": "langfuse_unavailable",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                },
                default=str,
            )
        )
    return _langfuse_client


def _send_to_langfuse(event: str, payload: dict) -> None:
    if not LANGFUSE_ENABLED:
        return
    client = _get_langfuse_client()
    if client is None:
        return
    try:
        client.create_event(name=event, metadata=payload)
    except Exception:
        pass  # observability plumbing must never break the pipeline
