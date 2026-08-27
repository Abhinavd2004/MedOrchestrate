"""Tests for backend.services.logging.

Covers: redact_sensitive()'s allowlist behavior (the core ask -- must
strip clinical_text/patient/free-text fields), log_event()'s automatic
timestamp + redaction + JSON-lines output, the rotating file handler
setup, and the Langfuse integration's off-by-default / graceful-
failure behavior (no network access, no real Langfuse server needed).
"""

import json
import logging
from logging.handlers import RotatingFileHandler

import pytest

import backend.services.logging as app_logging
from backend.services.logging import log_event, redact_sensitive


# --------------------------------------------------------------------------
# redact_sensitive: allowlist behavior
# --------------------------------------------------------------------------


def test_redact_sensitive_strips_clinical_text_and_patient_fields():
    data = {
        "case_id": "CASE001",
        "clinical_text": "45-year-old male with headache and dizziness, history of hypertension",
        "patient": {"age": 45, "sex": "male"},
        "medical_report": "MRI shows white matter hyperintensities",
    }

    redacted = redact_sensitive(data)

    assert redacted == {"case_id": "CASE001"}
    assert "clinical_text" not in redacted
    assert "patient" not in redacted
    assert "medical_report" not in redacted


def test_redact_sensitive_strips_name_mrn_and_other_pii_like_fields():
    data = {
        "case_id": "CASE001",
        "name": "Jane Doe",
        "mrn": "MRN-00123",
        "dob": "1980-01-01",
        "email": "jane@example.com",
        "phone": "555-0100",
        "address": "123 Main St",
    }

    redacted = redact_sensitive(data)

    assert redacted == {"case_id": "CASE001"}


def test_redact_sensitive_strips_api_keys_and_free_text_query():
    data = {
        "agent_name": "clinical",
        "api_key": "sk-ant-super-secret-value",
        "authorization": "Bearer sk-ant-super-secret-value",
        "query": "headache, dizziness; history of hypertension",  # symptom-derived free text
    }

    redacted = redact_sensitive(data)

    assert redacted == {"agent_name": "clinical"}


def test_redact_sensitive_keeps_the_documented_safe_fields():
    data = {
        "case_id": "CASE001",
        "agent_name": "fusion",
        "iteration": 0,
        "status": "success",
        "latency_ms": 12.34,
        "confidence": 0.8,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "event": "agent_call",
    }

    assert redact_sensitive(data) == data


def test_redact_sensitive_drops_unknown_fields_by_default_fail_closed():
    # An allowlist means a brand-new field added anywhere in the
    # codebase tomorrow is invisible to logging until deliberately
    # added here -- verify an arbitrary unknown key is dropped, not
    # passed through.
    redacted = redact_sensitive({"case_id": "CASE001", "some_new_field_nobody_reviewed_yet": "value"})
    assert redacted == {"case_id": "CASE001"}


# --------------------------------------------------------------------------
# log_event: automatic timestamp + redaction + JSON-lines output
# --------------------------------------------------------------------------


def test_log_event_emits_valid_json_with_automatic_timestamp(caplog):
    # Note: capsys can't see this output -- the StreamHandler bound to
    # sys.stdout at module-import time, before capsys ever gets a
    # chance to swap sys.stdout out; caplog is the correct tool for
    # asserting on `logging`-module output regardless of handler wiring.
    with caplog.at_level(logging.INFO, logger="medorchestrate"):
        log_event(
            "agent_call", case_id="CASE001", agent_name="clinical", iteration=0, status="success", latency_ms=1.5, confidence=0.8
        )

    payload = json.loads(caplog.records[-1].getMessage())

    assert payload["event"] == "agent_call"
    assert payload["case_id"] == "CASE001"
    assert payload["agent_name"] == "clinical"
    assert payload["iteration"] == 0
    assert payload["status"] == "success"
    assert payload["latency_ms"] == 1.5
    assert payload["confidence"] == 0.8
    assert "timestamp" in payload
    # ISO 8601 -- just sanity-check it parses back
    from datetime import datetime

    datetime.fromisoformat(payload["timestamp"])


def test_log_event_redacts_unsafe_fields_before_emitting(caplog):
    with caplog.at_level(logging.INFO, logger="medorchestrate"):
        log_event("agent_call", case_id="CASE001", clinical_text="raw patient narrative that must never be logged")

    message = caplog.records[-1].getMessage()
    assert "raw patient narrative" not in message
    payload = json.loads(message)
    assert "clinical_text" not in payload
    assert payload["case_id"] == "CASE001"


# --------------------------------------------------------------------------
# Rotating file handler
# --------------------------------------------------------------------------


def test_logger_has_both_stream_and_rotating_file_handlers():
    # A plain StreamHandler (stdout) -- not the RotatingFileHandler
    # subclass, which is also technically a StreamHandler.
    stream_handlers = [h for h in app_logging._logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1

    file_handlers = [h for h in app_logging._logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename.replace("\\", "/").endswith("logs/medorchestrate.log")


# --------------------------------------------------------------------------
# Langfuse: off by default, graceful failure when enabled
# --------------------------------------------------------------------------


def test_langfuse_disabled_by_default():
    assert app_logging.LANGFUSE_ENABLED is False


def test_send_to_langfuse_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(app_logging, "LANGFUSE_ENABLED", False)
    get_client_called = []
    monkeypatch.setattr(app_logging, "_get_langfuse_client", lambda: get_client_called.append(True))

    app_logging._send_to_langfuse("agent_call", {"case_id": "CASE001"})

    assert get_client_called == []  # never even tried to construct a client


def test_log_event_works_normally_with_langfuse_disabled(caplog):
    # The default state for this whole test suite -- confirms the app
    # runs fine with LANGFUSE_ENABLED off (no network access attempted).
    with caplog.at_level(logging.INFO, logger="medorchestrate"):
        log_event("agent_call", case_id="CASE001", status="success")
    assert json.loads(caplog.records[-1].getMessage())["case_id"] == "CASE001"


def test_send_to_langfuse_swallows_client_construction_failure(monkeypatch):
    monkeypatch.setattr(app_logging, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(app_logging, "_langfuse_client", None)
    monkeypatch.setattr(app_logging, "_langfuse_unavailable", False)

    class _BoomClient:
        def __init__(self, *a, **kw):
            raise RuntimeError("no credentials configured")

    monkeypatch.setitem(__import__("sys").modules, "langfuse", type("m", (), {"Langfuse": _BoomClient})())

    # Must not raise -- a bad/missing Langfuse setup should never break logging.
    app_logging._send_to_langfuse("agent_call", {"case_id": "CASE001"})
    assert app_logging._langfuse_unavailable is True


def test_send_to_langfuse_calls_create_event_when_client_available(monkeypatch):
    monkeypatch.setattr(app_logging, "LANGFUSE_ENABLED", True)

    calls = []

    class _FakeClient:
        def create_event(self, name, metadata):
            calls.append((name, metadata))

    monkeypatch.setattr(app_logging, "_get_langfuse_client", lambda: _FakeClient())

    app_logging._send_to_langfuse("agent_call", {"case_id": "CASE001", "status": "success"})

    assert calls == [("agent_call", {"case_id": "CASE001", "status": "success"})]


def test_send_to_langfuse_swallows_create_event_failure(monkeypatch):
    monkeypatch.setattr(app_logging, "LANGFUSE_ENABLED", True)

    class _FailingClient:
        def create_event(self, name, metadata):
            raise ConnectionError("simulated network failure")

    monkeypatch.setattr(app_logging, "_get_langfuse_client", lambda: _FailingClient())

    # Must not raise.
    app_logging._send_to_langfuse("agent_call", {"case_id": "CASE001"})
