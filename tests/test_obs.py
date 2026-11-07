"""Shipping R3's logs and AI traces to Axiom (r3/obs.py): AXIOM_TOKEN +
AXIOM_DATASET turn it on, AXIOM_ORG_ID and AXIOM_URL are optional, and full
prompts and answers go only with AXIOM_AI_CAPTURE_CONTENT=1."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from r3 import obs
from r3.events import EventLog, trace_scope

ENV = {"AXIOM_TOKEN": "xaat-test", "AXIOM_DATASET": "r3", "AXIOM_ORG_ID": "org-1"}


def test_axiom_is_off_without_a_token_and_a_dataset():
    assert obs.AxiomConfig.from_env({}) is None
    assert obs.AxiomConfig.from_env({"AXIOM_TOKEN": "x"}) is None
    assert obs.AxiomConfig.from_env({"AXIOM_DATASET": "r3"}) is None


def test_axiom_config_reads_the_axiom_variables():
    cfg = obs.AxiomConfig.from_env(ENV)
    assert cfg is not None
    assert (cfg.url, cfg.dataset, cfg.org_id, cfg.capture_content) == ("https://api.axiom.co", "r3", "org-1", False)
    assert obs.AxiomConfig.from_env({**ENV, "AXIOM_AI_CAPTURE_CONTENT": "1"}).capture_content is True
    assert obs.AxiomConfig.from_env({**ENV, "AXIOM_URL": "https://eu.axiom.co/"}).url == "https://eu.axiom.co"


def test_traces_go_to_axioms_otlp_endpoint_with_the_dataset_header():
    endpoint, headers = obs.otlp_target(obs.AxiomConfig.from_env(ENV))
    assert endpoint == "https://api.axiom.co/v1/traces"
    assert headers == {"Authorization": "Bearer xaat-test", "X-Axiom-Dataset": "r3", "X-Axiom-Org-ID": "org-1"}


def _record(msg: str, **extra) -> logging.LogRecord:
    rec = logging.LogRecord("r3.events", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_log_lines_are_batched_into_one_ingest_call():
    sent: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda req: (sent.append(req), httpx.Response(200, json={}))[1])
    shipper = obs.AxiomShipper(obs.AxiomConfig.from_env(ENV), "r3-portal", transport=transport, start=False)
    shipper.emit(_record("Marked job 1 applied", event={"kind": "job.status", "user_id": 3}))
    shipper.emit(_record("second"))
    assert shipper.flush() is True
    assert len(sent) == 1
    req = sent[0]
    assert str(req.url) == "https://api.axiom.co/v1/datasets/r3/ingest"
    assert req.headers["authorization"] == "Bearer xaat-test" and req.headers["x-axiom-org-id"] == "org-1"
    body = json.loads(req.content)
    assert [b["message"] for b in body] == ["Marked job 1 applied", "second"]
    first = body[0]
    assert first["service"] == "r3-portal" and first["level"] == "INFO" and first["logger"] == "r3.events"
    assert first["event"] == {"kind": "job.status", "user_id": 3} and "_time" in first
    assert shipper.flush() is True and len(sent) == 1  # nothing left to send


def test_a_failing_axiom_never_raises():
    def boom(req):
        raise httpx.ConnectError("down")
    shipper = obs.AxiomShipper(obs.AxiomConfig.from_env(ENV), "r3-portal", transport=httpx.MockTransport(boom),
                               start=False)
    shipper.emit(_record("x"))
    assert shipper.flush() is False
    shipper2 = obs.AxiomShipper(obs.AxiomConfig.from_env(ENV), "r3-portal",
                                transport=httpx.MockTransport(lambda r: httpx.Response(403)), start=False)
    shipper2.emit(_record("y"))
    assert shipper2.flush() is False


def test_the_shipper_never_ships_its_own_failure_lines():
    shipper = obs.AxiomShipper(obs.AxiomConfig.from_env(ENV), "r3-portal",
                               transport=httpx.MockTransport(lambda r: httpx.Response(200)), start=False)
    shipper.emit(logging.LogRecord("httpx", logging.INFO, __file__, 1, "HTTP Request: POST axiom", None, None))
    assert shipper.pending() == 0


# --- AI calls ------------------------------------------------------------------


@pytest.fixture
def tracing():
    """Tracing on for one test, with no Axiom; restored afterwards."""
    log = EventLog()
    recorded = []
    original = log.record

    def spy(*a, **k):
        ev = original(*a, **k)
        recorded.append(ev)
        return ev

    log.record = spy  # type: ignore[method-assign]
    handle = obs.setup_tracing("r3-test", None, events=log)
    try:
        yield recorded
    finally:
        handle.shutdown()


def test_every_model_call_becomes_an_ai_event_in_the_requests_trace(tracing):
    agent = Agent(TestModel(custom_output_text="Senior SRE at Acme"), name="r3-resume-reader")
    tid = "e" * 32
    with trace_scope(tid, user_id=3), obs.request_span("POST /api/resume/1/read", tid):
        agent.run_sync("read this")
    calls = [e for e in tracing if e.kind == "ai.call"]
    assert len(calls) == 1
    ev = calls[0]
    assert (ev.actor, ev.user_id, ev.trace_id) == ("ai", 3, tid)
    assert ev.data["agent"] == "r3-resume-reader" and ev.data["response_model"] == "test"
    assert ev.data["input_tokens"] > 0 and "duration_ms" in ev.data
    assert "Senior SRE" not in json.dumps(ev.data)  # content goes to Axiom only, never the table
    assert "r3-resume-reader" in ev.summary


def test_tracing_off_leaves_agents_uninstrumented():
    handle = obs.setup_tracing("r3-test", None, events=EventLog())
    handle.shutdown()
    agent = Agent(TestModel(custom_output_text="ok"))
    assert agent.run_sync("hi").output == "ok"
