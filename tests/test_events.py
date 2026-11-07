"""R3's audit trail (r3/events.py, r3/db/events.py): every human action and AI
decision becomes one append-only event, in Postgres and in the log stream
that ships to Axiom."""

from __future__ import annotations

import json
import logging

from r3.db.events import EVENTS_DDL, EventQueries
from r3.events import Event, EventLog, current_trace, new_trace_id, trace_scope


class MemoryStore:
    def __init__(self, fail: bool = False) -> None:
        self.rows: list[Event] = []
        self.fail = fail

    async def insert_events(self, events: list[Event]) -> None:
        if self.fail:
            raise RuntimeError("database is down")
        self.rows.extend(events)


def test_an_event_takes_its_account_and_trace_from_the_current_request():
    log = EventLog()
    with trace_scope("a" * 32, user_id=7):
        ev = log.record("job.status", "Marked job 12 applied", actor="user", subject="job:12",
                        data={"from": "new", "to": "applied"})
    assert (ev.user_id, ev.trace_id, ev.actor, ev.subject) == (7, "a" * 32, "user", "job:12")
    assert ev.data == {"from": "new", "to": "applied"}
    assert current_trace() is None  # the scope ends with the block


def test_an_event_outside_a_request_is_a_system_event_with_its_own_trace():
    ev = EventLog().record("run.finished", "Stats finished")
    assert ev.actor == "system" and ev.user_id is None and len(ev.trace_id) == 32


def test_secrets_never_reach_an_event_and_big_values_are_clipped():
    ev = EventLog().record("http.post", "POST /login", data={
        "password": "hunter22", "api_key": "sk-1", "Authorization": "Bearer x", "session_token": "t",
        "fields": {"username": "dan", "new_password": "p"}, "note": "x" * 5000})
    flat = json.dumps(ev.data)
    for secret in ("hunter22", "sk-1", "Bearer x", '"t"', '"p"'):
        assert secret not in flat
    assert ev.data["fields"]["username"] == "dan"
    assert len(ev.data["note"]) < 2100 and ev.data["note"].endswith("…")


def test_an_oversized_event_keeps_its_summary_but_drops_the_detail():
    ev = EventLog().record("ai.call", "x" * 900, data={f"k{i}": "v" * 1900 for i in range(20)})
    assert len(ev.summary) <= 300
    assert len(json.dumps(ev.data)) <= 8192 and ev.data.get("_clipped") is True


def test_every_event_is_also_a_structured_log_line(caplog):
    caplog.set_level(logging.INFO, logger="r3.events")
    with trace_scope("b" * 32, user_id=3):
        EventLog().record("resume.uploaded", "Uploaded CV", actor="user", subject="resume:1", data={"pages": 2})
    rec = next(r for r in caplog.records if r.name == "r3.events")
    assert rec.getMessage() == "Uploaded CV"
    assert rec.event["kind"] == "resume.uploaded" and rec.event["trace_id"] == "b" * 32
    assert rec.event["user_id"] == 3 and rec.event["data"] == {"pages": 2}


async def test_flush_writes_the_buffer_in_one_batch_and_empties_it():
    store = MemoryStore()
    log = EventLog(store)
    log.record("a", "one")
    log.record("b", "two")
    assert await log.flush() == 2
    assert [e.kind for e in store.rows] == ["a", "b"]
    assert await log.flush() == 0


async def test_a_database_failure_never_reaches_the_caller():
    log = EventLog(MemoryStore(fail=True))
    log.record("a", "one")
    assert await log.flush() == 0  # logged and dropped, never raised


def test_the_buffer_is_bounded():
    log = EventLog(MemoryStore(), max_buffer=3)
    for i in range(5):
        log.record("k", f"e{i}")
    assert [e.summary for e in log.pending()] == ["e2", "e3", "e4"]


def test_trace_ids_are_32_hex_characters():
    tid = new_trace_id()
    assert len(tid) == 32 and int(tid, 16) >= 0


# --- the table --------------------------------------------------------------


class RecordingPool:
    def __init__(self, rows=None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.rows = rows or []

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "OK"

    async def executemany(self, sql, args):
        self.calls.append((sql, tuple(args)))

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self.rows


def _queries(pool: RecordingPool) -> EventQueries:
    q = EventQueries()
    q.pool = pool  # type: ignore[assignment]
    return q


def test_the_events_table_is_append_only_with_no_foreign_keys():
    ddl = " ".join(EVENTS_DDL.split())
    assert "CREATE TABLE IF NOT EXISTS r3_events" in ddl
    assert "REFERENCES" not in ddl
    for col in ("trace_id text", "user_id bigint", "actor text", "kind text", "subject text", "data jsonb"):
        assert col in ddl, col
    for index in ("r3_events_user_at", "r3_events_subject", "r3_events_trace"):
        assert index in ddl, index


async def test_events_are_inserted_in_one_round_trip():
    pool = RecordingPool()
    with trace_scope("c" * 32, user_id=5):
        evs = [EventLog().record("a", "one", subject="job:1", data={"x": 1}), EventLog().record("b", "two")]
    await _queries(pool).insert_events(evs)
    sql, rows = pool.calls[-1]
    assert "INSERT INTO r3_events" in sql
    assert len(rows) == 2 and rows[0][2] == 5 and json.loads(rows[0][-1]) == {"x": 1}


async def test_listing_events_filters_by_account_subject_and_trace():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_events(user_id=5, subject="job:1", limit=20)
    sql, args = pool.calls[-1]
    assert "user_id = $1" in sql and "subject = $2" in sql and "ORDER BY at DESC, id DESC" in sql
    assert args == (5, "job:1", 20)
    await q.list_events(trace_id="d" * 32)
    sql, args = pool.calls[-1]
    assert "trace_id = $1" in sql and args == ("d" * 32, 200)
