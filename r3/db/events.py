"""Queries over ``r3_events``: R3's append-only audit trail (r3/events.py).

Rows are only ever inserted. No foreign keys on purpose: accounts, jobs and
resumes can be deleted or pruned, and their history must outlive them.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from r3.events import Event

from ._base import QueryBase

EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS r3_events (
    id       bigserial PRIMARY KEY,
    at       timestamptz NOT NULL DEFAULT now(),
    trace_id text NOT NULL DEFAULT '',
    user_id  bigint,
    actor    text NOT NULL,
    kind     text NOT NULL,
    subject  text NOT NULL DEFAULT '',
    summary  text NOT NULL DEFAULT '',
    data     jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS r3_events_user_at ON r3_events (user_id, at DESC);
CREATE INDEX IF NOT EXISTS r3_events_subject ON r3_events (subject, at DESC) WHERE subject <> '';
CREATE INDEX IF NOT EXISTS r3_events_trace ON r3_events (trace_id) WHERE trace_id <> ''"""

_INSERT = ("INSERT INTO r3_events (at, trace_id, user_id, actor, kind, subject, summary, data) "
           "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)")

_COLUMNS = "id, at, trace_id, user_id, actor, kind, subject, summary, data"


class EventRow(BaseModel):
    """A stored event."""

    id: int
    at: datetime
    trace_id: str
    user_id: int | None
    actor: str
    kind: str
    subject: str
    summary: str
    data: dict[str, Any]


def _row(r: Any) -> EventRow:
    d = dict(r)
    if isinstance(d.get("data"), str):
        d["data"] = json.loads(d["data"])
    return EventRow(**d)


class EventQueries(QueryBase):
    async def ensure_events_table(self) -> None:
        """Create r3_events and its indexes when they do not exist yet."""
        await self.pool.execute(EVENTS_DDL)

    async def insert_events(self, events: list[Event]) -> None:
        await self.pool.executemany(_INSERT, [
            (e.at, e.trace_id, e.user_id, e.actor, e.kind, e.subject, e.summary, json.dumps(e.data, default=str))
            for e in events])

    async def list_events(self, *, user_id: int | None = None, subject: str = "", trace_id: str = "",
                          kind_prefix: str = "", limit: int = 200) -> list[EventRow]:
        """The newest events matching every filter given."""
        where: list[str] = []
        args: list[Any] = []
        for col, value in (("user_id", user_id), ("subject", subject or None), ("trace_id", trace_id or None)):
            if value is not None:
                args.append(value)
                where.append(f"{col} = ${len(args)}")
        if kind_prefix:
            args.append(kind_prefix + "%")
            where.append(f"kind LIKE ${len(args)}")
        args.append(limit)
        sql = (f"SELECT {_COLUMNS} FROM r3_events" + (" WHERE " + " AND ".join(where) if where else "")
               + f" ORDER BY at DESC, id DESC LIMIT ${len(args)}")
        return [_row(r) for r in await self.pool.fetch(sql, *args)]
