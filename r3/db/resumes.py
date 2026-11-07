"""Queries over ``r3_resumes``: the job seeker's resume library (``/resume``),
which R3 owns and creates on demand like the label tables.

Every method takes the account's ``user_id`` and puts it in the WHERE
clause, so one account can never read or change another account's rows; a
resume id that belongs to someone else behaves exactly like a missing one.

Each account has at most one primary resume, the one R3 matches jobs
against. The first resume an account adds becomes primary; deleting
the primary promotes the most recently updated remaining one. A partial
unique index enforces "at most one" in the database, and the writes that
move the flag take a per-account advisory lock so two concurrent requests
cannot both decide they are first.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator

from ._base import QueryBase

# Where a resume came from. 'builder' is written by the interview builder,
# 'chat' by a resume made from a background described in the Assistant.
RESUME_SOURCES = ("upload", "paste", "builder", "chat")

RESUMES_DDL = """
CREATE TABLE IF NOT EXISTS r3_resumes (
    id          bigserial PRIMARY KEY,
    user_id     bigint NOT NULL,
    title       text NOT NULL,
    source      text NOT NULL,
    filename    text NOT NULL DEFAULT '',
    content     text NOT NULL,
    is_primary  boolean NOT NULL DEFAULT false,
    created_at  timestamptz DEFAULT now(),
    updated_at  timestamptz DEFAULT now()
)"""

RESUMES_USER_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS r3_resumes_user_idx ON r3_resumes (user_id)"""

RESUMES_PRIMARY_INDEX_DDL = """
CREATE UNIQUE INDEX IF NOT EXISTS r3_resumes_one_primary_idx ON r3_resumes (user_id) WHERE is_primary"""

# How an uploaded resume was read (r3.resume.parse): the parsed draft, the
# engine, whether the clean layout was proved faithful and, when it was not,
# the original text the follow-up answers are added to. Added after the table
# shipped, hence ALTER rather than CREATE.
RESUMES_STRUCTURE_DDL = """
ALTER TABLE r3_resumes ADD COLUMN IF NOT EXISTS structure jsonb"""

_SUMMARY_COLUMNS = ("id, user_id, title, source, filename, is_primary, created_at, updated_at, "
                    "length(content) AS chars")
_RESUME_COLUMNS = _SUMMARY_COLUMNS + ", content, structure"

_PROMOTION_ORDER = "updated_at DESC NULLS LAST, id DESC"


class ResumeSummary(BaseModel):
    """A row of r3_resumes without its text, for the list page."""

    id: int
    user_id: int
    title: str
    source: str
    filename: str = ""
    is_primary: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    chars: int = 0


class Resume(ResumeSummary):
    """A full row of r3_resumes."""

    content: str
    structure: dict[str, Any] | None = None

    @field_validator("structure", mode="before")
    @classmethod
    def _parse_structure(cls, raw: Any) -> Any:
        # asyncpg hands jsonb back as text unless a codec is set.
        return json.loads(raw) if isinstance(raw, str) else raw


def _lock_sql() -> str:
    # Serializes the flag-moving writes of one account; the key is namespaced
    # so it cannot collide with other advisory locks on the same database.
    # $1 is cast to bigint first so asyncpg binds the int user_id, not text.
    return "SELECT pg_advisory_xact_lock(hashtextextended('r3_resumes:' || $1::bigint::text, 0))"


class ResumeQueries(QueryBase):
    async def ensure_resume_tables(self) -> None:
        """Create r3_resumes and its indexes when they do not exist yet."""
        await self.pool.execute(RESUMES_DDL)
        await self.pool.execute(RESUMES_USER_INDEX_DDL)
        await self.pool.execute(RESUMES_PRIMARY_INDEX_DDL)
        await self.pool.execute(RESUMES_STRUCTURE_DDL)

    async def list_resumes(self, user_id: int) -> list[ResumeSummary]:
        """The account's resumes, primary first, then most recently updated."""
        rows = await self.pool.fetch(
            f"SELECT {_SUMMARY_COLUMNS} FROM r3_resumes WHERE user_id = $1 "
            "ORDER BY is_primary DESC, updated_at DESC NULLS LAST, id DESC", user_id)
        return [ResumeSummary(**dict(r)) for r in rows]

    async def get_resume(self, user_id: int, resume_id: int) -> Resume | None:
        row = await self.pool.fetchrow(
            f"SELECT {_RESUME_COLUMNS} FROM r3_resumes WHERE id = $2 AND user_id = $1", user_id, resume_id)
        return Resume(**dict(row)) if row else None

    async def primary_resume(self, user_id: int) -> Resume | None:
        row = await self.pool.fetchrow(
            f"SELECT {_RESUME_COLUMNS} FROM r3_resumes WHERE user_id = $1 AND is_primary", user_id)
        return Resume(**dict(row)) if row else None

    async def create_resume(self, user_id: int, *, title: str, source: str, content: str,
                            filename: str = "") -> Resume:
        """Insert a resume; it becomes primary when the account has none."""
        if source not in RESUME_SOURCES:
            raise ValueError(f"unknown resume source {source!r}")
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(_lock_sql(), user_id)
            row = await conn.fetchrow(
                "INSERT INTO r3_resumes (user_id, title, source, filename, content, is_primary) "
                "VALUES ($1, $2, $3, $4, $5, "
                "NOT EXISTS (SELECT 1 FROM r3_resumes WHERE user_id = $1 AND is_primary)) "
                f"RETURNING {_RESUME_COLUMNS}",
                user_id, title, source, filename, content)
        return Resume(**dict(row))

    async def set_resume_structure(self, user_id: int, resume_id: int, *, structure: dict[str, Any],
                                   content: str) -> Resume | None:
        """Store how a resume was read and the text to keep; None when the
        account has no such resume."""
        row = await self.pool.fetchrow(
            "UPDATE r3_resumes SET structure = $3::jsonb, content = $4, updated_at = now() "
            f"WHERE id = $2 AND user_id = $1 RETURNING {_RESUME_COLUMNS}",
            user_id, resume_id, json.dumps(structure), content)
        return Resume(**dict(row)) if row else None

    async def count_recent_uploads(self, user_id: int, since: datetime) -> int:
        """Uploads and chat-made resumes (each one a read) the account made
        since ``since``: the hourly limit."""
        return int(await self.pool.fetchval(
            "SELECT count(*) FROM r3_resumes WHERE user_id = $1 AND source IN ('upload', 'chat') "
            "AND created_at >= $2",
            user_id, since) or 0)

    async def update_resume(self, user_id: int, resume_id: int, *, title: str, content: str) -> Resume | None:
        """Replace the title and text; None when the account has no such resume."""
        row = await self.pool.fetchrow(
            "UPDATE r3_resumes SET title = $3, content = $4, updated_at = now() "
            f"WHERE id = $2 AND user_id = $1 RETURNING {_RESUME_COLUMNS}",
            user_id, resume_id, title, content)
        return Resume(**dict(row)) if row else None

    async def delete_resume(self, user_id: int, resume_id: int) -> bool:
        """Delete a resume, promoting another when it was the primary one.
        False when the account has no such resume."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(_lock_sql(), user_id)
            was_primary: Any = await conn.fetchval(
                "DELETE FROM r3_resumes WHERE id = $2 AND user_id = $1 RETURNING is_primary", user_id, resume_id)
            if was_primary is None:
                return False
            if was_primary:
                await conn.execute(
                    "UPDATE r3_resumes SET is_primary = true WHERE id = "
                    f"(SELECT id FROM r3_resumes WHERE user_id = $1 ORDER BY {_PROMOTION_ORDER} LIMIT 1)",
                    user_id)
        return True

    async def set_primary_resume(self, user_id: int, resume_id: int) -> bool:
        """Make one resume the account's only primary. False when the
        account has no such resume (nothing changes then)."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(_lock_sql(), user_id)
            if await conn.fetchval("SELECT 1 FROM r3_resumes WHERE id = $2 AND user_id = $1",
                                   user_id, resume_id) is None:
                return False
            # Clear first: the partial unique index forbids two primaries even
            # for the instant between the two statements.
            await conn.execute("UPDATE r3_resumes SET is_primary = false "
                               "WHERE user_id = $1 AND is_primary AND id <> $2", user_id, resume_id)
            await conn.execute("UPDATE r3_resumes SET is_primary = true WHERE id = $2 AND user_id = $1",
                               user_id, resume_id)
        return True
