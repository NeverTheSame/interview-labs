"""Queries over ``r3_resume_drafts``: the resume builder's one in-progress
draft per account (``/resume/build``), so a reload or a new sign-in picks
the interview up where it stopped.

``draft`` is a :class:`r3.resume.builder.ResumeDraft` and ``history`` the
list of :class:`r3.resume.builder.ChatEntry`, both as JSON. Saving the
draft to the library (``r3_resumes``) or starting over deletes the row.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ._base import QueryBase

RESUME_DRAFTS_DDL = """
CREATE TABLE IF NOT EXISTS r3_resume_drafts (
    user_id     bigint PRIMARY KEY,
    draft       jsonb NOT NULL,
    history     jsonb NOT NULL,
    updated_at  timestamptz DEFAULT now()
)"""


class StoredDraft(BaseModel):
    """A row of r3_resume_drafts; the JSON stays raw here and is validated
    by the builder routes."""

    user_id: int
    draft: dict[str, Any]
    history: list[dict[str, Any]]
    updated_at: datetime | None = None


def _json(value: Any) -> Any:
    # asyncpg hands jsonb back as text unless a codec is registered.
    return json.loads(value) if isinstance(value, str) else value


class ResumeDraftQueries(QueryBase):
    async def ensure_resume_draft_tables(self) -> None:
        """Create r3_resume_drafts when it does not exist yet."""
        await self.pool.execute(RESUME_DRAFTS_DDL)

    async def get_resume_draft(self, user_id: int) -> StoredDraft | None:
        row = await self.pool.fetchrow(
            "SELECT user_id, draft, history, updated_at FROM r3_resume_drafts WHERE user_id = $1", user_id)
        if row is None:
            return None
        return StoredDraft(user_id=row["user_id"], draft=_json(row["draft"]), history=_json(row["history"]),
                           updated_at=row["updated_at"])

    async def save_resume_draft(self, user_id: int, *, draft: dict[str, Any], history: list[dict[str, Any]]) -> None:
        await self.pool.execute(
            "INSERT INTO r3_resume_drafts (user_id, draft, history, updated_at) "
            "VALUES ($1, $2::jsonb, $3::jsonb, now()) "
            "ON CONFLICT (user_id) DO UPDATE SET draft = EXCLUDED.draft, history = EXCLUDED.history, "
            "updated_at = now()", user_id, json.dumps(draft), json.dumps(history))

    async def delete_resume_draft(self, user_id: int) -> bool:
        status = await self.pool.execute("DELETE FROM r3_resume_drafts WHERE user_id = $1", user_id)
        return status.endswith(" 1")
