"""Queries over ``r3_job_summaries``: the cached AI overview of a job, the
"About this role" text a regular account sees instead of the posting (OQ-6).

One row per job. Only model-written overviews are stored; the extractive
fallback is cheap and is recomputed, so a model key added later upgrades
every job. No foreign key to scraped_jobs on purpose: jobs get pruned, and
the table must not constrain the other application's deletes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from ._base import QueryBase

JOB_SUMMARIES_DDL = """
CREATE TABLE IF NOT EXISTS r3_job_summaries (
    job_id     bigint PRIMARY KEY,
    summary    text NOT NULL,
    model      text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
)"""


class JobSummary(BaseModel):
    """A row of r3_job_summaries."""

    job_id: int
    summary: str
    model: str = ""
    created_at: datetime | None = None


class JobSummaryQueries(QueryBase):
    async def ensure_job_summary_table(self) -> None:
        """Create r3_job_summaries when it does not exist yet."""
        await self.pool.execute(JOB_SUMMARIES_DDL)

    async def get_job_summary(self, job_id: int) -> JobSummary | None:
        row = await self.pool.fetchrow(
            "SELECT job_id, summary, model, created_at FROM r3_job_summaries WHERE job_id = $1", job_id)
        return None if row is None else JobSummary(**dict(row))

    async def save_job_summary(self, job_id: int, summary: str, model: str) -> None:
        await self.pool.execute(
            "INSERT INTO r3_job_summaries (job_id, summary, model, created_at) VALUES ($1, $2, $3, now()) "
            "ON CONFLICT (job_id) DO UPDATE SET summary = EXCLUDED.summary, model = EXCLUDED.model, "
            "created_at = now()", job_id, summary, model)
