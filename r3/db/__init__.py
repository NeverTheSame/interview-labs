"""Package db owns the Postgres connection pool and every query the portal
issues. Handlers never build SQL — they call typed methods on
:class:`Database`."""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

from .labels import (
    HISTORY_DDL,
    LABELS_DDL,
    JobSnapshot,
    Label,
    LabelCounts,
    LabelExport,
    LabelQueries,
    next_job_where,
)
from .jobs import (
    ALLOWED_STATUSES,
    EMPLOYMENT_FAMILIES,
    JOBS_SORT_SQL,
    USER_JOBS_DDL,
    USER_STATUS_SQL,
    Facets,
    FacetValue,
    Job,
    JobFilter,
    JobQueries,
    Page,
    SiteOption,
    Stats,
    build_job_where,
)
from .scout import Combo, Preset, PresetGroup, ScoutQueries, ScoutScope, employment_family_values
from .resumes import RESUME_SOURCES, Resume, ResumeQueries, ResumeSummary
from .resume_drafts import ResumeDraftQueries, StoredDraft
from .job_summaries import JobSummary, JobSummaryQueries
from .events import EventQueries, EventRow
from .settings import SettingsQueries
from .users import AccountNameTaken, User, UserQueries

__all__ = [
    "ALLOWED_STATUSES", "EMPLOYMENT_FAMILIES", "HISTORY_DDL", "JOBS_SORT_SQL", "LABELS_DDL",
    "Combo", "Database", "Facets", "FacetValue",
    "EventRow", "Job", "JobFilter", "JobSummary", "JobSnapshot", "Label", "LabelCounts", "LabelExport", "Page",
    "Preset", "PresetGroup", "ScoutScope", "SiteOption", "Stats", "StoredDraft", "AccountNameTaken", "User",
    "RESUME_SOURCES", "Resume", "ResumeSummary",
    "USER_JOBS_DDL", "USER_STATUS_SQL",
    "build_job_where", "employment_family_values", "next_job_where", "normalize_dsn",
]

# asyncpg forwards DSN query parameters it does not recognise to the server as
# session settings, and Postgres rejects the libpq-only ones (hosted Postgres
# connection strings often carry channel_binding), so they are dropped here
# rather than failing the connect with "unrecognized configuration parameter".
_LIBPQ_ONLY_PARAMS = frozenset({"channel_binding", "connect_timeout", "application_name", "options"})


def normalize_dsn(dsn: str) -> str:
    """Strip DSN parameters asyncpg cannot pass through."""
    parts = urlsplit(dsn)
    if not parts.query:
        return dsn
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in _LIBPQ_ONLY_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(kept)))


class Database(JobQueries, ScoutQueries, SettingsQueries, LabelQueries, UserQueries,
               ResumeQueries, ResumeDraftQueries, JobSummaryQueries, EventQueries):
    """Wraps the asyncpg pool; every query lives on one of the mixins."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def open(cls, dsn: str) -> "Database":
        """Connect to Postgres and verify the connection is usable."""
        # A hosted pooler may drop idle connections; keep the local pool small
        # and short-lived so we don't hand out dead sockets.
        pool = await asyncpg.create_pool(
            normalize_dsn(dsn),
            min_size=1,
            max_size=8,
            max_inactive_connection_lifetime=120,
            command_timeout=60,
        )
        try:
            await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=10)
        except BaseException:
            await pool.close()
            raise
        return cls(pool)

    async def close(self) -> None:
        """Release the pool."""
        await self.pool.close()

    async def ping(self) -> None:
        """Raise if the database is unreachable, for /readyz."""
        await self.pool.fetchval("SELECT 1")
