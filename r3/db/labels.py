"""Queries over ``r3_job_labels`` and ``r3_job_labels_history``: the
operator's labels on scraped postings (``/label``), which R3 owns and
creates on demand like ``r3_settings``. ``scraped_jobs`` is only read here.

One row per (job_id, rubric_version). Saving a label for a job that already
has one overwrites the row after copying the old one into the history
table, in one transaction, so a re-label never loses the earlier answer.

``job_id`` deliberately carries no foreign key to ``scraped_jobs``: old
postings get deleted, and a constraint would either block that or cascade
the labels away. The export
freezes the job's fields next to the answers, so a label stays usable after
its posting is gone.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Sequence

from pydantic import BaseModel, Field

from ._base import QueryBase
from .jobs import _JOB_COLUMNS, LIVE_SQL, Job, JobFilter, build_job_where

LABELS_DDL = """
CREATE TABLE IF NOT EXISTS r3_job_labels (
    id             bigserial PRIMARY KEY,
    job_id         bigint NOT NULL,
    rubric_version text NOT NULL,
    answers        jsonb NOT NULL,
    note           text NOT NULL DEFAULT '',
    labeled_at     timestamptz NOT NULL DEFAULT now(),
    labeler        text NOT NULL DEFAULT 'operator',
    session_id     text NOT NULL DEFAULT '',
    seconds_spent  integer NOT NULL DEFAULT 0,
    UNIQUE (job_id, rubric_version)
)"""

HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS r3_job_labels_history (
    id             bigserial PRIMARY KEY,
    label_id       bigint NOT NULL,
    job_id         bigint NOT NULL,
    rubric_version text NOT NULL,
    answers        jsonb NOT NULL,
    note           text NOT NULL DEFAULT '',
    labeled_at     timestamptz NOT NULL,
    labeler        text NOT NULL,
    session_id     text NOT NULL DEFAULT '',
    seconds_spent  integer NOT NULL DEFAULT 0,
    replaced_at    timestamptz NOT NULL DEFAULT now()
)"""

HISTORY_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS r3_job_labels_history_job_idx ON r3_job_labels_history (job_id, rubric_version)"""

_LABEL_COLUMNS = "id, job_id, rubric_version, answers, note, labeled_at, labeler, session_id, seconds_spent"

# What the export freezes next to the answers: every column the portal shows
# plus the full description, so the eval never has to read the live table.
_SNAPSHOT_FIELDS = ("id", "site_url", "site_name", "job_url", "apply_url", "position", "company", "location",
                    "posted_date", "scraped_at", "role_category", "employment", "salary", "salary_min", "salary_max",
                    "location_kind", "tech_tags", "status", "description")


class Label(BaseModel):
    """A row of r3_job_labels."""

    id: int
    job_id: int
    rubric_version: str
    answers: dict[str, Any]
    note: str = ""
    labeled_at: datetime
    labeler: str
    session_id: str = ""
    seconds_spent: int = 0


class LabelCounts(BaseModel):
    """The counter on the labeling page."""

    today: int
    total: int
    remaining: int  # unlabeled jobs in the current filter not yet shown this session


class JobSnapshot(BaseModel):
    """The job's fields as they were when the labels were exported."""

    id: int
    site_url: str
    site_name: str = ""
    job_url: str
    apply_url: str = ""
    position: str
    company: str = ""
    location: str = ""
    posted_date: datetime | date | None = None
    scraped_at: datetime
    role_category: str = ""
    employment: str = ""
    salary: str = ""
    salary_min: int | None = None
    salary_max: int | None = None
    location_kind: str = ""
    tech_tags: list[str] = Field(default_factory=list)
    status: str
    description: str = ""


class LabelExport(BaseModel):
    """One line of the JSONL export: the label and the job frozen beside it.
    ``job`` is None when the posting has since been deleted."""

    rubric_version: str
    job_id: int
    job: JobSnapshot | None
    answers: dict[str, Any]
    note: str = ""
    labeled_at: datetime
    labeler: str
    session_id: str = ""
    seconds_spent: int = 0


def next_job_where(f: JobFilter, rubric_version: str, seen: Sequence[int]) -> tuple[str, list[Any]]:
    """The WHERE clause for the labeling queue: the job filter, no label
    under ``rubric_version`` yet, and none of the ids shown this session."""
    clause, args = build_job_where(f)
    parts = [clause.removeprefix("WHERE ")] if clause else []
    args = list(args)
    args.append(rubric_version)
    parts.append(f"NOT EXISTS (SELECT 1 FROM r3_job_labels l WHERE l.job_id = j.id AND l.rubric_version = ${len(args)})")
    if seen:
        args.append([int(i) for i in seen])
        parts.append(f"NOT (j.id = ANY(${len(args)}::bigint[]))")
    return "WHERE " + " AND ".join(parts), args


# Newest live postings first, then everything else still unlabeled; the id
# breaks ties so the order is stable between requests.
NEXT_JOB_ORDER = f"{LIVE_SQL} DESC, j.scraped_at DESC, j.id DESC"


def _answers(raw: Any) -> dict[str, Any]:
    """asyncpg hands jsonb back as text unless a codec is set; accept both."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    return dict(raw) if isinstance(raw, dict) else {}


def _label(row: Any) -> Label:
    data = dict(row)
    data["answers"] = _answers(data["answers"])
    return Label(**data)


class LabelQueries(QueryBase):
    async def ensure_labels_tables(self) -> None:
        """Create the two label tables when they do not exist yet."""
        await self.pool.execute(LABELS_DDL)
        await self.pool.execute(HISTORY_DDL)
        await self.pool.execute(HISTORY_INDEX_DDL)

    async def get_label(self, job_id: int, rubric_version: str) -> Label | None:
        row = await self.pool.fetchrow(
            f"SELECT {_LABEL_COLUMNS} FROM r3_job_labels WHERE job_id = $1 AND rubric_version = $2",
            job_id, rubric_version)
        return _label(row) if row else None

    async def save_label(self, job_id: int, rubric_version: str, answers: dict[str, Any], note: str, *,
                         labeler: str, session_id: str, seconds_spent: int) -> tuple[Label, bool]:
        """Insert or overwrite the label. Returns the saved row and whether it
        replaced an earlier one (which is now in the history table)."""
        payload = json.dumps(answers, separators=(",", ":"), sort_keys=True)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                previous = await conn.fetchrow(
                    "SELECT id FROM r3_job_labels WHERE job_id = $1 AND rubric_version = $2 FOR UPDATE",
                    job_id, rubric_version)
                if previous is not None:
                    await conn.execute(
                        """INSERT INTO r3_job_labels_history
                               (label_id, job_id, rubric_version, answers, note, labeled_at, labeler, session_id,
                                seconds_spent)
                           SELECT id, job_id, rubric_version, answers, note, labeled_at, labeler, session_id,
                                  seconds_spent
                           FROM r3_job_labels WHERE id = $1""",
                        previous["id"])
                row = await conn.fetchrow(
                    f"""INSERT INTO r3_job_labels (job_id, rubric_version, answers, note, labeler, session_id,
                                                   seconds_spent)
                        VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7)
                        ON CONFLICT (job_id, rubric_version) DO UPDATE
                            SET answers = EXCLUDED.answers, note = EXCLUDED.note, labeled_at = now(),
                                labeler = EXCLUDED.labeler, session_id = EXCLUDED.session_id,
                                seconds_spent = EXCLUDED.seconds_spent
                        RETURNING {_LABEL_COLUMNS}""",
                    job_id, rubric_version, payload, note, labeler, session_id, seconds_spent)
        return _label(row), previous is not None

    async def next_job_to_label(self, f: JobFilter, rubric_version: str, seen: Sequence[int]) -> Job | None:
        """The next posting to show: the newest live one matching the
        filter that has no label yet and was not shown this session, then
        the rest of the unlabeled pool. With its full description."""
        clause, args = next_job_where(f, rubric_version, seen)
        row = await self.pool.fetchrow(
            f"""SELECT {_JOB_COLUMNS}, COALESCE(j.description, '') AS description
                FROM scraped_jobs j JOIN job_sites s ON s.url = j.site_url
                {clause}
                ORDER BY {NEXT_JOB_ORDER}
                LIMIT 1""",
            *args)
        return Job(**dict(row)) if row else None

    async def label_counts(self, f: JobFilter, rubric_version: str, seen: Sequence[int], since: datetime) -> LabelCounts:
        """Labels made since ``since`` (the labeler's local midnight), all
        labels under this rubric, and what is left in the current filter."""
        counted = await self.pool.fetchrow(
            """SELECT count(*) AS total, count(*) FILTER (WHERE labeled_at >= $2) AS today
               FROM r3_job_labels WHERE rubric_version = $1""",
            rubric_version, since)
        clause, args = next_job_where(f, rubric_version, seen)
        remaining: int = await self.pool.fetchval(
            f"SELECT count(*) FROM scraped_jobs j JOIN job_sites s ON s.url = j.site_url {clause}", *args)
        return LabelCounts(today=counted["today"], total=counted["total"], remaining=remaining)

    async def export_labels(self, rubric_version: str) -> list[LabelExport]:
        """Every label under the rubric, oldest first, with the job's current
        fields frozen beside it (``job`` is None for a deleted posting)."""
        rows = await self.pool.fetch(
            f"""SELECT l.id AS label_id, l.job_id, l.rubric_version, l.answers, l.note, l.labeled_at, l.labeler,
                       l.session_id, l.seconds_spent, j.id AS job_row_id,
                       {_JOB_COLUMNS}, COALESCE(j.description, '') AS description
                FROM r3_job_labels l
                LEFT JOIN scraped_jobs j ON j.id = l.job_id
                LEFT JOIN job_sites s ON s.url = j.site_url
                WHERE l.rubric_version = $1
                ORDER BY l.labeled_at, l.id""",
            rubric_version)
        out: list[LabelExport] = []
        for r in rows:
            data = dict(r)
            job = None
            if data.get("job_row_id") is not None:
                snapshot = {k: data.get(k) for k in _SNAPSHOT_FIELDS}
                snapshot["site_name"] = snapshot.get("site_name") or ""
                job = JobSnapshot(**snapshot)
            out.append(LabelExport(
                rubric_version=data["rubric_version"], job_id=data["job_id"], job=job,
                answers=_answers(data["answers"]), note=data["note"], labeled_at=data["labeled_at"],
                labeler=data["labeler"], session_id=data["session_id"], seconds_spent=data["seconds_spent"]))
        return out
