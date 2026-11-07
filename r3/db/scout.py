"""The Explore page (``/scout``): curated hunts and single-dimension presets, each with a
live count computed over the same query builder that serves /jobs."""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from ._base import QueryBase
from .jobs import (EMPLOYMENT_FAMILIES, SHARED_STATUS_SQL, USER_JOIN_SQL, USER_STATUS_SQL, FacetValue, JobFilter,
                   build_job_where)


class Preset(BaseModel):
    """One launchable search on the Explore page: a label, the jobs-page query
    it opens, and how many rows currently match."""

    label: str
    hint: str = ""
    query: str  # query string including the leading "?"
    count: int


class PresetGroup(BaseModel):
    """A titled row of presets."""

    title: str
    hint: str = ""
    items: list[Preset] = Field(default_factory=list)


class Combo(BaseModel):
    """A curated multi-filter hunt shown at the top of the Explore page."""

    label: str
    hint: str
    query: str
    count: int


class ScoutScope(BaseModel):
    """Selects which pool the Explore page counts over."""

    status: str = "new"  # "new" (default) or "" for every job

    def scope_sql(self, arg_n: int) -> tuple[str, list[Any]]:
        """Render the scope as a WHERE fragment plus its args."""
        if not self.status:
            return "TRUE", []
        return f"{SHARED_STATUS_SQL} = ${arg_n}", [self.status]

    def base_pairs(self) -> list[tuple[str, str]]:
        """The query-string pairs every preset link inherits."""
        return [("status", self.status)] if self.status else []


def employment_family_values(family: str) -> list[str] | None:
    """Expose the expansion for a family name."""
    return EMPLOYMENT_FAMILIES.get(family)


def _from_facets(vals: list[FacetValue], key: str, link: Callable[..., str]) -> list[Preset]:
    """Turn grouped counts into presets pointing at one filter key."""
    return [Preset(label=v.value, count=v.count, query=link(**{key: v.value})) for v in vals]


# The cents/dollars split in salary_max is normalized the same way the jobs
# filter does it (see build_job_where).
_SAL_EXPR = "(CASE WHEN salary_max >= 1000000 THEN salary_max / 100 ELSE salary_max END)"


class ScoutQueries(QueryBase):
    async def scout_presets(
        self, scope: ScoutScope, *, user_id: int | None = None
    ) -> tuple[list[PresetGroup], int]:
        """Build every preset group with live counts. Runs three grouped
        queries plus one bucket query rather than a count per preset. With a
        ``user_id`` the scope's status is that user's own (r3_user_jobs), the
        same effective status the jobs page filters on."""
        src = "scraped_jobs j"
        if user_id is None:
            where, args = scope.scope_sql(1)
        else:
            # $1 is the overlay join's user; the scope's status follows.
            src = f"scraped_jobs j {USER_JOIN_SQL}"
            where, args = ("TRUE", [user_id]) if not scope.status else (f"{USER_STATUS_SQL} = $2", [user_id, scope.status])
        base = scope.base_pairs()

        def link(**overrides: str) -> str:
            """Build a /jobs query string carrying the scope plus overrides."""
            pairs = dict(base)
            pairs.update({k: v for k, v in overrides.items() if v})
            return "?" + urlencode(pairs)

        total: int = await self.pool.fetchval(f"SELECT count(*) FROM {src} WHERE {where}", *args)

        # --- grouped dimensions ---------------------------------------------

        async def group_counts(expr: str) -> list[FacetValue]:
            # expr is one of two column names fixed in this module, never input.
            rows = await self.pool.fetch(
                f"""SELECT {expr} AS v, count(*) AS n FROM {src}
                    WHERE {where} AND {expr} IS NOT NULL AND {expr} <> ''
                    GROUP BY 1 ORDER BY 2 DESC""",
                *args,
            )
            return [FacetValue(value=r["v"], count=r["n"]) for r in rows]

        roles = await group_counts("role_category")
        locations = await group_counts("location_kind")

        # Tech tags need unnest, so they don't fit group_counts.
        tech_rows = await self.pool.fetch(
            f"""SELECT tag, count(*) AS n FROM {src}, unnest(tech_tags) AS tag
                WHERE {where} GROUP BY 1 ORDER BY 2 DESC LIMIT 18""",
            *args,
        )
        techs = [FacetValue(value=r["tag"], count=r["n"]) for r in tech_rows]

        # --- fixed buckets: employment families, salary bands, freshness ----

        # One pass with FILTER clauses; cheaper and atomic versus a count per band.
        n = len(args)
        b = await self.pool.fetchrow(
            f"""
            SELECT
              count(*) FILTER (WHERE employment = ANY(${n + 1}::text[])) AS emp_contract,
              count(*) FILTER (WHERE employment = ANY(${n + 2}::text[])) AS emp_fulltime,
              count(*) FILTER (WHERE employment = ANY(${n + 3}::text[])) AS emp_parttime,
              count(*) FILTER (WHERE employment = ANY(${n + 4}::text[])) AS emp_internship,
              count(*) FILTER (WHERE salary_period IN ('year','annual') AND {_SAL_EXPR} >= 100000) AS sal100,
              count(*) FILTER (WHERE salary_period IN ('year','annual') AND {_SAL_EXPR} >= 150000) AS sal150,
              count(*) FILTER (WHERE salary_period IN ('year','annual') AND {_SAL_EXPR} >= 200000) AS sal200,
              count(*) FILTER (WHERE salary_period IN ('year','annual') AND {_SAL_EXPR} >= 250000) AS sal250,
              count(*) FILTER (WHERE scraped_at >= now() - interval '1 day')   AS fresh1,
              count(*) FILTER (WHERE scraped_at >= now() - interval '7 days')  AS fresh7,
              count(*) FILTER (WHERE scraped_at >= now() - interval '30 days') AS fresh30,
              count(*) FILTER (WHERE location_kind IN ('remote-us','remote-ww','remote-ca')) AS remote_any,
              count(*) FILTER (WHERE salary IS NOT NULL AND salary <> '') AS has_salary
            FROM {src} WHERE {where}""",
            *args,
            EMPLOYMENT_FAMILIES["contract"],
            EMPLOYMENT_FAMILIES["fulltime"],
            EMPLOYMENT_FAMILIES["parttime"],
            EMPLOYMENT_FAMILIES["internship"],
        )

        # --- assemble -------------------------------------------------------

        groups = [
            PresetGroup(
                title="By role",
                hint="What the classifier tagged the posting as",
                items=_from_facets(roles, "role", link),
            ),
            PresetGroup(
                title="By employment type",
                hint="Case and c2c variants are grouped together",
                items=[
                    Preset(label="Contract", hint="incl. C2C", count=b["emp_contract"], query=link(employment_family="contract")),
                    Preset(label="Full-time", count=b["emp_fulltime"], query=link(employment_family="fulltime")),
                    Preset(label="Part-time", count=b["emp_parttime"], query=link(employment_family="parttime")),
                    Preset(label="Internship", count=b["emp_internship"], query=link(employment_family="internship")),
                ],
            ),
            PresetGroup(
                title="By salary floor",
                hint="Annual postings only — hourly rates have no comparable floor",
                items=[
                    Preset(label="$100k+", count=b["sal100"], query=link(min_salary="100000", sort="salary")),
                    Preset(label="$150k+", count=b["sal150"], query=link(min_salary="150000", sort="salary")),
                    Preset(label="$200k+", count=b["sal200"], query=link(min_salary="200000", sort="salary")),
                    Preset(label="$250k+", count=b["sal250"], query=link(min_salary="250000", sort="salary")),
                    Preset(label="Any stated salary", count=b["has_salary"], query=link(has_salary="1", sort="salary")),
                ],
            ),
            PresetGroup(
                title="By location",
                items=[
                    Preset(label="Remote (any)", hint="US, worldwide or Canada", count=b["remote_any"], query=link(remote="1")),
                    *_from_facets(locations, "location", link),
                ],
            ),
            PresetGroup(
                title="By freshness",
                hint="When the posting was added",
                items=[
                    Preset(label="Last 24 hours", count=b["fresh1"], query=link(days="1")),
                    Preset(label="Last 7 days", count=b["fresh7"], query=link(days="7")),
                    Preset(label="Last 30 days", count=b["fresh30"], query=link(days="30")),
                ],
            ),
            PresetGroup(
                title="By technology",
                hint="Most common tags in this pool",
                items=_from_facets(techs, "tech", link),
            ),
        ]
        return groups, total

    async def scout_combos(self, scope: ScoutScope, *, user_id: int | None = None) -> list[Combo]:
        """Count the curated cross-dimension hunts. Each is a real JobFilter
        run through the same builder the jobs page uses, so a combo can never
        drift from what its link actually shows."""
        st = scope.status
        # A job seeker did not scrape anything; to them these jobs are new.
        today = ("New today", "added in the last 24 hours") if user_id is not None else ("Scraped today", "landed in the last 24h")
        specs: list[tuple[str, str, JobFilter]] = [
            ("Fresh remote AI/ML", "new this week, remote, AI/ML",
             JobFilter(status=st, role_categories=["AI", "ML", "AI/ML"], remote=True, days=7)),
            ("Remote contract, any role", "contract or C2C, remote",
             JobFilter(status=st, employment_family="contract", remote=True)),
            ("High-paying remote", "$200k+ annual, remote",
             JobFilter(status=st, remote=True, min_salary=200000)),
            ("DevOps / SRE remote", "both categories, remote",
             JobFilter(status=st, remote=True, role_categories=["DevOps", "SRE"])),
            ("Kubernetes + remote", "k8s tagged, remote",
             JobFilter(status=st, tech_tag="kubernetes", remote=True)),
            ("Python + AI/ML", "python tagged, AI/ML",
             JobFilter(status=st, tech_tag="python", role_categories=["AI", "ML", "AI/ML"])),
            (*today, JobFilter(status=st, days=1)),
            ("Salary stated, remote", "postings that disclose pay",
             JobFilter(status=st, has_salary=True, remote=True)),
        ]

        out: list[Combo] = []
        for label, hint, f in specs:
            clause, args = build_job_where(f, user_id)
            # With a user, build_job_where binds them as $1 for the overlay join.
            join = "" if user_id is None else f" {USER_JOIN_SQL}"
            n: int = await self.pool.fetchval(
                f"""SELECT count(*) FROM scraped_jobs j{join}
                    JOIN job_sites st ON st.url = j.site_url {clause}""",
                *args,
            )
            out.append(Combo(label=label, hint=hint, count=n, query=f.query_string()))
        return out
