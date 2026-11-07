"""The jobs query builder: parameterization, whitelists and the filter <->
query-string round trip. No database needed."""

from urllib.parse import parse_qs

import pytest

from r3.db import EMPLOYMENT_FAMILIES, JOBS_SORT_SQL, JobFilter, build_job_where


@pytest.mark.parametrize(
    "name, f, want_parts, want_args",
    [
        ("search spans position, company and description",
         JobFilter(search="kafka"),
         ["j.position ILIKE $1", "j.company ILIKE $1", "j.description ILIKE $1"], 1),
        ("status and role combine with AND",
         JobFilter(status="new", role_category="AI/ML"),
         ["THEN 'new' ELSE 'expired' END = $1", "AND", "j.role_category = $2"], 2),
        ("role list supersedes single role",
         JobFilter(role_category="DevOps", role_categories=["DevOps", "SRE"]),
         ["j.role_category = ANY($1)"], 1),
        ("employment family supersedes exact employment",
         JobFilter(employment="contract", employment_family="contract"),
         ["j.employment = ANY($1)"], 1),
        ("remote expands to the three remote kinds",
         JobFilter(remote=True),
         ["remote-us", "remote-ww", "remote-ca"], 0),
        ("min salary normalizes cents and restricts to annual",
         JobFilter(min_salary=150000),
         ["salary_period IN ('year', 'annual')", "salary_max / 100", ">= $1"], 1),
        ("tech tag uses the array containment operator",
         JobFilter(tech_tag="kubernetes"),
         ["j.tech_tags @> ARRAY[$1]::text[]"], 1),
        ("days uses a parameterized interval",
         JobFilter(days=7),
         ["make_interval(days => $1)"], 1),
    ],
)
def test_build_job_where(name, f, want_parts, want_args):
    clause, args = build_job_where(f)
    assert clause.startswith("WHERE "), name
    for part in want_parts:
        assert part in clause, f"{name}: clause missing {part!r}\n{clause}"
    assert len(args) == want_args, name


def test_no_filters_produces_no_clause():
    clause, args = build_job_where(JobFilter())
    assert clause == "" and args == []


def test_unknown_employment_family_is_ignored_not_injected():
    clause, args = build_job_where(JobFilter(employment_family="bogus"))
    assert clause == "" and args == []


def test_build_job_where_parameterizes_input():
    """A filter value must never reach the SQL text itself — it belongs in args."""
    evil = "'; DROP TABLE scraped_jobs; --"
    clause, args = build_job_where(
        JobFilter(search=evil, status=evil, role_category=evil, site_url=evil, tech_tag=evil, employment=evil)
    )
    assert "DROP TABLE" not in clause
    assert any(isinstance(a, str) and "DROP TABLE" in a for a in args)


def test_placeholders_are_numbered_consecutively():
    clause, args = build_job_where(JobFilter(status="new", role_category="SRE", tech_tag="go", days=3))
    for i in range(1, len(args) + 1):
        assert f"${i}" in clause
    assert f"${len(args) + 1}" not in clause


def test_sort_whitelist():
    # An unrecognized sort key must fall back, never interpolate.
    for bad in ("", "bogus", "id; DROP TABLE x"):
        assert bad not in JOBS_SORT_SQL
    for key, sql in JOBS_SORT_SQL.items():
        assert sql, key


def test_query_string_round_trip():
    f = JobFilter(
        status="new", role_categories=["DevOps", "SRE"], remote=True, min_salary=200000,
        days=7, tech_tag="kubernetes", employment_family="contract", has_salary=True,
    )
    v = parse_qs(f.query_string().removeprefix("?"), keep_blank_values=True)
    assert v["role"] == ["DevOps", "SRE"]
    for key, want in {
        "status": "new", "remote": "1", "min_salary": "200000", "days": "7",
        "tech": "kubernetes", "employment_family": "contract", "has_salary": "1",
        # A salary floor is meaningless unsorted, so it forces the sort.
        "sort": "salary",
    }.items():
        assert v[key] == [want], key


def test_employment_families_cover_known_spellings():
    # These are the exact values present in scraped_jobs.employment; if a
    # scraper starts writing a new spelling this test is where it should fail.
    want = {
        "contract": "contract", "Contract": "contract", "c2c": "contract",
        "Contract (12 Months)": "contract",
        "fulltime": "fulltime", "Full-time": "fulltime",
        "parttime": "parttime", "Part-time": "parttime",
        "internship": "internship",
    }
    got = {v: fam for fam, vals in EMPLOYMENT_FAMILIES.items() for v in vals}
    assert got == want
