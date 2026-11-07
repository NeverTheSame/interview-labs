"""From a route to the panel's action, against the stub database and the
in-memory run service: jobs lists, stats snapshots (recorded as runs when a
runner is there), and the plain "cannot run" answers."""

from __future__ import annotations

import pytest

from r3 import db
from r3.agent.actions import job_filter, jobs_title, plan_action
from r3.agent.contracts import (
    AdviceAction,
    JobsAction,
    RouteDecision,
    RouteParams,
    StatsAction,
    UnavailableAction,
)
from tests.fakes import FakeRuns
from tests.fakes import test_settings as make_settings
from tests.test_routes import StubDB

MODEL_ID = "openrouter:openrouter/free"

FACETS = db.Facets(
    statuses=[db.FacetValue(value="new", count=40), db.FacetValue(value="applied", count=3)],
    role_categories=[db.FacetValue(value=v, count=10) for v in ("AI/ML", "SRE", "DevOps", "Data")],
    location_kinds=[db.FacetValue(value=v, count=10) for v in ("remote-us", "remote-ca", "remote-ww", "onsite-us")],
    tech_tags=[db.FacetValue(value=v, count=10) for v in ("python", "kubernetes")],
)


def route(intent: str, **params) -> RouteDecision:
    return RouteDecision(intent=intent, params=RouteParams(**params), router="rules", reason="test")


async def plan(decision: RouteDecision, stub: StubDB | None = None, runs: FakeRuns | None = None, *,
               settings=None, text: str = "please do the thing", facets=FACETS):
    return await plan_action(decision, database=stub or StubDB(), runs=runs or FakeRuns(),
                             settings=settings or make_settings(), model_id=MODEL_ID, text=text, facets=facets)


class DownStub(StubDB):
    async def list_jobs(self, f, page, size, *, user_id=None):
        raise ConnectionError("down")

    async def get_job(self, job_id, *, user_id=None):
        raise ConnectionError("down")

    async def overview(self, *, user_id=None):
        raise ConnectionError("down")


async def test_advice_needs_nothing():
    assert isinstance(await plan(route("advice")), AdviceAction)


async def test_jobs_lists_with_the_filter_query():
    stub = StubDB()
    action = await plan(route("jobs", roles=["site reliability"], remote=True, limit=5), stub)
    assert isinstance(action, JobsAction)
    f = stub.last_filter
    assert f.status == "new" and f.role_category == "SRE" and f.remote and stub.last_page == (1, 5)
    assert action.title == "101 new remote SRE jobs"
    assert action.filter_query == "?status=new&role=SRE&remote=1"
    assert action.total == 101 and action.jobs[0].id == 7 and action.jobs[0].position == "Senior ML Engineer"


async def test_jobs_filter_details():
    stub = StubDB()
    decision = route("jobs", roles=["machine learning", "Security"], status="applied", tech=["k8s"], days=7,
                     min_salary=200_000, employment="c2c", location="Canada", remote=True)
    action = await plan(decision, stub)
    f = stub.last_filter
    assert f.status == "applied" and f.role_categories == [] and f.role_category == "AI/ML"
    assert f.search == "Security"  # a role the database does not have is searched for
    assert f.tech_tag == "kubernetes" and f.days == 7 and f.min_salary == 200_000 and f.sort == "salary"
    assert f.employment_family == "contract" and f.location_kind == "remote-ca"
    assert action.title == ('101 applied remote Canada contract AI/ML jobs with kubernetes, matching "Security", '
                            'paying $200k+, from the last 7 days')


async def test_jobs_by_id_lists_exactly_those():
    action = await plan(route("jobs", job_ids=[7, 404]))
    assert isinstance(action, JobsAction) and [card.id for card in action.jobs] == [7] and action.total == 1
    assert action.title == "#7 Senior ML Engineer; not found: #404" and action.filter_query == ""


def test_job_filter_mapping():
    f = job_filter(RouteParams(roles=["sre", "devops"], status="all", tech=["rust"]), FACETS)
    assert f.status == "" and f.role_categories == ["SRE", "DevOps"] and f.tech_tag == "" and f.search == "rust"
    assert job_filter(RouteParams(location="US", remote=True), None).location_kind == "remote-us"
    assert job_filter(RouteParams(location="US"), FACETS).location_kind == ""  # remote-us or onsite-us: ambiguous
    assert job_filter(RouteParams(location="worldwide", remote=True), FACETS).location_kind == "remote-ww"
    assert job_filter(RouteParams(employment="Full-time"), FACETS).employment_family == "fulltime"


def test_jobs_title_singular():
    assert jobs_title(db.JobFilter(status="skipped"), 1) == "1 skipped job"
    assert jobs_title(db.JobFilter(), 3) == "3 jobs"


async def test_stats_snapshot_is_recorded_as_a_run():
    runs = FakeRuns()
    action = await plan(route("stats", days=7), runs=runs, text="how did the week go?")
    assert isinstance(action, StatsAction)
    assert action.snapshot.window_days == 7 and action.snapshot.total_jobs == 100 and action.snapshot.new_jobs == 40
    assert action.run is not None and action.run.kind == "stats"
    request = runs.requests[0]
    assert request.params.days == 7 and request.source == "chat" and request.model == MODEL_ID
    assert request.note == "how did the week go?"


async def test_stats_without_a_runner_still_answers():
    action = await plan(route("stats"), runs=FakeRuns(configured=False))
    assert isinstance(action, StatsAction) and action.run is None and action.snapshot.window_days == 30


@pytest.mark.parametrize("intent, params", [("jobs", {}), ("jobs", {"job_ids": [7]}), ("stats", {})])
async def test_database_failures_are_unavailable(intent, params):
    action = await plan(route(intent, **params), DownStub())
    assert isinstance(action, UnavailableAction) and action.intent == intent
    assert action.reason == "The database did not answer" and "down" not in action.hint


async def test_facets_are_fetched_when_not_given():
    stub = StubDB()
    await plan(route("jobs", roles=["ai"]), stub, facets=None)
    assert stub.last_filter.role_category == "AI/ML"  # the stub's only AI-ish category
