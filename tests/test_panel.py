"""The panel's suggestion rules (pure and deterministic) and the panel
assembly over the stub database and a fake run service."""

from datetime import datetime, timedelta, timezone

from r3.panel import build_suggestions, panel_state
from r3.runs.models import RunRequest, RunsInfo, RunStats, StatsParams, StatsSnapshot, WorkflowRun
from tests import fakes
from tests.test_routes import StubDB

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
TEMPORAL = RunsInfo(backend="temporal", configured=True, namespace="r3", task_queue="r3-controller")
NO_RUNS = RunsInfo(backend="none", configured=False, reason="Runs are switched off (R3_RUNS_BACKEND=none).")


def snapshot(**kw) -> StatsSnapshot:
    return StatsSnapshot(captured_at=NOW, **kw)


def run(kind: str, status: str, hours_ago: float, error: str = "") -> WorkflowRun:
    started = NOW - timedelta(hours=hours_ago)
    closed = None if status in ("running", "queued") else started + timedelta(minutes=2)
    return WorkflowRun(id=f"r3-{kind}-{hours_ago}", kind=kind, status=status, started_at=started, closed_at=closed,
                       error=error)


FRESH_STATS = [run("stats", "completed", 1)]


def ids(suggestions) -> list[str]:
    return [s.id for s in suggestions]


def test_quiet_pipeline_suggests_nothing():
    assert build_suggestions(snapshot(), TEMPORAL, FRESH_STATS) == []


def test_unreviewed_jobs_offer_review_and_a_short_list():
    review, top = build_suggestions(snapshot(new_jobs=1234), TEMPORAL, FRESH_STATS)
    assert (review.kind, review.href, review.tone, review.title) == (
        "link", "/jobs?status=new", "accent", "Review 1,234 unreviewed jobs")
    assert (top.kind, top.title, top.prompt) == (
        "prompt", "Show the top 3 newest jobs", "Show me the top 3 newest unreviewed jobs")
    assert build_suggestions(snapshot(new_jobs=1), TEMPORAL, FRESH_STATS)[0].title == "Review 1 unreviewed job"


def test_a_recent_failed_run_links_to_it():
    failed = run("stats", "failed", 2, error="Statistics failed: the database did not answer")
    (s,) = build_suggestions(snapshot(), TEMPORAL, [failed, *FRESH_STATS])
    assert (s.kind, s.tone, s.href, s.detail) == ("link", "warn", f"/controller#run={failed.id}",
                                                  "Statistics failed: the database did not answer")
    timed_out = run("stats", "timed_out", 3)
    assert build_suggestions(snapshot(), TEMPORAL, [timed_out, *FRESH_STATS])[0].href.endswith(timed_out.id)
    old = run("stats", "failed", 30)
    assert build_suggestions(snapshot(), TEMPORAL, [old, *FRESH_STATS]) == []


def test_statistics_are_offered_once_a_day():
    (s,) = build_suggestions(snapshot(), TEMPORAL, [])
    assert (s.id, s.kind, s.request.params) == ("stats-today", "run", StatsParams())
    assert ids(build_suggestions(snapshot(), TEMPORAL, [run("stats", "completed", 25)])) == ["stats-today"]
    assert ids(build_suggestions(snapshot(), TEMPORAL, [run("stats", "failed", 30)])) == ["stats-today"]
    # Temporal's statistics know about runs older than the recent list.
    known = snapshot(runs=RunStats(window_days=30, last_run_at={"stats": NOW - timedelta(hours=5)}))
    assert build_suggestions(known, TEMPORAL, []) == []


def test_without_a_runner_nothing_runnable_is_offered():
    out = build_suggestions(snapshot(new_jobs=3), NO_RUNS, [])
    assert ids(out) == ["review-new", "top3", "connect-temporal"]
    connect = out[-1]
    assert (connect.href, connect.tone, connect.detail) == ("/controller#setup", "neutral", NO_RUNS.reason)
    assert all(s.kind != "run" for s in out)


def test_order_is_warn_accent_neutral():
    busy = snapshot(new_jobs=40)
    failed = run("stats", "failed", 1, error="boom")
    out = build_suggestions(busy, TEMPORAL, [failed])
    assert ids(out) == [f"failed-{failed.id}", "review-new", "top3", "stats-today"]
    assert [s.tone for s in out] == ["warn", "accent", "neutral", "neutral"]
    assert build_suggestions(busy, TEMPORAL, [failed]) == out  # deterministic


async def test_panel_state_assembles_every_part():
    runs = fakes.FakeRuns()
    started = await runs.start(RunRequest(params=StatsParams(), source="panel"))
    state = await panel_state(StubDB(), runs, fakes.test_settings())
    assert state.runs_info.backend == "local" and state.runs_info.configured
    assert set(state.facts) == {"jobs", "tracking"}
    assert state.facts["jobs"][2] == {"value": "100", "label": "jobs stored", "warn": False}
    assert [r.id for r in state.recent_runs] == [started.id]
    assert state.stats is not None and state.stats.runs is not None and state.stats.runs.total == 1
    assert ids(state.suggestions) == ["review-new", "top3"]


async def test_panel_state_without_database_or_runner():
    stub = StubDB()

    async def down():
        raise ConnectionError("down")

    stub.overview = down
    state = await panel_state(stub, fakes.FakeRuns(configured=False, reason="Temporal unreachable"))
    assert state.facts == {} and state.stats is None and state.recent_runs == []
    assert state.runs_info.configured is False
    assert ids(state.suggestions) == ["connect-temporal"]


async def test_panel_state_for_a_regular_account_uses_job_seeker_wording():
    state = await panel_state(StubDB(), fakes.FakeRuns(), fakes.test_settings(), user_id=7, operator=False)
    labels = [f["label"] for items in state.facts.values() for f in items]
    assert not {"unreviewed", "jobs stored", "added 7d", "skipped"} & set(labels)
    assert {"new for you", "applied by you", "passed on"} <= set(labels)
    assert "stats-today" not in ids(state.suggestions)
    review = next(s for s in state.suggestions if s.id == "review-new")
    assert review.title.startswith("Look through your ") and review.title.endswith(" new jobs")
    assert "triage" not in review.detail
