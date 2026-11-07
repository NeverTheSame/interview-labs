"""The portal's embedded Temporal worker: with ``R3_EMBED_WORKER`` on and a
Temporal runner, a run posted to the API is executed by the portal process
itself, so one process serves the page and the runs."""

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from r3.runs.temporal import TemporalRuns
from r3.settings import ControllerSettings
from r3.web import create_app
from tests.fakes import ScriptedAdvisor
from tests.test_routes import StubDB

pytestmark = [pytest.mark.temporal, pytest.mark.asyncio(loop_scope="module")]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def env():
    try:
        environment = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter,
                                                            dev_server_log_level="error")
    except Exception as e:
        pytest.skip(f"Temporal dev server unavailable: {e}")
    yield environment
    await environment.shutdown()


async def test_embedded_worker_executes_runs_posted_to_the_api(env):
    settings = ControllerSettings(temporal_address="localhost:7233", temporal_task_queue="r3-embedded-test",
                                  embed_worker=True)
    runs = TemporalRuns(env.client, settings)
    app = create_app(database=StubDB(), settings=settings, runs=runs, advisor=ScriptedAdvisor(settings))
    async with app.router.lifespan_context(app):
        assert app.state.worker is not None, "the lifespan must start the embedded worker for a Temporal runner"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            started = await client.post("/api/runs", json={"params": {"kind": "stats", "days": 7}, "source": "test"})
            assert started.status_code == 202, started.text
            run_id = started.json()["id"]
            for _ in range(100):
                run = (await client.get(f"/api/runs/{run_id}")).json()
                if run["status"] not in ("queued", "running"):
                    break
                await asyncio.sleep(0.2)
            assert run["status"] == "completed", run
            assert run["result"]["total_jobs"] == 100  # the stub's overview, computed by the worker
            assert run["backend"] == "temporal"
    assert app.state.worker.task.done(), "shutdown stops the embedded worker"


async def test_no_embedded_worker_without_a_temporal_runner():
    from tests.fakes import FakeRuns

    settings = ControllerSettings(embed_worker=True)
    app = create_app(database=StubDB(), settings=settings, runs=FakeRuns(), advisor=ScriptedAdvisor(settings))
    async with app.router.lifespan_context(app):
        assert app.state.worker is None
