"""Command worker runs the Temporal worker that executes R3's runs (the
statistics workflow and its activities) against the database.

Run with ``r3-worker``. It reads the same environment as the portal:
``DATABASE_URL`` and the Temporal connection (``TEMPORAL_ADDRESS``,
``TEMPORAL_NAMESPACE``, ``TEMPORAL_API_KEY`` or
``TEMPORAL_TLS_CERT``/``TEMPORAL_TLS_KEY``, ``TEMPORAL_TASK_QUEUE``). Without
``TEMPORAL_ADDRESS`` it idles, so a deploy made before Temporal is configured
stays healthy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import timedelta
from typing import Awaitable, Callable, TypeVar

from dotenv import find_dotenv, load_dotenv
from temporalio.worker import Worker

from r3 import obs
from r3.events import default_log
from r3.db import Database
from r3.runs.activities import Activities
from r3.runs.temporal import connect_client
from r3.runs.workflows import WORKFLOWS
from r3.settings import ControllerSettings

log = logging.getLogger("r3.worker")

T = TypeVar("T")


async def _with_retry(what: str, fn: Callable[[], Awaitable[T]], stop: asyncio.Event) -> T | None:
    """Keep trying ``fn`` with backoff until it works or the worker is told
    to stop (None). A dependency that is briefly down at boot should not
    turn into a restart loop."""
    delay = 2.0
    while not stop.is_set():
        try:
            return await fn()
        except Exception as e:
            log.warning("%s failed: %s: %s; retrying in %.0fs", what, type(e).__name__, e, delay)
        try:
            await asyncio.wait_for(stop.wait(), delay)
        except TimeoutError:
            pass
        delay = min(delay * 2, 60.0)
    return None


def build_worker(client, settings: ControllerSettings, activities: Activities) -> Worker:
    """The worker for R3's task queue. Heartbeats are flushed every ten
    seconds so the panel's progress stays fresh; short activities get thirty
    seconds to finish when the worker shuts down."""
    return Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=WORKFLOWS,
        activities=activities.all(),
        max_heartbeat_throttle_interval=timedelta(seconds=10),
        default_heartbeat_throttle_interval=timedelta(seconds=10),
        graceful_shutdown_timeout=timedelta(seconds=30),
    )


class EmbeddedWorker:
    """The worker running inside the portal process (``R3_EMBED_WORKER``,
    on by default). The portal already holds the Temporal client and the
    database, so the worker reuses both; ``stop`` lets running activities
    finish, as the standalone command does."""

    def __init__(self, worker: Worker, task: asyncio.Task[None]) -> None:
        self.worker = worker
        self.task = task

    @classmethod
    async def start(cls, runs: object, database: Database, settings: ControllerSettings) -> "EmbeddedWorker | None":
        """Start polling when ``runs`` is the Temporal runner and embedding
        is on; otherwise nothing runs here (a separate ``r3-worker`` does)."""
        from r3.runs.temporal import TemporalRuns

        if not settings.embed_worker or not isinstance(runs, TemporalRuns):
            return None
        worker = build_worker(runs.client, settings, Activities(database))
        task = asyncio.create_task(worker.run(), name="temporal-worker")

        def _observe(done: asyncio.Task[None]) -> None:
            if not done.cancelled() and done.exception() is not None:
                log.error("embedded worker stopped: %r", done.exception())

        task.add_done_callback(_observe)
        log.info("embedded worker polling namespace=%s task_queue=%s", settings.temporal_namespace,
                 settings.temporal_task_queue)
        return cls(worker, task)

    async def stop(self) -> None:
        if self.task.done():
            return
        try:
            await self.worker.shutdown()
            await self.task
        except Exception:  # a worker that failed while stopping is already logged
            log.exception("embedded worker did not stop cleanly")


async def run_worker(settings: ControllerSettings, dsn: str, stop: asyncio.Event) -> None:
    """Connect, poll the task queue until ``stop`` is set, then shut down
    gracefully. A fatal worker error propagates."""
    database = await _with_retry("opening the database", lambda: Database.open(dsn), stop)
    if database is None:
        return
    # Activities record audit events (r3/events.py); this process writes them.
    try:
        await database.ensure_events_table()
    except Exception as e:
        log.warning("events table unavailable: %s: %s", type(e).__name__, e)
    default_log.store = database
    flusher = asyncio.create_task(_flush_events_forever(), name="audit-events")
    try:
        client = await _with_retry(f"connecting to Temporal at {settings.temporal_address}",
                                   lambda: connect_client(settings), stop)
        if client is None:
            return
        worker = build_worker(client, settings, Activities(database))
        log.info("worker polling address=%s namespace=%s task_queue=%s",
                 settings.temporal_address, settings.temporal_namespace, settings.temporal_task_queue)
        running = asyncio.create_task(worker.run(), name="temporal-worker")
        stopping = asyncio.create_task(stop.wait(), name="stop-signal")
        done, _ = await asyncio.wait({running, stopping}, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            stopping.cancel()
            running.result()  # a fatal worker error surfaces here
            return
        log.info("stopping: letting running activities finish (up to 30s)")
        await worker.shutdown()
        await running
    finally:
        flusher.cancel()
        await default_log.flush()
        default_log.store = None
        await database.close()
    log.info("worker stopped")


async def _flush_events_forever(interval: float = 1.0) -> None:
    while True:
        await asyncio.sleep(interval)
        await default_log.flush()


async def _main(settings: ControllerSettings, dsn: str) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    if not settings.temporal_configured or settings.runs_backend in ("local", "none"):
        why = ("TEMPORAL_ADDRESS is not set" if not settings.temporal_configured
               else f"R3_RUNS_BACKEND={settings.runs_backend}")
        log.warning("%s, so there is no task queue to poll; the worker idles until stopped. Runs execute in "
                    "the portal process instead.", why)
        await stop.wait()
        return
    await run_worker(settings, dsn, stop)


def main() -> None:
    # .env is a convenience for local runs; deployments set the environment.
    load_dotenv(find_dotenv(usecwd=True))
    shipper = obs.setup_logging("r3-worker", os.environ)
    tracing = obs.setup_tracing("r3-worker", obs.AxiomConfig.from_env(os.environ))
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        log.error("DATABASE_URL is not set")
        sys.exit(1)
    try:
        asyncio.run(_main(ControllerSettings.from_env(), dsn))
    except KeyboardInterrupt:
        pass
    finally:
        tracing.shutdown()
        if shipper is not None:
            shipper.close()


if __name__ == "__main__":
    main()
