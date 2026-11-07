"""Temporal workflow definitions, one per run kind, named as in
``WORKFLOW_TYPES`` so Temporal Cloud lists them as ``R3StatsWorkflow``. Each
workflow runs its flow from :mod:`r3.runs.flows` with activities executed by
Temporal, and answers the ``progress`` query with the flow's latest stage. A
flow that cannot finish fails the workflow with a readable message (an
``ApplicationError`` of type ``R3RunFailed``); cancellation propagates
untouched, so a cancelled run ends as CANCELED."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, TypeVar

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, is_cancelled_exception
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

with workflow.unsafe.imports_passed_through():
    from r3.runs import flows
    from r3.runs.models import WORKFLOW_TYPES, RunProgress, RunRequest, StatsSnapshot

T = TypeVar("T")


def _describe_timeout(step: flows.Step, err: TemporalTimeoutError) -> str:
    kind = err.type.name.replace("_", " ").lower() if err.type is not None else "timeout"
    if kind.startswith("heartbeat"):
        return (f"{step.label} stopped reporting for {flows.human_duration(step.heartbeat)}; "
                "the worker running it was probably lost")
    return f"{step.label} timed out after {flows.human_duration(step.timeout)} ({kind})"


class TemporalSteps:
    """Runs a flow's steps as Temporal activities."""

    async def run_activity(self, step: flows.Step, arg: Any, result_type: type[T]) -> T:
        retry = RetryPolicy(initial_interval=timedelta(seconds=2), backoff_coefficient=2.0,
                            maximum_interval=timedelta(seconds=30), maximum_attempts=step.attempts,
                            non_retryable_error_types=[flows.FATAL_ERROR_TYPE])
        args = [] if arg is None else [arg]
        try:
            return await workflow.execute_activity(
                step.name, args=args, result_type=result_type, start_to_close_timeout=step.timeout,
                heartbeat_timeout=step.heartbeat, retry_policy=retry, summary=step.label,
            )
        except ActivityError as e:
            if is_cancelled_exception(e):
                # The run was cancelled: let it propagate so the workflow
                # ends as CANCELED rather than FAILED.
                raise
            cause = e.cause
            if isinstance(cause, ApplicationError):
                raise flows.StepFailed(cause.message, fatal=cause.type == flows.FATAL_ERROR_TYPE) from None
            if isinstance(cause, TemporalTimeoutError):
                raise flows.StepFailed(_describe_timeout(step, cause)) from None
            raise flows.StepFailed(str(cause or e)) from None


class _RunWorkflow:
    """The shared body: progress bookkeeping and the failure contract."""

    def __init__(self) -> None:
        self._progress = RunProgress(stage="starting", message="Waiting for the first step")

    def _set_progress(self, stage: str, message: str = "", done: int = 0, total: int = 0) -> None:
        self._progress = RunProgress(stage=stage, message=message, done=done, total=total, updated_at=workflow.now())
        detail = f"{stage}: {message}" if message else stage
        if total:
            detail += f" ({done}/{total})"
        workflow.set_current_details(detail[:500])

    async def _execute(self, flow: flows.Flow, request: RunRequest) -> Any:
        try:
            return await flow(request, TemporalSteps(), self._set_progress)
        except flows.FlowFailed as e:
            self._set_progress("failed", str(e), self._progress.done, self._progress.total)
            raise ApplicationError(str(e), type=flows.RUN_FAILED_TYPE, non_retryable=True) from None


@workflow.defn(name=WORKFLOW_TYPES["stats"])
class StatsWorkflow(_RunWorkflow):
    @workflow.run
    async def run(self, request: RunRequest) -> StatsSnapshot:
        return await self._execute(flows.stats_flow, request)

    @workflow.query(name=flows.PROGRESS_QUERY)
    def progress(self) -> RunProgress:
        return self._progress


WORKFLOWS: list[type] = [StatsWorkflow]
