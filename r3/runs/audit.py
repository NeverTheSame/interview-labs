"""Run events for R3's audit trail (r3/events.py): a run's start, with who
asked for it and with what, wherever it runs (in-process or on Temporal)."""

from __future__ import annotations

from r3.events import record

from .models import WorkflowRun


def record_run_started(run: WorkflowRun) -> None:
    record("run.started", f"{run.kind.capitalize()} run started" + (f" from {run.source}" if run.source else ""),
           user_id=run.user_id, subject=f"run:{run.id}",
           data={"kind": run.kind, "params": run.params, "source": run.source, "requested_by": run.requested_by,
                 "model": run.model, "note": run.note, "backend": run.backend})
