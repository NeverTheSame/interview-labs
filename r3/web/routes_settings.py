"""The Settings page and its API: every configurable value, including
secrets, entered in the portal, stored in ``r3_settings`` and applied without
a redeploy; and a "Test connection" check per vendor.

``GET /api/settings`` never returns a secret (only whether it is set, where
it comes from and at most its last four characters); ``PUT`` never echoes
one back; check results are scrubbed of every secret in play.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from r3 import config_registry as registry
from r3.events import record as record_event
from r3.agent.models import clear_model_cache
from r3.connection_checks import CheckContext, run_check
from r3.settings_store import SettingsStore, SettingsUnavailable, changed_groups
from r3.worker import EmbeddedWorker

from .auth import require_admin

log = logging.getLogger("r3.web.settings")

# Operator console: every route here is for admins only.
router = APIRouter(tags=["settings"], dependencies=[Depends(require_admin)])


class SettingsUpdate(BaseModel):
    values: dict[str, str | None] = Field(default_factory=dict)


class TestRequest(BaseModel):
    values: dict[str, str | None] = Field(default_factory=dict)


def _store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def settings_state(app: FastAPI) -> dict[str, Any]:
    state = app.state.settings_store.state()
    state["applied"] = getattr(app.state, "settings_applied", None)
    state["pinned"] = bool(getattr(app.state, "settings_pinned", False))
    return state


async def apply_settings(app: FastAPI, changed: list[str] | None = None) -> dict[str, Any]:
    """Make the stored settings live: rebuild ``app.state.settings`` from the
    store, then whatever depends on the changed names. The advisor is rebuilt
    on every change (its settings snapshot also decides the runs hints), the
    model cache is cleared when a model or router name changed, and the run
    service and embedded worker are replaced when a Temporal name changed. ``changed=None`` rebuilds everything. Serialised by a lock so
    two saves never interleave their rebuilds."""
    async with app.state.settings_lock:
        store: SettingsStore = app.state.settings_store
        groups = changed_groups(changed) if changed is not None else set(registry.GROUPS_BY_ID)
        new = store.settings()
        app.state.settings = new
        rebuilt: list[str] = ["settings"]
        messages: list[str] = []
        ok = True
        if changed is not None and not changed:
            messages.append("Nothing changed.")
        elif groups:
            if groups & registry.MODEL_GROUPS:
                clear_model_cache()
            try:
                app.state.advisor = app.state.build_advisor(new)
                rebuilt.append("advisor")
            except Exception as e:
                ok = False
                log.exception("rebuilding the advisor failed")
                messages.append(f"The advisor could not be rebuilt ({type(e).__name__}); the previous one stays.")
            if groups & registry.RUNS_GROUPS:
                ok = await _rebuild_runs(app, new, rebuilt, messages) and ok
        info = app.state.runs.describe()
        result = {
            "ok": ok, "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "changed": list(changed or []), "rebuilt": rebuilt, "messages": messages,
            "runs": info.model_dump(mode="json") if hasattr(info, "model_dump") else dict(info),
            "worker": app.state.worker is not None,
        }
        app.state.settings_applied = result
        return result


async def _rebuild_runs(app: FastAPI, new: Any, rebuilt: list[str], messages: list[str]) -> bool:
    try:
        runs = await app.state.build_runs(new, app.state.db)
    except Exception as e:
        log.exception("rebuilding the run service failed")
        messages.append(f"The run service could not be rebuilt ({type(e).__name__}); the previous one stays.")
        return False
    old_runs, old_worker, owned = app.state.runs, app.state.worker, app.state.owns_runs
    # New requests go to the new runner at once; the old worker then finishes
    # its running activities (up to 30 s) before the old runner closes.
    app.state.runs, app.state.owns_runs, app.state.worker = runs, True, None
    rebuilt.append("runs")
    if old_worker is not None:
        await old_worker.stop()
    if owned:
        try:
            await old_runs.close()
        except Exception:
            log.exception("closing the previous run service failed")
    try:
        app.state.worker = await EmbeddedWorker.start(runs, app.state.db, new)
    except Exception as e:
        log.exception("starting the embedded worker failed")
        messages.append(f"The embedded worker did not start ({type(e).__name__}).")
        return False
    if app.state.worker is not None:
        rebuilt.append("worker")
    info = runs.describe()
    where = {"temporal": f"Temporal ({info.namespace})", "local": "in-process", "none": "off"}.get(info.backend,
                                                                                                    info.backend)
    messages.append(f"Runs now execute {where}." if info.backend != "none" else "Runs are off.")
    if info.reason:
        messages.append(info.reason)
    return True


# --- page -----------------------------------------------------------------------------


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    tmpl = request.app.state.templates.get_template("settings.html")
    state = settings_state(request.app)
    tests = [t for g in state["groups"] for t in g["tests"]]
    return HTMLResponse(tmpl.render(title="Settings", nav="settings", query={}, state=state, tests=tests))


# --- API --------------------------------------------------------------------------------


@router.get("/api/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    return settings_state(request.app)


@router.put("/api/settings")
async def put_settings(request: Request, body: SettingsUpdate) -> dict[str, Any]:
    app = request.app
    store = _store(request)
    if getattr(app.state, "settings_pinned", False) or not store.available:
        raise HTTPException(503, store.error or "Portal overrides are unavailable.")
    clean, errors = store.validate(body.values)
    if errors:
        raise HTTPException(400, {"message": "Some values are not valid.", "errors": errors})
    if not clean:
        return {**settings_state(app), "apply": None}
    try:
        changed = await store.save(app.state.db, clean)
    except SettingsUnavailable as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("saving settings failed")
        raise HTTPException(503, f"Could not save to r3_settings ({type(e).__name__}).")
    log.info("settings saved from the portal: %s", ", ".join(sorted(clean)))  # names only, never values
    record_event("settings.changed", f"Settings changed: {', '.join(sorted(clean))}", actor="user",
                 data={"changed": sorted(clean)})
    result = await apply_settings(app, changed)
    return {**settings_state(app), "apply": result}


@router.post("/api/settings/test/{test_id}")
async def test_settings(request: Request, test_id: str, body: TestRequest | None = None) -> dict[str, Any]:
    if test_id not in registry.TESTS_BY_ID:
        raise HTTPException(404, f"No connection test {test_id!r}")
    store = _store(request)
    clean, errors = store.validate((body.values if body else {}) or {})
    if errors:
        raise HTTPException(400, {"message": "Some values are not valid.", "errors": errors})
    app = request.app
    if clean:
        candidate = store.settings(clean)
    else:
        candidate = app.state.settings  # what is live, including injected settings
    ctx = CheckContext(settings=candidate, database=getattr(app.state, "db", None),
                       transport=getattr(app.state, "check_transport", None),
                       typesafe_transport=getattr(app.state, "check_typesafe_transport", None),
                       connect=getattr(app.state, "check_temporal_connect", None))
    secrets = store.secret_values(clean) + _live_secrets(candidate)
    async with asyncio.timeout(30):
        result = await run_check(test_id, ctx, secrets)
    return result


def _live_secrets(s: Any) -> list[str]:
    values = [*getattr(s, "api_keys", {}).values(), getattr(s, "temporal_api_key", ""),
              getattr(s, "temporal_tls_key", "")]
    return [v for v in values if isinstance(v, str) and len(v) >= 4]
