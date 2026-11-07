"""Routing and handlers. Handlers parse the request, call ``r3.db`` and
render a template; they never build SQL."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles

from r3 import db
from r3.events import default_log, job_name
from r3.events import record as record_event
from r3.agent.service import AdvisorService, build_advisor, effective_default_model
from r3.db import Database
from r3.job_summary import Summarizer, build_summarizer
from r3.pipelines import PIPELINES, pipeline_facts
from r3.resume.builder import Interviewer, build_resume_interviewer
from r3.resume import MAX_UPLOAD_BYTES
from r3.resume.parse import ResumeParser, build_parser
from r3.resume.roles import RoleKeywords, resume_roles
from r3.resume.skills import VocabularyCache, resume_skills
from r3.db.jobs import JOBS_SORT_MATCH
from r3.runs.service import RunService, build_run_service
from r3.settings import ControllerSettings, ModelChoice
from r3.settings_store import SettingsStore
from r3.worker import EmbeddedWorker

from . import routes_chat, routes_label, routes_runs, routes_settings
from . import routes_resume
from . import routes_resume_build
from . import routes_overview
from . import routes_resume_chat
from . import routes_events
from .audit import AuditTrail
from .inputs import MAX_DAYS, MAX_PAGE, MAX_SALARY, DbId, clean_query, clean_text
from .auth import route_path
from .auth import OperatorsOnly, SignInGate, User, current_user, operators_only_response, signup_open, template_viewer
from .ratelimit import RateLimiter
from .security import SameOriginGuard
from . import routes_auth
from .headers import SecurityHeaders
from .limits import UploadSizeLimit
from .templating import STATIC_DIR, build_env

log = logging.getLogger("r3.web")


def auth_rate_limit() -> int:
    """Failed sign-ins (and sign-ups) allowed per 15 minutes per client address
    and per account name; ``R3_AUTH_RATE_LIMIT`` overrides the default of 10."""
    try:
        return max(1, int(os.environ.get("R3_AUTH_RATE_LIMIT", "10")))
    except ValueError:
        return 10


def create_app(
    database: Database | None = None,
    dsn: str | None = None,
    root_path: str = "",
    settings: ControllerSettings | None = None,
    runs: RunService | None = None,
    advisor: AdvisorService | None = None,
    settings_store: SettingsStore | None = None,
    auth: bool | None = None,
    resume_interviewer: Interviewer | None = None,
    job_summarizer: Summarizer | None = None,
    resume_parser: ResumeParser | None = None,
) -> FastAPI:
    """Build the portal app. Pass ``database`` to reuse an open connection
    (tests do this with a stub); otherwise the pool is opened from ``dsn`` at
    startup and closed at shutdown. ``runs`` and ``advisor`` are injected the
    same way; when omitted they are built from the settings at startup, which
    is where the Temporal client connects. ``resume_interviewer`` (the resume
    builder's, :mod:`r3.resume.builder`) is injected the same way; omitted,
    it is the LLM interviewer when a chat provider has a key, else the
    scripted one. ``job_summarizer`` (the job teaser's overview,
    :mod:`r3.job_summary`) is injected the same way.

    The settings come from a :class:`SettingsStore`: the environment plus
    the overrides saved on the Settings page, loaded once the database is
    open. Passing ``settings`` instead pins them (tests, demo servers): the
    Settings page then shows the environment and refuses to save.

    ``auth`` turns sign-in on or off. ``None`` means on when the app opens its
    own database from ``dsn`` and off when a ``database`` is injected, so
    stub-database tests and demos run as :data:`r3.web.auth.LOCAL_USER`."""

    pinned = settings is not None and settings_store is None
    store = settings_store or SettingsStore(env=dict(os.environ), dsn=dsn or "")
    if pinned:
        store.available = False
        store.error = "Settings were fixed when this portal started, so portal overrides are off."
    initial = settings if pinned else store.settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if database is not None:
            app.state.db = database
        else:
            if not dsn:
                raise RuntimeError("DATABASE_URL is not set")
            app.state.db = await Database.open(dsn)
        # Accounts and sessions (r3/db/users.py), before any table that refers
        # to them. Not caught: with sign-in on and no account tables nobody
        # could get in, so the portal should fail to start instead.
        if app.state.auth:
            await app.state.db.ensure_user_tables()
        # The label tables are R3's own (r3/db/labels.py); create them so /label
        # works on a fresh database. A failure only disables labeling.
        if hasattr(app.state.db, "ensure_labels_tables"):
            try:
                await app.state.db.ensure_labels_tables()
            except Exception as e:
                log.warning("label tables unavailable: %s: %s", type(e).__name__, e)
        # r3_resumes is R3's own (r3/db/resumes.py). A failure only disables
        # the resume library.
        if hasattr(app.state.db, "ensure_resume_tables"):
            try:
                await app.state.db.ensure_resume_tables()
            except Exception as e:
                log.warning("resume table unavailable: %s: %s", type(e).__name__, e)
        # r3_resume_drafts holds the resume builder's in-progress interview
        # (r3/db/resume_drafts.py). A failure only disables the builder.
        if hasattr(app.state.db, "ensure_resume_draft_tables"):
            try:
                await app.state.db.ensure_resume_draft_tables()
            except Exception as e:
                log.warning("resume draft table unavailable: %s: %s", type(e).__name__, e)
        # r3_job_summaries caches the job teaser's AI overview
        # (r3/db/job_summaries.py). A failure only means no caching.
        if hasattr(app.state.db, "ensure_job_summary_table"):
            try:
                await app.state.db.ensure_job_summary_table()
            except Exception as e:
                log.warning("job summary table unavailable: %s: %s", type(e).__name__, e)
        # r3_events is the audit trail (r3/db/events.py, r3/events.py). A failure
        # only means events reach the logs (and Axiom) but not the table.
        audit_flusher = None
        if hasattr(app.state.db, "ensure_events_table"):
            try:
                await app.state.db.ensure_events_table()
            except Exception as e:
                log.warning("events table unavailable: %s: %s", type(e).__name__, e)
        if hasattr(app.state.db, "insert_events"):
            default_log.store = app.state.db
            audit_flusher = asyncio.create_task(_flush_events_forever())
        # r3_user_jobs holds each account's job status and notes (r3/db/jobs.py).
        # A failure surfaces on the Jobs pages, not as a dead portal.
        if hasattr(app.state.db, "ensure_user_jobs_table"):
            try:
                await app.state.db.ensure_user_jobs_table()
            except Exception as e:
                log.warning("user jobs table unavailable: %s: %s", type(e).__name__, e)
        if not pinned:
            # Portal overrides live in the database, so they are read here,
            # before anything is built from the settings.
            await store.load(app.state.db)
            app.state.settings = store.settings()
        current: ControllerSettings = app.state.settings
        app.state.owns_runs = runs is None
        app.state.runs = runs if runs is not None else await app.state.build_runs(current, app.state.db)
        app.state.advisor = advisor if advisor is not None else app.state.build_advisor(current)
        app.state.resume_interviewer = (resume_interviewer if resume_interviewer is not None
                                        else build_resume_interviewer(current))
        # The worker that executes Temporal runs lives in this process unless
        # R3_EMBED_WORKER=0 hands that job to a separate r3-worker.
        app.state.worker = await EmbeddedWorker.start(app.state.runs, app.state.db, current)
        try:
            yield
        finally:
            # Background AI overviews and resume reads end before the database closes.
            await routes_overview.cancel_overviews(app)
            await routes_resume_chat.cancel_reads(app)
            # A settings save may be swapping the runner; let it finish first.
            async with app.state.settings_lock:
                if app.state.worker is not None:
                    await app.state.worker.stop()
                if app.state.owns_runs:
                    await app.state.runs.close()
            if audit_flusher is not None:
                audit_flusher.cancel()
                await default_log.flush()
                default_log.store = None
            if database is None:
                await app.state.db.close()

    root_path = root_path.rstrip("/")
    app = FastAPI(title="R3", lifespan=lifespan, root_path=root_path,
                  docs_url=None, redoc_url=None, openapi_url=None)
    # app.state.settings is always the live ControllerSettings; the Settings
    # page replaces it (and what is built from it) when a value is saved.
    app.state.settings = initial
    app.state.settings_store = store
    app.state.settings_pinned = pinned
    app.state.settings_lock = asyncio.Lock()
    app.state.settings_applied = None
    # What a save rebuilds with; tests swap these for fakes.
    app.state.build_runs = build_run_service
    app.state.build_advisor = build_advisor
    # The job teaser's overview reads the live settings per call, so a key
    # saved in Settings takes effect without rebuilding it.
    app.state.job_summarizer = (job_summarizer if job_summarizer is not None
                                else build_summarizer(lambda: app.state.settings))
    routes_overview.init_state(app)
    # The Assistant's resume upload reads files with this (r3.resume.parse).
    app.state.resume_parser = (resume_parser if resume_parser is not None
                               else build_parser(lambda: app.state.settings))
    routes_resume_chat.init_state(app)
    app.state.templates = build_env(base_path=root_path)
    app.state.templates.globals["viewer"] = template_viewer
    app.state.templates.globals["signup_open"] = signup_open
    app.state.auth = (database is None) if auth is None else auth
    # Inside the sign-in gate, so every recorded action knows its account.
    app.add_middleware(AuditTrail)
    # Sign-in and sign-up attempts per client address and account name.
    app.state.auth_limiter = RateLimiter(limit=auth_rate_limit())
    if app.state.auth:
        app.add_middleware(SignInGate)
        app.include_router(routes_auth.router)
    # Added last so it runs first: a cross-site POST is refused before the
    # sign-in gate looks it up, and every response (401s included) gets the
    # security headers. Installed whether or not sign-in is on.
    app.add_middleware(SameOriginGuard)
    app.add_middleware(UploadSizeLimit)
    # Outermost, so every response carries the security headers.
    app.add_middleware(SecurityHeaders)
    app.add_exception_handler(OperatorsOnly, operators_only_response)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(routes_chat.router)
    app.include_router(routes_label.router)
    app.include_router(routes_runs.router)
    app.include_router(routes_settings.router)
    app.include_router(routes_resume_build.router)
    app.include_router(routes_resume.router)
    app.include_router(routes_overview.router)
    app.include_router(routes_resume_chat.router)
    app.include_router(routes_events.router)
    _register_routes(app)
    return app


async def _flush_events_forever(interval: float = 1.0) -> None:
    """Write buffered audit events to the database once a second."""
    while True:
        await asyncio.sleep(interval)
        await default_log.flush()


# --- rendering ----------------------------------------------------------------


def _db(request: Request) -> Database:
    return request.app.state.db


def _query(request: Request) -> dict[str, list[str]]:
    """The query string as key -> values, the shape the ``qs`` helper takes."""
    out: dict[str, list[str]] = {}
    for k, v in request.query_params.multi_items():
        out.setdefault(k, []).append(v)
    return out


def _render(request: Request, name: str, *, title: str, nav: str, **ctx: Any) -> HTMLResponse:
    """Write a full page (layout + body)."""
    tmpl = request.app.state.templates.get_template(f"{name}.html")
    return HTMLResponse(tmpl.render(title=title, nav=nav, query=_query(request), **ctx))


def _render_partial(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    """Write just a fragment, for HTMX swaps."""
    tmpl = request.app.state.templates.get_template(f"{name}.html")
    return HTMLResponse(tmpl.render(query=_query(request), **ctx))


def _render_macro(request: Request, name: str, macro: str, *args: Any) -> HTMLResponse:
    """Emit a single macro (a table row), the swap target for a row action."""
    module = request.app.state.templates.get_template(f"{name}.html").module
    return HTMLResponse(str(getattr(module, macro)(*args)))


def _is_htmx(request: Request) -> bool:
    """Whether the request came from an HTMX swap rather than a full
    navigation, so the handler can skip the surrounding chrome."""
    return request.headers.get("HX-Request") == "true"


def _atoi(s: str | None) -> int:
    """Parse a non-negative int; anything else is 0."""
    try:
        n = int((s or "").strip())
    except ValueError:
        return 0
    return n if n >= 0 else 0


_ERROR_SENTENCES = {
    404: "There is nothing at this address.",
    422: "Part of this address is not something this page can use.",
}


def _wants_error_page(request: Request) -> bool:
    """A browser opening a page gets an HTML error; HTMX swaps, the JSON API
    and everything else keep FastAPI's JSON."""
    return (request.method == "GET" and "text/html" in request.headers.get("accept", "")
            and not _is_htmx(request) and not route_path(request.scope).startswith("/api/"))


_ERROR_TITLES = {404: "Page not found", 422: "Address not valid"}


def _error_page(request: Request, status: int) -> HTMLResponse:
    tmpl = request.app.state.templates.get_template("error.html")
    html = tmpl.render(title=_ERROR_TITLES.get(status, str(status)), nav="", status=status, message=_ERROR_SENTENCES[status])
    return HTMLResponse(html, status_code=status)


async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
    if exc.status_code in _ERROR_SENTENCES and _wants_error_page(request):
        return _error_page(request, exc.status_code)
    return await http_exception_handler(request, exc)


async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
    """FastAPI's 422, with each error's echoed input clipped so a huge body
    is not sent back whole."""
    if _wants_error_page(request):
        return _error_page(request, 422)
    errors = jsonable_encoder(exc.errors())
    for err in errors:
        if "input" in err:
            text = err["input"] if isinstance(err["input"], str) else json.dumps(err["input"])
            if len(text) > 200:
                err["input"] = text[:200] + "..."
    return JSONResponse({"detail": errors}, status_code=422)


def picker_models(models: list[ModelChoice], *, admin: bool) -> list[ModelChoice]:
    """The Assistant's model picker. An id listed twice (``R3_CHAT_MODELS``
    repeating one, or the default prepended) shows once. Admins see each
    provider's route to a model; a regular account sees one entry per
    model name, since "OpenRouter · GPT-4.1 mini" and "OpenAI · GPT-4.1
    mini" are the same choice to them and the picker drops the provider."""
    seen: set[str] = set()
    out: list[ModelChoice] = []
    for m in models:
        key = m.id if admin else m.label.split(" · ", 1)[-1].strip().casefold()
        if m.id in seen or key in seen:
            continue
        seen.update({m.id, key})
        out.append(m)
    return out


async def _home(request: Request, d: Database, user: User) -> HTMLResponse:
    """A job seeker's start page: their own numbers, a get-started checklist
    drawn from real state, and one line about the shared corpus. Nothing
    operator-facing (per-site tables, other people's triage) is shown."""
    stats = await d.overview(user_id=user.id)
    primary = await d.primary_resume(user.id)
    resumes = await d.list_resumes(user.id)
    # The five new jobs that fit the resume best: its roles, then its skills.
    best: list[db.Job] = []
    _, skills, roles = await _resume_fit(request, d, user)
    if skills:
        top, _ = await d.list_jobs(db.JobFilter(status="new", sort=JOBS_SORT_MATCH), 1, 5,
                                   user_id=user.id, skills=skills, titles=roles.patterns())
        best = [j for j in top if j.match_count]
    return _render(
        request, "home", title="Home", nav="overview", seeker=True,
        account_name=user.account_name, stats=stats, primary=primary,
        resume_count=len(resumes), corpus_sites=stats.total_sites,
        best_matches=best,
    )


async def _resume_match(request: Request, d: Database, user: User) -> tuple[bool, list[str]]:
    """A regular account's input to the job match: whether they have a
    primary resume, and the corpus tech tags it shows. Admins get neither,
    so their pages are unchanged. The tag vocabulary is cached per app."""
    has_resume, skills, _ = await _resume_fit(request, d, user, roles=False)
    return has_resume, skills


async def _resume_fit(request: Request, d: Database, user: User, *,
                      roles: bool = True) -> tuple[bool, list[str], RoleKeywords]:
    """:func:`_resume_match` plus the roles the resume names (its headline
    and recent titles), which the best-match sort ranks job titles by."""
    if user.is_admin:
        return False, [], RoleKeywords()
    primary = await d.primary_resume(user.id)
    if primary is None or not primary.content.strip():
        return False, [], RoleKeywords()
    cache = getattr(request.app.state, "skill_vocab", None)
    if cache is None:
        cache = request.app.state.skill_vocab = VocabularyCache()
    skills = resume_skills(primary.content, await cache.get(d.tech_vocabulary))
    return True, skills, resume_roles(primary.content, primary.structure) if roles else RoleKeywords()


# --- routes -------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    # Liveness answers without the database, so a frequent health check
    # never keeps a scale-to-zero database awake; the database check lives
    # on /readyz.
    @app.api_route("/healthz", methods=["GET", "HEAD"], response_class=PlainTextResponse)
    async def liveness() -> PlainTextResponse:
        return PlainTextResponse("OK")

    @app.api_route("/readyz", methods=["GET", "HEAD"], response_class=PlainTextResponse)
    async def readiness(request: Request) -> PlainTextResponse:
        try:
            await _db(request).ping()
        except Exception:
            return PlainTextResponse("database unreachable", status_code=503)
        return PlainTextResponse("OK")

    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request, user: User = Depends(current_user)) -> HTMLResponse:
        d = _db(request)
        if not user.is_admin:
            # The Assistant is the product's front door: a new account lands in
            # the chat and is walked through the journey (owner decision).
            return RedirectResponse(f"{request.scope['root_path']}/controller", status_code=303)
        stats = await d.overview()
        facets = await d.job_facets()
        top = (await d.sites_with_jobs())[:10]
        return _render(request, "overview", title="Overview", nav="overview", stats=stats, facets=facets, top_sites=top)

    @app.get("/home", response_class=HTMLResponse)
    async def home_page(request: Request, user: User = Depends(current_user)) -> HTMLResponse:
        """The get-started checklist, kept off the main nav: the Assistant is
        the landing page; this is the at-a-glance version of the same journey."""
        return await _home(request, _db(request), user)

    @app.get("/controller", response_class=HTMLResponse)
    async def controller(request: Request, user: User = Depends(current_user)) -> HTMLResponse:
        """The workspace: chat on the left, the pipelines, suggestions, runs
        and statistics on the right. The pipeline cards carry live numbers,
        but the page must stay usable when the database is down, since the
        chat itself works without it. Admins get the operator page (runner,
        pipeline stages, custom model ids); a regular account gets the job
        seeker's "Assistant" with the same chat and its own runs and jobs."""
        settings: ControllerSettings = request.app.state.settings
        runs: RunService = request.app.state.runs
        d = _db(request)
        facts: dict[str, Any] | None = None
        if user.is_admin:  # the stage cards are operator-only; skip their queries otherwise
            try:
                async with asyncio.timeout(8):
                    stats, facets = await asyncio.gather(d.overview(), d.job_facets())
                facts = pipeline_facts(stats, facets)
            except Exception:
                log.warning("Pipeline facts unavailable for the controller page")
        return _render(
            request, "controller", title="Controller" if user.is_admin else "Assistant", nav="controller",
            settings=settings, user=user, is_admin=user.is_admin,
            # A portal with only an OpenAI key and the OpenRouter default still
            # chats: the page offers whichever provider has a key.
            chat_ready=bool(settings.available_models()),
            chat_model=effective_default_model(settings),
            chat_setup=settings.setup_hint(),
            models=picker_models(settings.available_models(), admin=user.is_admin),
            default_model=effective_default_model(settings),
            runs_info=runs.describe(),
            pipelines=PIPELINES, facts=facts, max_upload=MAX_UPLOAD_BYTES,
        )

    @app.get("/scout", response_class=HTMLResponse)
    async def scout(request: Request, user: User = Depends(current_user)) -> HTMLResponse:
        """The preset launcher ("Explore jobs"): curated hunts plus every
        single-dimension slice, each with a live count."""
        d = _db(request)
        # Default to the unreviewed pool — that is the actionable queue. "all"
        # widens it to every stored job.
        scope_param = request.query_params.get("scope", "")
        scope = db.ScoutScope(status="" if scope_param == "all" else "new")

        # The counts are the viewer's own triage over the shared corpus.
        groups, total = await d.scout_presets(scope, user_id=user.id)
        combos = await d.scout_combos(scope, user_id=user.id)
        return _render(
            request, "scout", title="Explore jobs", nav="scout", seeker=not user.is_admin,
            groups=groups, combos=combos, total=total, scope=scope_param,
        )

    @app.get("/jobs", response_class=HTMLResponse)
    async def jobs(request: Request, user: User = Depends(current_user)) -> HTMLResponse:
        # Status, notes and the status filter/counts are the signed-in user's
        # own (r3_user_jobs); everything else is the shared corpus.
        d = _db(request)
        q = clean_query(request.query_params)
        # A repeated ?role= means "any of these" (the DevOps+SRE style hunts).
        # The first value stays on the scalar field so the dropdown reflects it.
        roles = q.getlist("role")
        f = db.JobFilter(
            search=q.get("q", ""),
            status=q.get("status", ""),
            role_category=roles[0] if roles else "",
            location_kind=q.get("location", ""),
            site_url=q.get("site", ""),
            tech_tag=q.get("tech", ""),
            employment=q.get("employment", ""),
            min_salary=min(_atoi(q.get("min_salary")), MAX_SALARY),
            days=min(_atoi(q.get("days")), MAX_DAYS),
            sort=q.get("sort", ""),
            employment_family=q.get("employment_family", ""),
            remote=q.get("remote") == "1",
            has_salary=q.get("has_salary") == "1",
        )
        if len(roles) > 1:
            f.role_categories = roles
        page = min(_atoi(q.get("page")) or 1, MAX_PAGE)
        size = _atoi(q.get("size"))
        size = size if size in (25, 50, 100) else 50

        # A regular account with a resume gets the job match: each row shows
        # how many of its tags the resume covers, and "best match" is the
        # sort until they pick another. Admins and resume-less accounts get
        # the same query as before.
        has_resume, skills, roles = await _resume_fit(request, d, user)
        if skills and not f.sort:
            f.sort = JOBS_SORT_MATCH
        match = {"skills": skills, "titles": roles.patterns()} if skills else {}
        job_rows, pg = await d.list_jobs(f, page, size, user_id=user.id, **match)
        # seeker: a regular account sees the job-seeker row actions, not the
        # operator ones (expire, reset).
        view: dict[str, Any] = dict(jobs=job_rows, page=pg, filter=f, seeker=not user.is_admin,
                                    match_skills=skills, resume_hint=not user.is_admin and not has_resume)

        # An HTMX swap only replaces the results table, so skip the facet and
        # site queries the filter panel needs.
        if _is_htmx(request):
            return _render_partial(request, "jobs_table", oob_status=True, **view)

        view["facets"] = await d.job_facets(user_id=user.id)
        view["sites"] = await d.sites_with_jobs()
        return _render(request, "jobs", title="Jobs", nav="jobs", **view)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(request: Request, job_id: DbId, user: User = Depends(current_user)) -> HTMLResponse:
        job = await _db(request).get_job(job_id, user_id=user.id)
        if job is None:
            raise HTTPException(404, "job not found")
        _, skills = await _resume_match(request, _db(request), user)
        # With R3_JOB_POSTINGS=teaser a regular account gets the overview
        # instead of the posting (OQ-6); otherwise everyone reads the posting.
        postings = request.app.state.settings.postings_for(is_admin=user.is_admin)
        teaser = {} if postings else await routes_overview.page_overview(request, job)
        history = await routes_events.job_history(request, user.id, job.id)
        return _render(request, "job_detail", title=job.position, nav="jobs", job=job, postings=postings,
                       seeker=not user.is_admin, match_skills=skills, history=history, **teaser)

    @app.post("/jobs/{job_id}/status")
    async def job_status(
        request: Request, job_id: DbId, status: str = Form(""), redirect: str = Form(""),
        user: User = Depends(current_user),
    ) -> Any:
        """Update one job's status and re-render just that row, so the table
        updates in place without losing scroll position or filters. The status
        is the user's own (r3_user_jobs)."""
        d = _db(request)
        before = await d.get_job(job_id, user_id=user.id)
        try:
            await d.set_user_job_status(user.id, job_id, status)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except LookupError as e:
            raise HTTPException(404, str(e))
        if before is not None:
            record_event("job.status", f"You marked “{job_name(before.position, before.company)}” {status}",
                         actor="user", user_id=user.id, subject=f"job:{job_id}",
                         data={"from": before.status, "to": status, "via": "job page" if redirect else "jobs list"})

        # The detail page posts with redirect=1 and wants a normal navigation back.
        if redirect:
            return RedirectResponse(f"{request.scope['root_path']}/jobs/{job_id}", status_code=303)

        job = await d.get_job(job_id, user_id=user.id)
        if job is None:
            raise HTTPException(404, "job not found")
        _, skills = await _resume_match(request, d, user)
        return _render_macro(request, "_job_row", "job_row", job, not user.is_admin, skills, True)

    @app.post("/jobs/{job_id}/notes")
    async def job_notes(
        request: Request, job_id: DbId, notes: str = Form(""), user: User = Depends(current_user)
    ) -> RedirectResponse:
        try:
            await _db(request).set_user_job_notes(user.id, job_id, clean_text(notes))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except LookupError as e:
            raise HTTPException(404, str(e))
        return RedirectResponse(f"{request.scope['root_path']}/jobs/{job_id}", status_code=303)
