# interview-labs

**You claimed a skill on your resume. The interview is next week. Go touch it.**

Hands-on labs for the cloud and data skills that interviews ask about, built
on a stack that runs on your laptop: [MiniStack](https://ministack.org) (an
open-source AWS emulator), Redis, and a Snowflake stand-in. No cloud account,
no bill. Each lab gives you a realistic scenario, tasks, a `check` command that
tells you what is still wrong, and the questions an interviewer will ask.

| Lab | Level | You practise |
|---|---|---|
| `lambda-sqs-pipeline` | beginner | SQS → Lambda → DynamoDB, dead-letter queues, partial batch failures |
| `athena-data-lake` | beginner | S3 partition layout, the Glue Data Catalog, Athena queries |
| `redis-caching` | beginner | cache-aside, invalidation, rate limiting, sorted-set leaderboards |
| `step-functions-etl` | intermediate | retrying what can recover, failing fast on what cannot, alerting |
| `snowflake-elt` | intermediate | VARIANT/JSON, FLATTEN, QUALIFY, a MERGE that is safe to rerun |

## Quick start

```bash
uv sync --extra labs
docker compose up -d ministack redis   # local AWS on :4566, Redis on :56379
uv run labs list
uv run labs show lambda-sqs-pipeline   # the tasks, what is real, interview questions
uv run labs check lambda-sqs-pipeline  # what is done and what is not (--hint for hints)
uv run labs reset                      # wipe the local stack and start again
```

Not sure where to start? Paste your resume or the job posting:

```bash
pbpaste | uv run labs suggest
```

## Honest about what is real

A local stack is not AWS. Every lab says what really runs (Lambda executes
your code, Athena runs your SQL), what is a stand-in (IAM is not enforced;
the Snowflake lab runs on [fakesnow](https://github.com/tekumara/fakesnow)),
and what differs on the real service. Read that list before you claim the
skill in an interview: `labs show <lab>`.

Not covered yet: managed Kubernetes (MiniStack's EKS needs the Docker socket
mounted into its container) and Redshift (MiniStack has no Redshift).

## Writing a lab

See [labs/README.md](labs/README.md). Every lab's checks are proven by the
test suite: on an empty stack every check fails, after the reference solution
every check passes, and a common near miss is rejected.

## The job-search assistant

The repo also holds the web app the labs grew out of: a job-search portal
with an AI assistant that reads your resume, keeps your job list and answers
questions about it. It needs Postgres (`docker compose up -d postgres`) and a
model key for the chat.

R3 is a server-rendered portal over a Postgres database of job postings. You
browse and triage jobs, keep your resume, and, on `/controller`, get two
systems in one page:

- **An advisor you talk to** (left): a Pydantic AI agent with read-only tools
  over the database. It streams answers in real time, and you can switch the
  model (OpenRouter or OpenAI) per conversation.
- **Workflows that run when you ask for them** (right): every message is
  first classified; "show me new AI/ML jobs from this week" fills the Jobs
  tab, and "where do things stand?" becomes a Temporal workflow (stats) whose
  progress, result and run statistics show up in the panel and in Temporal's
  UI. Anything else is a conversation.

Python throughout: FastAPI + Jinja2 + asyncpg, Pydantic models at every
boundary, Pydantic AI for the agent and the router, the Temporal Python SDK
for the workflows. No build step, no CDN.

## Run

```bash
cp .env.example .env      # DATABASE_URL, then a model key (OPENROUTER_API_KEY or OPENAI_API_KEY)
uv sync
uv run r3-portal          # http://localhost:8090 — chat works; runs execute in-process until Temporal is set
uv run r3-label           # the same portal, with /label opened in your browser
```

R3 reads two tables it does not create, `job_sites` and `scraped_jobs`
(`tests/e2e/fixtures/schema.sql` is the minimal schema, `seed.sql` a
synthetic sample). Everything an account owns it creates on demand at startup.

With Temporal, for durable runs that you can watch in Temporal's UI:

```bash
uv run r3-temporal-dev                             # local Temporal + web UI on http://localhost:8233
TEMPORAL_ADDRESS=localhost:7233 uv run r3-portal   # portal + embedded worker
```

For Temporal Cloud set `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE` and
`TEMPORAL_API_KEY` (or the mTLS pair) as in `.env.example`.

| Variable | Required | Purpose |
|----------|----------|---------|
| `DATABASE_URL` | yes | Postgres connection string |
| `PORT`, `HOST` | no | Listen port (`8090`) and bind address |
| `R3_ROOT_PATH` | no | Path prefix when the portal is served under one behind a reverse proxy |
| `OPENROUTER_API_KEY`, `OPENAI_API_KEY` | one for chat | Providers offered in the model picker |
| `R3_CHAT_MODEL`, `R3_CHAT_MODELS` | no | Default model and the picker, as `provider:name` ids |
| `R3_TASK_MODEL`, `R3_PARSE_MODEL` | no | Models for the AI steps (resume builder and reader, job overviews) apart from the chat |
| `R3_ROUTER_MODE` | no | `rules`, `llm` (default), `jev`, `hybrid`: how a message is classified |
| `TYPESAFE_API_KEY` | for Jev | Enables the `jev`/`hybrid` router (Jev is TypeSafe's decision model) |
| `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, `TEMPORAL_API_KEY` | for durable runs | Temporal Cloud (or `localhost:7233`) |
| `R3_RUNS_BACKEND`, `R3_EMBED_WORKER` | no | `auto` picks Temporal when configured; the worker runs inside the portal by default |
| `R3_JOB_POSTINGS` | no | `full` (default) shows every account the posting; `teaser` gives regular accounts an overview instead |
| `R3_ADMIN_ACCOUNTS` | no | Comma-separated account names that are operators. Public sign-up refuses these names, so create them with `python -m r3.accounts create NAME` |
| `R3_AUTH_RATE_LIMIT`, `R3_INSECURE_COOKIES` | no | Sign-in/sign-up attempts per 15 minutes per address and per name (default 10); `1` drops the cookie `Secure` flag for plain-http use off localhost |
| `R3_SIGNUP` | no | `off` closes the public sign-up form (default: open). The first operator is made with `python -m r3.accounts create NAME --admin`; sign-up never grants admin |
| `R3_SETTINGS_KEY` | recommended | Encrypts secrets saved on `/settings`; without it secrets cannot be saved there (no fallback) |

Most of the variables above can also be entered on `/settings`; a value
saved there overrides the environment and applies at once.

Hosted Postgres connection strings often carry `channel_binding=require`;
asyncpg can't pass that through, so `r3.db.normalize_dsn` drops it before
connecting.

## Pages

| Route        | What it does |
|--------------|--------------|
| `/`          | The Assistant for a job seeker; for an operator, counters, role/location/status breakdowns and top sites |
| `/home`      | A job seeker's get-started checklist and best matches |
| `/controller`| The advisor chat on the left; runs, jobs and statistics on the right |
| `/scout`     | Explore: curated hunts + every single-dimension slice, with live counts |
| `/jobs`      | Filter, sort, paginate all jobs; triage rows in place |
| `/jobs/{id}` | Full posting (or the teaser), metadata, notes editor, what happened to it |
| `/resume`    | Upload, paste, edit, pick the primary resume; download as text or PDF; `/resume/build` writes one with you |
| `/events`    | The audit trail (operators) |
| `/label`     | Label one posting at a time against a fixed rubric, by key; `/label/export` is the JSONL export |
| `/settings`  | Enter keys and switches in the portal, test each connection, apply without a restart |
| `/healthz`   | `OK` while the process is up; never queries the database (liveness) |
| `/readyz`    | `OK`, or 503 when the database is unreachable (readiness) |

Every filter lives in the URL, so any view is linkable and refresh-safe.
Filtering swaps only the results table via HTMX; the surrounding page stays put.

### Controller

**Left: the advisor.** Type a message; the reply streams token by token.
The header's picker switches the model for the conversation (any provider
with a key; custom OpenRouter ids allowed), the "Include portal summary"
box adds the live counts to the model's context, and "Advice only" skips
workflow detection. Conversations survive a refresh in the tab; New chat
clears them. A resume file dropped on the chat is read and saved.

**Every message is routed first** (`r3/agent/router.py`): `advice`, `jobs`
or `stats`. The router runs on the chat model by default, on rules alone with
`R3_ROUTER_MODE=rules`, or on TypeSafe's Jev with `jev`/`hybrid` (one
calibrated request, no LLM; hybrid falls back to the LLM below a confidence
floor). Slash commands (`/jobs`, `/stats`, ...) never call a model.

| You say | What happens |
|---------|--------------|
| "Show me new AI/ML jobs from this week" | The **Jobs** tab lists them, with the `/jobs` link that reproduces the filter |
| "Where do things stand?" | The **Stats** tab fills and the snapshot is recorded as a **stats** run |
| anything else | The advisor answers, calling read-only tools for facts |

**Right: the panel.** *Overview* shows suggestion cards computed from the
data (N unreviewed, a run failed, no snapshot today) and the pipeline facts.
*Runs* lists every run newest first with live progress, results, errors,
Cancel, and an "Open in Temporal" link. *Jobs* filters the database. *Stats*
shows the snapshot: counts, status and role breakdowns, and run statistics
(totals, per-kind outcomes, durations) read from Temporal visibility.

### Runs and Temporal

Each run is a Temporal workflow (`R3StatsWorkflow`) started with a memo
(kind, who and what started it, the parameters), so Temporal's UI shows every
run with its summary, and `GET /api/stats` aggregates the same visibility data
for the Stats tab. The worker executes runs inside the portal process by
default; `R3_EMBED_WORKER=0` moves it to `uv run r3-worker`. Without Temporal
the same flows run in-process and the panel says so.

### Explore

`/scout` is the entry point for finding something to apply to. It defaults to
the viewer's **new** jobs and can widen to all jobs. Presets cover:

- **Curated hunts** — cross-dimension combos (fresh remote AI/ML, remote
  contract, $200k+ remote, DevOps+SRE remote, Kubernetes remote, …)
- **Role**, **employment type**, **salary floor**, **location**, **freshness**,
  **technology**

Every count is computed with the same query builder that serves `/jobs`, so a
preset's number always equals what you see after clicking it.

## Writes

`job_sites` and `scraped_jobs` are only read. Each account's triage lives in
`r3_user_jobs`: its status (new / applied / skipped / expired; moving to
`applied` stamps `applied_at`, moving away clears it) and its notes. A posting
whose `last_validation` is `expired` shows as expired to everyone until they
triage it themselves.

R3's own tables, created on demand with `CREATE TABLE IF NOT EXISTS`:
`r3_users` and `r3_sessions`, `r3_user_jobs`, `r3_resumes` and
`r3_resume_drafts`, `r3_job_summaries` (cached job overviews), `r3_events`
(the audit trail), `r3_settings` (portal overrides), `r3_job_labels` and
`r3_job_labels_history` (the operator's job labels).

Status values are checked against a fixed allowlist; every filter value is
passed as a query parameter, never interpolated into SQL.

## Known data issues

- **`salary_min` / `salary_max` can be inconsistently scaled.** Some yearly
  rows store cents (`56600000` meaning `$566,000`). The portal therefore
  **displays the raw `salary` string**, which is always faithful to the
  posting, and uses the parsed columns only for filtering and sorting. The
  salary filter normalizes the obvious cents case (a yearly figure ≥ $1M) and
  applies only to annual postings.
- **`employment` has several spellings** for one concept (`fulltime` /
  `Full-time`, `contract` / `Contract` / `c2c`). `r3/db/jobs.py` groups these
  into families so a preset returns what the label promises.

## Layout

```
r3/portal.py         entry point (r3-portal / python -m r3.portal)
r3/worker.py         the Temporal worker (embedded in the portal, or r3-worker)
r3/settings.py       env → ControllerSettings (models, router, Temporal)
r3/config_registry.py  every setting the Settings page manages (group, kind, secret, test)
r3/settings_store.py   env + portal overrides (r3_settings, secrets encrypted) → ControllerSettings
r3/connection_checks.py  the Settings page's "Test connection" checks
r3/labeling.py       the job-label rubric (job-label-rubric/v1) and the form parser
r3/text.py           description text, link safety and the resume-rewrite fact check
r3/agent/            router (rules/llm/jev/hybrid), actions, advisor + tools, models, SSE events
r3/runs/             run models, Temporal + local runners, flows, activities, workflows, stats
r3/resume/           resume extraction, reading, the builder interview, PDF rendering
r3/panel.py          suggestion rules and the panel state
r3/pipelines.py      the pipeline cards and their live facts
r3/db/               asyncpg pool + every query (jobs, explore, users, resumes, labels, events, settings)
r3/web/              FastAPI routes (pages, chat API, runs API, resume API), template helpers
  templates/         Jinja2 pages, HTMX partials and row macros (_*.html)
  static/            app.css, controller.js + vendored htmx (no CDN)
evals/               the evaluation suite (routing, actions, advisor)
tests/               pure logic, route, agent, runs (incl. a real local Temporal) tests
```

Templates and static files ship inside the package, so `pip install .`
produces a self-contained wheel.

## Test and evaluate

```bash
uv run pytest                      # everything; -m temporal starts a local Temporal server once
uv run python -m evals             # offline evals: routing, what the panel does, advisor turns
uv run python -m evals --live      # against real models (needs the provider keys)
```

The tests need no database or keys: the agent runs on Pydantic AI's
`FunctionModel`/`TestModel`, Temporal on its local dev server (downloaded on
first use). The browser suite in `tests/e2e` runs only with `R3_E2E_PG_URL`
set to an admin URL of a throwaway Postgres.
