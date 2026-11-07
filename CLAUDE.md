# interview-labs

Two parts in one repo:

- `labkit/` + `labs/`: hands-on interview-prep labs on a local stack. `labkit` is the
  `labs` CLI (catalog, checks runner, suggest); each `labs/<id>/` holds `lab.toml`,
  `README.md`, `check.py` and `solution/solve.py`. Rules for labs: `labs/README.md`.
- `r3/`: the job-search web app (FastAPI, Jinja, htmx, asyncpg, pydantic-ai, Temporal
  or in-process runs). The package is still named `r3`.

## Commands

- `uv sync --extra labs` installs everything; `docker compose up -d ministack redis`
  starts the lab stack (MiniStack pinned in `compose.yaml`).
- `uv run pytest -q` runs all tests. Lab tests marked `ministack` skip when nothing
  listens on :4566; browser tests in `tests/e2e` skip unless `R3_E2E_PG_URL` is set.
- `uv run labs check <lab>` checks one lab; `uv run labs solve <lab>` applies the
  reference solution (used by the tests).

## Conventions

- Test first. A new lab check must fail on an empty stack, pass after `solve`, and
  have a near miss in `tests/labkit/test_near_misses.py`.
- Every lab states what is real, what is a stand-in and what differs on AWS
  (`[fidelity]` in `lab.toml`). Never present an emulated behaviour as the real one.
- Original content only: no copied tutorials, real company data or personal data.
- Never commit `.env` or credentials. MiniStack takes any credentials (`test`/`test`).
