# Snowflake ELT: JSON, deduplication and MERGE

**Level:** intermediate · **Time:** about an hour · **Engine:** a Snowflake stand-in (fakesnow)

## Scenario

An app streams order events into `raw.order_events` as JSON. The same order
shows up several times (placed, then shipped or cancelled), sometimes late.
The analytics team needs clean tables: the current state of every order, one
row per order line, and a `core.orders` table kept up to date with MERGE.

`data.py` holds the raw events. Each check loads them into a fresh database,
so you never need to clean up.

## Before you start

```bash
uv sync --extra labs        # installs the Snowflake stand-in
```

No Docker is needed for this lab.

## Tasks

Write each query in `answers/`:

1. **`latest_orders.sql`**: one row per order, from its newest event.
2. **`order_lines.sql`**: one row per item of each order's newest event,
   with `line_total`.
3. **`upsert_orders.sql`**: a MERGE into `core.orders` that is safe to run twice.

```bash
uv run labs check snowflake-elt --hint
```

## What is real here, and what is not

Your SQL is real Snowflake SQL, but it runs on fakesnow, which translates it
to DuckDB. There are no warehouses, credits, roles, Time Travel or streams.
`uv run labs show snowflake-elt` lists the differences and the interview
questions.
