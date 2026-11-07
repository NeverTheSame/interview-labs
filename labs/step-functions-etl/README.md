# Orchestrate a nightly ETL with Step Functions

**Level:** intermediate · **Time:** about an hour · **Services:** Step Functions, Lambda, SQS

## Scenario

Every night a job pulls the previous day's data from a partner. Sometimes the
partner times out and a second try works; sometimes the file is corrupt and no
number of tries will help. On-call wants the first case handled quietly and the
second one reported at once, without burning retries.

The extract step is written for you: `starter/extract.py`. Your job is the
orchestration around it.

## Before you start

```bash
docker compose up -d ministack
export AWS_ENDPOINT_URL=http://localhost:4566 AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
```

## Tasks

1. **Function.** Deploy `starter/extract.py` as the Lambda `extract`.
2. **Alerts.** Create the SQS queue `etl-alerts`.
3. **Happy path.** Create the STANDARD state machine `nightly-etl` that calls
   `extract`; `{"date": "2026-11-01"}` succeeds with the row count.
4. **Retry.** `TransientError` is retried at least twice, with backoff.
5. **No retry.** `BadDataError` is not retried; any failure that gets through
   sends a message naming the error to `etl-alerts` and ends the run FAILED.

```bash
uv run labs check step-functions-etl --hint
```

Inputs for trying it by hand: `{"date": "2026-11-01", "fail": "transient"}`
and `{"date": "2026-11-01", "fail": "bad-data"}`.

`uv run labs show step-functions-etl` lists what is real here and the
interview questions.
