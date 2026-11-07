# Data lake: S3 + Glue Data Catalog + Athena

**Level:** beginner · **Time:** about 45 minutes · **Services:** S3, Glue, Athena

## Scenario

A shop drops a CSV of the day's orders into S3 every night. Analysts want to
answer questions with SQL without loading anything into a database. You set up
the lake: the raw files in a partitioned layout, a table definition in the Glue
Data Catalog, and the two queries the analysts run every morning.

The data is in `data/orders/dt=YYYY-MM-DD/`.

## Before you start

```bash
docker compose up -d ministack
export AWS_ENDPOINT_URL=http://localhost:4566 AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
```

## Tasks

1. **Lake.** Create the buckets `lake` and `lake-athena-results` and upload
   `data/orders/` to `s3://lake/raw/orders/`, keeping the `dt=` folders.
2. **Catalog.** Create the Glue database `shop` and the external table
   `orders` (order_id, customer, amount, country), partitioned by `dt`.
3. **Revenue.** Save the named query `revenue_by_country`: country and total
   amount, highest first.
4. **One day.** Save the named query `orders_on_day`: the number of orders on
   2026-11-02, filtered so Athena reads only that day's files.

```bash
uv run labs check athena-data-lake
uv run labs check athena-data-lake --hint
```

## What is real here, and what is not

Athena runs your SQL for real, but on DuckDB rather than Trino, and it reads
partitions straight from the folders. On AWS a new `dt=` folder stays invisible
until the catalog learns about it. `uv run labs show athena-data-lake` has the
full list and the interview questions.
