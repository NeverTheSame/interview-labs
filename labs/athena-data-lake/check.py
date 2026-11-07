"""Checks for the data lake. Expected answers are computed from the lab's own
data files, so editing the data keeps the checks right."""

import csv
import re
import time
from collections import defaultdict
from pathlib import Path

from labkit.runner import CheckFailed

DATA = Path(__file__).parent / "data" / "orders"


def _rows():
    for path in sorted(DATA.glob("dt=*/*.csv")):
        with path.open() as f:
            for row in csv.DictReader(f):
                yield path.parent.name.removeprefix("dt="), row


def _named(aws, name):
    ids = aws.athena.list_named_queries()["NamedQueryIds"]
    for q in aws.athena.batch_get_named_query(NamedQueryIds=ids)["NamedQueries"] if ids else []:
        if q["Name"] == name:
            return q
    raise CheckFailed(f"no named query {name!r}")


def _run(aws, query):
    qid = aws.athena.start_query_execution(
        QueryString=query["QueryString"], QueryExecutionContext={"Database": query.get("Database") or "shop"},
        ResultConfiguration={"OutputLocation": "s3://lake-athena-results/"})["QueryExecutionId"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = aws.athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if status["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.5)
    if status["State"] != "SUCCEEDED":
        raise CheckFailed(f"query {query['Name']!r} {status['State']}: {status.get('StateChangeReason', '')[:200]}")
    rows = aws.athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    return [[c.get("VarCharValue") for c in r["Data"]] for r in rows[1:]]  # first row is the header


def check_lake(aws):
    for bucket in ("lake", "lake-athena-results"):
        try:
            aws.s3.head_bucket(Bucket=bucket)
        except aws.s3.exceptions.ClientError:
            raise CheckFailed(f"bucket {bucket!r} not found")
    keys = {o["Key"] for o in aws.s3.list_objects_v2(Bucket="lake", Prefix="raw/orders/").get("Contents", [])}
    want = {f"raw/orders/{p.parent.name}/{p.name}" for p in DATA.glob("dt=*/*.csv")}
    missing = sorted(want - keys)
    if missing:
        raise CheckFailed(f"missing from s3://lake: {', '.join(missing)}")


def check_catalog(aws):
    try:
        table = aws.glue.get_table(DatabaseName="shop", Name="orders")["Table"]
    except aws.glue.exceptions.EntityNotFoundException:
        raise CheckFailed("Glue table shop.orders not found")
    sd = table.get("StorageDescriptor", {})
    if sd.get("Location", "").rstrip("/") != "s3://lake/raw/orders":
        raise CheckFailed(f"table location is {sd.get('Location')!r}; use s3://lake/raw/orders/")
    cols = {c["Name"]: c["Type"] for c in sd.get("Columns", [])}
    want = {"order_id": "string", "customer": "string", "amount": "double", "country": "string"}
    if cols != want:
        raise CheckFailed(f"columns are {cols}; want {want}")
    parts = [(p["Name"], p["Type"]) for p in table.get("PartitionKeys", [])]
    if parts != [("dt", "string")]:
        raise CheckFailed(f"partition keys are {parts}; partition by dt (string)")


def check_revenue(aws):
    got = _run(aws, _named(aws, "revenue_by_country"))
    totals = defaultdict(float)
    for _, row in _rows():
        totals[row["country"]] += float(row["amount"])
    want = sorted(totals.items(), key=lambda kv: -kv[1])
    try:
        rows = [(country, float(revenue)) for country, revenue, *_ in got]
    except (TypeError, ValueError):
        raise CheckFailed(f"expected (country, revenue) rows, got {got[:3]}")
    if [c for c, _ in rows] != [c for c, _ in want] or any(abs(a - b) > 0.01 for (_, a), (_, b) in zip(rows, want)):
        raise CheckFailed(f"got {rows}; expected {[(c, round(v, 2)) for c, v in want]}")


def check_one_day(aws):
    query = _named(aws, "orders_on_day")
    if not re.search(r"\bdt\b\s*(=|in\b|between\b)", query["QueryString"], re.I):
        raise CheckFailed("the query does not filter on the partition column dt, so it reads every day's files")
    got = _run(aws, query)
    want = sum(1 for dt, _ in _rows() if dt == "2026-11-02")
    if not got or got[0][0] is None or int(float(got[0][0])) != want:
        raise CheckFailed(f"got {got}; expected one row with {want}")
