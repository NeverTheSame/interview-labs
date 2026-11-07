"""Checks for the Snowflake ELT lab. Each check loads the raw events into a
fresh in-memory Snowflake stand-in (fakesnow), runs your SQL file and compares
what comes back with the answer computed from data.py."""

import importlib.util
import warnings
from decimal import Decimal
from pathlib import Path

from labkit.runner import CheckFailed

HERE = Path(__file__).parent


def _data():
    spec = importlib.util.spec_from_file_location("snowflake_elt_data", HERE / "data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sql(aws, name):
    path = (aws.code_dir or HERE / "answers") / f"{name}.sql"
    if not path.is_file():
        raise CheckFailed(f"{path} is missing")
    body = "\n".join(line for line in path.read_text().splitlines() if not line.strip().startswith("--")).strip()
    if not body:
        raise CheckFailed(f"{path.name} is empty: write the query")
    return body.rstrip(";")


def _session():
    try:
        import fakesnow
        import snowflake.connector
    except ImportError:
        raise CheckFailed("the Snowflake stand-in is not installed: run `uv sync --extra labs`")
    warnings.filterwarnings("ignore", module="fakesnow")
    patch = fakesnow.patch()
    patch.__enter__()
    cursor = snowflake.connector.connect(database="SHOP", schema="RAW").cursor()
    for statement in _data().SEED_SQL:
        cursor.execute(statement)
    return patch, cursor


def _run(aws, name, then=None):
    patch, cursor = _session()
    try:
        try:
            cursor.execute(_sql(aws, name))
            rows = cursor.fetchall()
            if then:
                cursor.execute(_sql(aws, name))  # run twice: an upsert must be safe to repeat
                cursor.execute(then)
                rows = cursor.fetchall()
        except CheckFailed:
            raise
        except Exception as e:
            raise CheckFailed(f"{name}.sql failed: {str(e).splitlines()[0][:200]}")
        return [tuple(Decimal(str(v)).quantize(Decimal("0.01")) if isinstance(v, (float, Decimal)) else v for v in r)
                for r in rows]
    finally:
        patch.__exit__(None, None, None)


def _compare(name, got, want):
    if sorted(got, key=str) != sorted(want, key=str):
        raise CheckFailed(f"{name}: got {len(got)} row(s) {sorted(got, key=str)[:4]}...; "
                          f"expected {len(want)} row(s) {want[:4]}...")


def check_latest_orders(aws):
    _compare("latest_orders", _run(aws, "latest_orders"), _data().expected_latest_orders())


def check_order_lines(aws):
    _compare("order_lines", _run(aws, "order_lines"), _data().expected_order_lines())


def check_upsert_orders(aws):
    got = _run(aws, "upsert_orders", then="select order_id, status, customer_id from core.orders")
    _compare("upsert_orders (core.orders after running it twice)", got, _data().expected_orders_after_merge())
