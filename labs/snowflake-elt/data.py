"""The raw order events the lab loads, and the answers they imply. Every
order can arrive more than once: placed, then shipped or cancelled, and a
late duplicate of the same event."""

import json
from decimal import Decimal

EVENTS = [
    # (event_id, loaded_at, payload)
    (1, "2026-11-01 08:00:00", {"order_id": "O-1", "customer_id": "c-1", "status": "placed",
                                "items": [{"sku": "kb-01", "qty": 1, "price": 89.0}, {"sku": "ms-02", "qty": 2, "price": 25.5}]}),
    (2, "2026-11-01 08:05:00", {"order_id": "O-2", "customer_id": "c-2", "status": "placed",
                                "items": [{"sku": "hd-07", "qty": 1, "price": 140.0}]}),
    (3, "2026-11-01 09:30:00", {"order_id": "O-1", "customer_id": "c-1", "status": "shipped",
                                "items": [{"sku": "kb-01", "qty": 1, "price": 89.0}, {"sku": "ms-02", "qty": 2, "price": 25.5}]}),
    (4, "2026-11-01 10:00:00", {"order_id": "O-3", "customer_id": "c-1", "status": "placed",
                                "items": [{"sku": "cb-11", "qty": 3, "price": 9.99}]}),
    (5, "2026-11-01 11:15:00", {"order_id": "O-2", "customer_id": "c-2", "status": "cancelled", "items": []}),
    (6, "2026-11-01 07:00:00", {"order_id": "O-4", "customer_id": "c-3", "status": "placed",
                                "items": [{"sku": "kb-01", "qty": 2, "price": 89.0}]}),
]

# core.orders before the MERGE: O-1 is stale, O-9 is not in today's events.
EXISTING_ORDERS = [("O-1", "placed", "c-1"), ("O-9", "shipped", "c-7")]

SEED_SQL = [
    "create schema if not exists raw",
    "create schema if not exists core",
    "create or replace table raw.order_events (event_id int, loaded_at timestamp_ntz, payload variant)",
    "create or replace table core.orders (order_id string, status string, customer_id string)",
    *(f"insert into raw.order_events select {i}, '{at}', parse_json('{json.dumps(p)}')" for i, at, p in EVENTS),
    *(f"insert into core.orders values ('{o}', '{s}', '{c}')" for o, s, c in EXISTING_ORDERS),
]


def latest():
    out = {}
    for _, at, p in sorted(EVENTS, key=lambda e: e[1]):
        out[p["order_id"]] = p
    return out


def expected_latest_orders():
    return sorted((p["order_id"], p["status"], p["customer_id"]) for p in latest().values())


def expected_order_lines():
    return sorted((p["order_id"], i["sku"], i["qty"], Decimal(str(i["qty"] * i["price"])).quantize(Decimal("0.01")))
                  for p in latest().values() for i in p["items"])


def expected_orders_after_merge():
    rows = {o: (o, s, c) for o, s, c in EXISTING_ORDERS}
    rows.update({o: (o, p["status"], p["customer_id"]) for o, p in latest().items()})
    return sorted(rows.values())
