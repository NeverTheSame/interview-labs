"""Your code for the Redis caching lab. Each function gets a redis.Redis
client `r` (decode_responses=True). See ../README.md for the tasks."""

import json  # noqa: F401  (you will probably want it)


def get_product(r, db, product_id: str) -> dict:
    """Return the product, from the cache key product:{id} when it is there,
    otherwise from db.load_product(product_id) and then cache it with a TTL."""
    return db.load_product(product_id)


def update_product(r, db, product_id: str, fields: dict) -> None:
    """Save the change with db.save_product(product_id, fields) and make sure
    the next get_product does not return the old version."""
    db.save_product(product_id, fields)


def allow_request(r, user_id: str, limit: int = 5, window: int = 60) -> bool:
    """Fixed-window rate limit on the key rate:{user_id}: True for the first
    `limit` calls in a window, False after that."""
    return True


def record_view(r, product_id: str) -> None:
    """Count a view of the product in the sorted set product_views."""


def top_products(r, n: int) -> list[str]:
    """The n most viewed product ids, most viewed first."""
    return []
