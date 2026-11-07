"""Reference solution for the Redis caching lab."""

import json

TTL = 300


def get_product(r, db, product_id: str) -> dict:
    key = f"product:{product_id}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)
    product = db.load_product(product_id)
    r.set(key, json.dumps(product), ex=TTL)
    return product


def update_product(r, db, product_id: str, fields: dict) -> None:
    db.save_product(product_id, fields)
    r.delete(f"product:{product_id}")  # write to the source first, then drop the cached copy


def allow_request(r, user_id: str, limit: int = 5, window: int = 60) -> bool:
    key = f"rate:{user_id}"
    pipe = r.pipeline()  # MULTI/EXEC: the count and its expiry are set together
    pipe.incr(key)
    pipe.expire(key, window, nx=True)
    count, _ = pipe.execute()
    return count <= limit


def record_view(r, product_id: str) -> None:
    r.zincrby("product_views", 1, product_id)


def top_products(r, n: int) -> list[str]:
    return r.zrevrange("product_views", 0, n - 1)
