"""Checks for the Redis caching lab. They call your functions in
answers/cache.py against the lab's Redis and a fake slow database that counts
its calls. Only the keys the tasks name are touched."""

import importlib.util
import uuid
from pathlib import Path

from labkit.runner import CheckFailed

HERE = Path(__file__).parent


class FakeDb:
    def __init__(self):
        self.products = {}
        self.loads = 0

    def load_product(self, product_id):
        self.loads += 1
        return dict(self.products[product_id])

    def save_product(self, product_id, fields):
        self.products[product_id].update(fields)


def _code(aws):
    path = (aws.code_dir or HERE / "answers") / "cache.py"
    spec = importlib.util.spec_from_file_location(f"redis_lab_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise CheckFailed(f"{path.name} does not load: {type(e).__name__}: {e}")
    return module


def _redis(aws):
    try:
        import redis
    except ImportError:
        raise CheckFailed("the redis client is not installed: run `uv sync --extra labs`")
    r = redis.Redis.from_url(aws.redis_url, decode_responses=True, socket_connect_timeout=2)
    try:
        r.ping()
    except redis.ConnectionError:
        raise CheckFailed(f"no Redis at {aws.redis_url}; start it with `docker compose up -d redis`")
    return r


def _product(r):
    pid = f"check-{uuid.uuid4().hex[:8]}"
    db = FakeDb()
    db.products[pid] = {"id": pid, "name": "Mechanical keyboard", "price": 89}
    r.delete(f"product:{pid}")
    return db, pid


def check_cache_aside(aws):
    code, r = _code(aws), _redis(aws)
    db, pid = _product(r)
    try:
        first, second = code.get_product(r, db, pid), code.get_product(r, db, pid)
        if first != db.products[pid] or second != db.products[pid]:
            raise CheckFailed(f"get_product returned {second}, expected {db.products[pid]}")
        if db.loads != 1:
            raise CheckFailed(f"the database was read {db.loads} times for two calls; the second should come from Redis")
        ttl = r.ttl(f"product:{pid}")
        if ttl == -2:
            raise CheckFailed(f"nothing cached under product:{pid}")
        if not 60 <= ttl <= 600:
            raise CheckFailed(f"product:{pid} has TTL {ttl}; give cached products a TTL between 60 and 600 seconds")
    finally:
        r.delete(f"product:{pid}")


def check_invalidation(aws):
    code, r = _code(aws), _redis(aws)
    db, pid = _product(r)
    try:
        code.get_product(r, db, pid)
        code.get_product(r, db, pid)
        if db.loads != 1:
            raise CheckFailed("get_product does not cache yet, so there is nothing to invalidate: do cache-aside first")
        code.update_product(r, db, pid, {"price": 79})
        after = code.get_product(r, db, pid)
        if after.get("price") != 79:
            raise CheckFailed(f"after update_product the price is still {after.get('price')}: the cache served the old copy")
        code.get_product(r, db, pid)
        if db.loads != 2:
            raise CheckFailed("after the update the product is no longer cached: cache it again on the next read")
    finally:
        r.delete(f"product:{pid}")


def check_rate_limit(aws):
    code, r = _code(aws), _redis(aws)
    user = f"check-{uuid.uuid4().hex[:8]}"
    key = f"rate:{user}"
    try:
        answers = [code.allow_request(r, user, limit=5, window=60) for _ in range(7)]
        if answers != [True] * 5 + [False] * 2:
            raise CheckFailed(f"7 calls with limit 5 returned {answers}")
        ttl = r.ttl(key)
        if ttl == -2:
            raise CheckFailed(f"no counter under {key}")
        if not 0 < ttl <= 60:
            raise CheckFailed(f"{key} has TTL {ttl}; the counter must expire with the window or the user stays blocked")
    finally:
        r.delete(key)


def check_leaderboard(aws):
    code, r = _code(aws), _redis(aws)
    tag = uuid.uuid4().hex[:6]
    views = {f"{tag}-a": 3, f"{tag}-b": 7, f"{tag}-c": 1, f"{tag}-d": 5}
    saved = r.zrange("product_views", 0, -1, withscores=True)
    r.delete("product_views")
    try:
        for pid, count in views.items():
            for _ in range(count):
                code.record_view(r, pid)
        got = code.top_products(r, 3)
        want = [f"{tag}-b", f"{tag}-d", f"{tag}-a"]
        if list(got) != want:
            raise CheckFailed(f"top_products(3) returned {got}, expected {want}")
    finally:
        r.delete("product_views")
        if saved:
            r.zadd("product_views", dict(saved))
