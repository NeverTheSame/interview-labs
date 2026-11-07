# Redis: cache-aside, invalidation, rate limiting, leaderboards

**Level:** beginner · **Time:** about 45 minutes · **Services:** Redis

## Scenario

A product page reads from a slow database, gets hammered by one scraper, and
the marketing team wants a "most viewed" list. You add Redis in front of it.

You write Python in `answers/cache.py`. The checks call your functions with a
Redis client and a fake slow database that counts its reads.

## Before you start

```bash
docker compose up -d redis      # Redis 7.4 on localhost:56379
uv sync --extra labs            # the redis client
```

## Tasks

1. **Cache-aside.** `get_product` serves `product:{id}` from Redis and caches
   misses with a TTL.
2. **Invalidation.** After `update_product`, readers see the new data.
3. **Rate limit.** `allow_request` allows `limit` calls per window per user,
   and the counter always expires.
4. **Leaderboard.** `record_view` and `top_products` with a sorted set.

```bash
uv run labs check redis-caching --hint
redis-cli -p 56379 monitor      # watch the commands your code sends
```

## What is real here, and what is not

Redis is real. What ElastiCache adds on AWS (replicas, failover, cluster
mode, TLS, VPC access) is not here. `uv run labs show redis-caching` lists the
differences and the interview questions.
