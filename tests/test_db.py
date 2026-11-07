"""DSN handling. Query modules are exercised against a real database by the
route smoke test; this covers the pure parts."""

from r3.db import normalize_dsn


def test_normalize_dsn_strips_libpq_only_params():
    dsn = "postgresql://u:p@db.example.com/db?sslmode=require&channel_binding=require"
    assert normalize_dsn(dsn) == "postgresql://u:p@db.example.com/db?sslmode=require"


def test_normalize_dsn_leaves_plain_dsn_alone():
    assert normalize_dsn("postgresql://u@h/db") == "postgresql://u@h/db"
