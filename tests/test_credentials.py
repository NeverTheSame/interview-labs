"""Passwords and session tokens are never stored as the visitor sent them:
passwords as a salted scrypt hash, session tokens as a SHA-256 digest. The real
``UserQueries`` SQL runs on a fake pool, so no database is touched."""

from __future__ import annotations

import hashlib
from typing import Any

from r3.db.users import (
    UserQueries,
    hash_session_token,
    password_matches,
    password_needs_rehash,
    stored_password,
)


class Pool:
    def __init__(self, row: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.row = row

    async def execute(self, sql, *args):
        self.calls.append((sql.strip(), args))

    async def fetchrow(self, sql, *args):
        self.calls.append((sql.strip(), args))
        return self.row


def _queries(row=None) -> tuple[UserQueries, Pool]:
    q = UserQueries()
    q.pool = Pool(row)  # type: ignore[assignment]
    return q, q.pool  # type: ignore[return-value]


def test_a_new_password_is_stored_as_a_salted_scrypt_hash():
    a, b = stored_password("correct horse battery"), stored_password("correct horse battery")
    assert a.startswith("scrypt$") and "correct horse" not in a
    assert a != b  # random salt
    assert password_matches(a, "correct horse battery")
    assert not password_matches(a, "correct horse batterY")


def test_a_legacy_plain_text_row_still_matches_and_wants_a_rehash():
    assert password_matches("pass1234", "pass1234")
    assert not password_matches("pass1234", "pass12345")
    assert password_needs_rehash("pass1234")
    assert not password_needs_rehash(stored_password("long enough password"))


def test_a_malformed_hash_never_matches():
    assert not password_matches("scrypt$16384$8$1$!!$!!", "anything")


async def test_login_against_a_plain_text_row_rehashes_it_in_place():
    q, pool = _queries({"id": 7, "account_name": "Morgan", "password": "pass1234", "is_admin": False})
    user = await q.check_login("morgan", "pass1234")
    assert user is not None and user.id == 7
    update = [(s, a) for s, a in pool.calls if s.startswith("UPDATE r3_users SET password")]
    assert len(update) == 1
    new_hash, uid, old = update[0][1]
    assert new_hash.startswith("scrypt$") and uid == 7 and old == "pass1234"
    assert password_matches(new_hash, "pass1234")


async def test_a_wrong_password_does_not_rehash_or_log_in():
    q, pool = _queries({"id": 7, "account_name": "Morgan", "password": "pass1234", "is_admin": False})
    assert await q.check_login("morgan", "nope") is None
    assert not any(s.startswith("UPDATE") for s, _ in pool.calls)


async def test_a_hashed_row_is_not_rewritten_on_login():
    q, pool = _queries({"id": 7, "account_name": "K", "password": stored_password("long enough password"),
                        "is_admin": False})
    assert await q.check_login("k", "long enough password") is not None
    assert not any(s.startswith("UPDATE") for s, _ in pool.calls)


async def test_an_unknown_account_is_refused():
    q, _ = _queries(None)
    assert await q.check_login("ghost", "whatever-password") is None


async def test_sessions_store_only_the_digest_of_the_token():
    q, pool = _queries()
    token = await q.create_session(3)
    (sql, args), = [c for c in pool.calls if c[0].startswith("INSERT INTO r3_sessions")]
    assert args == (hashlib.sha256(token.encode()).hexdigest(), 3)
    assert token not in args
    assert hash_session_token(token) == args[0]


async def test_session_lookup_and_delete_hash_the_incoming_token():
    q, pool = _queries({"id": 3, "account_name": "K", "is_admin": False})
    await q.session_user("raw-token")
    await q.delete_session("raw-token")
    digest = hash_session_token("raw-token")
    assert all(a == (digest,) for s, a in pool.calls if "r3_sessions" in s)
    assert len([1 for s, _ in pool.calls if "r3_sessions" in s]) == 2


async def test_ensure_user_tables_drops_sessions_holding_a_raw_token():
    q, pool = _queries()
    await q.ensure_user_tables()
    assert any(s.startswith("DELETE FROM r3_sessions") and "length(token)" in s for s, _ in pool.calls)
