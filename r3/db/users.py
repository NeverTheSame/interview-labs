"""Queries over ``r3_users`` and ``r3_sessions``: the portal's own accounts
and the sign-in sessions their browsers hold. R3 owns both tables and creates
them at startup, like ``r3_job_labels``.

Passwords are stored as a salted scrypt hash, never as typed
(:func:`stored_password`, :func:`password_matches`). Rows written before
hashing hold the password as plain text; :meth:`UserQueries.check_login`
recognises them, and on the first successful sign-in rewrites the row as a
hash, so the table converges without a bulk migration. A session token is
likewise stored only as its SHA-256 digest (:func:`hash_session_token`); the
browser cookie keeps the raw token. A real auth provider can replace all of
this later; nothing outside this module knows the storage format.

Account names are unique ignoring case (``Morgan`` and ``morgan`` are one
account) but are shown as they were typed at sign-up.

Operators (``User.is_admin``) see the whole console. Public sign-up never
makes one: an operator is an account whose ``is_admin`` column was set out of
band (``python -m r3.accounts create NAME --admin``), or whose name is in
``R3_ADMIN_ACCOUNTS`` (comma-separated, any case), read from the environment
each time an account is loaded so a restart is not needed to grant or withdraw
it. Because that list grants by name, the sign-up form refuses every name in it
(see ``r3.web.routes_auth``); such an account can only be created with the
command above.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import secrets

import asyncpg
from pydantic import BaseModel

from ._base import QueryBase

USERS_DDL = """
CREATE TABLE IF NOT EXISTS r3_users (
    id           bigserial PRIMARY KEY,
    account_name text NOT NULL,
    password     text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
)"""

# Added after the table first shipped, hence an ALTER rather than a column above.
USERS_ADMIN_DDL = """
ALTER TABLE r3_users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false"""

USERS_NAME_INDEX_DDL = """
CREATE UNIQUE INDEX IF NOT EXISTS r3_users_account_name_key ON r3_users (lower(account_name))"""

SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS r3_sessions (
    token      text PRIMARY KEY,
    user_id    bigint NOT NULL REFERENCES r3_users(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now()
)"""

SESSIONS_USER_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS r3_sessions_user_idx ON r3_sessions (user_id)"""

# Sessions created before tokens were hashed hold the raw token (43 characters);
# a digest is 64. Those rows can never match a lookup again, so they are
# removed: everyone signed in at the time of this change signs in once more.
SESSIONS_RAW_TOKEN_CLEANUP = "DELETE FROM r3_sessions WHERE length(token) <> 64"

# A session older than this no longer signs anyone in; the browser cookie is
# given the same lifetime.
SESSION_DAYS = 30


class User(BaseModel):
    """A signed-in account. ``r3.web.auth`` re-exports this as its ``User``."""

    id: int
    account_name: str
    # Operators see the whole console (settings, labeling);
    # everyone else sees only their own job search.
    is_admin: bool = False


class AccountNameTaken(Exception):
    """Raised by :meth:`UserQueries.create_user` when the name is already in
    use, compared without regard to case."""

    def __init__(self, account_name: str) -> None:
        super().__init__(f"account name {account_name!r} is already taken")
        self.account_name = account_name


ADMIN_ACCOUNTS_ENV = "R3_ADMIN_ACCOUNTS"

_CREATE_USER = """
INSERT INTO r3_users (account_name, password, is_admin)
VALUES ($1, $2, $3)
RETURNING id, account_name, is_admin"""


def admin_account_names() -> frozenset[str]:
    """The lower-cased names in ``R3_ADMIN_ACCOUNTS``, read at call time."""
    raw = os.environ.get(ADMIN_ACCOUNTS_ENV, "")
    return frozenset(name.strip().lower() for name in raw.split(",") if name.strip())


def account_user(user_id: int, account_name: str, is_admin: bool) -> User:
    """The ``User`` for a stored account: an operator when its column says so
    or when ``R3_ADMIN_ACCOUNTS`` names it. Every loaded account goes through
    here so the override applies everywhere."""
    admin = bool(is_admin) or account_name.lower() in admin_account_names()
    return User(id=user_id, account_name=account_name, is_admin=admin)


_SCRYPT_PREFIX = "scrypt$"
# 64 MiB of memory per hash (128 * N * r), ~0.1 s; the parameters are written
# into every stored hash, so they can be raised later without breaking old rows.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**16, 8, 1
_SCRYPT_MAXMEM = 128 * 1024 * 1024
_SCRYPT_MAX_N = 2**20


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=_SCRYPT_MAXMEM, dklen=32)


def stored_password(password: str) -> str:
    """What goes into ``r3_users.password`` for a new password:
    ``scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>`` with a random 16-byte salt."""
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    return f"{_SCRYPT_PREFIX}{_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def _parse_hash(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    """The parts of a stored hash, or None when ``stored`` is not one (a
    legacy plain-text row, or a damaged value)."""
    if not stored.startswith(_SCRYPT_PREFIX):
        return None
    try:
        n, r, p, salt, digest = stored[len(_SCRYPT_PREFIX):].split("$")
        parts = int(n), int(r), int(p), base64.b64decode(salt, validate=True), base64.b64decode(digest, validate=True)
    except ValueError:
        return None
    if not (1 < parts[0] <= _SCRYPT_MAX_N and parts[0] & (parts[0] - 1) == 0 and 0 < parts[1] <= 32
            and 0 < parts[2] <= 4 and parts[3] and parts[4]):
        return None
    return parts


def password_matches(stored: str, password: str) -> bool:
    """Whether ``password`` is the one ``stored`` was made from. A hash is
    recomputed with its own salt and parameters; a value that is not a hash is
    a legacy plain-text row and is compared directly. Both comparisons are
    constant-time so they leak nothing."""
    parts = _parse_hash(stored)
    if parts is None:
        if stored.startswith(_SCRYPT_PREFIX):
            return False  # a damaged hash is not a password
        return hmac.compare_digest(stored.encode(), password.encode())
    n, r, p, salt, digest = parts
    try:
        candidate = _scrypt(password, salt, n, r, p)
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, digest)


def password_needs_rehash(stored: str) -> bool:
    """Whether ``stored`` is a legacy plain-text value (or uses weaker
    parameters than today's) and should be rewritten after a good sign-in."""
    parts = _parse_hash(stored)
    return parts is None or parts[:3] != (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)


def hash_session_token(token: str) -> str:
    """What ``r3_sessions.token`` holds for a session: the SHA-256 hex digest
    of the raw token in the cookie. The token is 256 random bits, so a plain
    digest is enough; a database leak does not yield usable cookies."""
    return hashlib.sha256(token.encode()).hexdigest()


# Checked against when the account name does not exist, so an unknown name
# costs the same time as a wrong password.
_DUMMY_HASH = ""


def new_session_token() -> str:
    """An unguessable session id for the cookie."""
    return secrets.token_urlsafe(32)


class UserQueries(QueryBase):
    async def ensure_user_tables(self) -> None:
        """Create the account and session tables when they do not exist yet."""
        await self.pool.execute(USERS_DDL)
        await self.pool.execute(USERS_ADMIN_DDL)
        await self.pool.execute(USERS_NAME_INDEX_DDL)
        await self.pool.execute(SESSIONS_DDL)
        await self.pool.execute(SESSIONS_RAW_TOKEN_CLEANUP)
        await self.pool.execute(SESSIONS_USER_INDEX_DDL)

    async def create_user(self, account_name: str, password: str, *, is_admin: bool = False) -> User:
        """Add an account. Raises :class:`AccountNameTaken` when the name is in
        use; the unique index decides, so two racing sign-ups cannot both win.
        ``is_admin`` is for out-of-band callers (``r3.accounts``); the sign-up
        route never passes it, so no visitor can become an operator."""
        hashed = await asyncio.to_thread(stored_password, password)
        try:
            row = await self.pool.fetchrow(_CREATE_USER, account_name, hashed, is_admin)
        except asyncpg.UniqueViolationError:
            raise AccountNameTaken(account_name) from None
        return account_user(row["id"], row["account_name"], row["is_admin"])

    async def check_login(self, account_name: str, password: str) -> User | None:
        """The account these credentials belong to, or None. A wrong name and a
        wrong password look the same to the caller."""
        row = await self.pool.fetchrow(
            "SELECT id, account_name, password, is_admin FROM r3_users WHERE lower(account_name) = lower($1)",
            account_name,
        )
        global _DUMMY_HASH
        if row is None:
            _DUMMY_HASH = _DUMMY_HASH or await asyncio.to_thread(stored_password, secrets.token_urlsafe(16))
            await asyncio.to_thread(password_matches, _DUMMY_HASH, password)
            return None
        stored = row["password"]
        if not await asyncio.to_thread(password_matches, stored, password):
            return None
        if password_needs_rehash(stored):
            # Compare-and-swap on the old value so a concurrent password
            # change is never overwritten with a stale hash.
            await self.pool.execute("UPDATE r3_users SET password = $1 WHERE id = $2 AND password = $3",
                                    await asyncio.to_thread(stored_password, password), row["id"], stored)
        return account_user(row["id"], row["account_name"], row["is_admin"])

    async def create_session(self, user_id: int) -> str:
        """Start a session for the account and return its token."""
        token = new_session_token()
        await self.pool.execute("INSERT INTO r3_sessions (token, user_id) VALUES ($1, $2)",
                                hash_session_token(token), user_id)
        return token

    async def session_user(self, token: str) -> User | None:
        """The account a live session token belongs to, or None."""
        row = await self.pool.fetchrow(
            f"""SELECT u.id, u.account_name, u.is_admin FROM r3_sessions s JOIN r3_users u ON u.id = s.user_id
                WHERE s.token = $1 AND s.created_at > now() - interval '{SESSION_DAYS} days'""",
            hash_session_token(token),
        )
        return account_user(row["id"], row["account_name"], row["is_admin"]) if row else None

    async def user_by_id(self, user_id: int) -> User | None:
        """The account with this id, admin flag included; None when it is gone."""
        row = await self.pool.fetchrow("SELECT id, account_name, is_admin FROM r3_users WHERE id = $1", user_id)
        return None if row is None else account_user(row["id"], row["account_name"], row["is_admin"])

    async def delete_session(self, token: str) -> None:
        """End a session; a token that is already gone is not an error."""
        await self.pool.execute("DELETE FROM r3_sessions WHERE token = $1", hash_session_token(token))
