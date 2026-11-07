"""Command ``python -m r3.accounts``: create accounts outside the sign-up form.

    python -m r3.accounts create NAME [--admin]

The password is read from the terminal (never from an argument, so it does not
land in shell history). ``--admin`` sets the account's ``is_admin`` column: this
is how the first operator is made, because public sign-up never grants admin and
refuses any name listed in ``R3_ADMIN_ACCOUNTS``. Run it against the portal's
``DATABASE_URL``; it creates the account tables if they do not exist yet.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys

from dotenv import find_dotenv, load_dotenv

from r3.db import Database
from r3.db.users import AccountNameTaken
from r3.web.routes_auth import NAME_RE, PASSWORD_MIN


async def create_account(db: Database, name: str, password: str, admin: bool) -> str:
    """Create the account and say what happened. Raises ``ValueError`` for a
    name or password the sign-up form would refuse."""
    if not NAME_RE.fullmatch(name):
        raise ValueError("account name must be 3-40 characters: letters, digits, dot, underscore or dash")
    if len(password) < PASSWORD_MIN:
        raise ValueError(f"password must be at least {PASSWORD_MIN} characters")
    await db.ensure_user_tables()
    try:
        user = await db.create_user(name, password, is_admin=admin)
    except AccountNameTaken:
        raise ValueError(f"account name {name!r} is already taken") from None
    return f"created account {user.account_name!r} (id {user.id})" + (" as an operator" if admin else "")


async def _run(name: str, admin: bool) -> str:
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise ValueError("DATABASE_URL is not set")
    password = getpass.getpass("Password: ")
    if getpass.getpass("Repeat password: ") != password:
        raise ValueError("the two passwords do not match")
    db = await Database.open(dsn)
    try:
        return await create_account(db, name, password, admin)
    finally:
        await db.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m r3.accounts", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create an account")
    create.add_argument("name")
    create.add_argument("--admin", action="store_true", help="make it an operator")
    args = parser.parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    try:
        print(asyncio.run(_run(args.name, args.admin)))
    except ValueError as e:
        sys.exit(f"r3.accounts: {e}")


if __name__ == "__main__":
    main()
