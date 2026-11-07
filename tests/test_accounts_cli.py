"""The out-of-band way to make an account (and the first operator)."""

from __future__ import annotations

import pytest

from r3.accounts import create_account
from r3.db.users import AccountNameTaken, User


class Db:
    def __init__(self, taken=False):
        self.calls, self.taken = [], taken

    async def ensure_user_tables(self):
        self.calls.append("ensure")

    async def create_user(self, name, password, *, is_admin=False):
        if self.taken:
            raise AccountNameTaken(name)
        self.calls.append(("create", name, is_admin))
        return User(id=1, account_name=name, is_admin=is_admin)


async def test_create_account_can_make_an_operator():
    db = Db()
    msg = await create_account(db, "morgan", "a-long-enough-pass", True)
    assert db.calls == ["ensure", ("create", "morgan", True)] and "operator" in msg


@pytest.mark.parametrize("name, password, error", [
    ("x", "a-long-enough-pass", "account name"),
    ("morgan", "abc", "at least"),
])
async def test_create_account_applies_the_sign_up_rules(name, password, error):
    with pytest.raises(ValueError, match=error):
        await create_account(Db(), name, password, False)


async def test_create_account_reports_a_taken_name():
    with pytest.raises(ValueError, match="already taken"):
        await create_account(Db(taken=True), "morgan", "a-long-enough-pass", False)
