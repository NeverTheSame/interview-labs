"""Shared plumbing for the query mixins: the pool attribute and the small
helpers every module uses to build parameterized SQL."""

from __future__ import annotations

from typing import Any

import asyncpg


class QueryBase:
    """Every query mixin reads ``self.pool``; ``Database`` supplies it."""

    pool: asyncpg.Pool


class WhereBuilder:
    """Accumulates WHERE fragments and their positional arguments.

    ``add`` renders the clause with the next ``$n`` placeholder so a value is
    always passed as a parameter, never interpolated into the SQL text. The
    clause template uses ``{n}`` where the placeholder number goes.
    """

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.args: list[Any] = []

    def add(self, clause: str, value: Any) -> None:
        self.args.append(value)
        self.parts.append(clause.format(n=len(self.args)))

    def raw(self, clause: str) -> None:
        """Append a fragment that carries no user-supplied value."""
        self.parts.append(clause)

    @property
    def next_n(self) -> int:
        """The placeholder number the next appended argument will get."""
        return len(self.args) + 1

    def clause(self, prefix: str = "WHERE ") -> str:
        if not self.parts:
            return ""
        return prefix + " AND ".join(self.parts)
