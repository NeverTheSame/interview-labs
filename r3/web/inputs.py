"""Shared limits and cleaning for what users send: ids, numbers and text."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import Path
from starlette.datastructures import QueryParams

# A row id as a path parameter: Postgres bigint, so a larger number names no row.
DbId = Annotated[int, Path(ge=1, le=2**63 - 1)]

MAX_DAYS = 3650
MAX_SALARY = 10_000_000
MAX_PAGE = 100_000

# C0 control characters except tab and newline. Postgres text cannot hold NUL.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")


def clean_text(s: str) -> str:
    """Drop NUL and the other C0 control characters; tab and newline stay."""
    return _CONTROL.sub("", s)


def clean_query(q: QueryParams) -> QueryParams:
    """The query string with every value passed through :func:`clean_text`."""
    return QueryParams([(k, clean_text(v)) for k, v in q.multi_items()])
