"""Package web serves the R3 portal: a server-rendered browser over the jobs
database, with HTMX for in-place filtering and row actions."""

from .app import create_app

__all__ = ["create_app"]
