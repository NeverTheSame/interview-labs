"""Request-size limits that act before the body is parsed.

Starlette spools a multipart upload to memory/disk while it parses the form,
so a size check inside the route (``file.read(MAX + 1)``) runs only after the
whole body has been received. :class:`UploadSizeLimit` closes that gap for the
two resume-upload endpoints:

- a declared ``Content-Length`` over the limit is never read: the route is
  called with an empty body and told (``scope["r3.upload_too_large"]``) so it
  answers with its usual "larger than 5 MB" message and status 413;
- a body with no ``Content-Length`` (chunked) is counted as it arrives and cut
  off at the limit with a 413 JSON response.
"""

from __future__ import annotations

from typing import Any

from starlette.exceptions import HTTPException

from r3.resume import MAX_UPLOAD_BYTES

from .auth import route_path

__all__ = ["UPLOAD_PATHS", "UPLOAD_TOO_LARGE_KEY", "UploadSizeLimit", "too_large_message"]

UPLOAD_PATHS = frozenset({"/resume/upload", "/api/resume/upload"})
UPLOAD_TOO_LARGE_KEY = "r3.upload_too_large"
# The file plus multipart framing and the title field.
BODY_LIMIT = MAX_UPLOAD_BYTES + 64 * 1024


def too_large_message() -> str:
    return (f"That file is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB, the upload limit. "
            "Save a smaller copy or paste the text instead.")


class UploadSizeLimit:
    def __init__(self, app: Any, limit: int = BODY_LIMIT) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or route_path(scope) not in UPLOAD_PATHS:
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.strip().isdigit() and int(declared) > self.limit:
            scope[UPLOAD_TOO_LARGE_KEY] = True

            async def empty() -> dict[str, Any]:
                return {"type": "http.request", "body": b"", "more_body": False}

            await self.app(scope, empty, send)
            return

        seen = 0

        async def counting() -> dict[str, Any]:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    # Raised inside the body read, so the app's own handlers
                    # turn it into the 413 response; the rest is not read.
                    raise HTTPException(413, too_large_message())
            return message

        await self.app(scope, counting, send)
