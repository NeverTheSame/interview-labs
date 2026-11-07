"""Command portal serves the R3 web portal.

Run with ``r3-portal`` or ``python -m r3.portal``. ``r3-label`` is the same
portal with the labeling page opened in the browser once it answers, for a
labeling session that starts with one command.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn
from dotenv import find_dotenv, load_dotenv

from r3 import obs
from r3.web import create_app


# R3_AUTH values that start the portal with sign-in off. For a deployment that
# sits behind another application's authenticated proxy, which has already
# signed the person in; anything else (or unset) keeps R3's own sign-in on.
_AUTH_OFF = frozenset({"off", "0", "false", "no"})


def auth_enabled(value: str | None) -> bool:
    """Whether ``R3_AUTH`` (its raw value, None when unset) keeps sign-in on."""
    return (value or "").strip().lower() not in _AUTH_OFF


def main(open_path: str = "") -> None:
    """Serve the portal. With ``open_path`` (``r3-label``), open that page in
    the default browser as soon as the portal answers."""
    # .env is a convenience for local runs; in deployment the environment
    # is already populated, so a missing file is not an error. Read first:
    # the Axiom settings may live there.
    load_dotenv(find_dotenv(usecwd=True))
    shipper = obs.setup_logging("r3-portal", os.environ)
    tracing = obs.setup_tracing("r3-portal", obs.AxiomConfig.from_env(os.environ))
    log = logging.getLogger("r3")

    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        log.error("DATABASE_URL is not set")
        sys.exit(1)

    port = int(os.environ.get("PORT") or "8090")
    log.info("portal listening on http://localhost:%d", port)
    host = os.environ.get("HOST", "0.0.0.0")
    auth = auth_enabled(os.environ.get("R3_AUTH"))
    if not auth:
        log.warning("R3_AUTH=off: sign-in is disabled; every visitor is the local operator")
    config = uvicorn.Config(
        create_app(dsn=dsn, root_path=os.environ.get("R3_ROOT_PATH", ""), auth=auth),
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=120,
    )
    server = uvicorn.Server(config)
    if open_path:
        open_when_up(f"http://localhost:{port}{os.environ.get('R3_ROOT_PATH', '')}{open_path}", port)
    try:
        if host == "::":
            # asyncio's default IPv6 listener sets IPV6_V6ONLY; explicitly create
            # a dual-stack socket so IPv4 and IPv6 clients reach the same process.
            with socket.create_server((host, port), family=socket.AF_INET6, dualstack_ipv6=True) as listener:
                server.run(sockets=[listener])
        else:
            server.run()
    finally:
        # Send what is still queued for Axiom before the process exits.
        tracing.shutdown()
        if shipper is not None:
            shipper.close()


def open_when_up(url: str, port: int, *, timeout_s: float = 60.0, poll_s: float = 0.5) -> threading.Thread:
    """Open ``url`` in the browser once ``/healthz`` on ``port`` answers (or
    after ``timeout_s``, so a slow database still ends in an open page with
    its own error). Runs on a daemon thread so the server starts meanwhile."""

    def wait() -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(poll_s)
        webbrowser.open(url)

    thread = threading.Thread(target=wait, name="r3-open-browser", daemon=True)
    thread.start()
    return thread


def label() -> None:
    """``r3-label``: the portal with ``/label`` opened in the browser."""
    main(open_path="/label")


if __name__ == "__main__":
    main()
