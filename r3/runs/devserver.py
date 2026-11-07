"""Command r3-temporal-dev runs a local Temporal server with its web UI, for
developing runs without Temporal Cloud:

    uv run r3-temporal-dev            # Temporal on localhost:7233, UI on http://localhost:8233
    TEMPORAL_ADDRESS=localhost:7233 uv run r3-worker
    TEMPORAL_ADDRESS=localhost:7233 uv run r3-portal

It is the Temporal CLI's dev server (downloaded on first use and cached),
started through the SDK's test environment. Runs are kept in memory unless
``--db-file`` names a SQLite file."""

from __future__ import annotations

import argparse
import asyncio
import signal

from temporalio.testing import WorkflowEnvironment


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r3-temporal-dev",
        description="Run a local Temporal dev server with the web UI for R3. Point the portal and the worker "
                    "at it with TEMPORAL_ADDRESS=localhost:7233.")
    p.add_argument("--ip", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=7233, help="Temporal frontend port (default 7233)")
    p.add_argument("--ui-port", type=int, default=8233, help="web UI port (default 8233)")
    p.add_argument("--namespace", default="default", help="namespace to create (default 'default')")
    p.add_argument("--db-file", default="", help="SQLite file that keeps runs across restarts (default: in memory)")
    p.add_argument("--log-level", default="warn", help="dev server log level (default warn)")
    return p


async def _serve(args: argparse.Namespace) -> None:
    env = await WorkflowEnvironment.start_local(
        namespace=args.namespace, ip=args.ip, port=args.port, ui=True, ui_port=args.ui_port,
        dev_server_database_filename=args.db_file or None, dev_server_log_level=args.log_level,
    )
    host = "localhost" if args.ip in ("127.0.0.1", "0.0.0.0", "::1", "::") else args.ip
    print(f"Temporal dev server: {host}:{args.port} (namespace {args.namespace!r})", flush=True)
    print(f"Web UI:              http://{host}:{args.ui_port}/namespaces/{args.namespace}/workflows", flush=True)
    print(f"Use it:              TEMPORAL_ADDRESS={host}:{args.port} uv run r3-worker   (and the same for r3-portal)",
          flush=True)
    print("Ctrl-C stops it.", flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await stop.wait()
    finally:
        await env.shutdown()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
