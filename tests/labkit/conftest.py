import fcntl
import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

ENDPOINT = os.environ.get("LABS_ENDPOINT", "http://localhost:4566")


def pytest_configure(config):
    config.addinivalue_line("markers", "ministack: needs a running MiniStack (docker compose up -d ministack); skipped otherwise")


def _ministack_up() -> bool:
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/_ministack/health", timeout=2):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if _ministack_up():
        return
    skip = pytest.mark.skip(reason=f"MiniStack not reachable at {ENDPOINT}")
    for item in items:
        if "ministack" in item.keywords:
            item.add_marker(skip)


# The live tests reset the whole stack, and every checkout on this machine
# (worktrees included) shares it. Hold one lock per test session, so two
# runs at the same time wait for each other instead of wiping each other's
# state mid-test.
_LOCK_PATH = Path(tempfile.gettempdir()) / "interview-labs-stack.lock"
_held = None


@pytest.fixture(autouse=True)
def _stack_lock(request):
    global _held
    if "ministack" in request.keywords and _held is None:
        _held = _LOCK_PATH.open("w")
        fcntl.flock(_held, fcntl.LOCK_EX)
    yield
