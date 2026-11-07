import os
import urllib.request

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
