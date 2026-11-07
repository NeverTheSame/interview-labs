"""boto3 clients pointed at the local stack. MiniStack takes any credentials;
a 12-digit access key would select an account, ``test`` means the default."""

from __future__ import annotations

import json
import os
import urllib.request
from functools import cache
from pathlib import Path

DEFAULT_ENDPOINT = "http://localhost:4566"
REGION = "us-east-1"


class Aws:
    def __init__(self, endpoint: str | None = None, region: str = REGION):
        self.endpoint = endpoint or os.environ.get("LABS_ENDPOINT", DEFAULT_ENDPOINT)
        self.region = region
        self.redis_url = os.environ.get("LABS_REDIS_URL", "redis://localhost:56379/0")
        # Labs where the learner writes code or SQL read it from here; None
        # means the lab's own answers/ folder. A reference solution points it
        # at solution/ instead.
        self.code_dir: Path | None = None

    @cache
    def client(self, service: str):
        import boto3

        return boto3.client(service, endpoint_url=self.endpoint, region_name=self.region,
                            aws_access_key_id="test", aws_secret_access_key="test")

    def __getattr__(self, service: str):
        # aws.sqs, aws.lambda_ (``lambda`` is a keyword), aws.dynamodb ...
        if service.startswith("_"):
            raise AttributeError(service)
        return self.client(service.rstrip("_"))


def health(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{endpoint}/_ministack/health", timeout=3) as r:
        return json.load(r)


def reset_stack(endpoint: str) -> None:
    """Wipe every service back to empty."""
    req = urllib.request.Request(f"{endpoint}/_ministack/reset", method="POST")
    with urllib.request.urlopen(req, timeout=30):
        pass
