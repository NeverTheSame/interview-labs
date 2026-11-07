"""Against a real MiniStack: on a clean stack every check fails, after the
reference solution every check passes. This is what proves a check really
tests the work."""

import pytest

from labkit.aws import Aws, reset_stack
from labkit.catalog import load_catalog, load_checks, load_solution
from labkit.runner import run_checks

from .test_catalog import LABS

pytestmark = pytest.mark.ministack


@pytest.mark.parametrize("lab", load_catalog(LABS), ids=lambda lab: lab.id)
def test_checks_fail_then_pass(lab):
    aws = Aws()
    reset_stack(aws.endpoint)
    checks = load_checks(lab)
    before = run_checks(lab, checks, aws)
    assert not any(r.passed for r in before), [r.task.id for r in before if r.passed]
    load_solution(lab)(aws)
    after = run_checks(lab, checks, aws)
    assert all(r.passed for r in after), [(r.task.id, r.message) for r in after if not r.passed]
    reset_stack(aws.endpoint)
