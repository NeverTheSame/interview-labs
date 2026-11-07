"""A check is only useful if it rejects work that is almost right. Each case
applies the reference solution, breaks one thing a learner commonly gets
wrong, and expects exactly that task to fail."""

import json
from pathlib import Path

import pytest

from labkit.aws import Aws, reset_stack
from labkit.catalog import find_lab, load_checks, load_solution
from labkit.runner import run_checks

from .test_catalog import LABS

pytestmark = pytest.mark.ministack


def _no_partial_batches(aws):
    m = aws.lambda_.list_event_source_mappings(FunctionName="process-order")["EventSourceMappings"][0]
    aws.lambda_.update_event_source_mapping(UUID=m["UUID"], FunctionResponseTypes=[])


def _revenue_ascending(aws):
    q = next(q for q in aws.athena.batch_get_named_query(
        NamedQueryIds=aws.athena.list_named_queries()["NamedQueryIds"])["NamedQueries"] if q["Name"] == "revenue_by_country")
    aws.athena.delete_named_query(NamedQueryId=q["NamedQueryId"])
    aws.athena.create_named_query(Name="revenue_by_country", Database="shop",
                                  QueryString=q["QueryString"].replace("DESC", "ASC"))


def _retry_everything(aws):
    m = next(m for m in aws.stepfunctions.list_state_machines()["stateMachines"] if m["name"] == "nightly-etl")
    d = json.loads(aws.stepfunctions.describe_state_machine(stateMachineArn=m["stateMachineArn"])["definition"])
    d["States"]["Extract"]["Retry"][0]["ErrorEquals"] = ["States.ALL"]
    aws.stepfunctions.update_state_machine(stateMachineArn=m["stateMachineArn"], definition=json.dumps(d))


def _merge_without_update(aws):
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    shutil.copytree(LABS / "snowflake-elt" / "solution" / "answers", work, dirs_exist_ok=True)
    sql = (work / "upsert_orders.sql").read_text()
    (work / "upsert_orders.sql").write_text(
        sql.replace("when matched then update set status = source.status, customer_id = source.customer_id\n", ""))
    aws.code_dir = work


def _limiter_without_expiry(aws):
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    shutil.copytree(LABS / "redis-caching" / "solution" / "answers", work, dirs_exist_ok=True)
    code = (work / "cache.py").read_text()
    (work / "cache.py").write_text(code.replace("    pipe.expire(key, window, nx=True)\n    count, _ = pipe.execute()",
                                                "    (count,) = pipe.execute()"))
    aws.code_dir = work


@pytest.mark.parametrize("lab_id, break_it, task", [
    ("lambda-sqs-pipeline", _no_partial_batches, "wiring"),
    ("athena-data-lake", _revenue_ascending, "revenue"),
    ("step-functions-etl", _retry_everything, "no-retry"),
    ("snowflake-elt", _merge_without_update, "merge"),
    ("redis-caching", _limiter_without_expiry, "rate-limit"),
])
def test_near_miss_fails_its_task(lab_id, break_it, task):
    lab = find_lab(LABS, lab_id)
    aws = Aws()
    reset_stack(aws.endpoint)
    load_solution(lab)(aws)
    break_it(aws)
    result = run_checks(lab, load_checks(lab), aws, only=task)[0]
    reset_stack(aws.endpoint)
    assert not result.passed, f"{lab_id}/{task} passed although it was broken"
