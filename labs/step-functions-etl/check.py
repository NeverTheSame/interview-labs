"""Checks for the ETL state machine: each runs a real execution and reads its history."""

import json
import time

from labkit.runner import CheckFailed


def _machine(aws):
    for m in aws.stepfunctions.list_state_machines()["stateMachines"]:
        if m["name"] == "nightly-etl":
            return m
    raise CheckFailed("state machine 'nightly-etl' not found")


def _alerts_url(aws):
    try:
        return aws.sqs.get_queue_url(QueueName="etl-alerts")["QueueUrl"]
    except aws.sqs.exceptions.QueueDoesNotExist:
        raise CheckFailed("queue 'etl-alerts' not found")


def _execute(aws, payload, timeout=30):
    arn = _machine(aws)["stateMachineArn"]
    ex = aws.stepfunctions.start_execution(stateMachineArn=arn, input=json.dumps(payload))["executionArn"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = aws.stepfunctions.describe_execution(executionArn=ex)
        if run["status"] != "RUNNING":
            events = aws.stepfunctions.get_execution_history(executionArn=ex, maxResults=1000)["events"]
            return run, events
        time.sleep(0.5)
    raise CheckFailed(f"the execution with input {payload} was still running after {timeout} s")


def _extract_calls(events):
    return sum(1 for e in events if e["type"] in ("LambdaFunctionScheduled", "TaskScheduled")
               and "extract" in json.dumps(e.get("taskScheduledEventDetails") or e.get("lambdaFunctionScheduledEventDetails") or {}))


def _drain(aws, url):
    bodies = []
    while True:
        msgs = aws.sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=1).get("Messages", [])
        if not msgs:
            return bodies
        for m in msgs:
            bodies.append(m["Body"])
            aws.sqs.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])


def check_function(aws):
    try:
        fn = aws.lambda_.get_function_configuration(FunctionName="extract")
    except aws.lambda_.exceptions.ResourceNotFoundException:
        raise CheckFailed("function 'extract' not found")
    out = aws.lambda_.invoke(FunctionName="extract", Payload=json.dumps({"date": "2026-11-01"}))
    body = json.loads(out["Payload"].read() or b"{}")
    if out.get("FunctionError") or body.get("rows") is None:
        raise CheckFailed(f"'extract' did not return a row count: {body}")
    if fn.get("Handler") != "extract.handler":
        raise CheckFailed(f"handler is {fn.get('Handler')!r}; use extract.handler")


def check_alerts(aws):
    _alerts_url(aws)


def check_happy_path(aws):
    if _machine(aws).get("type", "STANDARD") != "STANDARD":
        raise CheckFailed("use a STANDARD workflow")
    run, _ = _execute(aws, {"date": "2026-11-01"})
    if run["status"] != "SUCCEEDED":
        raise CheckFailed(f"the run {run['status']}: {run.get('error')} {run.get('cause', '')[:200]}")
    if "1250" not in run.get("output", ""):
        raise CheckFailed(f"the output does not carry the row count: {run.get('output', '')[:200]}")


def check_retry(aws):
    run, events = _execute(aws, {"date": "2026-11-01", "fail": "transient"})
    calls = _extract_calls(events)
    if calls < 3:
        raise CheckFailed(f"'extract' ran {calls} time(s) on a TransientError; it should be retried at least twice")
    if run["status"] != "FAILED":
        raise CheckFailed(f"a run whose retries all fail should end FAILED, not {run['status']}")


def check_no_retry(aws):
    url = _alerts_url(aws)
    _drain(aws, url)
    run, events = _execute(aws, {"date": "2026-11-01", "fail": "bad-data"})
    calls = _extract_calls(events)
    if calls != 1:
        raise CheckFailed(f"'extract' ran {calls} times on a BadDataError; bad data should not be retried")
    if run["status"] != "FAILED":
        raise CheckFailed(f"a run that hit bad data should end FAILED, not {run['status']}")
    alerts = _drain(aws, url)
    if not any("BadDataError" in body for body in alerts):
        raise CheckFailed(f"no message naming BadDataError reached 'etl-alerts' (got {len(alerts)} message(s))")
