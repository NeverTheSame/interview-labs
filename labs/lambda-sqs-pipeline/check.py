"""Checks for the order pipeline. Each ``check_<name>`` gets an ``Aws`` and
raises CheckFailed with what to fix."""

import json
import time
import uuid

from labkit.runner import CheckFailed


def _queue(aws, name):
    try:
        url = aws.sqs.get_queue_url(QueueName=name)["QueueUrl"]
    except aws.sqs.exceptions.QueueDoesNotExist:
        raise CheckFailed(f"queue {name!r} not found")
    return url, aws.sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["All"])["Attributes"]


def _function(aws):
    try:
        return aws.lambda_.get_function_configuration(FunctionName="process-order")
    except aws.lambda_.exceptions.ResourceNotFoundException:
        raise CheckFailed("function 'process-order' not found")


def check_queues(aws):
    _, dlq = _queue(aws, "orders-dlq")
    _, orders = _queue(aws, "orders")
    if orders.get("FifoQueue") == "true":
        raise CheckFailed("'orders' should be a standard queue")
    if "RedrivePolicy" not in orders:
        raise CheckFailed("'orders' has no redrive policy, so a poison message is retried forever")
    policy = json.loads(orders["RedrivePolicy"])
    if policy.get("deadLetterTargetArn") != dlq["QueueArn"]:
        raise CheckFailed("the redrive policy of 'orders' does not point at 'orders-dlq'")
    if not 3 <= int(policy.get("maxReceiveCount", 0)) <= 5:
        raise CheckFailed(f"maxReceiveCount is {policy.get('maxReceiveCount')}; use 3 to 5")


def check_table(aws):
    try:
        table = aws.dynamodb.describe_table(TableName="orders")["Table"]
    except aws.dynamodb.exceptions.ResourceNotFoundException:
        raise CheckFailed("table 'orders' not found")
    keys = {k["AttributeName"]: k["KeyType"] for k in table["KeySchema"]}
    types = {a["AttributeName"]: a["AttributeType"] for a in table["AttributeDefinitions"]}
    if keys != {"order_id": "HASH"} or types.get("order_id") != "S":
        raise CheckFailed("the table's only key should be the string partition key 'order_id'")
    if table.get("BillingModeSummary", {}).get("BillingMode") != "PAY_PER_REQUEST":
        raise CheckFailed("use on-demand billing (PAY_PER_REQUEST)")


def check_function(aws):
    fn = _function(aws)
    if not fn.get("Runtime", "").startswith("python3"):
        raise CheckFailed(f"runtime is {fn.get('Runtime')!r}; use a python3.x runtime")
    order_id = f"check-{uuid.uuid4().hex[:8]}"
    event = {"Records": [
        {"messageId": "good", "body": json.dumps({"order_id": order_id, "amount": 7})},
        {"messageId": "bad", "body": "not json"},
    ]}
    out = aws.lambda_.invoke(FunctionName="process-order", Payload=json.dumps(event))
    payload = out["Payload"].read()
    if out.get("FunctionError"):
        raise CheckFailed(f"the function failed on a batch with one bad record: {payload[:300]!r}")
    if not aws.dynamodb.get_item(TableName="orders", Key={"order_id": {"S": order_id}}).get("Item"):
        raise CheckFailed("the good record in the batch was not written to 'orders'")
    failed = [f.get("itemIdentifier") for f in json.loads(payload or b"{}").get("batchItemFailures", [])]
    if failed != ["bad"]:
        raise CheckFailed(f"batchItemFailures should name only the bad record, got {failed}")


def check_wiring(aws):
    _, orders = _queue(aws, "orders")
    fn = _function(aws)
    maps = [m for m in aws.lambda_.list_event_source_mappings(FunctionName="process-order")["EventSourceMappings"]
            if m["EventSourceArn"] == orders["QueueArn"]]
    if not maps:
        raise CheckFailed("no event source mapping from 'orders' to 'process-order'")
    mapping = maps[0]
    if mapping.get("State") not in ("Enabled", "Enabling", "Creating"):
        raise CheckFailed(f"the mapping is {mapping.get('State')}")
    if "ReportBatchItemFailures" not in mapping.get("FunctionResponseTypes", []):
        raise CheckFailed("partial batch responses are off: one bad message makes the whole batch retry")
    need = 6 * fn["Timeout"]
    if int(orders["VisibilityTimeout"]) < need:
        raise CheckFailed(f"visibility timeout is {orders['VisibilityTimeout']}s; with a {fn['Timeout']}s "
                          f"function it should be at least {need}s")


def check_end_to_end(aws):
    url, _ = _queue(aws, "orders")
    order_id = f"e2e-{uuid.uuid4().hex[:8]}"
    aws.sqs.send_message_batch(QueueUrl=url, Entries=[
        {"Id": "bad", "MessageBody": "{broken"},
        {"Id": "good", "MessageBody": json.dumps({"order_id": order_id, "amount": 42})},
    ])
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if aws.dynamodb.get_item(TableName="orders", Key={"order_id": {"S": order_id}}).get("Item"):
            return
        time.sleep(1)
    raise CheckFailed(f"order {order_id} sent to 'orders' did not reach the table within 20 s")
