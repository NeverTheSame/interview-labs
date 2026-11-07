"""Reference solution. Learners can use the AWS CLI, Terraform or boto3; this
is what the checks are proven against."""

import io
import json
import zipfile

HANDLER = '''
import json
import boto3

table = boto3.resource("dynamodb").Table("orders")


def handler(event, context):
    failures = []
    for record in event["Records"]:
        try:
            order = json.loads(record["body"])
            table.put_item(Item={"order_id": str(order["order_id"]), "amount": str(order["amount"])})
        except (ValueError, KeyError, TypeError):
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}
'''


def _zip(source: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("app.py", source)
    return buf.getvalue()


def solve(aws) -> None:
    dlq = aws.sqs.create_queue(QueueName="orders-dlq")["QueueUrl"]
    dlq_arn = aws.sqs.get_queue_attributes(QueueUrl=dlq, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    queue = aws.sqs.create_queue(QueueName="orders", Attributes={
        "VisibilityTimeout": "180",
        "RedrivePolicy": json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": "3"}),
    })["QueueUrl"]
    queue_arn = aws.sqs.get_queue_attributes(QueueUrl=queue, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]

    aws.dynamodb.create_table(
        TableName="orders", BillingMode="PAY_PER_REQUEST",
        KeySchema=[{"AttributeName": "order_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "order_id", "AttributeType": "S"}])

    aws.lambda_.create_function(
        FunctionName="process-order", Runtime="python3.12", Handler="app.handler", Timeout=30,
        Role="arn:aws:iam::000000000000:role/process-order", Code={"ZipFile": _zip(HANDLER)})
    aws.lambda_.create_event_source_mapping(
        EventSourceArn=queue_arn, FunctionName="process-order", BatchSize=10,
        FunctionResponseTypes=["ReportBatchItemFailures"])
