"""Reference solution for the ETL lab."""

import io
import json
import zipfile
from pathlib import Path

STARTER = Path(__file__).resolve().parent.parent / "starter" / "extract.py"
ROLE = "arn:aws:iam::000000000000:role/nightly-etl"


def solve(aws) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(STARTER, "extract.py")
    fn = aws.lambda_.create_function(FunctionName="extract", Runtime="python3.12", Handler="extract.handler",
                                     Role=ROLE, Code={"ZipFile": buf.getvalue()})
    alerts = aws.sqs.create_queue(QueueName="etl-alerts")["QueueUrl"]
    definition = {
        "StartAt": "Extract",
        "States": {
            "Extract": {
                "Type": "Task", "Resource": fn["FunctionArn"], "Next": "Done",
                "Retry": [{"ErrorEquals": ["TransientError"], "IntervalSeconds": 1, "MaxAttempts": 2, "BackoffRate": 2}],
                "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "Alert"}],
            },
            "Alert": {
                "Type": "Task", "Resource": "arn:aws:states:::sqs:sendMessage", "Next": "Failed",
                "Parameters": {"QueueUrl": alerts, "MessageBody.$": "States.JsonToString($.error)"},
            },
            "Failed": {"Type": "Fail", "Error": "EtlFailed", "Cause": "see etl-alerts"},
            "Done": {"Type": "Succeed"},
        },
    }
    aws.stepfunctions.create_state_machine(name="nightly-etl", type="STANDARD", roleArn=ROLE,
                                           definition=json.dumps(definition))
