# Order pipeline: SQS → Lambda → DynamoDB

**Level:** beginner · **Time:** about 45 minutes · **Services:** SQS, Lambda, DynamoDB

## Scenario

A shop puts every new order on a queue. Your job is the worker behind it: a
Lambda function that takes orders off the queue and stores them in DynamoDB.
Orders arrive in batches, some are malformed, and the same order can arrive
twice. The pipeline must keep the good orders flowing and park the bad ones.

## Before you start

```bash
docker compose up -d ministack      # the local AWS, on http://localhost:4566
export AWS_ENDPOINT_URL=http://localhost:4566 AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
```

Build it any way you like: the AWS CLI, Terraform, or a boto3 script.

## Tasks

1. **Queues.** Create the standard queue `orders` and `orders-dlq`. A message
   that fails 3 to 5 times moves to `orders-dlq`.
2. **Table.** Create the DynamoDB table `orders`, partition key `order_id`
   (string), on-demand billing.
3. **Function.** Deploy the Python Lambda `process-order`. It stores each
   record's `{"order_id", "amount"}` body in `orders`. A record it cannot
   parse must not sink the rest of the batch.
4. **Wiring.** Map `orders` to `process-order` with partial batch responses on,
   and give the queue a visibility timeout of at least six times the
   function's timeout.
5. **End to end.** A good order sent next to a malformed one reaches the table.

Check your work at any time:

```bash
uv run labs check lambda-sqs-pipeline          # every task
uv run labs check lambda-sqs-pipeline wiring   # one task
uv run labs reset                              # start again from an empty stack
```

## What is real here, and what is not

Run `uv run labs show lambda-sqs-pipeline` for the full list. The short
version: the function, the polling and the dead-letter redrive really run.
IAM does not: any role is accepted, so on AWS you also need an execution role
with SQS and DynamoDB permissions.

## Be ready to answer

`labs show` lists the questions this lab prepares you for. Try answering them
out loud once the checks pass.
