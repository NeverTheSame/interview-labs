"""The nightly extract step. Deploy it as the Lambda `extract` (handler
extract.handler); you orchestrate it, you do not need to change it.

Input: {"date": "YYYY-MM-DD"}. Two inputs simulate the failures a real
source has: {"fail": "transient"} (the source timed out; trying again may
work) and {"fail": "bad-data"} (the file is corrupt; trying again never will)."""


class TransientError(Exception):
    pass


class BadDataError(Exception):
    pass


def handler(event, context):
    if event.get("fail") == "transient":
        raise TransientError("source timed out")
    if event.get("fail") == "bad-data":
        raise BadDataError(f"corrupt file for {event.get('date')}")
    return {"date": event["date"], "rows": 1250}
