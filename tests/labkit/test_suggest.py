"""Suggest labs from the skills a resume claims or a job posting asks for."""

from labkit.catalog import load_catalog
from labkit.suggest import suggest

from .test_catalog import LABS


def _ids(text):
    return [s.lab.id for s in suggest(load_catalog(LABS), text)]


def test_matches_skill_names_in_free_text():
    posting = "We need hands-on AWS Lambda and SQS experience; Redis is a plus."
    ids = _ids(posting)
    assert ids[0] == "lambda-sqs-pipeline"
    assert "redis-caching" in ids


def test_aliases_and_prefixes():
    assert set(_ids("ElastiCache, Step Functions")[:2]) == {"redis-caching", "step-functions-etl"}
    assert "athena-data-lake" in _ids("amazon athena")
    assert "athena-data-lake" in _ids("Athena")


def test_reports_what_matched():
    [top, *_] = suggest(load_catalog(LABS), "snowflake and window functions")
    assert top.lab.id == "snowflake-elt"
    assert set(top.matched) >= {"snowflake", "window functions"}


def test_whole_words_only():
    assert "redis-caching" not in _ids("we use predis-something and sqlite")
    assert _ids("") == []
