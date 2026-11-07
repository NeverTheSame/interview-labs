"""The pipeline definitions and the live facts on the Controller's cards."""

from r3 import db
from r3.pipelines import PIPELINE_KEYS, PIPELINES, pipeline_facts
from tests.test_routes import NOW

STATS = db.Stats(total_jobs=1234, new_jobs=40, applied_jobs=5, total_sites=4,
                 jobs_last7=10, jobs_last30=30, last_scrape_at=NOW)
FACETS = db.Facets(statuses=[db.FacetValue(value="new", count=40), db.FacetValue(value="skipped", count=7)],
                   role_categories=[], location_kinds=[], employments=[], tech_tags=[])


def test_pipeline_keys_are_unique_and_named():
    assert len(PIPELINE_KEYS) == len(PIPELINES)
    assert {p.key for p in PIPELINES} == {"jobs", "tracking"}
    assert all(p.name and p.summary and p.links for p in PIPELINES)


def test_facts_cover_every_pipeline_and_match_the_data():
    facts = pipeline_facts(STATS, FACETS)
    assert set(facts) == PIPELINE_KEYS
    by_label = {f.label: f for key in facts for f in facts[key]}
    assert by_label["jobs stored"].value == "1,234"
    assert by_label["unreviewed"].value == "40"
    assert by_label["added 7d"].value == "10"
    assert by_label["applied"].value == "5"
    assert by_label["skipped"].value == "7"


def test_a_missing_status_counts_as_zero():
    facets = db.Facets(statuses=[], role_categories=[], location_kinds=[], employments=[], tech_tags=[])
    assert [f.value for f in pipeline_facts(STATS, facets)["tracking"] if f.label == "skipped"] == ["0"]
