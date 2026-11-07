"""The labeling page end to end over the app against the stub database: the
rubric parser, the queue's WHERE clause, the page and its fragments, save /
overwrite / history, skip through ``seen``, the filter, the export's shape,
and the contract between the HTML and label.js (the keys are data on the
markup the script reads; the script never parses HTML)."""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from markupsafe import escape

from r3 import db
from r3.labeling import (
    OPTION_KEYS,
    QUESTIONS,
    RED_FLAG,
    RELEVANT,
    ROLE_BUCKET,
    RUBRIC_VERSION,
    SKIPPED,
    answered,
    clean_note,
    clean_seconds,
    clean_session_id,
    parse_answers,
    summarize,
)
from r3.web import create_app
from r3.web.routes_label import label_filter, parse_seen, parse_since
from r3.web.templating import STATIC_DIR
from tests.test_routes import JOB, NOW, StubDB

PREFIX = "/dashboard/controller/r3"

JOB2 = JOB.model_copy(update={
    "id": 9, "position": "Staff AI Engineer", "company": "Beta", "location": "Toronto, ON", "status": "skipped",
    "scraped_at": NOW - timedelta(days=2), "location_kind": "hybrid", "employment": "contract",
    "description": "Own the platform.\nMust hold an active US security clearance.",
})

FULL = {"relevant": "yes", "role_bucket": "platform_cloud", "work_mode": "remote", "employment": "fulltime",
        "eligible_from_canada": "yes", "seniority_fit": "fits", "red_flag": ["none"], "note": "looks real"}


@pytest.fixture
def stub():
    s = StubDB()
    s.label_pool = [JOB, JOB2]
    return s


@pytest.fixture
async def client(stub):
    app = create_app(database=stub)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def post(client, job_id, **fields):
    """Submit the rubric form; a list value posts the field several times (a multi-select)."""
    return await client.post(f"/label/{job_id}", data=fields)


# --- the rubric ----------------------------------------------------------------------------------


def test_rubric_is_versioned_and_every_question_has_a_cannot_tell_option():
    assert RUBRIC_VERSION == "job-label-rubric/v1"
    assert [q.key for q in QUESTIONS] == ["relevant", "role_bucket", "work_mode", "employment",
                                          "eligible_from_canada", "seniority_fit", "red_flag"]
    for q in QUESTIONS:
        assert q.unclear in q.values, q.key
        assert all(o.description.endswith(".") for o in q.options), q.key  # Jev reads these as the criteria
        assert len(q.options) <= 10 and q.keys == OPTION_KEYS[:len(q.options)]  # one key per option: 1-9, then 0
    assert RELEVANT.values == ("yes", "maybe", "no")  # keys 1 / 2 / 3
    assert ROLE_BUCKET.values == ("sre", "devops", "platform_cloud", "data_eng", "ml", "ai", "backend", "other_eng",
                                  "not_eng", "unclear")
    assert ROLE_BUCKET.keys[-1] == "0"  # the tenth option
    assert RED_FLAG.kind == "multi" and RED_FLAG.exclusive == "none"


def test_parse_answers_full_form():
    answers = parse_answers({k: ([v] if isinstance(v, str) else v) for k, v in FULL.items()})
    assert answers == {"relevant": "yes", "role_bucket": "platform_cloud", "work_mode": "remote",
                       "employment": "fulltime", "eligible_from_canada": "yes", "seniority_fit": "fits",
                       "red_flag": ["none"]}
    assert summarize(answers) == "relevant: yes · platform_cloud · remote · fulltime · yes · fits · none"


def test_parse_answers_no_with_nothing_else_stores_skipped():
    answers = parse_answers({"relevant": ["no"]})
    assert answers["relevant"] == "no"
    assert all(answers[q.key] == SKIPPED for q in QUESTIONS if q is not RELEVANT)
    assert summarize(answers) == "relevant: no"


def test_parse_answers_multi_select_keeps_option_order_and_drops_blanks():
    answers = parse_answers({"relevant": ["maybe"], "red_flag": ["staffing_spam", "", "clearance"]})
    assert answers["red_flag"] == ["clearance", "staffing_spam"]


@pytest.mark.parametrize("form, message", [
    ({}, "relevant is required"),
    ({"relevant": ["yes"], "role_bucket": ["wizard"]}, "role_bucket: 'wizard' is not one of"),
    ({"relevant": ["yes"], "red_flag": ["none", "clearance"]}, "'none' cannot be combined"),
    ({"relevant": ["yes", "no"]}, "pick one option"),
])
def test_parse_answers_rejects(form, message):
    with pytest.raises(ValueError, match=message):
        parse_answers(form)


def test_answered_excludes_skipped_and_the_cannot_tell_option():
    answers = {"relevant": "maybe", "role_bucket": "unclear", "work_mode": "remote", "employment": SKIPPED,
               "red_flag": ["none"], "eligible_from_canada": "no"}
    assert not answered(answers, "relevant") and not answered(answers, "role_bucket")
    assert answered(answers, "work_mode") and answered(answers, "eligible_from_canada")
    assert not answered(answers, "employment") and not answered(answers, "red_flag")
    assert answered({"red_flag": ["clearance"]}, "red_flag")


def test_cleaners():
    assert clean_note("  two\nlines\t here  ") == "two lines here"
    with pytest.raises(ValueError):
        clean_note("x" * 1001)
    assert clean_session_id("abc-DEF_123") == "abc-DEF_123" and clean_session_id("no spaces") == ""
    assert clean_seconds("12.7") == 12 and clean_seconds("-3") == 0 and clean_seconds("x") == 0
    assert clean_seconds("999999") == 24 * 3600


# --- the queue's SQL -------------------------------------------------------------------------------


def test_next_job_where_adds_the_label_and_seen_exclusions_after_the_filter():
    clause, args = db.next_job_where(db.JobFilter(role_category="AI/ML", days=7), RUBRIC_VERSION, [7, 9])
    assert clause.startswith("WHERE j.role_category = $1 AND j.scraped_at >= now() - make_interval(days => $2)")
    assert "NOT EXISTS (SELECT 1 FROM r3_job_labels l WHERE l.job_id = j.id AND l.rubric_version = $3)" in clause
    assert "NOT (j.id = ANY($4::bigint[]))" in clause
    assert args == ["AI/ML", 7, RUBRIC_VERSION, [7, 9]]


def test_next_job_where_without_filter_or_seen():
    clause, args = db.next_job_where(db.JobFilter(), RUBRIC_VERSION, [])
    assert clause == ("WHERE NOT EXISTS (SELECT 1 FROM r3_job_labels l WHERE l.job_id = j.id "
                      "AND l.rubric_version = $1)")
    assert args == [RUBRIC_VERSION] and "ANY" not in clause


def test_request_parsers():
    assert parse_seen("7, 9,x,9,") == [7, 9] and parse_seen(None) == []
    f = label_filter({"role": "SRE", "location": "remote-ca", "employment": "contract", "site": "https://a/", "days": "7"})
    assert (f.role_category, f.location_kind, f.employment, f.site_url, f.days) == ("SRE", "remote-ca", "contract",
                                                                                   "https://a/", 7)
    assert parse_since("2026-09-27T04:00:00.000Z").isoformat() == "2026-09-27T04:00:00+00:00"
    assert parse_since("2026-09-27T00:00:00").tzinfo is not None
    midnight = parse_since("garbage")
    assert (midnight.hour, midnight.minute, midnight.second) == (0, 0, 0) and midnight.tzinfo is not None


# --- the page -------------------------------------------------------------------------------------


async def test_startup_creates_the_label_tables(client, stub):
    assert hasattr(stub, "labels") and stub.labels == {} and stub.label_history == []


async def test_page_renders_the_first_card_with_the_rubric(client, stub):
    r = await client.get("/label")
    body = r.text
    assert r.status_code == 200 and "<!doctype html>" in body
    assert 'href="/label"  class="on"' in body  # the nav entry
    assert "/static/label.css?v=" in body and '<script src="/static/label.js?v=' in body
    assert 'id="r3-label" data-base-path="" data-rubric="job-label-rubric/v1"' in body
    assert 'id="label-filters"' in body and 'hx-get="/label/next"' in body
    for name in ("role", "location", "employment", "site", "days"):
        assert f'<select name="{name}"' in body
    # The newest unreviewed posting comes first: #7 is new, #9 is skipped.
    assert 'id="lab-card" data-job-id="7"' in body and "Senior ML Engineer" in body
    assert stub.last_seen == []
    # Full description, scrollable, not cut.
    assert '<div class="lab-desc"' in body and "<p>Build models.</p>" in body and "<p>Ship them.</p>" in body
    assert 'href="https://www.boards.example/jobs/7" target="_blank" rel="noopener">source ↗</a>' in body
    # Every question, every option, one digit per option; relevant first and big.
    for q in QUESTIONS:
        assert f'data-q="{q.key}" data-kind="{q.kind}"' in body
        for key, o in zip(q.keys, q.options):
            assert f'<label class="opt opt-{o.value}" data-key="{key}" title="{escape(o.description)}">' in body
            kind = "checkbox" if q.kind == "multi" else "radio"
            assert f'<input type="{kind}" name="{q.key}" value="{o.value}">' in body
    assert body.index('data-q="relevant"') < body.index('data-q="role_bucket"')
    assert '<fieldset class="lab-q big" data-q="relevant"' in body
    assert '<label class="opt opt-unclear" data-key="0"' in body and "<kbd>0</kbd><span>Unclear</span>" in body
    assert 'data-exclusive="none"' in body
    assert 'hx-post="/label/7"' in body and 'hx-include="#label-filters"' in body
    for key in ("Enter", "s", "u"):
        assert f"<kbd>{key}</kbd>" in body
    assert not re.search(r"\son[a-z]+\s*=", body), "no inline event handlers"
    # The counter: nothing labeled yet, two postings in the pool.
    assert re.search(r'id="label-counter"[^>]*>\s*<span><strong>0</strong> today</span>\s*<span><strong>0</strong> total'
                     r'</span>\s*<span><strong>2</strong> left in filter</span>', body)


async def test_filter_query_maps_to_the_job_filter(client, stub):
    await client.get("/label?role=AI%2FML&location=remote-ca&employment=contract&site=https%3A%2F%2Fx%2F&days=7")
    f = stub.last_filter
    assert (f.role_category, f.location_kind, f.employment, f.site_url, f.days) == ("AI/ML", "remote-ca", "contract",
                                                                                   "https://x/", 7)
    assert f.status == "" and f.search == ""  # the queue orders by status; it does not filter on it


async def test_next_is_a_fragment_with_the_counter_out_of_band(client, stub):
    r = await client.get("/label/next?seen=7", headers={"HX-Request": "true"})
    body = r.text
    assert r.status_code == 200 and "<!doctype html>" not in body and 'id="label-filters"' not in body
    assert 'id="label-counter" hx-swap-oob="true"' in body and 'id="label-flash"' in body
    assert 'data-job-id="9"' in body and stub.last_seen == [7]  # #7 was skipped this session
    assert "<strong>1</strong> left in filter" in body


async def test_nothing_left_state(client):
    r = await client.get("/label/next?seen=7,9", headers={"HX-Request": "true"})
    assert "Nothing left to label in this filter." in r.text and "data-job-id" not in r.text
    assert 'id="label-undo"' in r.text and "<strong>0</strong> left in filter" in r.text


async def test_save_stores_the_label_and_answers_with_the_next_card(client, stub):
    r = await post(client, 7, **FULL, seen="", session_id="tab-1", seconds_spent="41.6", since="2026-09-27T04:00:00Z")
    assert r.status_code == 200
    label = stub.labels[(7, RUBRIC_VERSION)]
    assert label.answers == parse_answers({k: ([v] if isinstance(v, str) else v) for k, v in FULL.items()})
    assert label.note == "looks real" and label.labeler == "operator" and label.session_id == "tab-1"
    assert label.seconds_spent == 41
    assert stub.writes == [("label", 7, "yes")]
    body = r.text
    assert 'id="label-flash"' in body and "Saved #7 Senior ML Engineer · relevant: yes · platform_cloud" in body
    assert "replaced" not in body
    assert 'data-job-id="9"' in body  # the next card, without another round trip
    assert stub.last_seen == [7]
    assert "<strong>1</strong> today</span>" in body and "<strong>1</strong> total</span>" in body
    assert "<strong>1</strong> left in filter" in body


async def test_save_no_alone_stores_the_rest_as_skipped(client, stub):
    r = await post(client, 7, relevant="no")
    assert r.status_code == 200
    answers = stub.labels[(7, RUBRIC_VERSION)].answers
    assert answers["relevant"] == "no"
    assert {k: v for k, v in answers.items() if k != "relevant"} == {q.key: SKIPPED for q in QUESTIONS if q is not RELEVANT}


async def test_relabel_overwrites_and_keeps_the_previous_row_in_history(client, stub):
    await post(client, 7, relevant="no")
    r = await post(client, 7, **FULL)
    assert r.status_code == 200 and "replaced the earlier label" in r.text
    assert len(stub.labels) == 1 and stub.labels[(7, RUBRIC_VERSION)].answers["relevant"] == "yes"
    assert [h.answers["relevant"] for h in stub.label_history] == ["no"]


@pytest.mark.parametrize("fields, status, needle", [
    ({"role_bucket": "sre"}, 400, "relevant is required"),
    ({"relevant": "yes", "work_mode": "mars"}, 400, "work_mode: 'mars' is not one of"),
    ({"relevant": "yes", "red_flag": ["none", "clearance"]}, 400, "cannot be combined"),
    ({"relevant": "yes", "note": "n" * 1001}, 400, "note is too long"),
])
async def test_save_rejects_bad_forms_without_writing(client, stub, fields, status, needle):
    r = await post(client, 7, **fields)
    assert r.status_code == status and needle in r.text
    assert stub.labels == {} and stub.writes == []


async def test_unknown_job_is_404(client):
    assert (await post(client, 999, relevant="yes")).status_code == 404
    assert (await client.get("/label/999")).status_code == 404


async def test_reopening_a_labeled_job_shows_the_previous_label(client, stub):
    await post(client, 7, relevant="maybe", role_bucket="sre", red_flag=["clearance", "staffing_spam"], note="hm")
    r = await client.get("/label/7")
    body = r.text
    assert r.status_code == 200 and "<!doctype html>" in body
    assert 'data-job-id="7" data-labeled="true"' in body
    assert "You labeled this" in body and "relevant: maybe · sre · clearance, staffing_spam · “hm”" in body
    assert '<input type="radio" name="relevant" value="maybe" checked>' in body
    assert '<input type="radio" name="role_bucket" value="sre" checked>' in body
    assert '<input type="checkbox" name="red_flag" value="clearance" checked>' in body
    assert '<input type="checkbox" name="red_flag" value="staffing_spam" checked>' in body
    assert '<input type="checkbox" name="red_flag" value="none">' in body
    assert body.count(" checked>") == 4
    assert 'value="hm"' in body
    # As a fragment (undo fetches it this way), the same card without the page.
    r = await client.get("/label/7", headers={"HX-Request": "true"})
    assert "<!doctype html>" not in r.text and 'data-job-id="7" data-labeled="true"' in r.text


async def test_export_is_jsonl_with_the_job_frozen_beside_the_answers(client, stub):
    await post(client, 7, **FULL)
    await post(client, 9, relevant="no")
    stub.label_pool = [JOB]  # #9 was pruned after it was labeled
    r = await client.get("/label/export")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers["x-r3-labels"] == "2"
    assert re.fullmatch(r'attachment; filename="r3-labels-job-label-rubric-v1-\d{8}\.jsonl"', r.headers["content-disposition"])
    lines = r.text.splitlines()
    assert len(lines) == 2 and r.text.endswith("\n")
    first, second = (json.loads(line) for line in lines)
    assert first["rubric_version"] == RUBRIC_VERSION and first["job_id"] == 7
    assert first["answers"]["relevant"] == "yes" and first["note"] == "looks real"
    assert first["labeler"] == "operator" and first["exported_at"].startswith("20")
    job = first["job"]
    assert job["id"] == 7 and job["position"] == "Senior ML Engineer" and job["company"] == "Acme AI"
    assert job["description"] == "Build models.\n\nShip them."  # the full text, frozen
    for key in ("site_url", "site_name", "job_url", "location", "employment", "salary", "salary_min", "salary_max",
                "location_kind", "tech_tags", "status", "posted_date", "scraped_at", "role_category"):
        assert key in job, key
    assert second["job_id"] == 9 and second["job"] is None and second["answers"]["role_bucket"] == SKIPPED


async def test_export_without_labels_is_empty(client):
    r = await client.get("/label/export")
    assert r.status_code == 200 and r.text == "" and r.headers["x-r3-labels"] == "0"


async def test_every_url_stays_under_the_proxy_prefix(stub):
    app = create_app(database=stub, root_path=PREFIX)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=f"http://test{PREFIX}/") as c:
            body = (await c.get("/label")).text
    links = re.findall(r'(?:href|src|action|hx-get|hx-post)="(/[^" ]*)"', body)
    assert links and all(link.startswith(PREFIX + "/") for link in links), links
    assert f'data-base-path="{PREFIX}"' in body


# --- label.js: the contract with the markup -----------------------------------------------------------


def test_label_js_keeps_its_contract():
    js = (STATIC_DIR / "label.js").read_text()
    # Server text is never parsed as HTML.
    assert not re.search(r"\.innerHTML|\.outerHTML|insertAdjacentHTML|document\.write|\bfetch\(", js)
    # Every request htmx.ajax makes goes through the base path.
    assert all(arg.startswith("url(") for arg in re.findall(r"htmx\.ajax\(\"GET\", ([^,]*),", js))
    assert 'url("/label/next")' in js and "url(`/label/${last.id}`)" in js
    # The keys: digits pick an option of the active question, Enter saves, s skips, u undoes, n notes.
    assert "/^[0-9]$/.test(e.key)" in js and 'e.key === "0" ? 10' in js
    for key in ('"Enter"', 'case "s"', 'case "u"', 'case "n"', '"Escape"'):
        assert key in js, key
    # The markup the script reads.
    for hook in ('"#lab-card[data-job-id]"', '".lab-q"', '".opt"', "dataset.exclusive", '"#label-form"', '"#label-skip"',
                 '"#label-undo"', '"#label-filters"', '"#label-note"', '"#label-flash"', '"#label-status"'):
        assert hook in js, hook
    # The session memory and what every request carries.
    for key in ("r3.label.seen.v1", "r3.label.undo.v1", "r3.label.session.v1"):
        assert key in js
    for param in ("p.seen =", "p.since =", "p.session_id =", "p.seconds_spent ="):
        assert param in js, param
    assert "htmx:configRequest" in js and "htmx:afterSwap" in js and "htmx:responseError" in js


def test_label_static_files_are_served(stub):
    import asyncio

    async def run():
        app = create_app(database=stub)
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                js = await c.get("/static/label.js")
                css = await c.get("/static/label.css")
        assert js.status_code == 200 and "javascript" in js.headers["content-type"]
        assert css.status_code == 200 and "css" in css.headers["content-type"]

    asyncio.run(run())



# --- r3-label: the portal plus the page in the browser ---------------------------------------------------


def test_open_when_up_opens_the_page_once_healthz_answers(monkeypatch):
    import r3.portal as portal

    answers = iter([OSError("not yet"), OSError("not yet"), 200])
    opened: list[str] = []

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(url, timeout):
        assert url == "http://127.0.0.1:8090/healthz"
        got = next(answers)
        if isinstance(got, Exception):
            raise got
        return Resp()

    monkeypatch.setattr(portal.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(portal.webbrowser, "open", lambda url: opened.append(url))
    thread = portal.open_when_up("http://localhost:8090/label", 8090, poll_s=0.01)
    thread.join(5)
    assert not thread.is_alive() and opened == ["http://localhost:8090/label"]


def test_r3_label_is_a_console_script():
    import tomllib

    scripts = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["r3-label"] == "r3.portal:label" and scripts["r3-portal"] == "r3.portal:main"
