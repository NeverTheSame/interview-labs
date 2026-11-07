"""Resume upload from the Assistant chat (/api/resume/...): the shared upload
rules, the read into the builder's structure, and the follow-ups. In-memory
stub, the heuristic reader (no network)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from r3.resume.builder import render_resume_text
from r3.resume.parse import HeuristicResumeParser, ParsedResume
from r3.web import create_app, routes_resume
from tests.test_resume import ALICE, BOB, ResumeStub, as_user
from tests.test_resume_extract import make_pdf
from tests.test_resume_parse import CLASSIC

GAPPY = ("Sam Lee\nsam@example.com\n\nWork History\nPlatform Engineer | Globex | 2020 - Present\n"
         "- Ran the platform for three product teams.\nSupport Engineer at Hooli\n- Answered tickets every day.\n\n"
         "Skills\nGo; Rust; Kubernetes")


class CountingParser(HeuristicResumeParser):
    def __init__(self) -> None:
        self.calls = 0
        self.gate: asyncio.Event | None = None

    async def parse(self, text):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return self.read(text)


@pytest.fixture
def stub():
    return ResumeStub()


@pytest.fixture
def parser():
    return CountingParser()


@pytest.fixture
async def make_client(stub, parser):
    app = create_app(database=stub, resume_parser=parser)
    clients = []
    async with app.router.lifespan_context(app):
        def make(user=ALICE) -> httpx.AsyncClient:
            c = httpx.AsyncClient(transport=httpx.ASGITransport(app=as_user(app, user)), base_url="http://test")
            clients.append(c)
            return c
        yield make
        for c in clients:
            await c.aclose()


def _file(name: str, data: bytes | str):
    return {"file": (name, data.encode() if isinstance(data, str) else data)}


# --- upload --------------------------------------------------------------------------


async def test_upload_saves_at_once_as_the_first_resume(make_client, stub):
    r = await make_client().post("/api/resume/upload", files=_file("sam-cv.txt", GAPPY))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"] == "sam-cv" and body["primary"] is True and body["chars"] == len(GAPPY)
    assert stub.resumes[body["resume_id"]]["source"] == "upload"


@pytest.mark.parametrize(("name", "data", "status", "words"), [
    ("cv.exe", "MZ binary", 415, "not supported"),
    ("cv.pdf", "not really a pdf", 400, "PDF"),
    ("cv.txt", "", 400, "empty"),
    ("cv.txt", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20, 422, "doesn't look like a resume"),
])
async def test_upload_refusals_explain_what_to_do(make_client, stub, name, data, status, words):
    r = await make_client().post("/api/resume/upload", files=_file(name, data))
    assert r.status_code == status and words in r.json()["detail"], r.text
    assert stub.resumes == {}


async def test_too_large_and_too_many_pages_are_refused(make_client, stub, monkeypatch):
    big = b"x" * (routes_resume.MAX_UPLOAD_BYTES + 1)
    r = await make_client().post("/api/resume/upload", files=_file("cv.txt", big))
    assert r.status_code == 413
    from r3.resume import extract
    monkeypatch.setattr(extract, "MAX_PDF_PAGES", 0)
    r = await make_client().post("/api/resume/upload", files=_file("cv.pdf", make_pdf(["Jane Doe", "Engineer"])))
    assert r.status_code == 413 and "pages" in r.json()["detail"] and stub.resumes == {}


async def test_a_slow_extraction_is_cut_off(make_client, stub, monkeypatch):
    import time

    def slow(filename, data):
        time.sleep(0.5)
        return CLASSIC

    monkeypatch.setattr(routes_resume, "extract_text", slow)
    monkeypatch.setattr(routes_resume, "EXTRACT_TIMEOUT_S", 0.05)
    r = await make_client().post("/api/resume/upload", files=_file("cv.txt", CLASSIC))
    assert r.status_code == 422 and "in time" in r.json()["detail"] and stub.resumes == {}


async def test_uploads_are_rate_limited_per_account(make_client, stub):
    stub.clock = datetime.now(timezone.utc) - timedelta(minutes=30)  # the uploads below are recent
    for i in range(routes_resume.UPLOADS_PER_HOUR):
        await stub.create_resume(ALICE.id, title=f"cv{i}", source="upload", content=CLASSIC)
    r = await make_client().post("/api/resume/upload", files=_file("cv.txt", CLASSIC))
    assert r.status_code == 429 and "hour" in r.json()["detail"]
    r = await make_client(BOB).post("/api/resume/upload", files=_file("cv.txt", CLASSIC))
    assert r.status_code == 200  # another account is not affected


async def test_the_resume_page_upload_follows_the_same_rules(make_client, stub):
    r = await make_client().post("/resume/upload", files=_file("cv.txt", "Lorem ipsum dolor sit amet. " * 20))
    assert r.status_code == 422 and "look like a resume" in r.text and stub.resumes == {}


# --- read and follow-ups ---------------------------------------------------------------


async def _upload(client, text=GAPPY) -> int:
    return (await client.post("/api/resume/upload", files=_file("cv.txt", text))).json()["resume_id"]


async def test_a_complete_resume_is_read_into_the_clean_layout(make_client, stub):
    c = make_client()
    rid = await _upload(c, CLASSIC)
    card = (await c.post(f"/api/resume/{rid}/read")).json()
    assert card["name"] == "Jane Doe" and card["faithful"] is True and card["done"] is True
    assert [r["company"] for r in card["roles"]] == ["Acme Corp", "Initech"] and card["open"] == []
    assert card["links"] == ["linkedin.com/in/janedoe"] and card["skills_count"] == 5
    row = stub.resumes[rid]
    assert row["structure"]["engine"] == "heuristic" and row["content"].startswith("Jane Doe\nSenior DevOps")


async def test_gaps_are_asked_in_order_and_folded_into_the_resume(make_client, stub):
    c = make_client()
    rid = await _upload(c)
    card = (await c.post(f"/api/resume/{rid}/read")).json()
    assert card["open"] == ["dates", "links"] and "Support Engineer" in card["question"]
    state = (await c.get(f"/api/resume/{rid}/followup")).json()
    assert state["topic"] == "dates" and state["question"] == card["question"]

    turn = (await c.post(f"/api/resume/{rid}/answer", json={"answer": "no idea"})).json()
    assert turn["note"] and turn["topic"] == "dates" and turn["off_topic"] is False
    turn = (await c.post(f"/api/resume/{rid}/answer", json={"answer": "what do you need?"})).json()
    assert turn["off_topic"] is True and turn["topic"] == "dates"  # the chat hands it to the assistant
    turn = (await c.post(f"/api/resume/{rid}/answer", json={"answer": "2017, 2020"})).json()
    assert turn["topic"] == "links" and "2017 – 2020" in turn["changed"]
    turn = (await c.post(f"/api/resume/{rid}/answer", json={"answer": "github.com/samlee"})).json()
    assert turn["done"] is True and turn["open"] == []
    content = stub.resumes[rid]["content"]
    assert "github.com/samlee" in content and "2017 – 2020" in content
    assert content == render_resume_text(
        __import__("r3.resume.builder", fromlist=["ResumeDraft"]).ResumeDraft(**stub.resumes[rid]["structure"]["draft"]))


async def test_an_unfaithful_read_keeps_the_original_and_adds_the_answers_below(make_client, stub, parser):
    async def lossy(text):
        parsed = parser.read(text)
        return ParsedResume(**{**parsed.model_dump(), "faithful": False, "text": text})

    parser.parse = lossy
    c = make_client()
    rid = await _upload(c)
    card = (await c.post(f"/api/resume/{rid}/read")).json()
    assert card["faithful"] is False and stub.resumes[rid]["content"] == GAPPY
    await c.post(f"/api/resume/{rid}/answer", json={"answer": "2017, 2020"})
    await c.post(f"/api/resume/{rid}/answer", json={"answer": "github.com/samlee"})
    content = stub.resumes[rid]["content"]
    assert content.startswith(GAPPY) and "Added in R3" in content
    assert "Support Engineer at Hooli: 2017 – 2020" in content and "Profile link: github.com/samlee" in content


async def test_concurrent_reads_share_one_parse(make_client, parser):
    c = make_client()
    rid = await _upload(c)
    parser.gate = asyncio.Event()
    first = asyncio.ensure_future(c.post(f"/api/resume/{rid}/read"))
    second = asyncio.ensure_future(c.post(f"/api/resume/{rid}/read"))
    await asyncio.sleep(0.05)
    parser.gate.set()
    a, b = await asyncio.gather(first, second)
    assert parser.calls == 1 and a.json()["name"] == b.json()["name"] == "Sam Lee"


async def test_another_accounts_resume_is_404_everywhere(make_client):
    rid = await _upload(make_client(ALICE))
    bob = make_client(BOB)
    for r in (await bob.post(f"/api/resume/{rid}/read"), await bob.get(f"/api/resume/{rid}/followup"),
              await bob.post(f"/api/resume/{rid}/answer", json={"answer": "2017, 2020"})):
        assert r.status_code == 404


async def test_answers_need_a_read_resume_and_a_sane_length(make_client):
    c = make_client()
    rid = await _upload(c)
    assert (await c.post(f"/api/resume/{rid}/answer", json={"answer": "2017"})).status_code == 409
    await c.post(f"/api/resume/{rid}/read")
    assert (await c.post(f"/api/resume/{rid}/answer", json={"answer": "  "})).status_code == 400
    assert (await c.post(f"/api/resume/{rid}/answer", json={"answer": "x" * 8001})).status_code == 413


async def test_the_resume_page_lists_what_is_missing_with_a_way_back_to_the_chat(make_client):
    c = make_client()
    rid = await _upload(c)
    page = (await c.get(f"/resume/{rid}")).text
    assert "Missing:" not in page  # not read yet
    await c.post(f"/api/resume/{rid}/read")
    page = (await c.get(f"/resume/{rid}")).text
    assert "Missing:" in page and "dates for Support Engineer" in page and "a profile link" in page
    assert f'href="/controller?resume={rid}"' in page and "Finish in the Assistant" in page
    await c.post(f"/api/resume/{rid}/answer", json={"answer": "2017, 2020"})
    await c.post(f"/api/resume/{rid}/answer", json={"answer": "github.com/samlee"})
    assert "Missing:" not in (await c.get(f"/resume/{rid}")).text
