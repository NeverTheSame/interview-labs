"""The resume library routes (/resume) over an in-memory stand-in for
:class:`r3.db.resumes.ResumeQueries`."""

import io
import re
import zipfile
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from r3.db.resumes import RESUME_SOURCES, Resume, ResumeSummary
from r3.resume import MAX_UPLOAD_BYTES
from r3.web import create_app
from r3.web.auth import User
from r3.web.routes_resume import download_filename
from tests.test_resume_extract import make_docx, make_pdf
from tests.test_routes import StubDB

TEXT = "Jane Doe\nSenior Platform Engineer\n\nBuilt Kubernetes platforms for eight years at scale."
XSS = '<script>alert("x")</script>'
PREFIX = "/apps/r3"


class ResumeStub(StubDB):
    """Mirrors ResumeQueries: every method is scoped by user_id."""

    def __init__(self) -> None:
        super().__init__()
        self.resumes: dict[int, dict] = {}
        self.next_id = 1
        self.clock = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def _tick(self) -> datetime:
        self.clock += timedelta(minutes=1)
        return self.clock

    def _mine(self, user_id: int, resume_id: int) -> dict | None:
        row = self.resumes.get(resume_id)
        return row if row and row["user_id"] == user_id else None

    @staticmethod
    def _full(row: dict) -> Resume:
        return Resume(**row, chars=len(row["content"]))

    async def ensure_resume_tables(self) -> None:
        pass

    async def list_resumes(self, user_id):
        rows = [r for r in self.resumes.values() if r["user_id"] == user_id]
        rows.sort(key=lambda r: (not r["is_primary"], -r["updated_at"].timestamp(), -r["id"]))
        return [ResumeSummary(**{k: v for k, v in r.items() if k != "content"}, chars=len(r["content"]))
                for r in rows]

    async def get_resume(self, user_id, resume_id):
        row = self._mine(user_id, resume_id)
        return self._full(row) if row else None

    async def primary_resume(self, user_id):
        return next((self._full(r) for r in self.resumes.values() if r["user_id"] == user_id and r["is_primary"]),
                    None)

    async def create_resume(self, user_id, *, title, source, content, filename="", job_id=None):
        assert source in RESUME_SOURCES
        now = self._tick()
        first = not any(r["user_id"] == user_id and r["is_primary"] for r in self.resumes.values())
        row = dict(id=self.next_id, user_id=user_id, title=title, source=source, filename=filename,
                   content=content, job_id=job_id, is_primary=first, created_at=now, updated_at=now, structure=None)
        self.resumes[row["id"]] = row
        self.next_id += 1
        return self._full(row)

    async def update_resume(self, user_id, resume_id, *, title, content):
        row = self._mine(user_id, resume_id)
        if row is None:
            return None
        row.update(title=title, content=content, updated_at=self._tick())
        return self._full(row)

    async def delete_resume(self, user_id, resume_id):
        row = self._mine(user_id, resume_id)
        if row is None:
            return False
        del self.resumes[resume_id]
        rest = [r for r in self.resumes.values() if r["user_id"] == user_id]
        if row["is_primary"] and rest:
            max(rest, key=lambda r: (r["updated_at"], r["id"]))["is_primary"] = True
        return True

    async def set_resume_structure(self, user_id, resume_id, *, structure, content):
        row = self._mine(user_id, resume_id)
        if row is None:
            return None
        row.update(structure=structure, content=content, updated_at=self._tick())
        return self._full(row)

    async def count_recent_uploads(self, user_id, since):
        return sum(1 for r in self.resumes.values()
                   if r["user_id"] == user_id and r["source"] in ("upload", "chat") and r["created_at"] >= since)

    async def set_primary_resume(self, user_id, resume_id):
        if self._mine(user_id, resume_id) is None:
            return False
        for r in self.resumes.values():
            if r["user_id"] == user_id:
                r["is_primary"] = r["id"] == resume_id
        return True


def as_user(app, user: User | None):
    """Wrap the app so every request carries ``user`` the way the sign-in
    middleware would put it on ``request.state``."""
    async def wrapped(scope, receive, send):
        if scope["type"] == "http" and user is not None:
            scope.setdefault("state", {})["user"] = user
        await app(scope, receive, send)
    return wrapped


ALICE = User(id=11, account_name="alice")
BOB = User(id=22, account_name="bob")


@pytest.fixture
def stub():
    return ResumeStub()


@pytest.fixture
async def make_client(stub):
    app = create_app(database=stub)
    clients = []
    async with app.router.lifespan_context(app):
        def make(user: User | None = None) -> httpx.AsyncClient:
            c = httpx.AsyncClient(transport=httpx.ASGITransport(app=as_user(app, user)), base_url="http://test")
            clients.append(c)
            return c
        yield make
        for c in clients:
            await c.aclose()


@pytest.fixture
def client(make_client):
    return make_client(None)


def primary_ids(stub, user_id):
    return [r["id"] for r in stub.resumes.values() if r["user_id"] == user_id and r["is_primary"]]


async def test_empty_state_explains_what_to_do(client):
    page = await client.get("/resume")
    assert page.status_code == 200
    assert "You have not added a resume yet." in page.text
    assert "matches the skills in your primary resume against the skills each job posting lists" in page.text
    assert 'enctype="multipart/form-data"' in page.text and 'action="/resume/paste"' in page.text
    assert 'href="/resume" class="on"' in page.text


async def test_paste_creates_primary_and_redirects(client, stub):
    r = await client.post("/resume/paste", data={"title": "", "content": TEXT + "\r\n\r\n\r\n"})
    assert r.status_code == 303 and r.headers["location"] == "/resume/1"
    row = stub.resumes[1]
    assert (row["title"], row["source"], row["content"], row["is_primary"], row["user_id"]) == \
        ("Pasted resume", "paste", TEXT, True, 0)
    listing = await client.get("/resume")
    assert "Pasted resume" in listing.text and "primary" in listing.text and "not added" not in listing.text


@pytest.mark.parametrize("content, status, needle", [
    ("too short", 400, "at least 50"),
    ("   \n\n  ", 400, "at least 50"),
    ("x" * 100_001, 413, "the limit is 100,000"),
])
async def test_paste_validation_keeps_what_was_typed(client, stub, content, status, needle):
    r = await client.post("/resume/paste", data={"title": "My CV & more", "content": content})
    assert r.status_code == status
    assert needle in r.text and 'role="alert"' in r.text
    assert 'value="My CV &amp; more"' in r.text
    if content.strip():
        assert content[:200] in r.text
    assert stub.resumes == {}


async def test_upload_each_type_titles_from_filename(client, stub):
    files = [("cv.txt", TEXT.encode()), ("notes.md", ("# " + TEXT).encode()),
             ("Jane Doe CV.docx", make_docx(TEXT.split("\n"))), ("resume.pdf", make_pdf(["Jane Doe", "Platform Engineer", "x" * 60]))]
    for i, (name, data) in enumerate(files, 1):
        r = await client.post("/resume/upload", files={"file": (name, data)}, data={"title": ""})
        assert r.status_code == 303, r.text
        assert r.headers["location"] == f"/resume/{i}"
    assert [r["title"] for r in stub.resumes.values()] == ["cv", "notes", "Jane Doe CV", "resume"]
    assert stub.resumes[3]["filename"] == "Jane Doe CV.docx" and "Senior Platform Engineer" in stub.resumes[3]["content"]
    assert "Jane Doe" in stub.resumes[4]["content"]
    assert primary_ids(stub, 0) == [1]
    r = await client.post("/resume/upload", files={"file": ("../../etc/x.txt", TEXT.encode())}, data={"title": " Mine "})
    assert stub.resumes[5]["filename"] == "x.txt" and stub.resumes[5]["title"] == "Mine"


@pytest.mark.parametrize("name, data, status, needle", [
    ("cv.pages", b"whatever", 415, "Word (.docx)"),
    ("cv.txt", b"a" * (MAX_UPLOAD_BYTES + 1), 413, "larger than 5 MB"),
    ("cv.txt", b"", 400, "empty"),
    ("cv.pdf", make_pdf(None), 400, "paste"),
    ("cv.docx", b"garbage", 400, "damaged"),
    ("cv.txt", b"short", 400, "at least 50"),
])
async def test_upload_errors(client, stub, name, data, status, needle):
    r = await client.post("/resume/upload", files={"file": (name, data)})
    assert r.status_code == status
    assert needle in r.text and 'role="alert"' in r.text
    assert stub.resumes == {}


async def test_oversized_upload_does_not_report_the_truncated_size(client):
    r = await client.post("/resume/upload", files={"file": ("cv.txt", b"a" * (6 * 1024 * 1024))})
    assert r.status_code == 413 and "larger than 5 MB" in r.text and "5.0 MB" not in r.text


async def test_upload_without_file(client):
    r = await client.post("/resume/upload", data={"title": "x"})
    assert r.status_code == 400 and "Choose a resume file" in r.text


async def test_xss_is_escaped_everywhere(client, stub):
    await client.post("/resume/paste", data={"title": XSS, "content": XSS + " " + TEXT})
    for path in ("/resume", "/resume/1"):
        page = await client.get(path)
        assert XSS not in page.text
        assert "&lt;script&gt;" in page.text
    r = await client.post("/resume/paste", data={"title": XSS, "content": "<b>" + XSS})
    assert XSS not in r.text and "&lt;script&gt;" in r.text


async def test_view_edit_and_validation(client, stub):
    await client.post("/resume/paste", data={"title": "Base", "content": TEXT})
    page = await client.get("/resume/1")
    assert page.status_code == 200 and 'value="Base"' in page.text and "Built Kubernetes platforms" in page.text
    r = await client.post("/resume/1", data={"title": "  Platform  CV ", "content": TEXT + "\nGo, Python"})
    assert r.status_code == 303 and r.headers["location"] == "/resume/1"
    assert stub.resumes[1]["title"] == "Platform CV" and stub.resumes[1]["content"].endswith("Go, Python")
    r = await client.post("/resume/1", data={"title": "", "content": TEXT})
    assert r.status_code == 400 and "Give the resume a title" in r.text
    r = await client.post("/resume/1", data={"title": "T", "content": "tiny edit"})
    assert r.status_code == 400 and "tiny edit" in r.text
    assert stub.resumes[1]["title"] == "Platform CV"


async def test_primary_switch_and_delete_promotes(client, stub):
    for title in ("A", "B", "C"):
        await client.post("/resume/paste", data={"title": title, "content": TEXT})
    assert primary_ids(stub, 0) == [1]
    r = await client.post("/resume/3/primary")
    assert r.status_code == 303 and r.headers["location"] == "/resume"
    assert primary_ids(stub, 0) == [3]
    await client.post("/resume/1", data={"title": "A2", "content": TEXT})  # A is now the most recently updated
    r = await client.post("/resume/3/delete")
    assert r.status_code == 303 and 3 not in stub.resumes
    assert primary_ids(stub, 0) == [1]
    await client.post("/resume/2/delete")
    assert primary_ids(stub, 0) == [1]
    await client.post("/resume/1/delete")
    assert stub.resumes == {}
    assert "You have not added a resume yet." in (await client.get("/resume")).text


async def test_download_headers(client):
    await client.post("/resume/paste", data={"title": 'Jane "CV" / 2026 ✓', "content": TEXT})
    r = await client.get("/resume/1/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.headers["content-disposition"].startswith('attachment; filename="Jane-CV-2026.txt"')
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.text == TEXT + "\n"
    assert download_filename("../..") == "resume.txt" and download_filename("✓✓") == "resume.txt"


async def test_pdf_download_renders_the_users_own_resume(client):
    await client.post("/resume/paste", data={"title": "Jane CV", "content": TEXT})
    r = await client.get("/resume/1/pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith('attachment; filename="Jane-CV.pdf"')
    assert (await client.get("/resume/999/pdf")).status_code == 404


async def test_users_cannot_touch_each_others_resumes(make_client, stub):
    alice, bob = make_client(ALICE), make_client(BOB)
    await alice.post("/resume/paste", data={"title": "Alice CV", "content": TEXT})
    await bob.post("/resume/paste", data={"title": "Bob CV", "content": TEXT})
    assert stub.resumes[1]["user_id"] == ALICE.id and stub.resumes[2]["user_id"] == BOB.id
    assert primary_ids(stub, ALICE.id) == [1] and primary_ids(stub, BOB.id) == [2]  # each user's first is primary
    listing = await bob.get("/resume")
    assert "Bob CV" in listing.text and "Alice CV" not in listing.text
    for method, path, data in [("GET", "/resume/1", None), ("GET", "/resume/1/download", None),
                               ("GET", "/resume/1/pdf", None),
                               ("POST", "/resume/1", {"title": "pwned", "content": TEXT}),
                               ("POST", "/resume/1/primary", None), ("POST", "/resume/1/delete", None)]:
        r = await bob.request(method, path, data=data)
        assert r.status_code == 404, (method, path)
    assert stub.resumes[1]["title"] == "Alice CV" and primary_ids(stub, BOB.id) == [2]
    assert (await alice.get("/resume/1")).status_code == 200
    assert (await alice.get("/resume/999")).status_code == 404
    assert (await alice.get("/resume/abc")).status_code == 404


async def test_everything_stays_under_root_path(stub):
    app = create_app(database=stub, root_path=PREFIX)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://test{PREFIX}/") as c:
            r = await c.post("/resume/paste", data={"content": TEXT})
            assert r.headers["location"] == f"{PREFIX}/resume/1"
            r = await c.post("/resume/upload", files={"file": ("cv.txt", TEXT.encode())})
            assert r.headers["location"] == f"{PREFIX}/resume/2"
            assert (await c.post("/resume/2/primary")).headers["location"] == f"{PREFIX}/resume"
            for path in ("/resume", "/resume/1"):
                page = await c.get(path)
                assert page.status_code == 200
                links = re.findall(r'(?:href|src|action)="(/[^" ]*)"', page.text)
                assert any("/resume/1/download" in link for link in links)
                assert all(link.startswith(PREFIX + "/") for link in links), links


def _count_body_bytes(monkeypatch):
    """Total request-body bytes the app read from the server."""
    from starlette.requests import Request

    seen = {"bytes": 0}
    original = Request.stream

    async def counting(self):
        async for chunk in original(self):
            seen["bytes"] += len(chunk)
            yield chunk

    monkeypatch.setattr(Request, "stream", counting)
    return seen


async def test_oversized_upload_is_not_read_at_all(client, stub, monkeypatch):
    """R3-12: a declared Content-Length over the limit is refused before the
    multipart parser (which would spool the whole file) reads any of it."""
    seen = _count_body_bytes(monkeypatch)
    r = await client.post("/resume/upload", files={"file": ("cv.txt", b"a" * (6 * 1024 * 1024))})
    assert r.status_code == 413 and "larger than 5 MB" in r.text and 'role="alert"' in r.text
    assert stub.resumes == {} and seen["bytes"] == 0


async def test_oversized_assistant_upload_is_refused_the_same_way(client, monkeypatch):
    seen = _count_body_bytes(monkeypatch)
    r = await client.post("/api/resume/upload", files={"file": ("cv.txt", b"a" * (6 * 1024 * 1024))})
    assert r.status_code == 413 and "larger than 5 MB" in r.json()["detail"]
    assert seen["bytes"] == 0


async def test_chunked_oversized_upload_is_cut_off(client):
    async def body():
        yield (b'--x\r\nContent-Disposition: form-data; name="file"; filename="cv.txt"\r\n'
               b"Content-Type: text/plain\r\n\r\n")
        for _ in range(100):  # 100 MB, no Content-Length
            yield b"a" * 1_000_000

    r = await client.post("/resume/upload", content=body(), headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413 and "larger than 5 MB" in r.json()["detail"]


async def test_upload_just_under_the_limit_still_works(client, stub):
    r = await client.post("/resume/upload", files={"file": ("cv.txt", TEXT.encode())})
    assert r.status_code in (200, 303)
