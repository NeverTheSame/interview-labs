"""A portal served under a path prefix (R3_ROOT_PATH) keeps every link under it."""

import re

import httpx
import pytest

from r3.web import create_app
from tests.test_routes import StubDB

PREFIX = "/apps/r3"


@pytest.fixture
async def client():
    app = create_app(database=StubDB(), root_path=PREFIX)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://test{PREFIX}/") as client:
            yield client


@pytest.mark.parametrize("path", ["/", "/controller", "/jobs", "/jobs/7", "/scout", "/settings"])
async def test_all_internal_links_stay_under_proxy(client, path):
    response = await client.get(path)
    assert response.status_code == 200
    links = re.findall(r'(?:href|src|action|hx-get|hx-post|data-api-url)="(/[^" ]*)"', response.text)
    assert links
    assert all(link.startswith(PREFIX + "/") for link in links), links


async def test_partial_macro_redirect_and_static(client):
    response = await client.get("/jobs", headers={"HX-Request": "true"})
    assert f'hx-get="{PREFIX}/jobs' in response.text
    response = await client.post("/jobs/7/status", data={"status": "applied"})
    assert f'href="{PREFIX}/jobs/7"' in response.text
    response = await client.post("/jobs/7/notes", data={"notes": "hello"})
    assert response.headers["location"] == PREFIX + "/jobs/7"
    response = await client.post("/jobs/7/status", data={"status": "new", "redirect": "1"})
    assert response.headers["location"] == PREFIX + "/jobs/7"
    assert (await client.get("/static/controller.js")).status_code == 200
    page = await client.get("/controller")
    assert f'data-base-path="{PREFIX}"' in page.text
