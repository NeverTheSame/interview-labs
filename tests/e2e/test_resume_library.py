"""The resume library: upload, paste, edit, primary, delete, download (item 3)."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from tests.e2e.conftest import RESUME_TEXT, api_paste_resume, api_signup
from tests.test_resume_extract import make_docx, make_pdf


def _rows(page):
    return page.locator("table.resume-list tbody tr")


def _upload(page, name: str, mime: str, data: bytes, title: str = "") -> None:
    page.goto("/resume")
    page.get_by_label(re.compile("^Resume file")).set_input_files({"name": name, "mimeType": mime, "buffer": data})
    if title:
        page.locator("#upload-title").fill(title)
    page.get_by_role("button", name="Upload resume").click()


def test_new_account_sees_empty_state(page):
    api_signup(page)
    page.goto("/resume")
    expect(page.get_by_text("You have not added a resume yet.")).to_be_visible()
    expect(page.get_by_role("link", name=re.compile("Build one by answering"))).to_be_visible()


def test_upload_txt_docx_pdf_extracts_text_and_first_is_primary(page):
    api_signup(page)
    _upload(page, "plain.txt", "text/plain", ("TXT marker line\n" + RESUME_TEXT).encode())
    expect(page.locator("main")).to_contain_text("TXT marker line")
    _upload(page, "word.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            make_docx(["DOCX marker paragraph", *RESUME_TEXT.splitlines()]))
    expect(page.locator("main")).to_contain_text("DOCX marker paragraph")
    _upload(page, "paper.pdf", "application/pdf", make_pdf(["PDF marker line", "Site Reliability Engineer",
                                                           "Kubernetes Terraform AWS Prometheus Go Python"]))
    expect(page.locator("main")).to_contain_text("PDF marker line")

    page.goto("/resume")
    expect(_rows(page)).to_have_count(3)
    primary = _rows(page).filter(has=page.locator(".pill.primary"))
    expect(primary).to_have_count(1)
    expect(primary).to_contain_text("plain.txt")


def test_paste_rules_and_unsupported_type(page):
    api_signup(page)
    page.goto("/resume")
    page.locator("#paste-content").fill("Too short to be a resume.")
    page.get_by_role("button", name="Save resume").click()
    expect(page.get_by_role("alert")).to_contain_text("a resume needs at least 50")
    expect(page.locator("#paste-content")).to_have_value("Too short to be a resume.")

    _upload(page, "photo.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    expect(page.get_by_role("alert")).to_contain_text("not supported")

    page.locator("#paste-title").fill("Pasted SRE resume")
    page.locator("#paste-content").fill(RESUME_TEXT)
    page.get_by_role("button", name="Save resume").click()
    expect(page.get_by_role("heading", name="Pasted SRE resume")).to_be_visible()


def test_edit_switch_primary_delete_and_download(page):
    api_signup(page)
    api_paste_resume(page, "First resume", RESUME_TEXT)
    api_paste_resume(page, "Second resume", "Second version\n" + RESUME_TEXT)

    page.goto("/resume")
    page.get_by_role("link", name="Second resume", exact=True).click()
    page.get_by_label("Title").fill("Second resume, edited")
    page.get_by_label("Resume text").fill("Edited body line\n" + RESUME_TEXT)
    page.get_by_role("button", name="Save changes").click()
    expect(page.get_by_role("heading", name="Second resume, edited")).to_be_visible()
    page.reload()
    expect(page.get_by_label("Resume text")).to_have_value(re.compile("^Edited body line"))

    with page.expect_download() as dl:
        page.get_by_role("link", name="Download .txt").click()
    assert "Edited body line" in open(dl.value.path(), encoding="utf-8").read()

    page.get_by_role("button", name="Make primary").click()
    page.goto("/resume")
    expect(_rows(page).filter(has=page.locator(".pill.primary"))).to_contain_text("Second resume, edited")

    # Deleting the primary promotes the remaining resume.
    page.get_by_role("link", name="Second resume, edited", exact=True).click()
    page.get_by_role("button", name="Delete").click()
    page.goto("/resume")
    expect(_rows(page)).to_have_count(1)
    expect(_rows(page).filter(has=page.locator(".pill.primary"))).to_contain_text("First resume")
