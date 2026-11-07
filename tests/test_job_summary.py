"""The job teaser's overview (OQ-6): a regular account never sees the
posting, only two or three sentences about it. The LLM summarizer and its
extractive fallback must never hand out a link, and the cache table is R3's
own."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from r3.db.job_summaries import JOB_SUMMARIES_DDL, JobSummaryQueries
from r3.db.jobs import Job
from r3.job_summary import ExtractiveSummarizer, LLMSummarizer, extract_overview, strip_links

DESCRIPTION = (
    "Acme builds launch software for small satellites. Apply at https://acme.example/careers/7 or write to "
    "jobs@acme.example with your CV. You will run our Kubernetes platform on AWS. You will own CI/CD and "
    "on-call. We offer equity."
)


def make_job(**over) -> Job:
    base = dict(id=7, site_url="https://boards.example/jobs", site_name="Example Boards",
                job_url="https://boards.example/jobs/7", apply_url="https://acme.example/apply/7",
                position="Platform Engineer", company="Acme", location="Remote, US",
                scraped_at=datetime(2026, 10, 1, tzinfo=timezone.utc), status="new",
                tech_tags=["kubernetes", "aws"], description=DESCRIPTION)
    return Job(**(base | over))


def _no_links(text: str) -> None:
    for leak in ("http", "www.", "@", "acme.example", "boards.example"):
        assert leak not in text, (leak, text)


# --- link stripping ----------------------------------------------------------


def test_strip_links_removes_urls_hosts_and_emails_but_keeps_tech_names():
    text = ("See https://a.example/x?y=1, www.b.example, careers.c.example/jobs/9 and hr@d.example. "
            "We use Node.js, CI/CD and ASP.NET, e.g. daily.")
    out = strip_links(text)
    for leak in ("https://", "a.example", "www.", "b.example", "c.example/jobs", "hr@", "d.example"):
        assert leak not in out, leak
    for keep in ("Node.js", "CI/CD", "ASP.NET", "e.g. daily"):
        assert keep in out, keep
    assert "  " not in out and " ," not in out


# --- extractive fallback -----------------------------------------------------


def test_extract_overview_is_the_first_sentences_without_links():
    ov = extract_overview(make_job())
    assert ov.engine == "extract" and ov.model == ""
    assert ov.text.startswith("Acme builds launch software for small satellites.")
    assert "Kubernetes platform on AWS" in ov.text
    assert "equity" not in ov.text  # three sentences at most
    _no_links(ov.text)


def test_extract_overview_without_a_description_describes_the_role_from_its_fields():
    ov = extract_overview(make_job(description=""))
    assert "Platform Engineer" in ov.text and "Acme" in ov.text and "kubernetes" in ov.text
    _no_links(ov.text)


def test_extract_overview_is_capped():
    ov = extract_overview(make_job(description="word " * 2000))
    assert len(ov.text) <= 600


async def test_extractive_summarizer_never_uses_a_model():
    s = ExtractiveSummarizer()
    assert s.uses_model() is False
    assert (await s.overview(make_job())).engine == "extract"


# --- LLM summarizer ----------------------------------------------------------


async def test_llm_overview_is_the_models_text_with_links_stripped():
    said = ("Acme is hiring a platform engineer to run Kubernetes on AWS. Details: https://acme.example/7. "
            "The role owns CI/CD and on-call.")
    ov = await LLMSummarizer(TestModel(custom_output_args={"summary": said})).overview(make_job())
    assert ov.engine == "llm" and ov.model == "test"
    assert "run Kubernetes on AWS" in ov.text and "owns CI/CD" in ov.text
    _no_links(ov.text)


async def test_llm_prompt_carries_the_posting_but_no_link():
    seen: list[str] = []

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(repr(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"summary": "A platform role running Kubernetes on AWS at Acme."})])

    ov = await LLMSummarizer(FunctionModel(answer)).overview(make_job())
    assert ov.engine == "llm"
    prompt = seen[0]
    assert "Platform Engineer" in prompt and "launch software for small satellites" in prompt
    for leak in ("https://", "acme.example", "boards.example", "jobs@"):
        assert leak not in prompt, leak


async def test_llm_without_a_key_uses_the_extract():
    def no_key():
        raise ValueError("no chat provider has a key")

    s = LLMSummarizer(no_key)
    assert s.uses_model() is False
    assert (await s.overview(make_job())).engine == "extract"


async def test_llm_failure_timeout_and_empty_answer_fall_back_to_the_extract():
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider down")

    async def hang(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(5)
        raise AssertionError("not reached")

    assert (await LLMSummarizer(FunctionModel(boom)).overview(make_job())).engine == "extract"
    assert (await LLMSummarizer(FunctionModel(hang), timeout=0.05).overview(make_job())).engine == "extract"
    empty = LLMSummarizer(TestModel(custom_output_args={"summary": "  https://acme.example/7  "}))
    assert (await empty.overview(make_job())).engine == "extract"


# --- the cache table ---------------------------------------------------------


class RecordingPool:
    def __init__(self, row=None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.row = row

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        return self.row

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "INSERT 0 1"


def _queries(pool: RecordingPool) -> JobSummaryQueries:
    q = JobSummaryQueries()
    q.pool = pool  # type: ignore[assignment]
    return q


def test_summary_table_is_keyed_by_job_without_a_foreign_key():
    ddl = " ".join(JOB_SUMMARIES_DDL.split())
    assert "CREATE TABLE IF NOT EXISTS r3_job_summaries" in ddl
    assert "job_id bigint PRIMARY KEY" in ddl and "REFERENCES" not in ddl


async def test_summary_reads_and_upserts_one_row_per_job():
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    pool = RecordingPool(row={"job_id": 7, "summary": "S.", "model": "m", "created_at": now})
    q = _queries(pool)
    await q.ensure_job_summary_table()
    got = await q.get_job_summary(7)
    assert got is not None and (got.job_id, got.summary, got.model) == (7, "S.", "m")
    await q.save_job_summary(7, "New.", "m2")
    sql, args = pool.calls[-1]
    assert "INSERT INTO r3_job_summaries" in sql and "ON CONFLICT (job_id) DO UPDATE" in sql
    assert args == (7, "New.", "m2")
    assert all("scraped_jobs" not in s for s, _ in pool.calls)  # the shared table is never touched

    pool.row = None
    assert await q.get_job_summary(8) is None


# --- review fixes ----------------------------------------------------------


def test_strip_links_removes_career_and_ats_hosts_ips_and_phones():
    text = ("Apply at careers.acme.com today, or at acme.workable.com, jobs.lever.co/acme and workable.com. "
            "Server 192.168.0.1:8080/apply. Call +1 (415) 555-0100 or (415) 555-0101. "
            "We use Node.js, ASP.NET, socket.io and Vue.js in the U.S. on v1.2; pay $120,000 - $150,000, 401(k).")
    out = strip_links(text)
    for leak in ("careers.acme.com", "acme.workable", "lever.co", "workable.com", "192.168", "555-010", "415"):
        assert leak not in out, leak
    for keep in ("Node.js", "ASP.NET", "socket.io", "Vue.js", "U.S.", "v1.2", "$120,000 - $150,000", "401(k)"):
        assert keep in out, keep


def test_extract_reads_an_html_posting_as_text():
    html = ('<div class="content-intro"><p><strong>About Acme</strong></p><p>Acme builds logistics software used '
            'by thousands of carriers. Our platform team keeps it fast.</p></div><p>You will own SLOs and on-call.</p>'
            '<p>We run Kubernetes on AWS.</p>')
    ov = extract_overview(make_job(description=html))
    assert "<" not in ov.text and "&" not in ov.text
    assert ov.text.startswith("Acme builds logistics software used by thousands of carriers.")


def test_extract_never_shows_more_than_half_of_a_short_posting():
    two = extract_overview(make_job(description="Acme ships rockets for small satellites. You will run Kubernetes."))
    assert two.text == "Acme ships rockets for small satellites."
    one = extract_overview(make_job(description="Acme ships rockets for small satellites and needs help."))
    assert "ships rockets" not in one.text and "Platform Engineer" in one.text  # built from the fields
    many = extract_overview(make_job(description=" ".join(f"Sentence number {i} is here." for i in range(10))))
    assert many.text == "Sentence number 0 is here. Sentence number 1 is here."


async def test_llm_prompt_reads_an_html_posting_as_text():
    seen: list[str] = []

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(repr(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"summary": "A platform role running Kubernetes on AWS at Acme."})])

    await LLMSummarizer(FunctionModel(answer)).overview(make_job(description="<p>Acme <b>builds</b> rockets.</p>"))
    assert "<p>" not in seen[0] and "<b>" not in seen[0] and "Acme builds rockets." in seen[0]
