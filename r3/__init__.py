"""R3 — a job-search portal.

A server-rendered portal over a Postgres database of job postings: browse
and triage jobs, keep your resume, and chat with an assistant about both.
``r3.db`` owns every query, ``r3.web`` the FastAPI app and templates, and
``r3.portal`` is the process entry point.
"""
