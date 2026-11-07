"""The stages the Controller shows beside the chat.

Each pipeline is one stage of a job search: finding postings worth a look,
then tracking what came of them. A pipeline is a description, the portal
pages that touch its data, and the live facts worth glancing at while
chatting."""

from __future__ import annotations

from dataclasses import dataclass

from r3.db import Facets, Stats


@dataclass(frozen=True)
class Link:
    href: str
    label: str


@dataclass(frozen=True)
class Pipeline:
    key: str
    name: str
    summary: str
    links: tuple[Link, ...] = ()


PIPELINES: tuple[Pipeline, ...] = (
    Pipeline(
        key="jobs", name="Jobs",
        summary="Browse the stored postings and triage the unreviewed ones.",
        links=(Link("/scout", "Explore"), Link("/jobs?status=new", "Unreviewed")),
    ),
    Pipeline(
        key="tracking", name="Tracking",
        summary="What you applied to and what you passed on.",
        links=(Link("/jobs?status=applied", "Applied"), Link("/jobs?status=skipped", "Skipped")),
    ),
)

PIPELINE_KEYS: frozenset[str] = frozenset(p.key for p in PIPELINES)


@dataclass(frozen=True)
class Fact:
    """One glanceable number on a pipeline card."""

    value: str
    label: str
    warn: bool = False


def pipeline_facts(stats: Stats, facets: Facets) -> dict[str, list[Fact]]:
    """The live facts for each pipeline card, keyed by pipeline. Built from
    the same models the overview page already uses, so the numbers match
    what it shows."""
    by_status = {v.value: v.count for v in facets.statuses}
    return {
        "jobs": [
            Fact(f"{stats.new_jobs:,}", "unreviewed"),
            Fact(f"{stats.jobs_last7:,}", "added 7d"),
            Fact(f"{stats.total_jobs:,}", "jobs stored"),
        ],
        "tracking": [
            Fact(f"{stats.applied_jobs:,}", "applied"),
            Fact(f"{by_status.get('skipped', 0):,}", "skipped"),
        ],
    }
