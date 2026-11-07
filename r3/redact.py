"""Taking links out of text a regular account reads (the OQ-6 teaser): a
job seeker never gets a posting's URL or anything that leads to it.

Covered: URLs with a scheme, ``www.`` hosts, hosts with a path, hosts of
three or more labels (careers.acme.com), job-board and applicant-tracking
domains at any depth (acme.workable.com, workable.com), IP addresses,
email addresses and phone numbers. A bare two-label name survives on
purpose (acme.com reads like the company name the teaser already shows, and
the rule would also hit Node.js, ASP.NET or socket.io).
"""

from __future__ import annotations

import re

# Job boards and applicant-tracking systems: where a posting is published.
BOARD_DOMAINS = (
    "workable.com", "lever.co", "greenhouse.io", "ashbyhq.com", "myworkdayjobs.com", "workday.com",
    "recruitee.com", "bamboohr.com", "smartrecruiters.com", "jobvite.com", "breezy.hr", "teamtailor.com",
    "icims.com", "personio.de", "personio.com", "linkedin.com", "indeed.com", "glassdoor.com", "wellfound.com",
    "dice.com", "ziprecruiter.com", "remoteok.com", "weworkremotely.com",
)

LINK_PATTERNS = (
    re.compile(r"(?i)\b(?:https?|ftp)://\S+"),
    re.compile(r"(?i)\bwww\.\S+"),
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    # A host with a path (careers.acme.com/jobs/7).
    re.compile(r"(?i)\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}/\S*"),
    # A host of three or more labels ending in a real top-level domain.
    re.compile(r"(?i)\b(?:[a-z0-9-]+\.){2,}[a-z]{2,}\b"),
    re.compile(r"(?i)\b(?:[a-z0-9-]+\.)*(?:" + "|".join(re.escape(d) for d in BOARD_DOMAINS) + r")\b"),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:/\S*)?"),
    # Phone numbers: international (+...) and North American (415) 555-0100.
    re.compile(r"\+\d[\d\s().-]{7,}\d"),
    re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
)


def strip_links(text: str) -> str:
    """``text`` with every link-like token removed and the spacing repaired."""
    for pattern in LINK_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?)])", r"\1", text)
    return re.sub(r"\( *\)", "", text).strip()


def has_link(text: str) -> bool:
    return any(p.search(text) for p in LINK_PATTERNS)


def without_link_sentences(text: str) -> str:
    """``text`` minus every sentence that carries a link (a sentence with a
    URL is usually about the link, so stripping only the URL leaves
    "apply at  soon")."""
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s and not has_link(s)]
    return " ".join(kept)
