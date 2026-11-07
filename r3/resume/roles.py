"""Which roles a resume is after, as keywords a job title can be ranked by.

The Jobs page's best-match sort ranks by title fit before tag overlap: a
DevOps seeker's DevOps postings come before a Technical Writer posting whose
only tag happens to be on the resume. The keywords come from the resume's
headline (the target role under the name) and its most recent role titles.

Deterministic and cheap, no LLM. Generic words ("senior", "engineer") are
dropped, so what is left says what the role is about, and a few spellings
are folded into one keyword ("site reliability" is "sre"). The SQL side gets
regular expressions built from these keywords only, bound as parameters.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .builder import ResumeDraft
from .parse import draft_from_labels, heuristic_labels

# How many of the most recent roles count, after the headline.
RECENT_ROLES = 2

# A "title" longer than this is a sentence (a summary line read as the
# headline), not a role.
MAX_TITLE_WORDS = 8

# GENERIC words say nothing about what a role is: seniority, level, the role
# noun itself, connectives and employment terms.
GENERIC = frozenset({
    "senior", "sr", "junior", "jr", "lead", "leader", "staff", "principal", "chief", "head", "mid", "entry",
    "level", "associate", "intern", "trainee", "apprentice", "graduate", "i", "ii", "iii", "iv", "v",
    "engineer", "engineers", "engineering", "eng", "developer", "developers", "dev", "specialist", "manager",
    "management", "administrator", "admin", "analyst", "architect", "consultant", "contractor", "technician",
    "officer", "coordinator", "professional", "expert", "member", "team",
    "and", "or", "of", "the", "for", "at", "in", "on", "to", "with", "an", "as",
    "remote", "hybrid", "onsite", "contract", "freelance", "temporary", "part", "full", "time",
})

# SYNONYMS: a keyword and every spelling a job title may use for it. A title
# on the resume is folded to the keyword; a job title matches any spelling.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "sre": ("sre", "site reliability"),
    "devops": ("devops", "dev ops", "devsecops"),
    "infrastructure": ("infrastructure", "infra"),
    "platform": ("platform",),
    "cloud": ("cloud",),
    "backend": ("backend", "back end"),
    "frontend": ("frontend", "front end"),
    "fullstack": ("fullstack", "full stack"),
    "ml": ("ml", "machine learning"),
    "qa": ("qa", "quality assurance"),
}

_WORD = re.compile(r"[a-z0-9]+")
_KEYWORD = re.compile(r"[a-z0-9]+(?: [a-z0-9]+)*")
# A headline often lists more after its role: "DevOps Engineer | AWS | K8s".
_TRAILER = re.compile(r"[|:;•·]")


def _spelling(spelling: str, *, sql: bool) -> str:
    """One spelling as a whole-word pattern: words may be joined by any run of
    non-word characters ("site-reliability") and the last may be plural."""
    words = spelling.split()
    plural = "" if words[-1].endswith("s") else "s?"
    start, end = (r"\m", r"\M") if sql else (r"\b", r"\b")
    return start + r"\W*".join(words) + plural + end


_FOLD = [(re.compile("|".join(_spelling(s, sql=False) for s in spellings)), keyword)
         for keyword, spellings in SYNONYMS.items()]


def title_keywords(title: str) -> list[str]:
    """The keywords one role title names, in order: synonyms folded, generic
    words dropped. A sentence rather than a title names none."""
    text = _TRAILER.split(title.lower(), 1)[0]
    for pattern, keyword in _FOLD:
        text = pattern.sub(f" {keyword} ", text)
    words = _WORD.findall(text)
    if len(words) > MAX_TITLE_WORDS:
        return []
    out: list[str] = []
    for w in words:
        if w not in GENERIC and not w.isdigit() and len(w) > 1 and w not in out:
            out.append(w)
    return out


def title_patterns(keywords: Iterable[str]) -> list[str]:
    """Postgres regular expressions (for ``~*``) that match a job title naming
    any of ``keywords``, in every spelling. Anything that is not a plain
    keyword of letters and digits is left out."""
    out: list[str] = []
    for k in keywords:
        if not _KEYWORD.fullmatch(k):
            continue
        for spelling in SYNONYMS.get(k, (k,)):
            p = _spelling(spelling, sql=True)
            if p not in out:
                out.append(p)
    return out


@dataclass(frozen=True)
class RoleKeywords:
    """The roles a resume names. ``target`` is the role it is after (the
    headline, or the latest role that names one); ``recent`` the other recent
    roles, without the target's keywords. Empty when the resume names none."""

    target: tuple[str, ...] = ()
    recent: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.target or self.recent)

    def patterns(self) -> tuple[list[str], list[str]] | None:
        """The (target, recent) title patterns for ``list_jobs(titles=...)``,
        or None when there is nothing to rank by."""
        return (title_patterns(self.target), title_patterns(self.recent)) if self else None


def role_keywords(headline: str, titles: Sequence[str]) -> RoleKeywords:
    """Keywords from a headline and role titles, most recent first. The first
    of them that names a role is the target; the rest of the window (the
    headline and :data:`RECENT_ROLES` roles) is recent."""
    named = [k for k in (title_keywords(t) for t in [headline, *titles[:RECENT_ROLES]]) if k]
    if not named:
        return RoleKeywords()
    target = tuple(named[0])
    recent: list[str] = []
    for keywords in named[1:]:
        recent += [k for k in keywords if k not in target and k not in recent]
    return RoleKeywords(target=target, recent=tuple(recent))


def resume_roles(content: str, structure: Mapping[str, Any] | None) -> RoleKeywords:
    """The roles a stored resume names: from how it was read (``structure``,
    written by the upload reader) when that has a headline or titles, else
    from a rule-based read of its text (a builder or pasted resume)."""
    raw = (structure or {}).get("draft") or {}
    try:
        draft = ResumeDraft.model_validate(raw)
    except ValidationError:  # an older or damaged read: the text still works
        draft = ResumeDraft()
    if not (draft.headline or any(j.title for j in draft.experience)):
        if not content.strip():
            return RoleKeywords()
        draft = draft_from_labels(content, heuristic_labels(content))
    return role_keywords(draft.headline, [j.title for j in draft.experience])
