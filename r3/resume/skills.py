"""Which of the corpus's tech tags a resume evidences.

Deterministic and cheap: every job row already carries lower-case
``tech_tags``, so a resume's skills are simply the tags from that vocabulary
that appear in its text as whole words or phrases. No LLM is involved; the
result feeds the SQL-level match (``list_jobs(..., skills=...)``).

The alias table is deliberately small and local.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Iterable

# ALIASES maps another spelling to the tag it evidences. An alias only counts
# when its tag is in the vocabulary.
ALIASES: dict[str, str] = {
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "golang": "go",
    "js": "javascript",
    "ts": "typescript",
    "google cloud": "gcp",
    "nodejs": "node.js",
}

# STRICT tags are also ordinary English words or single letters, so they only
# count in their canonical capitalisation and as standalone tokens: "Go" but
# not "going", "go-to-market" or "Google"; "R" but not "R&D"; "REST" but not
# "the rest of the team".
STRICT: dict[str, str] = {"go": "Go", "r": "R", "c": "C", "rest": "REST", "react": "React"}

# A word character on either side of a match means it is part of a longer
# word. "+" and "#" trail too, so "c" never matches inside "C++" or "C#".
_BEFORE = r"(?<![A-Za-z0-9_])"
_AFTER = r"(?![A-Za-z0-9_+#])"
_ALIAS_BEFORE = r"(?<![A-Za-z0-9_.])"
# Standalone-token delimiters for STRICT tags.
_STRICT_BEFORE = r"(?:^|(?<=[\s,;/(|]))"
_STRICT_AFTER = r"(?=$|[\s,;/)|]|\.(?:\s|$))"


def _phrase(term: str) -> str:
    """Escape a tag for a regex, letting a space also match a hyphen or a
    run of whitespace ("github actions" / "GitHub-Actions")."""
    return r"[\s\-]+".join(re.escape(part) for part in term.split())


def _pattern(term: str, *, alias: bool = False) -> re.Pattern[str]:
    # A tag that starts with punctuation (".net") may follow a word: "ASP.NET".
    # An alias must stand alone: the "js" in "Node.js" is not an alias hit.
    before = _ALIAS_BEFORE if alias else _BEFORE if term[:1].isalnum() else ""
    return re.compile(before + _phrase(term) + _AFTER, re.IGNORECASE)


def resume_skills(resume_text: str, vocabulary: Iterable[str]) -> list[str]:
    """Return the tags from ``vocabulary`` that ``resume_text`` evidences,
    sorted. Matching is whole-word/phrase and case-insensitive, except for
    the few :data:`STRICT` tags."""
    if not resume_text or not resume_text.strip():
        return []
    vocab = {t.strip().lower() for t in vocabulary if t and t.strip()}
    found: set[str] = set()
    for tag in vocab:
        if tag in STRICT:
            pat = re.compile(_STRICT_BEFORE + re.escape(STRICT[tag]) + _STRICT_AFTER)
        else:
            pat = _pattern(tag)
        if pat.search(resume_text):
            found.add(tag)
    for alias, tag in ALIASES.items():
        if tag in vocab and tag not in found and _pattern(alias, alias=True).search(resume_text):
            found.add(tag)
    return sorted(found)


class VocabularyCache:
    """Holds the distinct tech-tag vocabulary for ``ttl`` seconds per
    process, so a page view does not re-scan every job's tags. The clock is
    injectable for tests."""

    def __init__(self, ttl: float = 300.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl = ttl
        self.clock = clock
        self._value: list[str] | None = None
        self._at = 0.0

    async def get(self, load: Callable[[], Awaitable[list[str]]]) -> list[str]:
        now = self.clock()
        if self._value is None or now - self._at >= self.ttl:
            self._value = await load()
            self._at = now
        return self._value
