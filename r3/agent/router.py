"""Workflow detection ("herald"): decide whether a chat message asks for a
workflow (``jobs | stats``) or is a question for the advisor (``advice``),
and pull out the parameters it names.

Modes (``settings.router_mode``):

- ``rules``: keyword/regex classification and extraction, no model.
- ``llm``: one structured call (:class:`RouteCall`) on ``settings.router_model``,
  or the turn's chat model when that is empty. Any failure falls back to the
  rules (``router="fallback"``, the reason names the failure class). The
  rules' verdict is handed to the LLM as a hint.
- ``jev``: one TypeSafe Jev request per turn and no LLM request. Jev picks the
  intent and fills the closed-set fields (role families, remote, status,
  employment) of :func:`jev_route_type`; numbers and free text (job ids,
  limit, days, salary, search, tech, location) come from the rules
  extractor. A failure falls back to the rules.
- ``hybrid``: Jev first; when it raises, or its intent confidence is below
  ``settings.jev_confidence``, the whole route comes from the LLM path
  (``router="llm"``, Jev's lean kept in the reason), and from the rules if the
  LLM fails too.

Fast path (every mode, no model call), see :func:`fast_path`. It fires only when
the rules can be sure:

1. the message starts with a slash command: ``/jobs``, ``/stats``, ``/advice``
   (``/ask``, ``/help``);
2. the whole message is a greeting or thanks ("hi", "thanks", ...): advice.

Everything else is decided by the configured mode.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, WithJsonSchema, create_model
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from r3.db import EMPLOYMENT_FAMILIES, Facets
from r3.settings import ControllerSettings

from .contracts import INTENT_DESCRIPTIONS, INTENTS, ChatMessage, Intent, RouteDecision, RouteParams
from .models import build_model, history_messages

log = logging.getLogger("r3.agent.router")

LLM_TIMEOUT_S = 20.0
JEV_TIMEOUT_S = 12.0
HISTORY_WINDOW = 6  # messages of context the models see
HISTORY_CHARS = 1500  # per message

INTENT_QUESTION = "Which of these does the operator's latest message ask for?"


# --- vocabulary ---------------------------------------------------------------------

# The role buckets postings are classified into. AI and ML are separate buckets now; older rows say "AI/ML".
CANONICAL_ROLES: tuple[str, ...] = ("SRE", "DevOps", "Data", "AI", "ML")
_FAMILY_CANONICAL: dict[str, tuple[str, ...]] = {
    "ai": ("AI", "ML"), "sre": ("SRE",), "devops": ("DevOps",), "data": ("Data",), "other": ("Other",),
}
_FAMILY_LABEL: dict[str, str] = {
    "ai": "AI, machine learning, LLM or MLOps roles",
    "sre": "Site reliability engineering (SRE) roles",
    "devops": "DevOps, platform, infrastructure or cloud engineering roles",
    "data": "Data engineering or analytics roles",
}
STATUS_WORDS: dict[str, str] = {
    "new": "new", "applied": "applied", "skipped": "skipped", "expired": "expired",
}

_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
}
_NUM = r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
# Spanish and Russian words for a job, for an operator who switches language (kept to the words the
# Controller's commands need; "resumen" is Spanish for a summary, so it is never a resume).
_ML_JOB_NOUN = r"(?:trabajos?|vacantes?|ofertas?|empleos?|puestos?|вакансии|вакансий|вакансию|вакансия|работы|работу)"
_JOB_NOUN = (r"(?:jobs?|postings?|positions?|roles?|openings?|listings?|opportunit(?:y|ies)|vacanc(?:y|ies)"
             r"|gigs?|matches|results|leads|contracts|" + _ML_JOB_NOUN + r")")
# Facet tags that are also everyday words ("go", "data"): the scan of the facet list skips them.
_TECH_STOPWORDS = frozenset({
    "go", "c", "r", "ai", "ml", "llm", "data", "cloud", "security", "platform", "backend", "frontend", "api",
    "apis", "web", "mobile", "sre", "devops", "remote", "senior", "lead", "staff", "jobs", "new", "all", "rest",
    "design", "testing", "support", "sales", "product", "ops", "infra", "infrastructure", "networking", "it",
    "agile", "scrum", "saas", "startup", "analytics", "automation", "monitoring", "observability", "research",
})
_TECH_ALIASES: dict[str, tuple[str, ...]] = {
    "kubernetes": ("kubernetes", "k8s"), "terraform": ("terraform",), "aws": ("aws", "amazon web services"),
    "gcp": ("gcp", "google cloud"), "azure": ("azure",), "python": ("python",), "golang": ("golang",),
    "rust": ("rust",), "java": ("java",), "kafka": ("kafka",), "spark": ("spark", "pyspark"),
    "airflow": ("airflow",), "snowflake": ("snowflake",), "dbt": ("dbt",), "pytorch": ("pytorch",),
    "tensorflow": ("tensorflow",), "docker": ("docker",), "ansible": ("ansible",), "linux": ("linux",),
    "postgres": ("postgres", "postgresql"), "prometheus": ("prometheus",), "grafana": ("grafana",),
    "datadog": ("datadog",), "helm": ("helm",), "argocd": ("argocd", "argo cd"), "jenkins": ("jenkins",),
    "typescript": ("typescript",), "scala": ("scala",), "databricks": ("databricks",), "bigquery": ("bigquery",),
    "sql": ("sql",), "langchain": ("langchain",), "openai": ("openai",),
}
# Other spellings the facet list may use for the same tag.
_TECH_FACET_ALIASES: dict[str, tuple[str, ...]] = {"golang": ("go",), "postgres": ("postgresql",), "gcp": ("google cloud",)}


def _norm(text: str) -> str:
    """Lowercase, straight quotes, single spaces."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text.strip().lower())


def _num(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 0)


def _clip(text: str, n: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _dedupe(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _blank(text: str, span: tuple[int, int]) -> str:
    start, end = span
    return text[:start] + " " * (end - start) + text[end:]


# --- roles, tech and status against the database's vocabulary -------------------------------

def role_family(role: str) -> str | None:
    """Which bucket family a free-text role belongs to: ai | sre | devops | data | other."""
    low = role.strip().lower()
    tokens = set(re.findall(r"[a-z0-9]+", low))
    if tokens & {"ai", "ml", "llm", "llms", "genai", "mlops", "nlp"} or any(
        phrase in low for phrase in ("machine learning", "artificial intelligence", "deep learning", "computer vision",
                                     "generative")):
        return "ai"
    if tokens & {"sre", "sres"} or "reliability" in low:
        return "sre"
    if tokens & {"devops", "devsecops", "platform", "infra", "infrastructure"} or "cloud engineer" in low:
        return "devops"
    if tokens & {"data", "analytics", "etl"}:
        return "data"
    if low == "other":
        return "other"
    return None


def _family_matches(family: str, values: Sequence[str]) -> list[str]:
    return [v for v in values if role_family(v) == family]


def normalize_roles(roles: Sequence[str], facets: Facets | None) -> tuple[list[str], list[str]]:
    """Map free-text roles onto the database's role categories.

    Returns ``(known, unknown)``: ``known`` are exact category strings from
    ``facets`` (case-insensitive match, or every category of the role's family:
    "machine learning" → the AI-ish ones), or the canonical spellings
    (SRE, DevOps, Data, AI, ML) when no facets are at hand; ``unknown`` are
    the roles that match nothing the database has, as written."""
    values = [v.value for v in facets.role_categories if v.value] if facets else []
    known: list[str] = []
    unknown: list[str] = []
    for role in roles:
        role = role.strip()
        if not role:
            continue
        exact = next((v for v in values if v.lower() == role.lower()), None)
        if exact is not None:
            known.append(exact)
            continue
        family = role_family(role)
        if family is None:
            (unknown if values else known).append(role)
        elif values:
            matches = _family_matches(family, values)
            (known.extend(matches) if matches else unknown.append(role))
        else:
            known.extend(_FAMILY_CANONICAL[family])
    return _dedupe(known), _dedupe(unknown)


def normalize_tech(tag: str, facets: Facets | None) -> str:
    """A tech tag in the facet list's spelling when it has one, else lowercase."""
    low = tag.strip().lower()
    group = next((key for key, aliases in _TECH_ALIASES.items() if low == key or low in aliases), low)
    spellings = {group, *(_TECH_ALIASES.get(group, ())), *(_TECH_FACET_ALIASES.get(group, ())), low}
    if facets:
        for value in facets.tech_tags:
            if value.value.lower() in spellings:
                return value.value
    return group


# --- rules: extraction -------------------------------------------------------------------

_ROLE_TEXT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ai", re.compile(r"\b(?:ai/ml|ml/ai|ai|ml|machine[- ]learning|artificial intelligence|llms?|gen ?ai"
                      r"|generative ai|mlops|deep learning|nlp|computer vision)\b")),
    ("sre", re.compile(r"\b(?:sres?|site[- ]reliability)\b")),
    ("devops", re.compile(r"\b(?:devops|devsecops|dev ops)\b|\b(?:platform|infra(?:structure)?|cloud)\b"
                          r"(?=(?:\s+[\w/&+-]+){0,2}?\s+(?:engineer\w*|roles?|jobs?|positions?|openings?|postings?|teams?)\b)")),
    ("data", re.compile(r"\bdata\b(?=(?:\s+[\w/&+-]+){0,2}?\s+(?:engineer\w*|roles?|jobs?|positions?|openings?"
                        r"|postings?|science|scientists?|analysts?|analytics|platform)\b)|\b(?:analytics engineer\w*|etl)\b")),
)

_SALARY_K = re.compile(r"\$?\s?(\d{2,3}(?:\.\d{1,2})?)\s?k\b\+?")
_SALARY_FULL = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d{5,7})\b\+?")
_SALARY_CTX = re.compile(r"\b(?:salary|pay|paying|pays|comp|compensation|base|earn\w*|at least|over|above|minimum|min"
                         r"|more than)\b[^.?!\n]{0,20}?\b(\d{1,3}(?:,\d{3})+|\d{5,7})\b")

_TIME_UNIT = r"(days?|d|weeks?|wks?|w|months?|mo|hours?|hrs?|h)"
_DAYS_N = re.compile(r"\b(?:last|past|previous|within(?: the)?(?: last| past)?|in the (?:last|past)|over the (?:last|past)"
                     r"|from the (?:last|past)|for the (?:last|past)|since)\s+" + _NUM + r"\s*" + _TIME_UNIT + r"\b")
_DAYS_AGO = re.compile(r"\b" + _NUM + r"\s*(days?|weeks?|months?|hours?)\s+(?:ago|old)\b")
_DAYS_COMPACT = re.compile(r"(?<![\w$#])(\d{1,3})(d|w|h)\b")
_DAYS_WORDS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(?:today|last 24 hours|past 24 hours|this morning|tonight|hoy|сегодня)\b"), 1),
    (re.compile(r"\b(?:yesterday|ayer|вчера)\b"), 2),
    (re.compile(r"\b(?:this|past|last|previous) (?:fortnight|two weeks)\b"), 14),
    (re.compile(r"\b(?:this|past|last|previous) week\b|\bweekly\b|\besta semana\b|\bla semana pasada\b"
                r"|\bза (?:последнюю |эту )?неделю\b|\bна (?:этой|прошлой) неделе\b"), 7),
    (re.compile(r"\b(?:this|past|last|previous) month\b|\bmonthly\b|\beste mes\b|\bel mes pasado\b"
                r"|\bза (?:последний |этот )?месяц\b|\bв этом месяце\b"), 30),
    (re.compile(r"\b(?:this|past|last) quarter\b"), 90),
    (re.compile(r"\b(?:this|past|last) year\b"), 365),
)
_UNIT_DAYS: dict[str, float] = {"d": 1, "day": 1, "w": 7, "wk": 7, "week": 7, "mo": 30, "month": 30,
                                "h": 1 / 24, "hr": 1 / 24, "hour": 1 / 24}

_LIMIT_TOP = re.compile(r"\b(?:top|first|best|latest|newest|next|most recent|only)\s+" + _NUM
                        + r"\b(?!\s*(?:days?|weeks?|months?|hours?|d\b|w\b|h\b|k\b|%))")
_LIMIT_N = re.compile(r"(?<![#\d$.,])\b" + _NUM + r"\s+(?:[a-z][\w/+#.-]*\s+){0,3}?(?:jobs?|postings?|positions?|roles?"
                      r"|openings?|listings?|results|ones|matches)\b")
_LIMIT_FEW = re.compile(r"\ba (couple|few|handful)\b(?: of)?(?:\s+[\w/+#.-]+){0,3}?\s+" + _JOB_NOUN)

_HASH_ID = re.compile(r"#\s?(\d{1,9})\b")
_KEYWORD_IDS = re.compile(r"\b(?:jobs?|ids?|job ids?|postings?|numbers?|nos?\.|trabajos?|vacantes?|ofertas?|вакансии"
                          r"|вакансию|вакансия|вакансий|номер)\s*#?\s*(\d{1,9}(?:\s*(?:,|and|&|\+|or)\s*#?\s*\d{1,9})*)\b")
_QUOTED = re.compile(r'"([^"]{2,80})"')
_SEARCH_KW = re.compile(r"\b(?:mentioning|mentions|matching|containing|contains|with the (?:word|keyword|term|phrase)s?"
                        r"|keywords?|titled|called|named)\s+([\w+#./&' -]{2,60}?)(?=\s*(?:$|[,.;!?]|\b(?:jobs?|roles?"
                        r"|positions?|postings?|and|in|from|on|posted|for|that|which|with|at)\b))")
_COMPANY = re.compile(r"\bat\s+([A-Z][\w&.'-]*(?:\s+(?:[A-Z][\w&.'-]*|&))*)")
_NOT_COMPANIES = frozenset({"us", "usa", "canada", "remote", "europe", "uk", "the", "a", "an", "least", "most", "all",
                            "home", "work", "night", "noon", "monday", "tuesday", "wednesday", "thursday", "friday",
                            "saturday", "sunday", "once", "first", "last", "r3", "these", "those", "this", "that",
                            "my", "our", "your", "it", "them", "any", "some", "i", "senior", "junior", "staff", "lead",
                            "principal"})
_EMPLOYMENT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("contract", re.compile(r"\b(?:contracts?|contractors?|contracting|c2c|corp[- ]to[- ]corp|1099|freelance)\b")),
    ("fulltime", re.compile(r"\b(?:full[- ]?time|fte|permanent|w2)\b")),
    ("parttime", re.compile(r"\bpart[- ]?time\b")),
    ("internship", re.compile(r"\b(?:intern|interns|internships?)\b")),
)

@dataclass
class _Scan:
    """Numbers found in a message, and the text left once they are blanked out."""

    text: str
    min_salary: int | None = None
    days: int | None = None


def _scan_numbers(t: str) -> _Scan:
    scan = _Scan(text=t)
    for pattern in (_SALARY_K, _SALARY_FULL, _SALARY_CTX):
        for m in pattern.finditer(scan.text):
            raw = m.group(1).replace(",", "")
            if pattern is _SALARY_K and m.group(0).strip().rstrip("+") in ("401k", "403k"):
                continue
            try:
                value = int(float(raw) * 1000) if pattern is _SALARY_K else int(raw)
            except ValueError:
                continue
            if 20_000 <= value <= 10_000_000:
                scan.min_salary = scan.min_salary or value
                scan.text = _blank(scan.text, m.span())
    for pattern in (_DAYS_N, _DAYS_AGO, _DAYS_COMPACT):
        for m in pattern.finditer(scan.text):
            unit = m.group(2).rstrip("s") or "d"
            if unit == "wk":
                unit = "w"
            n = _num(m.group(1))
            if n <= 0:
                continue
            days = max(1, round(n * _UNIT_DAYS.get(unit, 1) + (0.49 if unit in ("h", "hr", "hour") else 0)))
            scan.days = scan.days or days
            scan.text = _blank(scan.text, m.span())
    if scan.days is None:
        for pattern, days in _DAYS_WORDS:
            if m := pattern.search(scan.text):
                scan.days = days
                scan.text = _blank(scan.text, m.span())
                break
    if scan.days is not None:
        scan.days = min(max(scan.days, 1), 365)
    return scan


def _roles_in(t: str, facets: Facets | None) -> list[str]:
    hits: list[tuple[int, str]] = []
    for family, pattern in _ROLE_TEXT:
        if m := pattern.search(t):
            hits.append((m.start(), family))
    families = [family for _, family in sorted(hits)]
    roles: list[str] = []
    values = [v.value for v in facets.role_categories if v.value] if facets else []
    for family in families:
        matches = _family_matches(family, values) if values else []
        roles.extend(matches or _FAMILY_CANONICAL[family])
    # Categories the database has beyond the canonical buckets ("Security", say), named verbatim.
    for value in values:
        if role_family(value) is None and len(value) >= 3 and re.search(r"\b" + re.escape(value.lower()) + r"\b", t):
            roles.append(value)
    return _dedupe(roles)[:10]


def _tech_in(t: str, facets: Facets | None) -> list[str]:
    hits: list[tuple[int, str]] = []
    for key, aliases in _TECH_ALIASES.items():
        for alias in aliases:
            if m := re.search(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", t):
                hits.append((m.start(), normalize_tech(key, facets)))
                break
    if facets:
        taken = {tag.lower() for _, tag in hits}
        for value in facets.tech_tags:
            low = value.value.lower()
            if len(low) < 3 or low in _TECH_STOPWORDS or low in taken:
                continue
            if m := re.search(r"(?<![\w-])" + re.escape(low) + r"(?![\w-])", t):
                hits.append((m.start(), value.value))
    return _dedupe([tag for _, tag in sorted(hits)])[:10]


def _remote_and_location(raw: str, t: str) -> tuple[bool | None, str]:
    remote: bool | None = None
    if re.search(r"\b(?:not remote|non-remote|no remote|in[- ]person)\b", t):
        remote = False
    elif re.search(r"\bremote(?:ly)?\b|\bwfh\b|\bwork from home\b|\bremot[oa]s?\b|\bудал[её]нн\w*|\bудал[её]нк\w*", t):
        remote = True
    hybrid = re.search(r"\bhybrid\b", t)
    onsite = re.search(r"\b(?:on[- ]?site|in[- ]office|office[- ]based)\b", t)
    if remote is True and (hybrid or onsite):
        remote = None  # "remote or hybrid": either will do
    elif remote is None and (hybrid or onsite):
        remote = False
    location = ""
    if re.search(r"\b(?:canada|canadian|toronto|vancouver|montreal|ottawa|calgary)\b", t):
        location = "Canada"
    elif (re.search(r"\bUS\b|\bU\.S\.(?:A\.)?", raw)
          or re.search(r"\b(?:usa|united states|america|us[- ](?:based|only|remote|citizens?)|remote[- ]us)\b", t)):
        location = "US"
    elif re.search(r"\b(?:worldwide|world[- ]wide|anywhere|global(?:ly)?|international)\b", t):
        location = "worldwide"
    elif re.search(r"\b(?:europe|european|eu|emea)\b", t):
        location = "Europe"
    elif re.search(r"\b(?:uk|united kingdom|london)\b", t):
        location = "UK"
    elif remote is False and hybrid:
        location = "hybrid"
    elif remote is False and onsite:
        location = "onsite"
    return remote, location


def _employment_in(t: str) -> str:
    hits = [(m.start(), family) for family, pattern in _EMPLOYMENT if (m := pattern.search(t))]
    return min(hits)[1] if hits else ""


def _status_in(t: str, intent: str) -> str:
    if intent == "advice":
        return ""
    if re.search(r"\b(?:applied|applications? sent)\b", t):
        return "applied"
    if re.search(r"\bskipped\b", t):
        return "skipped"
    if re.search(r"\bexpired\b", t):
        return "expired"
    if re.search(r"\b(?:new|unreviewed|not reviewed|untriaged)\b", t):
        return "new"
    if re.search(r"\b(?:all|every|any)\s+(?:of\s+the\s+)?(?:jobs|statuses)\b|\bregardless of status\b|\bany status\b", t):
        return "all"
    return ""


def _limit_in(t: str) -> int | None:
    for pattern in (_LIMIT_TOP, _LIMIT_N):
        if m := pattern.search(t):
            if (n := _num(m.group(1))) > 0:
                return min(n, 50)
    if m := _LIMIT_FEW.search(t):
        return {"couple": 2, "few": 3, "handful": 5}[m.group(1)]
    return None


def _ids_in(t: str) -> list[int]:
    ids: list[tuple[int, int]] = []
    for m in _HASH_ID.finditer(t):
        ids.append((m.start(), int(m.group(1))))
    for m in _KEYWORD_IDS.finditer(t):
        ids.extend((m.start(), int(n)) for n in re.findall(r"\d+", m.group(1)))
    return [i for i in _dedupe([n for _, n in sorted(ids)]) if i > 0][:25]


def _company_in(raw: str) -> str:
    """A capitalised name after "at" ("open positions at Acme AI")."""
    if m := _COMPANY.search(raw):
        words = m.group(1).split()
        while words and words[-1].lower() in _NOT_COMPANIES | {"&"}:
            words.pop()
        if words and words[0].lower() not in _NOT_COMPANIES:
            return " ".join(words)
    return ""


def _explicit_search(raw: str, t: str) -> str:
    """A quoted phrase or a "mentioning X" / "containing X" keyword: a text search the user asked for."""
    for m in (_QUOTED.search(raw.replace("“", '"').replace("”", '"')), _SEARCH_KW.search(t)):
        if m and len(candidate := m.group(1).strip(" .,'-")) >= 2:
            return candidate[:200]
    return ""


def _search_in(raw: str, t: str, taken: Sequence[str], company: str = "") -> str:
    """An explicit search (quoted, "mentioning X"), else a company; a company
    that is already a role or tech filter is left to that filter."""
    if explicit := _explicit_search(raw, t):
        return explicit
    candidate = company.strip(" .,'-")
    single_role = " " not in candidate and role_family(candidate) is not None
    if len(candidate) >= 2 and candidate.lower() not in {x.lower() for x in taken} and not single_role:
        return candidate[:200]
    return ""


def rules_params(text: str, history: Sequence[ChatMessage] = (), intent: str = "jobs",
                 facets: Facets | None = None) -> RouteParams:
    """Everything the rules can read off a message: roles (mapped onto the
    database's categories when ``facets`` are given), remote/hybrid/onsite,
    location, tech tags, employment family, a salary floor ("200k+"), a
    recency window ("last 10 days", "this week"), a count ("top 5"), job ids
    ("#123", "job 123"), a search term and a status ("applied jobs")."""
    raw = text.strip()
    t = fix_command_typos(_norm(raw))
    scan = _scan_numbers(t)
    company = _company_in(raw)
    named = t.replace(company.lower(), " ") if company else t  # "Acme AI" is not an AI role
    roles = _roles_in(named, facets)
    tech = _tech_in(named, facets)
    remote, location = _remote_and_location(raw, t)
    employment = _employment_in(t)
    search = _search_in(raw, t, [*tech, *roles], company)
    if search and search.lower() in {x.lower() for x in tech}:
        tech = [x for x in tech if x.lower() != search.lower()]  # "mentioning Kafka" is a text search
    return RouteParams(
        roles=roles,
        remote=remote,
        location=location,
        tech=tech,
        employment=employment,
        min_salary=scan.min_salary,
        days=scan.days,
        limit=_limit_in(scan.text),
        job_ids=_ids_in(scan.text),
        search=search,
        status=_status_in(t, intent),
    )


# --- rules: classification ---------------------------------------------------------------

_OPINION = re.compile(
    r"\b(?:should (?:i|we)|is it (?:worth|wise|a good idea)|worth (?:it|applying|taking|accepting|doing|pursuing"
    r"|considering|going for|a shot|a try|my time)|is (?:a|an|the|this|that)\b[^.?!\n]{0,60}?\bworth"
    r"|do you think|what do you think"
    r"|would you (?:recommend|suggest|advise)|what would you|how (?:do|can|should|would) (?:i|we)|how to"
    r"|any (?:advice|tips|suggestions|ideas|thoughts)|help me (?:decide|choose|think|understand|prepare|figure)"
    r"|which (?:one|job|role|of (?:these|them|those)) (?:should|would|is best|fits)|pros and cons|tell me (?:about|more)"
    r"|explain\w*|why|what(?:'s| is| are) the (?:best|difference|point)|how does|how do (?:\w+\s+){1,3}?work"
    r"|what does|what is an?"
    r"|what are some|recommend\w*|advice|tips?|thoughts on|opinions?|prepare for|interview\w*)\b"
    r"|\b(?:rewrite|reword|rephrase|polish|proofread|edit)\b(?![^.?!\n]{0,40}\b(?:pdfs?|cvs?|resumes?)\b)")
# "tell me about job 7", "details on #7": show that job, not an opinion.
_JOB_DETAIL = re.compile(r"\b(?:tell me (?:more )?about|details? (?:of|for|on|about)|more (?:info|information|details) "
                         r"(?:on|about)|show(?: me)?|open|pull up|look up|what(?:'s| is))\s+(?:the\s+)?"
                         r"(?:job|posting|position|role)?\s*(?:#\s?|(?:number|no\.|id)\s*)?\d{1,9}\b")
_ML_STATS = re.compile(r"\b(?:estadísticas?|estadisticas?|métricas|metricas|números|numeros|progreso|статистик\w*"
                       r"|цифры|цифр|прогресс\w*|метрик\w*)(?!\w)")

_STATS = re.compile(r"\b(?:estadísticas?|estadisticas?|métricas|metricas|números|numeros|progreso|статистик\w*"
                       r"|цифры|цифр|прогресс\w*|метрик\w*)(?!\w)")

_STATS = re.compile(
    r"\b(?:stats|statistics|metrics|kpis?|dashboard|analytics|funnel|conversion rate|throughput)\b"
    r"|\b(?:the|our|my|key|latest) numbers\b|^numbers\b|\bhow many\b|\bhow much\b|\bprogress\b"
    r"|\b(?:summary|summari[sz]e|overview|recap|report|status|health|state)\s+(?:of|on|for)\s+(?:the\s+|my\s+|our\s+)?"
    r"(?:pipelines?|search|job search|hunt|runs?|workflows?|week|month)\b"
    r"|\b(?:pipeline|search|job search|weekly|monthly|daily)\s+(?:summary|overview|recap|report|status|stats)\b"
    r"|\bhow (?:did|are|were|have|has) (?:the |my |our )?(?:runs?|workflows?|pipelines?|things|we|i)\b"
    r"(?:\s+\w+){0,3}?\s+(?:go|gone|going|doing|done|been)\b"
    r"|\bhow(?:'s| is) (?:the |my |our )?(?:pipeline|search|job search|hunt)\b"
    r"|^(?:summary|overview|recap|report)\b"
    r"|\bwhere do (?:things|we|i) stand\b|\bwhere are we at\b|\bstatus update\b|\bhow are things (?:going|looking)\b"
    # What a past run did is a number to look up.
    r"|\b(?:the|my|our)\s+(?:last|latest|previous|prior|recent|most recent)\s+(?:\w+\s+){0,2}?runs?\b"
    r"|\bwhat did (?:the |our |my )?(?:last |latest |previous )?runs?\b")
_JOBS = re.compile(r"\b" + _JOB_NOUN + r"\b|\b(?:unreviewed|backlog|inbox|c2c)\b"
                   r"|\bapplied (?:to|for)\b|\b(?:i|we)(?:'ve| have)? applied\b|\banything new\b|\bwhat's new\b")

# "skip #99", "forget #7": setting a job aside is not asking to see it.
_DISMISS = re.compile(r"\b(?:drop|remove|skip|exclude|ignore|cancel|forget)\b")
_BY_ID = "refers to a specific job id"


# --- rules: command typos -------------------------------------------------------------------

# A misspelt command word ("stast please") is read as the word when it starts a clause and is one edit
# (a missing, extra, swapped or wrong letter) away from one. Nouns are never corrected, and neither are
# the real words one edit away from a command.
_TYPO_VERBS: tuple[str, ...] = ("stats",)
_NOT_TYPOS = frozenset({
    "state", "states", "stars", "starts", "stags", "stabs", "stays", "slats", "spats", "seats", "stets", "stoats",
    "status", "tats", "sats",
})
_CLAUSE_WORD = re.compile(r"(?:^|[.!?;,]\s*|\b(?:then|and)\s+)(?:(?:please|pls|plz|can you|could you|would you"
                          r"|will you|now|go|ok|okay|also|just|so|and|then|hey|hi|hello)\s+)*([^\W\d_]+)")


def _one_edit(word: str, verb: str) -> bool:
    """Whether ``word`` is ``verb`` with one letter missing, extra, wrong, or swapped with its neighbour."""
    if word == verb or abs(len(word) - len(verb)) > 1:
        return False
    if len(word) == len(verb):
        diff = [i for i, (a, b) in enumerate(zip(word, verb, strict=True)) if a != b]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and word[diff[0]] == verb[diff[1]] and word[diff[1]] == verb[diff[0]])
    short, long = (word, verb) if len(word) < len(verb) else (verb, word)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def fix_command_typos(t: str) -> str:
    """The message with a misspelt command verb at the start of a clause corrected."""
    out = t
    for m in reversed(list(_CLAUSE_WORD.finditer(t))):
        word = m.group(1)
        fixed = None
        if len(word) >= 4 and word not in _NOT_TYPOS:
            fixed = next((verb for verb in _TYPO_VERBS if _one_edit(word, verb)), None)
        if fixed:
            out = out[:m.start(1)] + fixed + out[m.end(1):]
    return out


# --- rules: classification ---------------------------------------------------------------------

def classify(text: str, facets: Facets | None = None) -> tuple[Intent, str]:
    """The rules' intent for a message, and why.

    Order: a job-detail phrase → jobs; an opinion or explanation → advice;
    then a job id on its own ("is #7 still open?") → jobs; then stats (a
    misspelt "stats" at the start of a clause counts); then jobs; else
    advice."""
    t = fix_command_typos(_norm(text))
    if _JOB_DETAIL.search(t):
        return "jobs", "asks about a specific job"
    if _OPINION.search(t):
        return "advice", "asks for advice or an explanation"
    if _ids_in(_scan_numbers(t).text) and not _DISMISS.search(t):
        return "jobs", _BY_ID
    if m := _STATS.search(t):
        if m.group(0).startswith("how m") and _JOBS.search(t) and _names_a_filter(text, t, facets):
            return "jobs", "counts jobs matching a filter"
        return "stats", "asks for statistics or progress"
    if _ML_STATS.search(t):
        return "stats", "asks for statistics or progress"
    if _JOBS.search(t):
        return "jobs", "asks to list or filter stored jobs"
    return "advice", "no workflow keywords"


def _names_a_filter(raw: str, t: str, facets: Facets | None) -> bool:
    remote, location = _remote_and_location(raw, t)
    return bool(_roles_in(t, facets) or remote is not None or location or _tech_in(t, facets)
                or _employment_in(t) or _scan_numbers(t).min_salary or _search_in(raw, t, (), _company_in(raw)))


# --- fast path --------------------------------------------------------------------------

_SLASH = re.compile(r"^/(jobs|stats|advice|ask|help)\b\s*(.*)$", re.S)
_SLASH_INTENT: dict[str, Intent] = {"jobs": "jobs", "stats": "stats", "advice": "advice", "ask": "advice",
                                    "help": "advice"}
_GREETING = re.compile(r"^(?:hi|hello|hey|hiya|yo|howdy|good (?:morning|afternoon|evening)|thanks|thank you|thx|ty"
                       r"|cheers|thanks a lot|thank you so much|ok thanks|okay thanks|great,? thanks)(?: r3)?[\s.!]*$")


def fast_path(text: str, history: Sequence[ChatMessage] = (), facets: Facets | None = None) -> RouteDecision | None:
    """The decisions the rules can be sure of without a model; see the module
    docstring for the exact conditions. ``None`` means the configured mode decides."""
    raw = text.strip()
    t = _norm(raw)
    if m := _SLASH.match(t):
        intent = _SLASH_INTENT[m.group(1)]
        rest = re.sub(r"^/\w+", "", raw, count=1).strip()
        params = rules_params(rest, history, intent, facets) if rest else RouteParams()
        return RouteDecision(intent=intent, params=params, router="rules", reason=f"slash command /{m.group(1)}")
    if _GREETING.match(t):
        return RouteDecision(intent="advice", router="rules", reason="greeting or thanks")
    return None


# --- advice topics: what no router may turn into a workflow ------------------------------------------

# The reasons :func:`steer` gives; the advisor's narration keys its guidance on them.
RESUME_HELP = "asks for help making a resume"
PRICING = "asks what R3 costs"

_RESUME_NOUN = r"(?:resume|résumé|cv|curriculum vitae)"
_RESUME_FILLER = (r"(?:me|us|up|out|a|an|my|new|fresh|proper|good|great|strong|professional|simple|basic|first|real"
                  r"|whole|full|complete|the|one|this|that|it|of|into|for me|from scratch)")
# Making one resume for the person ("make me a resume from this", "help me build my resume").
_RESUME_MAKE = re.compile(
    r"\b(?:make|create|generate|build|produce|prepare|write|draft|craft|put together|start|do|redo|update)\b"
    r"(?:\s+" + _RESUME_FILLER + r")*\s+" + _RESUME_NOUN + r"(?![\w-])"
    r"|\bturn\s+(?:this|that|it|these|those|the above|my\s+[\w-]+)\s+into\s+(?:a\s+|an\s+|my\s+)?" + _RESUME_NOUN
    + r"(?![\w-])"
    r"|\b" + _RESUME_NOUN + r"\s+(?:from|out of|based on|using)\s+(?:this|that|it|these|those|my|the above|above|what)\b"
    # Their own resume as a PDF: every saved resume has "Download PDF".
    r"|\bpdf\s+(?:version\s+|copy\s+)?(?:of|from)\s+(?:my|the|this)\s+(?:own\s+)?" + _RESUME_NOUN + r"(?![\w-])"
    r"|\b(?:my|the)\s+" + _RESUME_NOUN + r"\s+(?:as|in|into)\s+(?:a\s+)?pdf\b")
_MAKE_ONE = re.compile(r"\b(?:make|create|build|write|draft)\s+(?:me\s+|us\s+)?one\b[^.?!\n]{0,40}?"
                       r"\b(?:from|out of|based on|using)\b")
_R3_ITSELF = (r"(?:it|this|that|r3|the (?:app|site|service|portal|assistant|tool|platform|product)"
              r"|this (?:app|site|service|portal|assistant|tool|platform|product))")
_PRICE = re.compile(
    r"\bhow much\b[^.?!\n]{0,30}?\b(?:cost|costs|charge|charges|fee|fees|price)\b"
    r"|\bwhat(?:'s| is| are| does| do)\b[^.?!\n]{0,25}?\b(?:price|prices|pricing|cost|costs|fee|fees)\b"
    r"(?![^.?!\n]{0,20}\b(?:of living|jobs?|roles?)\b)"
    r"|\b(?:pricing|subscriptions?|paid plans?|free (?:trial|tier|plan|version))\b"
    r"|\bis\s+" + _R3_ITSELF + r"\s+(?:really\s+|actually\s+|completely\s+)?(?:free|paid)\b"
    r"|\b(?:do|will|would|should|must) (?:i|we) (?:have to |need to )?pay\b"
    r"|\b" + _R3_ITSELF + r"\s+costs?\b")


def _wants_resume_help(t: str) -> bool:
    return bool(_RESUME_MAKE.search(t) or (_MAKE_ONE.search(t) and re.search(r"\b" + _RESUME_NOUN + r"\b", t)))


# A resume request carries the person's background when it, or the message it points back to, is this long.
BACKGROUND_WORDS = 25
_POINTS_BACK = re.compile(r"\b(?:this|that|these|those|it|above|what i (?:wrote|said|sent|pasted|typed))\b")


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def resume_background(text: str, history: Sequence[ChatMessage] = ()) -> str:
    """The background a resume request ("make me a resume from this") hands
    over: the message itself when it is long enough to describe someone, or,
    when it points back ("this", "the above"), the person's previous message.
    "" when there is none, and the Assistant explains how to make one."""
    if _word_count(text) >= BACKGROUND_WORDS:
        return text
    if not _POINTS_BACK.search(_norm(text)):
        return ""
    previous = next((m.content for m in reversed(history) if m.role == "user"), "")
    return previous if _word_count(previous) >= BACKGROUND_WORDS else ""


def advice_topic(text: str, history: Sequence[ChatMessage] = ()) -> str | None:
    """Why a message must be answered in conversation whatever a router
    reads into it: a request to make a resume (the builder, the upload button
    or the Resume page do that), or a question about what R3 costs. None
    otherwise."""
    t = fix_command_typos(_norm(text))
    if _wants_resume_help(t):
        return RESUME_HELP
    if _PRICE.search(t):
        return PRICING
    return None


def steer(decision: RouteDecision, text: str, history: Sequence[ChatMessage] = ()) -> RouteDecision:
    """The last word on an :func:`advice_topic` message, whichever router
    decided: advice, with the topic as the reason."""
    topic = advice_topic(text, history)
    if topic is None:
        return decision
    return RouteDecision(intent="advice", router=decision.router, reason=topic)


def rules_route(text: str, history: Sequence[ChatMessage] = (), facets: Facets | None = None) -> RouteDecision:
    """The rules' decision: the fast path, else :func:`classify` plus
    :func:`rules_params`, then :func:`steer`. Also the last-resort fallback of
    every other mode."""
    if (decision := fast_path(text, history, facets)) is not None:
        return decision
    intent, why = classify(text, facets)
    decision = RouteDecision(intent=intent, params=rules_params(text, history, intent, facets), router="rules",
                             reason=why)
    return steer(decision, text, history)


def _grounded_ids(ids: Sequence[int], text: str, history: Sequence[ChatMessage]) -> list[int]:
    """Only ids that appear in the message or the recent conversation: a model
    must not invent jobs to show."""
    seen = {int(n) for n in re.findall(r"\d{1,9}", " ".join([text, *(m.content for m in history[-HISTORY_WINDOW:])]))}
    return [i for i in _dedupe(ids) if isinstance(i, int) and i > 0 and i in seen][:25]


# --- the LLM route --------------------------------------------------------------------------

def _described_literal(options: Sequence[str], descriptions: Mapping[str, str] | None = None) -> Any:
    """``Literal[options]`` whose JSON schema describes each option, which is
    how both the LLM and Jev read what an option means."""
    descriptions = descriptions or {}
    schema = {"anyOf": [{"const": o, "type": "string", **({"description": descriptions[o]} if descriptions.get(o) else {})}
                        for o in options]}
    return Annotated[Literal[tuple(options)], WithJsonSchema(schema)]  # type: ignore[valid-type]


def _described_optional(options: Sequence[str], descriptions: Mapping[str, str]) -> Any:
    """``Literal[options] | None``, described per option. The schema keeps the
    options nested under one ``anyOf`` branch beside ``null``: Jev reads an
    optional field as exactly "X or null" (pick one, or none of these), and
    Pydantic would otherwise flatten the options and ``null`` into one list."""
    inner = {"anyOf": [{"const": o, "type": "string", **({"description": descriptions[o]} if descriptions.get(o) else {})}
                       for o in options]}
    return Annotated[Optional[Literal[tuple(options)]],  # type: ignore[valid-type]
                     WithJsonSchema({"anyOf": [inner, {"type": "null"}]})]


IntentChoice = _described_literal(INTENTS, INTENT_DESCRIPTIONS)


class RouteCall(BaseModel):
    """The workflow the operator's latest message asks for, with the job filters it names."""

    intent: IntentChoice = Field(description=INTENT_QUESTION)  # type: ignore[valid-type]
    roles: list[str] = Field(default_factory=list, description="Role categories the message names.")
    remote: bool | None = Field(default=None, description="true for remote, false for on-site/hybrid, null if unsaid.")
    location: str = Field(default="", description="Country or region named (US, Canada, worldwide, ...).")
    tech: list[str] = Field(default_factory=list, description="Technology tags named (kubernetes, python, ...).")
    employment: str = Field(default="", description="fulltime | contract | parttime | internship, if named.")
    min_salary: int | None = Field(default=None, description="Annual salary floor in USD, if named.")
    days: int | None = Field(default=None, description="Recency window in days, if named ('this week' = 7).")
    limit: int | None = Field(default=None, description="How many jobs, if named ('top 5' = 5).")
    job_ids: list[int] = Field(default_factory=list, description="Job ids named or referred to; never invented.")
    search: str = Field(default="", description="A keyword or company to search for, if named.")
    status: str = Field(default="", description="new | applied | skipped | expired | all, if named.")
    reason: str = Field(default="", description="One short sentence on why this intent.")


_EXAMPLES = """Examples:
"show me remote devops jobs paying 200k+" -> jobs, roles [DevOps], remote true, min_salary 200000
"list the jobs I applied to this week" -> jobs, status applied, days 7
"how many new data engineering jobs are there?" -> jobs, roles [Data], status new
"tell me more about job 7" -> jobs, job_ids [7]
"how did the runs go this week?" -> stats, days 7
"where do things stand?" -> stats
"покажи статистику" -> stats
"should I apply to #12?" -> advice
"rewrite my summary line for SRE roles" -> advice
"why did the last stats run fail?" -> advice"""


def llm_instructions(facets: Facets | None, now: datetime | None = None, hint: RouteDecision | None = None) -> str:
    """The router's instructions: intents, conventions, the database's vocabulary and examples,
    and the keyword pre-classifier's verdict as a hint when there is one."""
    now = now or datetime.now(timezone.utc)
    roles = [v.value for v in facets.role_categories][:15] if facets and facets.role_categories else list(CANONICAL_ROLES)
    tech = [v.value for v in facets.tech_tags][:30] if facets and facets.tech_tags else []
    statuses = [v.value for v in facets.statuses] if facets and facets.statuses else list(STATUS_WORDS)
    intents = "\n".join(f"- {name}: {desc}" for name, desc in INTENT_DESCRIPTIONS.items())
    return f"""You route one chat message in R3, the Controller of a job search, to a workflow.
Today is {now:%A %Y-%m-%d}.

Intents:
{intents}

Conventions:
- Listing, filtering or counting stored jobs is jobs. A request to see one specific job ("tell me about job 7") is
  jobs with its id; put the ids the user names (#123, "job 123") in job_ids and never invent ids.
- A question asking for an opinion or an explanation, or help writing text, is advice. So is asking for help making
  a resume ("make me a resume from this", "help me build my resume"), and a question about what R3 costs.
- The operator may write in Spanish or Russian, or misspell a word: route by meaning.
- Leave every field empty that the message does not state.
- Role categories in the database: {", ".join(roles)} (AI-ish words like "machine learning", "ml", "llm" mean the AI/ML
  categories; "site reliability" means SRE; "platform" or "infrastructure" means DevOps).
- Statuses: {", ".join(statuses)}; "unreviewed" means new, "all" means every status.
{f"- Common tech tags: {', '.join(tech)}." if tech else ""}
{_EXAMPLES}{_hint_line(hint)}"""


def _hint_line(hint: RouteDecision | None) -> str:
    if hint is None:
        return ""
    return (f"\n\nA keyword pre-classifier read this message as: {hint.intent} because {hint.reason or 'its keywords'}. "
            "Override it only if the message clearly asks for something else.")


def llm_prompt(text: str, history: Sequence[ChatMessage]) -> str:
    recent = list(history)[-HISTORY_WINDOW:]
    if not recent:
        return f"Latest user message:\n{text}"
    lines = [f"{m.role}: {_clip(m.content, HISTORY_CHARS)}" for m in recent]
    return "Conversation so far (oldest first):\n" + "\n".join(lines) + f"\n\nLatest user message:\n{text}"


def employment_family(value: str) -> str:
    """An employment spelling ("Full-time", "c2c") as its family key in
    ``r3.db.EMPLOYMENT_FAMILIES`` (fulltime | contract | parttime | internship), or ""."""
    low = value.strip().lower().replace("-", "").replace(" ", "")
    if low in EMPLOYMENT_FAMILIES:
        return low
    for family, spellings in EMPLOYMENT_FAMILIES.items():
        if low in {s.lower().replace("-", "").replace(" ", "") for s in spellings}:
            return family
    return _employment_in(_norm(value)) or ""


def _clamp(value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    return min(max(int(value), low), high)


def _params_from_call(call: RouteCall, text: str, history: Sequence[ChatMessage], facets: Facets | None) -> RouteParams:
    known, unknown = normalize_roles(call.roles, facets)
    salary = call.min_salary if call.min_salary and call.min_salary >= 1000 else None
    return RouteParams(
        roles=(known + unknown)[:10],
        remote=call.remote,
        location=call.location.strip()[:120],
        tech=[normalize_tech(t, facets) for t in call.tech if t.strip()][:10],
        employment=employment_family(call.employment)[:40],
        min_salary=_clamp(salary, 0, 10_000_000),
        days=_clamp(call.days, 1, 365) if call.days else None,
        limit=_clamp(call.limit, 1, 50) if call.limit else None,
        job_ids=_grounded_ids(call.job_ids, text, history),
        search=call.search.strip()[:200],
        status=call.status,
    )


_LLM_AGENT: Agent[None, RouteCall] = Agent(output_type=RouteCall, defer_model_check=True, name="r3-router-llm", retries=1)


# --- the Jev route ---------------------------------------------------------------------------

JEV_ROUTE_DOC = ("Route the operator's latest message in R3, the controller of their job search: the workflow it asks "
                 "for and the job filters it names. Jobs already in the database are listed with 'jobs'.")
_CANONICAL_JEV_ROLES: tuple[str, ...] = ("SRE", "DevOps", "Data", "AI/ML")
_ROLE_DESCRIPTIONS: dict[str, str] = {"AI": "AI engineering, LLM and generative-AI roles",
                                      "ML": "Machine learning and MLOps roles",
                                      "AI/ML": "AI, machine learning, LLM or MLOps roles"}


def jev_role_options(facets: Facets | None) -> tuple[str, ...]:
    """The role families Jev picks from: the database's categories (the 12
    commonest, without "Other") when at hand, else the canonical buckets."""
    if facets:
        values = _dedupe(v.value.strip() for v in facets.role_categories
                         if v.value and v.value.strip() and v.value.strip().lower() != "other")[:12]
        if len(values) >= 2:
            return tuple(values)
    return _CANONICAL_JEV_ROLES


def _role_description(value: str) -> str:
    if value in _ROLE_DESCRIPTIONS:
        return _ROLE_DESCRIPTIONS[value]
    family = role_family(value)
    return _FAMILY_LABEL.get(family or "", "")


@lru_cache(maxsize=16)
def jev_route_type(role_options: tuple[str, ...] = _CANONICAL_JEV_ROLES) -> type[BaseModel]:
    """The output type Jev fills in one request: the intent and the
    closed-set fields, each field description being the question Jev answers.
    Only field types TypeSafe's Jev supports: a Literal pick-one, a list of a
    Literal (one yes/no per option) and ``Literal | None`` (pick one or
    none)."""
    role = _described_literal(role_options, {o: _role_description(o) for o in role_options})
    remote = _described_optional(("remote", "onsite_or_hybrid"),
                                 {"remote": "Remote work", "onsite_or_hybrid": "On-site or hybrid work"})
    status = _described_optional(("new", "applied", "skipped", "expired"), {
        "new": "New, unreviewed jobs", "applied": "Jobs already applied to",
        "skipped": "Jobs the operator skipped", "expired": "Expired postings"})
    employment = _described_optional(("fulltime", "contract"),
                                     {"fulltime": "Full-time, permanent employment", "contract": "Contract or C2C work"})
    return create_model(  # type: ignore[call-overload]
        "JevRoute",
        __doc__=JEV_ROUTE_DOC,
        intent=(IntentChoice, Field(description=INTENT_QUESTION)),
        roles=(list[role], Field(default_factory=list,  # type: ignore[valid-type]
                                 description="Which of these role families does the message ask about, if any?")),
        remote=(remote, Field(default=None, description=(
            "Does the message ask for remote work, for on-site/hybrid work, or say nothing about it?"))),
        status=(status, Field(default=None, description=(
            "Which job status does the message refer to, if any (unreviewed = new)?"))),
        employment=(employment, Field(default=None, description=(
            "Which employment type does the message ask for, if any?"))),
    )


@lru_cache(maxsize=16)
def _jev_agent(route_type: type[BaseModel]) -> Agent[None, Any]:
    return Agent(output_type=route_type, defer_model_check=True, name="r3-router-jev", retries=0)


def jev_field_confidence(provider_details: Mapping[str, Any] | None, field: str) -> float | None:
    """A field's confidence from a TypeSafe response: ``provider_details`` is
    ``{"confidence": {field: 0..1}, "probabilities": {field: {option: p}},
    "scores": {...}}`` (pick-one fields have both; a bounded float has
    neither, its value being the probability)."""
    confidence = (provider_details or {}).get("confidence")
    value = confidence.get(field) if isinstance(confidence, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)
    return None


def _decision_from_jev(pick: BaseModel, confidence: float | None, text: str, history: Sequence[ChatMessage],
                       facets: Facets | None) -> RouteDecision:
    intent: Intent = pick.intent  # type: ignore[attr-defined]
    base = rules_params(text, history, intent, facets) if intent != "advice" else RouteParams()
    remote = {"remote": True, "onsite_or_hybrid": False}.get(pick.remote or "")  # type: ignore[attr-defined]
    status = pick.status or ""  # type: ignore[attr-defined]
    params = base.model_copy(update={
        "roles": list(pick.roles)[:10],  # type: ignore[attr-defined]
        "remote": remote,
        "status": RouteParams(status=status).status if intent != "advice" else "",
        "employment": pick.employment or "",  # type: ignore[attr-defined]
    })
    shown = f"{confidence:.2f}" if confidence is not None else "n/a"
    return RouteDecision(intent=intent, params=params, confidence=confidence, router="jev",
                         reason=f"Jev picked {intent} (confidence {shown})")


# --- the router ------------------------------------------------------------------------------

class IntentRouter:
    """Routes one message with the mode in ``settings.router_mode``.

    ``models`` injects ready-made models instead of building them from the
    settings: ``"llm"`` for the LLM route and ``"jev"`` for Jev (tests pass a
    ``FunctionModel``/``TestModel``). Without ``"llm"``, a model registered
    under the router's model id (or ``"chat"``, when the router follows the
    chat model) is used before one is built."""

    def __init__(self, settings: ControllerSettings, *, models: Mapping[str, Model] | None = None) -> None:
        self.settings = settings
        self.models: dict[str, Model] = dict(models or {})

    async def route(self, text: str, history: Sequence[ChatMessage], *, model_id: str,
                    facets: Facets | None = None) -> RouteDecision:
        history = list(history)
        if (decision := fast_path(text, history, facets)) is not None:
            return decision
        decision = await self._by_mode(text, history, model_id=model_id, facets=facets)
        return steer(decision, text, history)

    async def _by_mode(self, text: str, history: list[ChatMessage], *, model_id: str,
                       facets: Facets | None) -> RouteDecision:
        mode = self.settings.router_mode
        rules = rules_route(text, history, facets)  # the hint for the LLM and every mode's fallback
        if mode == "rules":
            return rules
        if mode == "llm":
            try:
                return await self._llm(text, history, model_id=model_id, facets=facets, hint=rules)
            except Exception as exc:
                return self._fallback(text, history, facets, f"LLM router failed ({_failure(exc)})")
        try:
            jev = await self._jev(text, history, facets)
        except Exception as exc:
            if mode == "jev":
                return self._fallback(text, history, facets, f"Jev router failed ({_failure(exc)})")
            lean = f"Jev failed ({_failure(exc)})"
        else:
            if mode == "jev" or (jev.confidence is not None and jev.confidence >= self.settings.jev_confidence):
                return jev
            shown = f"{jev.confidence:.2f}" if jev.confidence is not None else "n/a"
            lean = f"Jev leaned {jev.intent} at {shown}, below {self.settings.jev_confidence:.2f}"
        try:
            decision = await self._llm(text, history, model_id=model_id, facets=facets, hint=rules)
        except Exception as exc:
            return self._fallback(text, history, facets, f"{lean}; LLM router failed ({_failure(exc)})")
        return decision.model_copy(update={"reason": _clip(f"{decision.reason} · {lean}")})

    # --- paths -----------------------------------------------------------------------------

    async def _llm(self, text: str, history: list[ChatMessage], *, model_id: str,
                   facets: Facets | None, hint: RouteDecision | None = None) -> RouteDecision:
        model = self._llm_model(model_id)
        async with asyncio.timeout(LLM_TIMEOUT_S):
            result = await _LLM_AGENT.run(
                llm_prompt(text, history), model=model, instructions=llm_instructions(facets, hint=hint),
                model_settings=ModelSettings(temperature=0.0), usage_limits=UsageLimits(request_limit=3),
            )
        call = result.output
        return RouteDecision(intent=call.intent, params=_params_from_call(call, text, history, facets), router="llm",
                             reason=_clip(call.reason or f"LLM picked {call.intent}"))

    async def _jev(self, text: str, history: list[ChatMessage], facets: Facets | None) -> RouteDecision:
        model = self._jev_model()
        agent = _jev_agent(jev_route_type(jev_role_options(facets)))
        recent = history_messages(history[-HISTORY_WINDOW:], max_chars=HISTORY_CHARS)
        async with asyncio.timeout(JEV_TIMEOUT_S):
            result = await agent.run(text, model=model, message_history=recent, usage_limits=UsageLimits(request_limit=2))
        confidence = jev_field_confidence(result.response.provider_details, "intent")
        return _decision_from_jev(result.output, confidence, text, history, facets)

    def _fallback(self, text: str, history: list[ChatMessage], facets: Facets | None, why: str) -> RouteDecision:
        log.warning("Router fell back to rules: %s", why)
        decision = rules_route(text, history, facets)
        return decision.model_copy(update={"router": "fallback", "confidence": None,
                                           "reason": _clip(f"{why}; rules: {decision.reason}")})

    # --- models -----------------------------------------------------------------------------

    def _llm_model(self, model_id: str) -> Model:
        if "llm" in self.models:
            return self.models["llm"]
        router_model = self.settings.router_model
        if router_model:
            return self.models.get(router_model) or build_model(router_model, self.settings)
        return self.models.get(model_id) or self.models.get("chat") or build_model(model_id, self.settings)

    def _jev_model(self) -> Model:
        if "jev" in self.models:
            return self.models["jev"]
        model_id = self.settings.typesafe_model
        if ":" not in model_id:
            model_id = f"typesafe:{model_id}"
        return self.models.get(model_id) or build_model(model_id, self.settings)


def _failure(exc: BaseException) -> str:
    """The failure's class (and HTTP status), never the provider's body."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} {status}" if isinstance(status, int) else type(exc).__name__


async def route_message(text: str, history: Sequence[ChatMessage], *, settings: ControllerSettings, model_id: str,
                        facets: Facets | None = None, models: Mapping[str, Model] | None = None) -> RouteDecision:
    """Route one message without building a service (evals, scripts)."""
    return await IntentRouter(settings, models=models).route(text, history, model_id=model_id, facets=facets)
