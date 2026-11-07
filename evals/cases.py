"""The routing dataset: representative things an operator types, and what the
Controller must make of each. Every case fixes the intent; ``params`` lists
the parameters that must be extracted (a subset match, with role names
normalised to the role buckets); ``action`` is the panel action the operator
must see against the test database (which has one new AI/ML job, #7).

Add cases here when the router gets something wrong in the field; keep the
expected values honest about what the message actually says. Tag a case
"hard" only when it needs inference beyond its words (a vague ask, a
condition, the jobs listed earlier), never because a router fails it: hard
cases are reported separately and do not count towards the thresholds in
``evals/harness.py``."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class RoutingCase:
    name: str
    text: str
    intent: str
    params: dict = field(default_factory=dict)  # subset of RouteParams that must match
    action: str | None = None  # expected Action.kind against StubDB + FakeRuns
    history: tuple[Turn, ...] = ()
    tags: tuple[str, ...] = ()


# Role names as the evals compare them. The router may return either the
# operator's word or the database's category; ``normalise_role`` maps both.
ROLE_ALIASES = {
    "sre": "SRE", "site reliability": "SRE", "site reliability engineer": "SRE", "reliability": "SRE",
    "devops": "DevOps", "platform": "DevOps", "platform engineer": "DevOps", "infrastructure": "DevOps",
    "data": "Data", "data engineer": "Data", "data engineering": "Data",
    "ai": "AI/ML", "ml": "AI/ML", "ai/ml": "AI/ML", "machine learning": "AI/ML", "ai engineer": "AI/ML",
    "ml engineer": "AI/ML", "llm": "AI/ML", "aiml": "AI/ML",
}


def normalise_role(value: str) -> str:
    key = value.strip().lower().replace("-", " ").replace("_", " ")
    return ROLE_ALIASES.get(key, value.strip())


LISTED_JOBS = (
    Turn("user", "Show me new AI jobs"),
    Turn("assistant", "Here are the newest AI/ML jobs:\n#7 Senior ML Engineer at Acme AI (Remote, US)\n#9 Staff AI Engineer at Beta (Remote)\n#12 ML Platform Lead at Gamma (Hybrid)"),
)
LINKED_JOBS = (
    Turn("user", "Show me new AI jobs"),
    Turn("assistant", "Here are the newest AI/ML jobs:\n- [Senior ML Engineer at Acme AI](/jobs/7) (Remote, US)\n- [Staff AI Engineer at Beta](/jobs/9) (Remote)\n- [ML Platform Lead at Gamma](/jobs/12) (Hybrid)"),
)
PASTED_BACKGROUND = (
    Turn("user", "My background: 6 years as an SRE at Bluefjord, Kubernetes and Terraform, led on-call for 40 services."),
    Turn("assistant", "That is a strong SRE background: reliability ownership at scale and infrastructure as code."),
)

CASES: list[RoutingCase] = [
    # --- advice: conversation, no workflow --------------------------------------------
    RoutingCase("advice-positioning", "How should I position myself for platform engineering roles?", "advice", action="advice"),
    RoutingCase("advice-contract", "Is a 6-month C2C contract worth taking over a full-time offer?", "advice", action="advice"),
    RoutingCase("advice-pipeline-opinion", "What do you think of my pipeline so far? Be honest.", "advice", action="advice", tags=("ambiguous",)),
    RoutingCase("advice-rewrite", "Rewrite my summary line so it reads better for SRE roles.", "advice", action="advice"),
    RoutingCase("advice-interview", "How do I prepare for a Kubernetes-heavy SRE interview next week?", "advice", action="advice"),
    RoutingCase("advice-greeting", "hey", "advice", action="advice"),
    RoutingCase("advice-thanks", "thanks, that helps", "advice", action="advice"),
    RoutingCase("advice-should-apply", "should I apply to #7?", "advice", action="advice"),

    # --- jobs: show what is already stored ------------------------------------------------
    RoutingCase("jobs-new-ai-week", "Show me new AI/ML jobs from this week", "jobs", {"roles": ["AI/ML"], "days": 7}, action="jobs"),
    RoutingCase("jobs-available", "what jobs are available for me right now?", "jobs", action="jobs"),
    RoutingCase("jobs-remote-contract-salary", "list remote contract roles paying 200k+", "jobs", {"remote": True, "min_salary": 200000}, action="jobs"),
    RoutingCase("jobs-unreviewed", "what's in the unreviewed pool?", "jobs", action="jobs"),
    RoutingCase("jobs-applied", "which jobs have I applied to?", "jobs", {"status": "applied"}, action="jobs"),
    RoutingCase("jobs-top", "show the top 5 newest devops jobs", "jobs", {"roles": ["DevOps"], "limit": 5}, action="jobs"),
    RoutingCase("jobs-slash", "/jobs sre remote", "jobs", {"roles": ["SRE"], "remote": True}, action="jobs", tags=("fast-path",)),
    RoutingCase("jobs-search", "any postings mentioning Kafka?", "jobs", {"search": "kafka"}, action="jobs", tags=("hard",)),
    RoutingCase("jobs-detail", "tell me more about job 7", "jobs", {"job_ids": [7]}, action="jobs", tags=("detail",)),
    RoutingCase("jobs-new-this-week", "show me the new remote SRE jobs that came in this week", "jobs", {"roles": ["SRE"], "remote": True, "days": 7}, action="jobs"),

    # --- stats: numbers, progress, how the runs went ----------------------------------------------
    RoutingCase("stats-standing", "Where do things stand?", "stats", action="stats"),
    RoutingCase("stats-numbers", "give me the numbers", "stats", action="stats"),
    RoutingCase("stats-word", "statistics", "stats", action="stats"),
    RoutingCase("stats-runs-week", "how did the runs go this week?", "stats", {"days": 7}, action="stats"),
    RoutingCase("stats-dashboard", "show me a dashboard of the pipeline", "stats", action="stats"),
    RoutingCase("stats-count", "how many jobs came in over the last 7 days?", "stats", {"days": 7}, action="stats"),
    RoutingCase("stats-slash", "/stats", "stats", action="stats", tags=("fast-path",)),
    RoutingCase("stats-progress", "am I making progress? what's the applied count?", "stats", action="stats"),
    RoutingCase("stats-last-run", "what did the last run count?", "stats", action="stats"),

    # --- follow-ups that depend on history ------------------------------------------------------
    RoutingCase("followup-yes-nothing-asked", "yes", "advice", history=LISTED_JOBS, action="advice", tags=("history",)),

    # --- typos, one-word commands, a word or two of another language --------------------------------
    RoutingCase("typo-jobs", "shwo me remote devops jobs", "jobs", {"roles": ["DevOps"], "remote": True}, action="jobs", tags=("typo",)),
    RoutingCase("typo-stats", "stast please", "stats", action="stats", tags=("typo",)),
    RoutingCase("short-numbers", "numbers?", "stats", action="stats", tags=("short",)),
    RoutingCase("non-english-jobs", "zeig mir neue SRE jobs", "jobs", {"roles": ["SRE"]}, action="jobs", tags=("non-english",)),
    RoutingCase("non-english-stats-es", "muéstrame las estadísticas de esta semana", "stats", {"days": 7}, action="stats", tags=("non-english",)),
    RoutingCase("non-english-stats-ru", "покажи статистику за неделю", "stats", {"days": 7}, action="stats", tags=("non-english",)),
    RoutingCase("non-english-jobs-ru", "покажи новые вакансии sre", "jobs", {"roles": ["SRE"]}, action="jobs", tags=("non-english",)),

    # --- one job, one page, one number -----------------------------------------------------------
    RoutingCase("detail-salary", "what's the salary on job 7?", "jobs", {"job_ids": [7]}, action="jobs", tags=("detail",)),
    RoutingCase("detail-still-open", "is #7 still open?", "jobs", {"job_ids": [7]}, action="jobs", tags=("detail",)),
    RoutingCase("jobs-c2c-contracts", "any c2c devops contracts?", "jobs", {"roles": ["DevOps"], "employment": "contract"}, action="jobs"),
    RoutingCase("jobs-open-page", "open the jobs page", "jobs", action="jobs"),

    # --- QA, October 2026: a resume to make is advice; prices are not announced --------
    RoutingCase("resume-from-paste", "make me a resume from this", "advice", action="advice", history=PASTED_BACKGROUND, tags=("history", "qa")),
    RoutingCase("resume-starter", "Help me build my resume", "advice", action="advice", tags=("qa",)),
    RoutingCase("pricing", "how much does this cost?", "advice", action="advice", tags=("qa",)),
    RoutingCase("application-status", "did my application go through?", "advice", action="advice", tags=("qa",)),
    RoutingCase("jobs-from-linked-list", "show me those jobs again", "jobs", action="jobs", history=LINKED_JOBS, tags=("history",)),
]

BY_INTENT: dict[str, list[RoutingCase]] = {}
for _case in CASES:
    BY_INTENT.setdefault(_case.intent, []).append(_case)
