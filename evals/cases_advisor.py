"""The advisor dataset: one chat turn per case, with the route scripted so the
advisor is judged on its own (the routing dataset judges the router). The
scripted route goes through ``plan_action`` against the test database (one
new AI/ML job, #7 Senior ML Engineer at Acme AI) and an in-memory run service,
so every case leads to a known panel action; the checks then ask whether the
turn streamed in order and the answer says what that action requires: the
jobs listed, the numbers of a snapshot.

Offline the answer comes from a scripted narrator (a FunctionModel that
writes from the narration the advisor's instructions carry); live it comes
from the chat model and is graded by an LLM judge (and Jev, when
``TYPESAFE_API_KEY`` is set) against :data:`ADVISOR_RUBRIC`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.cases import Turn


@dataclass(frozen=True)
class AdvisorCase:
    name: str
    text: str
    route: dict[str, Any]  # the scripted RouteDecision: intent, params
    action: str  # the panel action kind the route leads to
    history: tuple[Turn, ...] = ()
    runs: str = "ok"  # "down": the run service refuses every run (the snapshot still answers)
    tags: tuple[str, ...] = field(default_factory=tuple)


ADVISOR_CASES: list[AdvisorCase] = [
    AdvisorCase("jobs-list", "Show me new AI/ML jobs from this week",
                {"intent": "jobs", "params": {"roles": ["AI/ML"], "days": 7}}, "jobs"),
    AdvisorCase("jobs-detail", "tell me more about job 7", {"intent": "jobs", "params": {"job_ids": [7]}}, "jobs",
                tags=("tools",)),
    AdvisorCase("jobs-missing", "tell me about job 99", {"intent": "jobs", "params": {"job_ids": [99]}}, "jobs"),
    AdvisorCase("stats-numbers", "give me the numbers", {"intent": "stats"}, "stats"),
    AdvisorCase("stats-runs-down", "show me the stats", {"intent": "stats"}, "stats", runs="down"),
    AdvisorCase("advice", "How should I position myself for platform engineering roles?", {"intent": "advice"},
                "advice", tags=("tools",)),
]

ADVISOR_RUBRIC = """The output is one turn of R3, the controller of an operator's job search. `panel` says what the
panel on the right shows for this turn; only the panel starts runs and lists jobs.
`answer` is the assistant's reply on the left. The answer passes only if all of these hold:
1. It does not claim that the assistant itself started a run, applied or changed anything (saying that the
   panel shows something is fine).
2. When the panel lists jobs, the answer names those jobs (as Markdown links to /jobs/<id>, with no #id in the
   text).
3. It is plain text: no Markdown tables, headings or bold.
4. It has at most 8 sentences."""
