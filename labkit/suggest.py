"""Which labs practise the skills a resume claims or a job posting asks for.
Matching is on whole words, case-insensitive, and ignores the "aws"/"amazon"
prefix, so "Lambda", "AWS Lambda" and "aws lambda" are the same skill."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import Lab

# Other names a posting uses for a lab's skill.
ALIASES = {
    "redis": ["elasticache", "valkey"],
    "amazon elasticache": ["elasticache"],
    "aws step functions": ["step functions", "stepfunctions", "state machine"],
    "amazon sqs": ["sqs", "simple queue service"],
    "aws glue": ["glue"],
    "event-driven architecture": ["event driven", "event-driven"],
    "elt": ["etl"],
}


@dataclass
class Suggestion:
    lab: Lab
    matched: list[str]


def _bare(skill: str) -> str:
    return re.sub(r"^(aws|amazon)\s+", "", skill.lower()).strip()


def _found(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def suggest(labs: list[Lab], text: str) -> list[Suggestion]:
    text = text.lower()
    out = []
    for lab in labs:
        matched = [skill for skill in lab.skills
                   if any(_found(_bare(name), text) for name in [skill, *ALIASES.get(skill, [])])]
        if matched:
            out.append(Suggestion(lab, matched))
    return sorted(out, key=lambda s: (-len(s.matched), s.lab.id))
