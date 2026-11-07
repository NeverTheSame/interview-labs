"""A deterministic check on the advisor's final reply. The advisor never
starts, retries or fixes anything (the panel starts runs), yet a model will
say "I'm running the workflow for you now" after a failure, or promise "no
more 403s" (QA L16). Unless the panel started a run in this
turn, :func:`honest_reply` replaces the first sentence that claims a run
started, is running or is done (or that an error is fixed) with
:data:`HONEST_LINE` and drops any further such sentence.

The patterns are deliberately narrow: first-person action claims ("I'm
running", "I've started", "let me run that"), a run described as started or
in progress ("your stats are being generated"), a gerund promising work now
("Starting the workflow now."), and fix promises. Negations ("I
can't start runs", "no run was started") never match."""

from __future__ import annotations

import logging
import re

log = logging.getLogger("r3.agent.honesty")

HONEST_LINE = ("Nothing has started for this yet: I can't start, retry or fix a run myself; runs start from the "
               "panel next to this chat.")

# Verbs that claim an action on their own ("I've applied", "I'm submitting").
_STRONG = (r"applied(?!\s+(?:the\s+)?filters?)|applying|submitted|submitting"
           r"|re-?ran|re-?running|rerun|retried|retrying|restarted|restarting|kicked off|kicking off|launched|launching"
           r"|triggered|triggering|queued|queuing")
# Verbs that claim an action only with a run as their object ("I'm running the workflow", "let me run that").
_WEAK = (r"running(?!\s+(?:through|over|by)\b)|run|started|starting|start(?!\s+(?:by|with|from|off)\b)|generating"
         r"|generated|generate|rendering|rendered|render|creating|created|create|sending|sent|send|marking|marked|mark"
         r"|fixing|fixed|fix|processing|processed|preparing|prepared|prepare|working on|handling|handled|handle")
# A run as the object of a weak verb: within the next few words ("running the stats workflow").
_RUN_NOUN = re.compile(r"^(?:\s+\S+){0,3}?\s+(?:runs?|workflows?|pipelines?|applications?)\b",
                       re.I)
_PRONOUN = re.compile(r"^\s+(?:it|that|this|them|those|these)\b", re.I)
_ADVERBS = (r"(?:\s+(?:now|just|already|also|currently|actually|quickly|right now|gone ahead and|went ahead and"
            r"|go ahead and|going to|gonna|about to))*")
_ACTOR = re.compile(
    r"(?:\b(?:i|we)(?:'m|'ve|'ll|'re|\s+am|\s+have|\s+will|\s+are)?|\blet me)" + _ADVERBS
    + r"\s+(?:(?P<strong>" + _STRONG + r")|(?P<weak>" + _WEAK + r"))\b(?P<rest>[^.!?\n]*)", re.I)
_PROGRESS = re.compile(
    r"\b(?:runs?|workflows?|pipelines?|applications?|requests?|it|everything)"
    r"(?:\s+(?:is|are|has|have|was|were)|'s|'re)\s+(?:(?:now|already|just|currently)\s+)*(?:been\s+)?"
    r"(?:running|started|underway|under way|in progress|kicked off|launched|queued|on (?:its|their|the) way"
    r"|being (?:generated|created|prepared|processed|submitted|sent|made))\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|nothing|none|never)\s+$", re.I)
_NOW_GERUND = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|sure|great|got it|done|alright|on it|perfect)\W+)?(?:now\s+)?(?:running|starting|kicking off"
    r"|launching|submitting|generating|applying|retrying|restarting|re-?running)\b"
    r"[^.!?\n]*?\b(?:now|for you|right away)\b", re.I)
_FIXED = re.compile(
    r"\bno more (?:403s?|4\d\ds?|5\d\ds?|errors?|failures?|problems?|issues?)\b"
    r"|\b(?:it|this|that|the (?:error|problem|issue|bug|403|failure))(?:\s+(?:is|has been|was)|'s)\s+(?:now\s+)?"
    r"(?:fixed|resolved|sorted(?: out)?)\b"
    r"|\b(?:should|will) (?:now )?work now\b", re.I)
# An application said to have reached the employer. R3 only records that the user
# applied, so this is false whether or not a run started (QA D2: "Yes, your
# application went through!").
APPLY_LINE = ("R3 has recorded it as applied for you, but R3 doesn't send applications: nothing went to the "
              "employer. To apply, use the employer's own application form.")
_APP = r"(?:applications?|resumes?|cvs?)"
_SENT_OUT = re.compile(
    r"\b" + _APP + r"\b[^.!?\n]*?\b(?:went|gone|got|has gone|have gone)\s+through\b"
    r"|\b" + _APP + r"\b[^.!?\n]*?\b(?:is|are|was|were|has been|have been|got)\s+(?:successfully\s+)?"
    r"(?:submitted|sent|delivered|received|forwarded)\b"
    r"|\b(?:i|we)(?:'ve|\s+have)?\s+(?:successfully\s+)?(?:submitted|sent|delivered|forwarded)\b[^.!?\n]*?\b"
    + _APP + r"\b"
    r"|\b(?:employer|company|recruiter|hiring (?:team|manager)|[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+"
    r"(?:has|have|now has|now have)\s+(?:received|got)\s+(?:your|the)\s+" + _APP + r"\b"
    r"|\bsuccessfully applied\b", re.I)
# A denial ("nothing was sent") or a condition ("once your application is submitted") is no claim.
_DENIES = re.compile(r"\b(?:not|never|nothing|no|once|after|when|if|before|until|whether)\b|n't\b", re.I)
_MARKER = re.compile(r"^(\s*(?:[-*+•]|\d{1,3}[.)])\s+)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def claims_a_run(sentence: str, *, looked_up_runs: bool = False) -> bool:
    """Whether ``sentence`` says a run started, is running or is done, or that
    an error is fixed. ``looked_up_runs``: the advisor read the run history
    this turn, so a run described as running is one it looked up (a
    first-person claim still counts)."""
    for m in _ACTOR.finditer(sentence):
        if m.group("strong") or _RUN_NOUN.match(m.group("rest")) or _PRONOUN.match(m.group("rest")):
            return True
    if not looked_up_runs:
        for m in _PROGRESS.finditer(sentence):
            if not _NEGATION.search(sentence[:m.start()]):
                return True
    return bool(_NOW_GERUND.search(sentence) or _FIXED.search(sentence))


def claims_sent_out(sentence: str) -> bool:
    """Whether ``sentence`` says an application reached the employer (went
    through, was submitted or sent, was received); a denial never counts."""
    m = _SENT_OUT.search(sentence)
    return m is not None and not _DENIES.search(sentence)


def honest_reply(text: str, *, run_started: bool, looked_up_runs: bool = False) -> str:
    """``text`` with its claims replaced (see the module docstring). Run claims
    stay when ``run_started`` (the panel started a run in this turn); claims
    that an application reached the employer never stay."""
    replaced = applied = False
    lines: list[str] = []
    for line in text.split("\n"):
        marker = _MARKER.match(line)
        prefix = marker.group(1) if marker else ""
        body = line[len(prefix):]
        if not body.strip():
            lines.append(line)
            continue
        kept: list[str] = []
        for sentence in _SENTENCE_END.split(body):
            if claims_sent_out(sentence):
                if not applied:
                    kept.append(APPLY_LINE)
                    applied = True
                continue
            if not run_started and claims_a_run(sentence, looked_up_runs=looked_up_runs):
                if not replaced:
                    kept.append(HONEST_LINE)
                    replaced = True
                continue
            kept.append(sentence)
        if kept:
            lines.append(prefix + " ".join(kept))
    if not (replaced or applied):
        return text
    log.info("Replaced a claim in the advisor's reply: %s",
             " and ".join(k for k, v in (("a run started or an error is fixed", replaced),
                                         ("an application reached the employer", applied)) if v))
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return out or (APPLY_LINE if applied else HONEST_LINE)
