"""Follow-ups after an uploaded resume is read (r3/resume/followup.py): only
the builder's hard requirements, deterministic, one at a time."""

from __future__ import annotations

from r3.resume.builder import Contact, Job, ResumeDraft
from r3.resume.followup import DATES_HELP, LINK_HELP, answer_followup, next_followup


def gappy() -> ResumeDraft:
    return ResumeDraft(full_name="Sam Lee", contact=Contact(email="sam@example.com"),
                       experience=[Job(title="Platform Engineer", company="Globex", start="2020", end="Present"),
                                   Job(title="Support Engineer", company="Hooli")])


def test_dates_first_then_the_link_then_done():
    turn = next_followup(gappy())
    assert turn.topic == "dates" and "Support Engineer" in turn.question
    turn = answer_followup(turn.draft, "2017, 2020")
    assert turn.draft.experience[1].start == "2017" and turn.draft.experience[1].end == "2020"
    assert turn.topic == "links" and "LinkedIn" in turn.question and "2017 – 2020" in turn.changed
    turn = answer_followup(turn.draft, "my github is github.com/samlee")
    assert turn.done and turn.draft.contact.links == ["github.com/samlee"]


def test_an_unreadable_answer_repeats_the_question_and_changes_nothing():
    turn = answer_followup(gappy(), "I don't remember")
    assert turn.note == DATES_HELP and turn.topic == "dates"
    assert (turn.draft.experience[1].start, turn.draft.experience[1].end) == ("", "")
    done_dates = answer_followup(gappy(), "Jan 2017, present").draft
    turn = answer_followup(done_dates, "I'd rather not")
    assert turn.note == LINK_HELP and turn.draft.contact.links == []


def test_skip_is_recorded_like_the_builder_does():
    turn = answer_followup(gappy(), "skip")
    assert "dates" in turn.draft.skipped and turn.topic == "links"
    turn = answer_followup(turn.draft, "skip")
    assert turn.done and turn.draft.skipped == ["dates", "links"]


def test_natural_date_answers_are_understood():
    for said, start, end in (("2021 - present", "2021", "present"), ("Jul 2021 – now", "Jul 2021", "now"),
                             ("from 2019 to 2022", "2019", "2022"), ("since 2020", "2020", "present"),
                             ("Jan, 2018 — Dec, 2019", "Jan, 2018", "Dec, 2019"), ("2017, 2020", "2017", "2020")):
        turn = answer_followup(gappy(), said)
        assert (turn.draft.experience[1].start, turn.draft.experience[1].end) == (start, end), said


def test_the_question_names_the_role_plainly():
    turn = next_followup(gappy())
    assert turn.question.startswith("When did you start and finish as Support Engineer at Hooli?")
    assert "answer as" not in turn.question


def test_a_question_back_is_handed_to_the_assistant_not_refused():
    turn = answer_followup(gappy(), "what do you need?")
    assert turn.off_topic and turn.note == "" and turn.draft == gappy()
    turn = answer_followup(gappy(), "no idea, sorry")
    assert not turn.off_topic and turn.note  # an attempt at an answer gets the example
