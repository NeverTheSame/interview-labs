"""Role keywords from a resume: the roles its headline and recent titles
name, which the best-match sort ranks job titles by before tag overlap."""

import re

import pytest

from r3.resume.roles import RoleKeywords, resume_roles, role_keywords, title_keywords, title_patterns


def _matches(patterns: list[str], title: str) -> bool:
    # The patterns are Postgres AREs for ~*; \m and \M are its word boundaries.
    return any(re.search(p.replace(r"\m", r"\b").replace(r"\M", r"\b"), title, re.I) for p in patterns)


@pytest.mark.parametrize("title, want", [
    ("DevOps / Platform Engineer", ["devops", "platform"]),
    ("Senior Staff Software Engineer II", ["software"]),
    ("Sr. Principal Lead Specialist", []),
    ("Jr Engineer III", []),
    ("Senior Site Reliability Engineer", ["sre"]),
    ("Site-Reliability Eng", ["sre"]),
    ("SRE Manager", ["sre"]),
    ("Infra Engineer", ["infrastructure"]),
    ("Cloud Infrastructure Engineer", ["cloud", "infrastructure"]),
    ("Dev Ops Engineer", ["devops"]),
    ("DevSecOps Lead", ["devops"]),
    ("Linux system administrator", ["linux", "system"]),
    ("Back-End Developer", ["backend"]),
    ("Machine Learning Engineer", ["ml"]),
    # A headline that lists more after a bar or colon names its role first.
    ("Senior DevOps Engineer | AWS Certified | Kubernetes", ["devops"]),
    ("Platform engineer: Python, Kubernetes and AWS. Terraform too.", ["platform"]),
    ("", []),
])
def test_title_keywords_drop_generic_words_and_map_synonyms(title, want):
    assert title_keywords(title) == want


def test_a_sentence_is_not_a_title():
    assert title_keywords("Experienced engineer who loves building reliable systems for fast growing teams") == []


def test_patterns_match_synonyms_as_whole_words():
    sre = title_patterns(["sre"])
    for title in ("Site Reliability Engineer", "Senior SRE", "SRE, Payments", "Staff Site-Reliability Engineer"):
        assert _matches(sre, title), title
    for title in ("Presales Engineer", "Reliability Engineer, Storage", "DevOps Engineer"):
        assert not _matches(sre, title), title
    infra = title_patterns(["infrastructure"])
    assert _matches(infra, "Infra Engineer") and _matches(infra, "Cloud Infrastructure Engineer")
    devops = title_patterns(["devops"])
    assert _matches(devops, "Dev-Ops Lead") and _matches(devops, "DevSecOps Engineer")
    assert not _matches(devops, "Developer Advocate")
    platform = title_patterns(["platform"])
    assert _matches(platform, "Platforms Team") and not _matches(platform, "Platformer")


def test_patterns_are_built_from_letters_and_digits_only():
    # The patterns are bound as data, but a malformed one would still fail
    # the query, so anything that is not a plain keyword is left out.
    assert title_patterns(["c++", "a.*", "", "x|y"]) == []
    assert title_patterns(["linux", "linux"]) == [r"\mlinuxs?\M"]


def test_headline_is_the_target_and_recent_roles_come_second():
    roles = role_keywords("DevOps / Platform Engineer", ["Linux system administrator", "IT help desk", "Barista"])
    assert roles.target == ("devops", "platform")
    # Only the two most recent roles count.
    assert roles.recent == ("linux", "system", "it", "help", "desk")


def test_without_a_headline_the_latest_role_is_the_target():
    roles = role_keywords("", ["Senior Site Reliability Engineer", "Systems Engineer"])
    assert roles == RoleKeywords(target=("sre",), recent=("systems",))
    # A title of only generic words names nothing, so the next one is the target.
    assert role_keywords("Engineer", ["Senior Engineer", "Cloud Engineer"]).target == ("cloud",)


def test_recent_keywords_never_repeat_a_target_keyword():
    roles = role_keywords("DevOps / Cloud Engineer", ["Cloud Engineer", "Junior Developer"])
    assert roles == RoleKeywords(target=("devops", "cloud"), recent=())


def test_an_empty_resume_has_no_keywords_and_no_patterns():
    assert not role_keywords("", [])
    assert not resume_roles("", None)
    assert resume_roles("   \n", {}).patterns() is None


def test_patterns_carry_both_tiers():
    target, recent = role_keywords("SRE", ["Systems Engineer"]).patterns()
    assert _matches(target, "Site Reliability Engineer") and not _matches(target, "Systems Engineer")
    assert _matches(recent, "Linux Systems Engineer") and not _matches(recent, "Senior SRE")


def test_resume_roles_prefers_the_stored_read():
    structure = {"draft": {"headline": "Data Engineer", "experience": [{"title": "Analytics Engineer"}]}}
    roles = resume_roles("Pat Doe\nDevOps Engineer\n", structure)
    assert roles == RoleKeywords(target=("data",), recent=("analytics",))


def test_resume_roles_reads_the_text_when_nothing_is_stored():
    # The layout the resume builder saves: name, headline, contact, sections.
    text = ("Maya Okafor\nDevOps / Platform Engineer\nmaya.okafor@example.com | 204-555-0147\n\n"
            "EXPERIENCE\nLinux system administrator — Mercy Valley Health, Winnipeg\nMarch 2019 – Present\n"
            "- Maintained RHEL servers with Ansible\n\nIT help desk — Brightline Telecom\n"
            "June 2017 – February 2019\n- Handled tickets\n\nSKILLS\nLinux, Bash, AWS, Python\n")
    roles = resume_roles(text, None)
    assert roles.target == ("devops", "platform")
    assert roles.recent == ("linux", "system", "it", "help", "desk")


def test_a_stored_read_without_titles_falls_back_to_the_text():
    text = "Dana Doe\nSite Reliability Engineer\ndana@example.com\n"
    assert resume_roles(text, {"draft": {"headline": "", "experience": []}}).target == ("sre",)


def test_a_damaged_stored_read_falls_back_to_the_text():
    text = "Dana Doe\nSite Reliability Engineer\ndana@example.com\n"
    assert resume_roles(text, {"draft": {"experience": "not a list"}}).target == ("sre",)
    assert resume_roles(text, {"draft": ["not", "a", "draft"]}).target == ("sre",)
