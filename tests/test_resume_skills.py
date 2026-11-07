"""Resume skills: which corpus tech tags a resume evidences, and the
per-process vocabulary cache. Pure functions; no database."""

from r3.resume.skills import VocabularyCache, resume_skills

VOCAB = [
    "aws", "kubernetes", "terraform", "python", "go", "c++", "c#", "c", "r", "ci/cd", "node.js",
    "github actions", "postgresql", "javascript", "typescript", "gcp", ".net", "rest", "java",
    "helm", "bash", "react",
]


def test_matches_whole_words_case_insensitively():
    text = "Built AWS infrastructure with Terraform and PYTHON; ran Kubernetes clusters with Helm."
    assert resume_skills(text, VOCAB) == ["aws", "helm", "kubernetes", "python", "terraform"]


def test_no_partial_word_hits():
    # "javascript" must not give "java"; "bashful" is not "bash"; "awsome" is not "aws".
    assert resume_skills("JavaScript developer, a bashful awsome person", VOCAB) == ["javascript"]


def test_tags_with_punctuation():
    text = "Wrote C++ services, some C#, a Node.js API and ASP.NET apps; owned CI/CD on GitHub Actions."
    assert resume_skills(text, VOCAB) == [".net", "c#", "c++", "ci/cd", "github actions", "node.js"]


def test_phrase_tags_allow_hyphen_or_extra_space():
    assert resume_skills("GitHub-Actions and github   actions", ["github actions"]) == ["github actions"]


def test_aliases():
    text = "Ran k8s on Google Cloud with Postgres, wrote Golang, JS and TS."
    assert resume_skills(text, VOCAB) == ["gcp", "go", "javascript", "kubernetes", "postgresql", "typescript"]


def test_alias_counts_only_when_its_tag_is_in_the_vocabulary():
    assert resume_skills("k8s and postgres", ["aws"]) == []


def test_go_is_not_found_in_going_google_or_lowercase_prose():
    text = "Going forward I joined Google; ready to go-to-market and go live."
    assert "go" not in resume_skills(text, VOCAB)
    assert "go" in resume_skills("Languages: Go, Python", VOCAB)
    assert "go" in resume_skills("Services in Go.", VOCAB)


def test_single_letter_tags_only_as_standalone_tokens():
    assert resume_skills("Led R&D, C-level reporting, C++ and C# work, Plan C-3", ["r", "c"]) == []
    assert resume_skills("Languages: R, C, Python", ["r", "c", "python"]) == ["c", "python", "r"]
    assert resume_skills("Statistics in R (tidyverse)", ["r"]) == ["r"]


def test_english_word_tags_need_their_tech_spelling():
    assert resume_skills("handled the rest of the team; react to incidents", VOCAB) == []
    assert resume_skills("Designed REST APIs with React", VOCAB) == ["react", "rest"]


def test_empty_resume_or_vocabulary():
    assert resume_skills("", VOCAB) == []
    assert resume_skills("   \n", VOCAB) == []
    assert resume_skills("Python and AWS", []) == []


async def test_vocabulary_cache_reloads_only_after_ttl():
    now = [100.0]
    loads: list[int] = []

    async def load():
        loads.append(1)
        return ["aws", "go"][: len(loads)]

    cache = VocabularyCache(ttl=300, clock=lambda: now[0])
    assert await cache.get(load) == ["aws"]
    now[0] += 299
    assert await cache.get(load) == ["aws"] and len(loads) == 1
    now[0] += 1
    assert await cache.get(load) == ["aws", "go"] and len(loads) == 2
