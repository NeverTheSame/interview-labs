"""R3_TASK_MODEL: one model for R3's AI steps (the resume builder, the resume
reader, job overviews) apart from the chat's own model. Unset, the
steps follow the chat default as before."""

from __future__ import annotations

import pytest

from r3.agent.service import effective_default_model, task_model_id
from r3.settings import ControllerSettings

LUNA = "openrouter:openai/gpt-6-luna"
FREE = "openrouter:openrouter/free"


def _settings(**env: str) -> ControllerSettings:
    return ControllerSettings.from_env({"OPENROUTER_API_KEY": "k", **env})


def test_task_model_is_pinned_apart_from_the_chat_model():
    s = _settings(R3_CHAT_MODEL=FREE, R3_TASK_MODEL=LUNA)
    assert effective_default_model(s) == FREE
    assert task_model_id(s) == LUNA


def test_task_model_follows_the_chat_default_when_unset():
    s = _settings(R3_CHAT_MODEL=FREE)
    assert s.task_model == ""
    assert task_model_id(s) == FREE


def test_task_model_without_a_key_falls_back_to_the_chat_default():
    """A pinned model whose provider has no key is not used: the steps keep
    working on the chat's model instead of failing every call."""
    s = _settings(R3_CHAT_MODEL=FREE, R3_TASK_MODEL="openai:gpt-4.1-mini")
    assert task_model_id(s) == FREE


def test_reader_uses_parse_model_then_task_model():
    from r3.resume.parse import reader_model_id

    assert reader_model_id(_settings(R3_CHAT_MODEL=FREE, R3_TASK_MODEL=LUNA)) == LUNA
    pinned = _settings(R3_CHAT_MODEL=FREE, R3_TASK_MODEL=LUNA, R3_PARSE_MODEL="openrouter:acme/small:free")
    assert reader_model_id(pinned) == "openrouter:acme/small:free"


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The model ids the steps ask r3.agent.models.build_model for."""
    import r3.agent.models as models

    ids: list[str] = []

    def fake_build(model_id: str, settings: object) -> object:
        ids.append(model_id)
        return object()

    monkeypatch.setattr(models, "build_model", fake_build)
    return ids


def test_job_overview_runs_on_the_task_model(built: list[str]):
    from r3.job_summary import build_summarizer

    s = _settings(R3_CHAT_MODEL=FREE, R3_TASK_MODEL=LUNA)
    build_summarizer(lambda: s)._resolve()
    assert built == [LUNA]


def test_resume_builder_runs_on_the_task_model(built: list[str]):
    from r3.resume.builder import LLMInterviewer, build_resume_interviewer

    s = _settings(R3_CHAT_MODEL=FREE, R3_TASK_MODEL=LUNA)
    interviewer = build_resume_interviewer(s)
    assert isinstance(interviewer, LLMInterviewer)
    interviewer._model()
    assert built == [LUNA]


def test_task_model_is_on_the_settings_page():
    from r3.config_registry import spec

    entry = spec("R3_TASK_MODEL")
    assert entry.kind == "model" and entry.group == "models"
