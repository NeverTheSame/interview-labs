"""``AdvisorService`` runs one chat turn and streams :mod:`events`. The
web layer only knows this protocol; :class:`Advisor` is the real one (router
→ panel action → advisor agent), :class:`UnconfiguredAdvisor` answers when no
provider key is set, and tests inject fakes."""

from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing
from datetime import datetime, timezone
from typing import AsyncIterator, Mapping, Protocol

from pydantic_ai.models import Model

from r3.db import Database, Facets
from r3.db.users import User
from r3.runs.service import RunService
from r3.settings import ControllerSettings

from .actions import fetch_facets, plan_action
from .advisor import TURN_TIMEOUT_S, stream_advice
from .contracts import ChatTurnRequest, ResumeFromTextAction, RouteDecision
from .events import ActionEvent, DoneEvent, ErrorEvent, FinalEvent, RouteEvent
from .models import build_model
from .router import RESUME_HELP, IntentRouter, advice_topic, resume_background
from .tools import AdvisorDeps, scoped_user_id

log = logging.getLogger("r3.agent.service")

MIN_ADVICE_BUDGET_S = 20.0


class AdvisorService(Protocol):
    settings: ControllerSettings

    def stream_turn(self, turn: ChatTurnRequest, database: Database, runs: RunService, *,
                    user: User | None = None) -> AsyncIterator[object]: ...


class UnconfiguredAdvisor:
    """What answers when no provider key is set: one error, then done."""

    def __init__(self, settings: ControllerSettings) -> None:
        self.settings = settings

    async def stream_turn(self, turn: ChatTurnRequest, database: Database, runs: RunService, *,
                          user: User | None = None) -> AsyncIterator[object]:
        yield ErrorEvent(message=self.settings.setup_hint(), retryable=False)
        yield DoneEvent()


# The Assistant's whole answer when a resume request carries a background: the page makes the resume.
MAKING_RESUME = "Making a resume from what you wrote. What I read appears below in a moment."


def effective_default_model(settings: ControllerSettings) -> str:
    """The model a turn uses when the client names none: the configured
    default, or the first model whose provider has a key when the default's
    provider has none (so a portal with only an OpenAI key still chats)."""
    if settings.chat_ready:
        return settings.default_model
    available = settings.available_models()
    return available[0].id if available else settings.default_model


def task_model_id(settings: ControllerSettings) -> str:
    """The model R3's AI steps run on (the resume builder and reader, job
    overviews): ``R3_TASK_MODEL`` when its provider has a key,
    else the chat's effective default."""
    pinned = settings.task_model
    if pinned and settings.provider_ready(pinned.split(":", 1)[0]):
        return pinned
    return effective_default_model(settings)


def custom_models_allowed(settings: ControllerSettings, user: User | None) -> bool:
    """Whether ``user`` may type a model id that is not on the list: admins
    (and the local operator) per the settings, regular accounts never."""
    return settings.allow_custom_models and (user is None or user.is_admin)


def turn_model(settings: ControllerSettings, requested: str | None, user: User | None = None) -> str:
    """The model id a turn runs on. A regular account may only pick from
    ``settings.available_models()``; any other id silently falls back to the
    default. Admins keep :meth:`ControllerSettings.resolve_model`'s rules."""
    requested = (requested or "").strip() or None
    if requested and user is not None and not user.is_admin and \
            requested not in {m.id for m in settings.available_models()}:
        requested = None
    return settings.resolve_model(requested or effective_default_model(settings))


class Advisor:
    """The Controller's chat: route the message, plan the panel action, then
    stream the advisor's answer.

    ``models`` injects ready-made models instead of building them from the
    settings (tests pass ``FunctionModel``/``TestModel``): ``"chat"`` (or a
    model id) for the advisor, ``"llm"`` and ``"jev"`` for the router."""

    def __init__(self, settings: ControllerSettings, *, models: Mapping[str, Model] | None = None,
                 router: IntentRouter | None = None) -> None:
        self.settings = settings
        self.models: dict[str, Model] = dict(models or {})
        self.router = router or IntentRouter(settings, models=self.models)

    def chat_model(self, model_id: str) -> Model:
        return self.models.get(model_id) or self.models.get("chat") or build_model(model_id, self.settings)

    async def stream_turn(self, turn: ChatTurnRequest, database: Database, runs: RunService, *,
                          user: User | None = None) -> AsyncIterator[object]:
        """``user`` is the signed-in account (None: the local operator): the
        tools, job lists and runs of the turn are its own."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            model_id = turn_model(self.settings, turn.model, user)
            if model_id.startswith("typesafe:"):
                raise ValueError("Jev classifies messages but cannot chat; pick another model")
            model = self.chat_model(model_id)
        except ValueError as exc:
            yield ErrorEvent(message=str(exc), retryable=False)
            yield DoneEvent()
            return
        try:
            facets: Facets | None = await fetch_facets(database, user_id=scoped_user_id(user))
            if turn.mode == "advice":
                # No workflow either way; a topic (a resume request, a price question) still gets its guidance.
                decision = RouteDecision(intent="advice", router="rules",
                                         reason=advice_topic(turn.text, turn.history) or "advice mode")
            else:
                decision = await self.router.route(turn.text, turn.history, model_id=model_id, facets=facets)
            yield RouteEvent(decision=decision)
            background = resume_background(turn.text, turn.history) if (
                turn.mode != "advice" and decision.reason == RESUME_HELP) else ""
            if background:
                # No chat model: the page posts the text to /api/resume/from-text and shows what R3 read.
                yield ActionEvent(action=ResumeFromTextAction(text=background))
                yield FinalEvent(text=MAKING_RESUME)
                yield DoneEvent()
                return
            action = await plan_action(decision, database=database, runs=runs, settings=self.settings,
                                       model_id=model_id, text=turn.text, facets=facets, user=user)
            yield ActionEvent(action=action)
            deps = AdvisorDeps(database=database, runs=runs, settings=self.settings, decision=decision, action=action,
                               include_context=turn.include_context and self.settings.include_context,
                               now=datetime.now(timezone.utc), facets=facets, user=user)
            budget = max(MIN_ADVICE_BUDGET_S, TURN_TIMEOUT_S - (loop.time() - started))
            async with aclosing(stream_advice(turn, deps, model, model_id, timeout_s=budget)) as events:
                async for event in events:
                    yield event
        except Exception:
            log.exception("Chat turn failed")
            yield ErrorEvent(message="Something went wrong on the server while answering. Please try again.")
        yield DoneEvent()


def build_advisor(settings: ControllerSettings) -> AdvisorService:
    """The real advisor when any chat provider has a key, else the one that
    explains how to configure it."""
    if settings.available_models():
        return Advisor(settings)
    return UnconfiguredAdvisor(settings)
