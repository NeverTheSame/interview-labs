"""Chat API: the streaming turn and the model picker. Owned by the agent side."""

from __future__ import annotations

import logging
from contextlib import aclosing, nullcontext
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from r3.agent.contracts import ChatTurnRequest
from r3.agent.events import (ActionEvent, DoneEvent, ErrorEvent, FinalEvent, RouteEvent, ToolCallEvent,
                             ToolResultEvent, sse)
from r3.agent.models import openrouter_catalogue
from r3.agent.service import custom_models_allowed, effective_default_model
from r3.settings import PROVIDERS, ControllerSettings
from r3.events import record as record_event
from r3.web.auth import current_user

log = logging.getLogger("r3.web.chat")

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


@router.post("/chat/stream")
async def chat_stream(request: Request, turn: ChatTurnRequest) -> StreamingResponse:
    """One chat turn as Server-Sent Events (see ``r3.agent.events``). An
    invalid body is a 422 before the stream starts. The turn is the signed-in
    account's: its jobs, resume and runs."""
    state = request.app.state
    user = current_user(request)
    asked = next((m.content for m in reversed(turn.messages) if m.role == "user"), "")
    record_event("chat.message", f"You asked: {asked}", actor="user", user_id=user.id,
                 data={"text": asked, "model": turn.model, "mode": turn.mode, "include_context": turn.include_context,
                       "history": len(turn.messages) - 1})
    stream = state.advisor.stream_turn(turn, state.db, state.runs, user=user)
    return StreamingResponse(_events(request, stream, user.id),
                             media_type="text/event-stream", headers=SSE_HEADERS)


class _Turn:
    """What the assistant decided in one turn, gathered from its stream for
    the audit trail: the route, the tools it called, the actions it proposed
    and its answer."""

    def __init__(self) -> None:
        self.route: dict[str, Any] | None = None
        self.tools: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.final: FinalEvent | None = None
        self.errors: list[str] = []

    def see(self, event: object) -> None:
        if isinstance(event, RouteEvent):
            self.route = event.decision.model_dump(mode="json")
        elif isinstance(event, ToolCallEvent):
            self.tools.append({"name": event.name, "args": event.args})
        elif isinstance(event, ToolResultEvent):
            call = next((t for t in reversed(self.tools) if t["name"] == event.name and "result" not in t), None)
            if call is not None:
                call["result"] = event.summary
        elif isinstance(event, ActionEvent):
            self.actions.append(event.action.model_dump(mode="json"))
        elif isinstance(event, FinalEvent):
            self.final = event
        elif isinstance(event, ErrorEvent):
            self.errors.append(event.message)

    def record(self, user_id: int) -> None:
        text = self.final.text if self.final else ""
        intent = (self.route or {}).get("intent", "")
        summary = (f"R3 answered{f' ({intent})' if intent else ''}: {text}" if text
                   else f"R3 could not answer: {self.errors[-1]}" if self.errors else "R3 gave no answer")
        record_event("chat.reply", summary, actor="ai", user_id=user_id,
                     data={"text": text, "model": self.final.model if self.final else "",
                           "usage": self.final.usage.model_dump() if self.final else None, "route": self.route,
                           "tools": self.tools, "actions": self.actions, "errors": self.errors})


async def _events(request: Request, stream: AsyncIterator[object], user_id: int) -> AsyncIterator[str]:
    done = False
    turn = _Turn()
    try:
        async with aclosing(stream) if hasattr(stream, "aclose") else nullcontext(stream) as events:  # type: ignore[attr-defined]
            async for event in events:
                turn.see(event)
                yield sse(event)  # type: ignore[arg-type]
                done = isinstance(event, DoneEvent)
                if done or await request.is_disconnected():
                    break
    except Exception:
        # The advisor handles its own failures; this only guards the stream's shape.
        log.exception("Chat stream failed")
        turn.errors.append("the chat stream failed on the server")
        yield sse(ErrorEvent(message="The chat stream failed on the server. Please try again."))
        done = False
    finally:
        turn.record(user_id)
    if not done and not await request.is_disconnected():
        yield sse(DoneEvent())


@router.get("/models")
async def models(request: Request, live: bool = False) -> dict[str, Any]:
    """The model picker: the configured models whose provider has a key, the
    default, whether a custom id may be typed, and which providers are ready.
    ``?live=1`` adds OpenRouter's public catalogue (empty when unreachable or
    without an OpenRouter key). A regular account may not type custom ids and
    gets no catalogue."""
    settings: ControllerSettings = request.app.state.settings
    user = current_user(request)
    body: dict[str, Any] = {
        "default": effective_default_model(settings),
        "models": [{"id": m.id, "label": m.label, "provider": m.provider} for m in settings.available_models()],
        "custom_allowed": custom_models_allowed(settings, user),
        "providers": {p: settings.provider_ready(p) for p in PROVIDERS},
    }
    if live and user.is_admin:
        body["catalogue"] = await openrouter_catalogue() if settings.provider_ready("openrouter") else []
    return body
