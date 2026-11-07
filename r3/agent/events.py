"""The Server-Sent Events a chat turn streams to the browser. Each event is
``event: <type>`` followed by one JSON ``data:`` line. The order for a turn is
``route`` → (``action``)? → (``tool_call`` | ``tool_result`` | ``delta``)* →
``final`` → ``done``; ``error`` may replace ``final`` and is followed by
``done``."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from .contracts import Action, RouteDecision


class RouteEvent(BaseModel):
    type: Literal["route"] = "route"
    decision: RouteDecision


class ActionEvent(BaseModel):
    type: Literal["action"] = "action"
    action: Action


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    name: str
    summary: str = ""


class DeltaEvent(BaseModel):
    type: Literal["delta"] = "delta"
    text: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


class FinalEvent(BaseModel):
    type: Literal["final"] = "final"
    text: str
    model: str = ""
    usage: Usage = Field(default_factory=Usage)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    retryable: bool = True


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


ChatEvent = Annotated[
    Union[RouteEvent, ActionEvent, ToolCallEvent, ToolResultEvent, DeltaEvent, FinalEvent, ErrorEvent, DoneEvent],
    Field(discriminator="type"),
]
chat_event_adapter: TypeAdapter[Any] = TypeAdapter(ChatEvent)


def sse(event: BaseModel) -> str:
    """Encode one event for the wire. JSON never contains a raw newline, so
    one ``data:`` line is enough."""
    payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.type}\ndata: {payload}\n\n"  # type: ignore[attr-defined]


def parse_sse(body: str) -> list[Any]:
    """Decode a stream back into events; tests and the evals use it."""
    events: list[Any] = []
    for block in body.split("\n\n"):
        data = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            events.append(chat_event_adapter.validate_json("\n".join(data)))
    return events
