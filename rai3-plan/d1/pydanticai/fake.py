"""Scripted fake chat model (test support, not counted as slice code)."""

from __future__ import annotations

import itertools

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..common.script import Script, Text


def scripted_model(script: Script) -> FunctionModel:
    ids = itertools.count()

    def respond(messages, info: AgentInfo) -> ModelResponse:
        seen = [
            (_role(p), str(getattr(p, "content", "")))
            for m in messages
            for p in m.parts
        ]
        step = script.next(seen)
        if isinstance(step, Text):
            return ModelResponse(parts=[TextPart(step.text)])
        return ModelResponse(
            parts=[ToolCallPart(c.name, c.args, f"call_{next(ids)}") for c in step]
        )

    return FunctionModel(respond)


def _role(part) -> str:
    return {
        "user-prompt": "human",
        "tool-return": "tool",
        "retry-prompt": "tool",
        "text": "ai",
        "tool-call": "ai",
    }.get(part.part_kind, part.part_kind)
