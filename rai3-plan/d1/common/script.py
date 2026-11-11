"""Scripted, deterministic chat model plan shared by both fake models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Call:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Text:
    text: str


Step = list[Call] | Text


@dataclass
class Script:
    """Steps the fake model returns in order. `seen` records, per model call, the plain text of
    every message the model received (role, text), so tests can check what the LLM saw."""

    steps: list[Step]
    seen: list[list[tuple[str, str]]] = field(default_factory=list)
    _i: int = 0

    def next(self, messages: list[tuple[str, str]]) -> Step:
        self.seen.append(messages)
        if self._i >= len(self.steps):
            return Text("script exhausted")
        step = self.steps[self._i]
        self._i += 1
        return step


def read_loop(n: int) -> Script:
    """n READ tool calls, then a final answer: n + 1 model calls."""
    return Script(
        [[Call("get_latest_message", {"topic": "/d1/status"})] for _ in range(n)]
        + [Text("done")]
    )
