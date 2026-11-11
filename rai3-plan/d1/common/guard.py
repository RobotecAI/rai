"""Framework-neutral fail-closed hook core. Each slice adapts it to its framework's hook point."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .robot import Effect


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    effect: Effect
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class Deny:
    reason: str


SafetyHook = Callable[[ToolCall], "Deny | None | Awaitable[Deny | None]"]


async def check(
    hooks: Sequence[SafetyHook], call: ToolCall, timeout_s: float
) -> str | None:
    """Run hooks in order. Return None to allow, or the denial text. A hook that raises or
    times out denies the call (fail closed)."""
    for hook in hooks:
        name = getattr(hook, "__name__", type(hook).__name__)
        try:
            async with asyncio.timeout(timeout_s):
                decision = hook(call)
                if inspect.isawaitable(decision):
                    decision = await decision
        except TimeoutError:
            return f"Denied: safety hook {name} timed out after {timeout_s}s"
        except Exception as e:  # noqa: BLE001 - fail closed on any hook error
            return f"Denied: safety hook {name} failed ({type(e).__name__}: {e})"
        if isinstance(decision, Deny):
            return f"Denied by {name}: {decision.reason}"
    return None


# ---- hooks used by the tests ----


def crashing_hook(call: ToolCall) -> None:
    raise RuntimeError("hook bug")


async def hanging_hook(call: ToolCall) -> None:
    await asyncio.sleep(30)


def deny_actuate(call: ToolCall) -> Deny | None:
    return Deny("actuation disabled") if call.effect is Effect.ACTUATE else None


@dataclass
class Recorder:
    calls: list[ToolCall] = field(default_factory=list)

    def __call__(self, call: ToolCall) -> None:
        self.calls.append(call)
