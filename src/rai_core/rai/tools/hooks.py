# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent-side tool hooks: an ordered, fail-closed chain run by every `RaiTool` call.

A `ToolHook` is `async (ToolCallContext) -> Allow | Deny`. A hook that raises, times out
or returns anything else denies the call. Connector-side egress hooks are separate.
"""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HOOK_TIMEOUT_S = 2.0


class Effect(StrEnum):
    """What a tool does to the world. Unknown means ACTUATE (most restrictive)."""

    READ = "read"
    WRITE = "write"
    ACTUATE = "actuate"


@dataclass(frozen=True)
class ToolCallContext:
    name: str
    effect: Effect
    targets: tuple[str, ...]
    args: Mapping[str, Any]


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Deny:
    reason: str


ToolHook = Callable[[ToolCallContext], Awaitable[Allow | Deny]]


@dataclass(frozen=True)
class _Entry:
    hook: ToolHook
    timeout_s: float


class ToolHooks:
    """Ordered hook chain. Hooks run in insertion order; the first denial wins."""

    def __init__(self) -> None:
        self._entries: tuple[_Entry, ...] = ()
        self._lock = threading.Lock()

    def add(
        self, hook: ToolHook, *, timeout_s: float = DEFAULT_HOOK_TIMEOUT_S
    ) -> ToolHook:
        with self._lock:
            self._entries = (*self._entries, _Entry(hook, timeout_s))
        return hook

    def remove(self, hook: ToolHook) -> None:
        with self._lock:
            self._entries = tuple(e for e in self._entries if e.hook is not hook)

    def clear(self) -> None:
        with self._lock:
            self._entries = ()

    def __len__(self) -> int:
        return len(self._entries)


global_tool_hooks = ToolHooks()
"""Hooks run before the per-tool hooks of every `RaiTool`."""


async def run_tool_hooks(
    call: ToolCallContext, chains: Iterable[ToolHooks]
) -> Deny | None:
    """Run every hook of `chains` in order. Return None to allow, or the first denial."""
    for chain in chains:
        for entry in chain._entries:
            name = getattr(entry.hook, "__name__", type(entry.hook).__name__)
            try:
                async with asyncio.timeout(entry.timeout_s):
                    decision = await entry.hook(call)
            except TimeoutError:
                failure = f"timed out after {entry.timeout_s}s"
            except Exception as e:  # fail closed on any hook error
                failure = f"failed ({type(e).__name__}: {e})"
            else:
                if isinstance(decision, Allow):
                    continue
                if isinstance(decision, Deny):
                    return Deny(f"Denied by {name}: {decision.reason}")
                failure = f"returned {decision!r}, expected Allow or Deny"
            logger.warning("Tool %s denied: hook %s %s", call.name, name, failure)
            return Deny(f"Denied: hook {name} {failure}")
    return None
