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

"""RateLimit: a token bucket per tool name and per (kind, name, type) egress key."""

import threading
import time
from collections.abc import Hashable, Iterable
from typing import Optional, Union

from rai.communication.ros2 import egress as eg
from rai.tools import hooks as th
from rai.tools.hooks import Effect, ToolCallContext

from ._base import SafetyHook


class RateLimit(SafetyHook):
    """Allows ``rate`` calls per ``per_s`` seconds per key, with bursts of up to
    ``burst`` (default: rate) and denies the rest: fail closed.

    Keys: a tool's name on the agent side; (kind, resolved name, type) on the
    connector side, so each topic is limited on its own. Streams are not
    exempt: a cmd_vel stream faster than the limit gets EgressDenied for the
    messages over it, which is the point (set the rate above the stream's).

    Matching: egress by ``types``/``names``/``kinds`` (neither types nor names:
    every write); tools by ``tool_effects`` (WRITE and ACTUATE by default) and,
    if given, ``tool_names``.
    """

    def __init__(
        self,
        rate: float,
        per_s: float = 1.0,
        *,
        burst: Optional[float] = None,
        types: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
        kinds: Iterable[str] = eg.WRITE_KINDS,
        tool_names: Optional[Iterable[str]] = None,
        tool_effects: Iterable[Effect] = (Effect.WRITE, Effect.ACTUATE),
        name: str = "RateLimit",
    ) -> None:
        if rate <= 0 or per_s <= 0:
            raise ValueError("rate and per_s must be positive")
        super().__init__(
            name, types=types, names=names, kinds=kinds, tool_effects=tool_effects
        )
        self.rate = rate
        self.per_s = per_s
        self.burst = float(burst if burst is not None else rate)
        self.tool_names = None if tool_names is None else frozenset(tool_names)
        self._refill = rate / per_s  # tokens per second
        self._buckets: dict[Hashable, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: Hashable) -> Optional[float]:
        """Take a token for key. None if allowed, else seconds until one is free."""
        now = time.monotonic()
        with self._lock:
            tokens, then = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - then) * self._refill)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return (1.0 - tokens) / self._refill
            self._buckets[key] = (tokens - 1.0, now)
            return None

    def _reason(self, wait_s: float) -> str:
        return f"rate limited to {self.rate:g} per {self.per_s:g} s, retry in {wait_s:.1f} s"

    async def _tool(self, call: ToolCallContext) -> Union[th.Allow, th.Deny]:
        if call.effect not in self.tool_effects or (
            self.tool_names is not None and call.name not in self.tool_names
        ):
            return th.Allow()
        wait_s = self.take(("tool", call.name))
        return th.Allow() if wait_s is None else th.Deny(self._reason(wait_s))

    def _egress(self, egress: eg.Egress) -> Union[eg.Allow, eg.Deny]:
        wait_s = self.take((egress.kind, egress.name, egress.type))
        return eg.Allow() if wait_s is None else eg.Deny(self._reason(wait_s))
