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

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Aggregator(ABC, Generic[T]):
    """One topic's recent messages, summarized only when the agent asks.

    :meth:`start` subscribes to ``topic`` on ``connector`` (anything with
    ``subscribe(topic, callback, *, msg_type, maxsize)`` returning an object
    with ``close()``) and keeps the last ``window`` messages with their arrival
    time; :meth:`stop` closes the subscription. Awaiting the aggregator (it is
    a :data:`rai.agents.StateSource`) returns :meth:`summarize` of the messages
    at most ``max_age_s`` old. The summary is cached for ``cache_s`` seconds and
    after that recomputed only if a new message arrived, so nothing runs (and no
    model is called) while nobody asks.
    """

    def __init__(
        self,
        connector: Any,
        topic: str,
        *,
        msg_type: Any = None,
        window: int = 30,
        max_age_s: float | None = 5.0,
        cache_s: float = 5.0,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.connector, self.topic, self.msg_type = connector, topic, msg_type
        self.window, self.max_age_s, self.cache_s = window, max_age_s, cache_s
        self._buffer: deque[tuple[float, T]] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._subscription: Any = None
        self._cached: str | list[Any] | None = None
        self._cached_at = float("-inf")
        self._newest: float | None = None  # arrival time of the newest summarized

    def start(self) -> None:
        """Subscribe to the topic. Idempotent."""
        if self._subscription is None:
            self._subscription = self.connector.subscribe(
                self.topic, self.push, msg_type=self.msg_type, maxsize=self.window
            )

    def stop(self) -> None:
        """Close the subscription. Idempotent; the buffer is kept."""
        subscription, self._subscription = self._subscription, None
        if subscription is not None:
            subscription.close()

    def push(self, msg: T) -> None:
        """Buffer a message (thread-safe; the oldest drops when full)."""
        with self._lock:
            self._buffer.append((time.monotonic(), msg))

    def messages(self) -> list[tuple[float, T]]:
        """``(arrival time.monotonic(), message)`` pairs at most ``max_age_s`` old,
        oldest first."""
        cutoff = float("-inf")
        if self.max_age_s is not None:
            cutoff = time.monotonic() - self.max_age_s
        with self._lock:
            return [(t, m) for t, m in self._buffer if t >= cutoff]

    @abstractmethod
    async def summarize(self, msgs: Sequence[T]) -> str | list[Any] | None:
        """Text or LangChain content blocks describing ``msgs`` (never empty)."""

    async def __call__(self) -> str | list[Any] | None:
        if time.monotonic() - self._cached_at < self.cache_s:
            return self._cached
        stamped = self.messages()
        newest = stamped[-1][0] if stamped else None
        if newest != self._newest or self._cached_at == float("-inf"):
            if stamped:
                self._cached = await self.summarize([m for _, m in stamped])
            else:
                age = (
                    ""
                    if self.max_age_s is None
                    else f" in the last {self.max_age_s:g} s"
                )
                self._cached = f"no messages on {self.topic}{age}"
            self._newest = newest
        self._cached_at = time.monotonic()
        return self._cached

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.topic!r}, len={len(self._buffer)})"
