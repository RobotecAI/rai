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

import logging
import os
import threading
from collections import deque
from typing import Any, Protocol


class ObservabilitySink(Protocol):
    """Minimal interface for observability sinks."""

    def record(self, event: dict[str, Any]) -> None: ...

    def start(self) -> None:  # pragma: no cover - default no-op
        return None

    def flush(self) -> None:  # pragma: no cover - default no-op
        return None

    def stop(self) -> None:  # pragma: no cover - default no-op
        return None


class NoOpSink:
    """Sink that does nothing."""

    def record(self, event: dict[str, Any]) -> None:
        return None

    def start(self) -> None:
        return None

    def flush(self) -> None:
        return None

    def stop(self) -> None:
        return None


class LoggingSink:
    """Simple sink that logs events (placeholder for real transports)."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ObservabilityLoggingSink")

    def record(self, event: dict[str, Any]) -> None:
        # Log at debug to minimize noise; callable is non-blocking/low cost.
        self.logger.debug("observability event: %s", event)


class StdoutSink:
    """Sink that prints events to stdout; handy for local debugging."""

    def __init__(self, prefix: str = "observability") -> None:
        self.prefix = prefix

    def record(self, event: dict[str, Any]) -> None:
        print(f"{self.prefix}: {event}")


class BufferedSink:
    """Wraps another sink, buffers on failure, and drops oldest when full."""

    def __init__(
        self,
        inner: ObservabilitySink,
        maxlen: int = 256,
        logger: logging.Logger | None = None,
    ) -> None:
        self.inner = inner
        self.maxlen = maxlen
        self.logger = logger or logging.getLogger("BufferedSink")
        self._lock = threading.Lock()
        self._buf: deque[dict[str, Any]] = deque()

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            # Try to flush buffered events first.
            self._flush_locked()
            self._record_one_locked(event)

    def _record_one_locked(self, event: dict[str, Any]) -> None:
        try:
            self.inner.record(event)
        except Exception:
            if len(self._buf) >= self.maxlen:
                self._buf.popleft()
                self.logger.debug("BufferedSink dropping oldest event (buffer full)")
            self._buf.append(event)

    def _flush_locked(self) -> None:
        pending = list(self._buf)
        self._buf.clear()
        for ev in pending:
            try:
                self.inner.record(ev)
            except Exception:
                # Re-queue remaining events and stop to avoid tight retry loops.
                self._buf.extend(pending[pending.index(ev) :])
                break

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def start(self) -> None:
        if hasattr(self.inner, "start"):
            try:
                self.inner.start()
            except Exception:
                pass

    def stop(self) -> None:
        if hasattr(self.inner, "stop"):
            try:
                self.inner.stop()
            except Exception:
                pass


def default_buffer_size() -> int:
    try:
        return int(os.getenv("RAI_OBS_BUFFER_SIZE", "256"))
    except Exception:
        return 256
