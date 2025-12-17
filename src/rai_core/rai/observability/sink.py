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
import time
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
    """Wraps another sink, buffers under backpressure, drops oldest when full.

    Critical property: `record()` stays lightweight and does not synchronously
    flush buffered events on the caller thread.
    """

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
        # Condition variable used to wake the worker thread when:
        # - new events are enqueued (`record()`), or
        # - we are stopping (`stop()`), so the worker can exit promptly.
        self._condition = threading.Condition(self._lock)
        self._buf: deque[dict[str, Any]] = deque()
        self._running = False
        self._thread: threading.Thread | None = None
        self._backoff_s = 0.05
        self._max_backoff_s = 2.0

    def record(self, event: dict[str, Any]) -> None:
        # Hot path: enqueue only; never do a synchronous flush here.
        with self._condition:
            if len(self._buf) >= self.maxlen:
                self._buf.popleft()
                self.logger.debug("BufferedSink dropping oldest event (buffer full)")
            self._buf.append(event)
            self._condition.notify()

    def _worker(self) -> None:
        backoff = self._backoff_s
        while True:
            with self._condition:
                while self._running and not self._buf:
                    self._condition.wait(timeout=0.5)
                if not self._running and not self._buf:
                    return
                ev = self._buf.popleft()
            try:
                self.inner.record(ev)
                backoff = self._backoff_s
            except Exception:
                # Put the event back at the front (best effort) and back off.
                with self._condition:
                    if len(self._buf) >= self.maxlen:
                        self._buf.popleft()
                        self.logger.debug(
                            "BufferedSink dropping oldest event (buffer full)"
                        )
                    self._buf.appendleft(ev)
                time.sleep(backoff)
                backoff = min(self._max_backoff_s, backoff * 2.0)

    def flush(self) -> None:
        # Best-effort flush; may block caller, but this is not on the hot path.
        pending: list[dict[str, Any]]
        with self._condition:
            pending = list(self._buf)
            self._buf.clear()
        for ev in pending:
            try:
                self.inner.record(ev)
            except Exception:
                # Re-queue remaining events and stop.
                with self._condition:
                    for back in pending[pending.index(ev) :]:
                        if len(self._buf) >= self.maxlen:
                            self._buf.popleft()
                        self._buf.append(back)
                break

    def start(self) -> None:
        if hasattr(self.inner, "start"):
            try:
                self.inner.start()
            except Exception:
                pass
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._worker, name="BufferedSinkWorker", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        if hasattr(self.inner, "stop"):
            try:
                self.inner.stop()
            except Exception:
                pass
        with self._condition:
            self._running = False
            self._condition.notify_all()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None


def default_buffer_size() -> int:
    try:
        return int(os.getenv("RAI_OBS_BUFFER_SIZE", "256"))
    except Exception:
        return 256
