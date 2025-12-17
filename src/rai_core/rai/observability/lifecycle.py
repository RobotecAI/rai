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

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .correlation import get_observability_context
from .sink import NoOpSink, ObservabilitySink

# Example emitted event (flat dict), e.g. heartbeat:
# {
#   "schema_version": "v1",
#   "event_type": "heartbeat",
#   "timestamp_s": 1734401234.56,
#   "component_type": "agent",
#   "agent_name": "planner",
#   "request_id": "abc123",
#   "run_id": "run-42",
#   "foo": "bar"
# }


def heartbeat_period_s() -> Optional[float]:
    """Return heartbeat period in seconds if enabled via env, else None."""
    # Heartbeats are opt-in via environment variable to avoid background threads
    # by default (useful in tests/CLI tools).
    # Controls whether heartbeat is enabled and, if enabled, the interval between emissions.
    raw = os.getenv("RAI_OBS_HEARTBEAT_SEC")
    if not raw:
        return None
    try:
        # Accept ints/floats; reject non-numeric values by returning None.
        period = float(raw)
    except Exception:
        return None
    # Non-positive periods effectively disable the heartbeat.
    if period <= 0:
        return None
    return period


def emit_event(
    *,
    sink: Optional[ObservabilitySink],
    event_type: str,
    component_type: str,
    agent_name: Optional[str] = None,
    connector_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a lightweight lifecycle/heartbeat event (best-effort)."""
    # If no sink is configured, fall back to a no-op sink so callers don't need
    # to branch on observability being enabled.
    s = sink or NoOpSink()
    try:
        # Keep this schema small/flat; sinks can enrich/transform downstream.
        event: Dict[str, Any] = {
            "schema_version": "v1",
            "event_type": event_type,
            "timestamp_s": time.time(),
            "component_type": component_type,  # "agent" | "connector"
        }
        if agent_name is not None:
            event["agent_name"] = agent_name
        if connector_name is not None:
            event["connector_name"] = connector_name
        # Add correlation/trace context (e.g. request_id, run_id) if present.
        event.update(get_observability_context())
        if extra:
            # Caller-provided fields override defaults if keys collide.
            event.update(extra)
        s.record(event)
    except Exception:
        # Best-effort: never raise into caller.
        return None


@dataclass
class HeartbeatHandle:
    # A small handle that lets callers stop the background heartbeat thread.
    stop_event: threading.Event
    thread: threading.Thread

    def stop(self) -> None:
        # Signal thread and wait briefly for clean exit.
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def start_heartbeat(
    *,
    sink: Optional[ObservabilitySink],
    component_type: str,
    agent_name: Optional[str] = None,
    connector_name: Optional[str] = None,
    period_s: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[HeartbeatHandle]:
    """Start periodic heartbeat emission in a daemon thread (opt-in via env)."""
    # Prefer explicit period_s if provided; otherwise read env var. Returning
    # None means "heartbeat disabled" (no thread started).
    period = period_s if period_s is not None else heartbeat_period_s()
    if not period:
        return None

    stop_event = threading.Event()

    def _run() -> None:
        # First heartbeat quickly, then periodically.
        while not stop_event.is_set():
            emit_event(
                sink=sink,
                event_type="heartbeat",
                component_type=component_type,
                agent_name=agent_name,
                connector_name=connector_name,
                extra=extra,
            )
            # Use wait() so stop() can interrupt sleep promptly.
            stop_event.wait(period)

    t = threading.Thread(
        target=_run,
        name=f"ObsHeartbeat({component_type}:{agent_name or connector_name or 'unknown'})",
        daemon=True,
    )
    t.start()
    return HeartbeatHandle(stop_event=stop_event, thread=t)
