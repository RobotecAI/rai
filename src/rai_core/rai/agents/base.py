# Copyright (C) 2024 Robotec.AI
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

# Used for best-effort cleanup of the observability heartbeat thread if the agent
# object is garbage-collected without an explicit `stop()` call.
import weakref
from abc import ABC, abstractmethod
from typing import Optional

from rai.communication.base_connector import BaseConnector
from rai.observability import ObservabilitySink, build_sink_from_env
from rai.observability.lifecycle import emit_event, start_heartbeat


class BaseAgent(ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        observability_sink: Optional[ObservabilitySink] = None,
        observability_endpoint: Optional[str] = None,
    ):
        """Initializes a new agent instance, logger, and optional observability sink."""
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        self.observability_sink = observability_sink or build_sink_from_env(
            endpoint=observability_endpoint
        )
        # Lifecycle + optional heartbeat for "silent failure" detection.
        emit_event(
            sink=self.observability_sink,
            event_type="agent_start",
            component_type="agent",
            agent_name=self.name,
        )
        hb = start_heartbeat(
            sink=self.observability_sink,
            component_type="agent",
            agent_name=self.name,
        )
        self._observability_heartbeat = hb
        # Best-effort cleanup on GC/interpreter shutdown without altering stop() behavior:
        # if the HeartbeatHandle becomes unreachable, ensure we stop its background thread.
        #
        # This `weakref.finalize(...)` pattern is common as a safety net for background
        # resources (threads, sockets, temp files) when callers might forget to call
        # an explicit shutdown method. Caveats:
        # - It is not a replacement for explicit shutdown (cleanup timing is not deterministic).
        # - It may not run on hard-kill exits (e.g., SIGKILL / abrupt process termination).
        if hb is not None:
            weakref.finalize(hb, hb.stop)

    def attach_connectors(self, *connectors: object) -> None:
        """Annotate connectors with agent context without changing their constructors."""
        for conn in connectors:
            try:
                if isinstance(conn, BaseConnector):
                    conn.attach_observability(
                        agent_name=self.name, sink=self.observability_sink
                    )
                else:
                    setattr(conn, "agent_name", self.name)
            except Exception:
                # Best effort; do not raise
                continue

    def __setattr__(self, name: str, value: object) -> None:
        """Automatically inject agent context into assigned connectors."""
        # Use super().__setattr__ first to ensure the attribute is set
        super().__setattr__(name, value)

        # Then inspect and inject if it looks like a connector
        # We avoid importing BaseConnector to prevent circular imports, use duck typing
        if isinstance(value, BaseConnector):
            value.attach_observability(
                agent_name=self.name, sink=self.observability_sink
            )

    @abstractmethod
    def run(self):
        """Starts the agent's main execution loop.
        In some cases, concrete run implementation may not be needed.
        In that case use pass as a placeholder."""
        pass

    @abstractmethod
    def stop(self):
        """Gracefully terminates the agent's execution and cleans up resources."""
        pass
