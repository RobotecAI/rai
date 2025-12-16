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
from abc import ABC, abstractmethod
from typing import Optional

from rai.communication.base_connector import BaseConnector
from rai.observability import ObservabilitySink, build_sink_from_env


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

    def attach_connectors(self, *connectors: object) -> None:
        """Annotate connectors with agent context without changing their constructors."""
        for conn in connectors:
            try:
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
            value.agent_name = self.name
            value.connector_name = value.__class__.__name__
            value.observability_sink = self.observability_sink

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
