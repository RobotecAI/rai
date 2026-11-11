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

"""The manifest for agents: in the system prompt (middleware) or as a READ tool."""

import logging
import time
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from rai.tools import Effect, RaiTool

from .manifest import ManifestSource

logger = logging.getLogger(__name__)


def with_manifest(system: SystemMessage | str | None, text: str) -> SystemMessage:
    """`system` with the manifest text appended."""
    if system is None:
        return SystemMessage(text)
    if isinstance(system, str):
        return SystemMessage(f"{system}\n\n{text}")
    if isinstance(system.content, str):
        return SystemMessage(f"{system.content}\n\n{text}")
    return SystemMessage([*system.content, {"type": "text", "text": text}])


class ManifestPrompt(AgentMiddleware[Any, Any]):
    """Appends the robot manifest to the system prompt of every model call.

    The manifest is never written into the history, so a refresh replaces it
    instead of adding to it. Before a model call the graph is checked for
    changes at most every `min_interval_s`, and the manifest is rebuilt only
    when it changed. A failed check keeps the last manifest.
    """

    def __init__(self, source: ManifestSource, *, min_interval_s: float = 1.0):
        super().__init__()
        self.source = source
        self.min_interval_s = min_interval_s
        self._text: str | None = None
        self._checked = float("-inf")

    def text(self) -> str:
        now = time.monotonic()
        if self._text is None or now - self._checked >= self.min_interval_s:
            self._checked = now
            try:
                self._text = self.source.render()
            except Exception as e:
                if self._text is None:
                    raise
                logger.warning("Manifest refresh failed, keeping the last: %r", e)
        return self._text

    def _request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        return request.override(
            system_message=with_manifest(request.system_message, self.text())
        )

    def wrap_model_call(self, request, handler):  # type: ignore[override]
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):  # type: ignore[override]
        return await handler(self._request(request))


class GetRobotManifestInput(BaseModel):
    format: Literal["text", "json"] = Field(
        "text", description="text: compact summary; json: every graph entry"
    )


class GetRobotManifestTool(RaiTool):
    """Pull the current manifest (rebuilt when the graph changed)."""

    default_effect = Effect.READ
    name: str = "get_robot_manifest"
    description: str = (
        "Get the robot manifest: what the robot is, its limits and safety notes,"
        " its capabilities, and the live ROS 2 topics, services and actions with"
        " their types and which of them move hardware (MOTION)."
    )
    args_schema: type[BaseModel] = GetRobotManifestInput
    source: ManifestSource = Field(exclude=True)

    async def _acall(self, format: str = "text") -> str:
        if format == "json":
            return self.source.current().model_dump_json(indent=1)
        return self.source.render()
