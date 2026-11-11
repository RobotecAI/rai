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

"""Base classes of the ROS 2 tools: connector, access lists, resolved targets."""

from typing import Any, ClassVar, Dict, List, Mapping, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import ConfigDict, Field

from rai.communication.ros2 import GoalHandle, ROS2Connector, compat
from rai.communication.ros2.api.conversion import ros2_message_to_dict
from rai.communication.ros2.runtime import message_class
from rai.tools.base import RaiTool


def from_dict(msg_type: Any, payload: Optional[Dict[str, Any]]) -> Any:
    """A native ROS message of msg_type (class or "pkg/msg/Name") from an LLM dict."""
    return compat.set_message_fields(message_class(msg_type)(), payload or {})


def to_dict(msg: Any) -> Dict[str, Any]:
    """A ROS message as a plain dict for the LLM."""
    return ros2_message_to_dict(msg)


def type_name(msg: Any) -> str:
    """ "pkg/msg/Name" of a ROS message instance."""
    package, kind = type(msg).__module__.split(".")[:2]
    return f"{package}/{kind}/{type(msg).__name__}"


class BaseROS2Tool(RaiTool):
    """A RAI tool on a ROS2Connector.

    readable, writable and forbidden are compared on resolved names, so
    "cmd_vel", "/cmd_vel" and remapped forms match. None means no restriction;
    forbidden wins over both lists.

    Targets are resolved names: `target_fields` maps tool fields (configured
    names, e.g. an action name) to their kind, `target_args` maps call
    arguments (names chosen by the LLM) to theirs.
    """

    target_fields: ClassVar[Mapping[str, str]] = {}
    target_args: ClassVar[Mapping[str, str]] = {}

    connector: ROS2Connector = Field(exclude=True)
    readable: Optional[List[str]] = None
    writable: Optional[List[str]] = None
    forbidden: Optional[List[str]] = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.target_fields and not self.targets:
            self.metadata["targets"] = tuple(  # type: ignore[index]
                self.connector.resolve(getattr(self, field), kind)
                for field, kind in self.target_fields.items()
            )

    def call_targets(self, kwargs: Mapping[str, Any]) -> tuple[str, ...]:
        return self.targets + tuple(
            self.connector.resolve(kwargs[arg], kind)
            for arg, kind in self.target_args.items()
            if kwargs.get(arg)
        )

    def _contains(self, names: List[str], name: str, kind: str) -> bool:
        resolved = self.connector.resolve(name, kind)
        return any(self.connector.resolve(n, kind) == resolved for n in names)

    def _forbidden(self, name: str, kind: str = "topic") -> bool:
        return self.forbidden is not None and self._contains(self.forbidden, name, kind)

    def is_readable(self, topic: str) -> bool:
        if self._forbidden(topic):
            return False
        return self.readable is None or self._contains(self.readable, topic, "topic")

    def is_writable(self, name: str, service: bool = False) -> bool:
        """Check a topic, action (service=False) or service (service=True) name."""
        kind = "service" if service else "topic"
        if self._forbidden(name, kind):
            return False
        return self.writable is None or self._contains(self.writable, name, kind)

    def check_readable(self, topic: str) -> None:
        if not self.is_readable(topic):
            raise ValueError(f"Topic {topic} is not readable")

    def check_writable(self, name: str, kind: str = "topic") -> None:
        if not self.is_writable(name, service=kind == "service"):
            raise ValueError(f"{kind.capitalize()} {name} is not writable")


class GoalRegistry:
    """Goals a toolkit started, by the id the LLM sees. Keeps at most maxlen;
    finished goals are forgotten first."""

    def __init__(self, maxlen: int = 32) -> None:
        self.maxlen = maxlen
        self._goals: Dict[str, GoalHandle] = {}

    def add(self, handle: GoalHandle) -> str:
        self._goals[handle.id] = handle
        while len(self._goals) > self.maxlen:
            done = [k for k, h in self._goals.items() if h.done]
            del self._goals[done[0] if done else next(iter(self._goals))]
        return handle.id

    def get(self, goal_id: str) -> GoalHandle:
        try:
            return self._goals[goal_id]
        except KeyError:
            raise ValueError(f"Unknown action ID: {goal_id}") from None

    def latest(self) -> Optional[GoalHandle]:
        return next(reversed(self._goals.values()), None)

    def ids(self) -> List[str]:
        return list(self._goals)


class BaseROS2Toolkit(BaseToolkit):
    """Builds tools that share a connector and the readable, writable and
    forbidden lists (see BaseROS2Tool)."""

    connector: ROS2Connector
    readable: Optional[List[str]] = None
    writable: Optional[List[str]] = None
    forbidden: Optional[List[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build(self, *tools: type[BaseTool], **kwargs: Any) -> List[BaseTool]:
        access = dict(
            connector=self.connector,
            readable=self.readable,
            writable=self.writable,
            forbidden=self.forbidden,
        )
        return [tool(**access, **kwargs) for tool in tools]
