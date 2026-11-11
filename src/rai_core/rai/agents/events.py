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

"""Typed events an agent emits while it works.

UIs, HRI output and recorders consume them with :meth:`BaseAgent.subscribe` or
:meth:`BaseAgent.events` instead of LangChain callbacks. A turn is one
:meth:`ReActAgent.ainvoke`; every event of a turn carries its id.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage

    from rai.agents.base import AgentState
    from rai.tools import Effect


@dataclass(frozen=True, kw_only=True)
class AgentEvent:
    agent: str
    t: float = field(default_factory=time.time)


@dataclass(frozen=True, kw_only=True)
class StateChanged(AgentEvent):
    old: "AgentState"
    new: "AgentState"


@dataclass(frozen=True, kw_only=True)
class TurnEvent(AgentEvent):
    turn: str


@dataclass(frozen=True, kw_only=True)
class ModelStart(TurnEvent):
    pass


@dataclass(frozen=True, kw_only=True)
class ModelEnd(TurnEvent):
    message: "AIMessage"


@dataclass(frozen=True, kw_only=True)
class ToolStart(TurnEvent):
    call_id: str
    name: str
    args: dict[str, Any]
    effect: "Effect"


@dataclass(frozen=True, kw_only=True)
class ToolEnd(TurnEvent):
    call_id: str
    name: str
    effect: "Effect"
    status: Literal["success", "error", "denied"]
    content: Any


@dataclass(frozen=True, kw_only=True)
class FinalMessage(TurnEvent):
    message: "AIMessage"


@dataclass(frozen=True, kw_only=True)
class Interrupted(TurnEvent):
    pass


@dataclass(frozen=True, kw_only=True)
class AgentError(TurnEvent):
    error: BaseException
