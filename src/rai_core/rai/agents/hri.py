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

"""Agent output to a human, driven by the agent's event stream.

::

    agent.subscribe(HRIOutput(hri_connector_sender(hri_connector, "/to_human")))
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rai.agents.events import (
    AgentError,
    AgentEvent,
    FinalMessage,
    Interrupted,
    ModelEnd,
    ToolEnd,
    ToolStart,
)


@dataclass(frozen=True)
class HRIChunk:
    """One piece of what the agent says in a turn. ``end`` marks its last piece."""

    text: str
    turn: str
    seq: int
    end: bool


class HRIOutput:
    """Sends the agent's text to ``send``: what the model says while it uses
    tools, then the final answer with ``end=True``. An interrupted or failed turn
    that already said something ends with an empty chunk.

    With ``tool_status=True`` tool starts, errors and denials are sent as well.
    """

    def __init__(
        self, send: Callable[[HRIChunk], None], *, tool_status: bool = False
    ) -> None:
        self.send = send
        self.tool_status = tool_status
        self._seq: dict[str, int] = {}

    def _say(self, turn: str, text: str, end: bool = False) -> None:
        seq = self._seq.pop(turn, 0)
        if not end:
            self._seq[turn] = seq + 1
        self.send(HRIChunk(text, turn, seq, end))

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case ModelEnd(message=m) if m.tool_calls and m.text:
                self._say(event.turn, m.text)
            case FinalMessage(message=m):
                self._say(event.turn, m.text, end=True)
            case Interrupted() | AgentError() if event.turn in self._seq:
                self._say(event.turn, "", end=True)
            case ToolStart(name=name) if self.tool_status:
                self._say(event.turn, f"Running {name}.")
            case ToolEnd(status="error" | "denied") if self.tool_status:
                self._say(event.turn, f"{event.name}: {event.content}")


def hri_connector_sender(connector: Any, target: str) -> Callable[[HRIChunk], None]:
    """A ``send`` for an :class:`~rai.communication.HRIConnector` such as
    ``ROS2HRIConnector``: publishes each chunk to ``target`` as an AI message."""
    from langchain_core.messages import AIMessage

    def send(chunk: HRIChunk) -> None:
        message = connector.build_message(
            AIMessage(chunk.text), chunk.turn, chunk.seq, chunk.end
        )
        connector.send_message(message, target)

    return send
