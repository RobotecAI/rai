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

"""A chat model that plays tool calls, and the setup functions of `rai mcp serve` tests."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Plays `steps` in order: a final text, or a list of (tool name, args) calls."""

    steps: list[Any]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, str):
            message = AIMessage(step)
        else:
            calls = [
                {"name": n, "args": a, "id": f"call_{len(messages)}_{i}"}
                for i, (n, a) in enumerate(step)
            ]
            message = AIMessage("", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


def deny_motion(server: Any, connector: Any) -> None:
    """`rai mcp serve --setup tests/rai_mcp/scripted.py:deny_motion`: no MOTION egress."""
    from rai.communication.ros2.egress import MOTION, Deny

    def no_motion(egress):
        return Deny("motion is off in this test")

    connector.egress.add(no_motion, types=MOTION)
