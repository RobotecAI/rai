# Copyright (C) 2026 Robotec.AI
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

"""Offline regression tests for ToolRunner CWE-209 exception sanitization."""

from logging import Logger
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from rai.agents.langchain.core.tool_runner import ToolRunner


@tool
def leaky_tool(x: str) -> str:
    """A tool that raises with a sensitive message."""
    raise RuntimeError(
        "DB connection failed: postgres://admin:SUPER_SECRET_PW@10.0.0.5:5432/prod"
    )


def test_tool_runner_does_not_leak_exception_details_to_llm():
    logger = MagicMock(spec=Logger)
    runner = ToolRunner(tools=[leaky_tool], logger=logger)
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "leaky_tool",
                        "args": {"x": "hi"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
    result = runner.invoke(state)
    msg = result["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    content = msg.content
    assert "SUPER_SECRET_PW" not in content
    assert "10.0.0.5" not in content
    assert "postgres://" not in content
    assert "leaky_tool" in content
    assert "RuntimeError" in content
    logger.exception.assert_called()


def test_tool_runner_success_path_unchanged():
    @tool
    def ok_tool(x: str) -> str:
        """Return ok."""
        return f"ok:{x}"

    runner = ToolRunner(tools=[ok_tool], logger=MagicMock(spec=Logger))
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ok_tool",
                        "args": {"x": "hi"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
    result = runner.invoke(state)
    msg = result["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status != "error"
    assert "ok:hi" in str(msg.content)
