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

"""Tests for CWE-209: Exception detail leakage in ToolRunner error handling.

Verifies that internal exception details (file paths, connection strings,
stack traces) are NOT leaked in ToolMessage content sent to the LLM/user.
"""

import logging

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rai.agents.langchain.core import ToolRunner


# --- Tools that simulate failures with sensitive information ---


@tool
def tool_with_file_path_error() -> str:
    """A tool that fails exposing internal file paths."""
    raise FileNotFoundError(
        "/home/robot/secrets/config.yaml: No such file or directory"
    )


@tool
def tool_with_connection_error() -> str:
    """A tool that fails exposing connection details."""
    raise ConnectionError(
        "Failed to connect to database at postgres://admin:s3cret@10.0.0.5:5432/robotdb"
    )


@tool
def tool_with_runtime_error() -> str:
    """A tool that fails with a generic runtime error."""
    raise RuntimeError(
        "Internal server error in module /opt/rai/src/rai_core/rai/tools/internal.py:42"
    )


class StrictInput(BaseModel):
    x: float = Field(..., description="Required numeric input")


@tool(args_schema=StrictInput)
def tool_with_validation_schema(x: float) -> str:
    """A tool with strict validation that will fail on bad input."""
    return f"Result: {x}"


# --- Tests ---


def _run_tool_and_get_error_message(tool_fn, tool_call: ToolCall) -> str:
    """Helper: runs a single tool call and returns the error ToolMessage content."""
    runner = ToolRunner(tools=[tool_fn], logger=logging.getLogger(__name__))
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    output = runner.invoke(state)
    # The error message is the last ToolMessage
    tool_msgs = [m for m in output["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1, "Expected exactly one ToolMessage"
    assert tool_msgs[0].status == "error", "Expected error status"
    return tool_msgs[0].content


def test_file_path_not_leaked():
    """Exception containing file paths must not appear in ToolMessage content."""
    call = ToolCall(
        name="tool_with_file_path_error", args={}, id="test-file-path-leak"
    )
    content = _run_tool_and_get_error_message(tool_with_file_path_error, call)
    assert "/home/robot" not in content, (
        f"Internal file path leaked in error message: {content}"
    )
    assert "config.yaml" not in content, (
        f"Internal file name leaked in error message: {content}"
    )


def test_connection_string_not_leaked():
    """Exception containing connection strings must not appear in ToolMessage content."""
    call = ToolCall(
        name="tool_with_connection_error", args={}, id="test-conn-string-leak"
    )
    content = _run_tool_and_get_error_message(tool_with_connection_error, call)
    assert "postgres://" not in content, (
        f"Connection string leaked in error message: {content}"
    )
    assert "s3cret" not in content, (
        f"Password leaked in error message: {content}"
    )
    assert "10.0.0.5" not in content, (
        f"Internal IP leaked in error message: {content}"
    )


def test_internal_module_path_not_leaked():
    """Exception containing internal module paths must not appear in ToolMessage content."""
    call = ToolCall(
        name="tool_with_runtime_error", args={}, id="test-module-path-leak"
    )
    content = _run_tool_and_get_error_message(tool_with_runtime_error, call)
    assert "/opt/rai" not in content, (
        f"Internal module path leaked in error message: {content}"
    )
    assert "internal.py" not in content, (
        f"Internal file name leaked in error message: {content}"
    )


def test_error_message_is_generic():
    """Error messages should be generic and not contain raw exception details."""
    call = ToolCall(
        name="tool_with_file_path_error", args={}, id="test-generic-msg"
    )
    content = _run_tool_and_get_error_message(tool_with_file_path_error, call)
    # Should contain tool name so the LLM knows which tool failed
    assert "tool_with_file_path_error" in content, (
        f"Error message should reference the tool name: {content}"
    )


def test_validation_error_no_raw_details():
    """ValidationError should not leak raw pydantic error details to ToolMessage."""
    # Pass wrong type to trigger validation error
    call = ToolCall(
        name="tool_with_validation_schema",
        args={"x": "not_a_number"},
        id="test-validation-leak",
    )
    content = _run_tool_and_get_error_message(tool_with_validation_schema, call)
    # Should not contain internal pydantic error JSON with input values
    assert "not_a_number" not in content, (
        f"User input value leaked back in validation error: {content}"
    )


def test_unknown_tool_error_no_raw_exception():
    """Calling a non-existent tool should not leak raw exception messages."""
    call = ToolCall(name="nonexistent_tool", args={}, id="test-keyerror")
    runner = ToolRunner(
        tools=[tool_with_file_path_error], logger=logging.getLogger(__name__)
    )
    state = {"messages": [AIMessage(content="", tool_calls=[call])]}
    output = runner.invoke(state)
    tool_msgs = [m for m in output["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    content = tool_msgs[0].content
    # Should mention the tool name
    assert "nonexistent_tool" in content, (
        f"Error message should reference the tool name: {content}"
    )
    # Should not contain raw str(e) value (for KeyError, str(e) = "'nonexistent_tool'")
    # The generic message should not embed repr of exception details
    assert "Traceback" not in content, (
        f"Stack trace leaked in error message: {content}"
    )
