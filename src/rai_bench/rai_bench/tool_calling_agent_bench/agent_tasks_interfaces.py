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

import logging
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel

loggers_type = logging.Logger


class Result(BaseModel):
    success: bool = False
    errors: list[str] = []


class ToolCallingAgentTask(ABC):
    """
    Abstract class for tool calling agent tasks. Contains methods for requested tool calls verification.
    """

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.expected_tools: List[BaseTool] = []
        self.result = Result()

    @abstractmethod
    def get_prompt(self) -> str:
        """Returns the task instruction - the prompt that will be passed to agent"""
        pass

    @abstractmethod
    def verify_tool_calls(self, response: dict[str, Any]):
        pass

    def _check_tool_call(
        self, tool_call: ToolCall, expected_name: str, expected_args: dict[str, Any]
    ) -> bool:
        """
        Helper method to check if a tool call has the expected name and arguments.

        Args:
            tool_call: The tool call to check
            expected_name: The expected name of the tool
            expected_args: The expected arguments dictionary

        Returns:
            bool: True if the tool call matches the expected name and args, False otherwise
        """
        error_occurs = False
        if tool_call["name"] != expected_name:
            error_msg = f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
            self.logger.error(
                f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}."
            )
            error_occurs = True

        if tool_call["args"] != expected_args:
            self.logger.error(
                f"Expected args for tool call should be {expected_args}, but got {tool_call['args']}."
            )
            error_occurs = True
        if error_occurs:
            return False
        return True

    def _check_multiple_tool_calls(
        self, message: AIMessage, expected_tool_calls: list[dict[str, Any]]
    ) -> bool:
        """
        Helper method to check multiple tool calls in a single AIMessage.

        Args:
            message: The AIMessage to check
            expected_calls: A list of dictionaries, each containing expected 'name' and 'args' for a tool call

        Returns:
            bool: True if all tool calls match expected patterns, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(
            message, len(expected_tool_calls)
        ):
            return False

        matched_calls = [False] * len(expected_tool_calls)
        error_occurs = False

        for tool_call in message.tool_calls:
            found_match = False

            for i, expected in enumerate(expected_tool_calls):
                if matched_calls[i]:
                    continue

                if (
                    tool_call["name"] == expected["name"]
                    and tool_call["args"] == expected["args"]
                ):
                    matched_calls[i] = True
                    found_match = True
                    break

            if not found_match:
                error_msg = f"Tool call {tool_call['name']} with args {tool_call['args']} does not match any expected call"
                self.logger.error(error_msg)
                self.result.errors.append(error_msg)
                error_occurs = True

        return not error_occurs

    def _check_tool_calls_num_in_ai_message(
        self, message: AIMessage, expected_num: int
    ) -> bool:
        """
        Helper method to check number of tool calls in a single AIMessage.

        Args:
            message: The AIMessage to check
            expected_num: The expected number of tool calls

        Returns:
            bool: True if the number of tool calls in the message matches the expected number, False otherwise
        """
        if len(message.tool_calls) != expected_num:
            error_msg = f"Expected number of tool calls should be {expected_num}, but got {len(message.tool_calls)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
            return False
        return True


class ROS2ToolCallingAgentTask(ToolCallingAgentTask, ABC):
    """
    Abstract class for ROS2 agent tasks.
    """

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger)

    def _is_ai_message_requesting_get_ros2_topics_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the only tool
        to get ROS2 topics names and types correctly.
        """
        error_occurs = False
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            error_occurs = True

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_topics_names_and_types",
            expected_args={},
        ):
            error_occurs = True
        if error_occurs:
            return False
        return True
