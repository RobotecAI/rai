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
from typing import Any, List, Literal

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from langchain_core.tools import BaseTool
from pydantic import BaseModel

loggers_type = logging.Logger


class Result(BaseModel):
    success: bool = False
    errors: list[str] = []


class ToolCallingAgentTask(ABC):
    """Abstract class for tool calling agent tasks. Contains methods for requested tool calls verification.

    Parameters
    ----------
    logger : loggers_type | None, optional
        Logger, by default None
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
        self._complexity: Literal["easy", "medium", "hard"]
        self._recursion_limit: int = DEFAULT_RECURSION_LIMIT

    @property
    @abstractmethod
    def complexity(self) -> Literal["easy", "medium", "hard"]:
        raise NotImplementedError

    @property
    def recursion_limit(self) -> int:
        """The number of allowed steps for agent.

        Returns
        -------
        int
            The number of allowed steps
        """
        return self._recursion_limit

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt that will be passed to agent

        Returns
        -------
        str
            System prompt
        """
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the task instruction - the prompt that will be passed to agent.

        Returns
        -------
        str
            Prompt
        """
        pass

    @abstractmethod
    def verify_tool_calls(self, response: dict[str, Any]):
        """Verify correctness of the tool calls from the agent's response.

        Note
        ----
        This method should set self.result.success to True if the verification is successful and append occuring errors related to verification to self.result.errors.

        Parameters
        ----------
        response : dict[str, Any]
            Agent's response
        """
        pass

    def _check_tool_call(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_args: dict[str, Any],
        expected_optional_args: dict[str, Any] = {},
    ) -> bool:
        """Helper method to check if a tool call has the expected name and arguments.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call to check
        expected_name : str
            The expected name of the tool
        expected_args : dict[str, Any]
            The expected arguments dictionary that must be present
        expected_optional_args : dict[str, Any], optional
            Optional arguments dictionary that can be present but don't need to be (e.g. timeout). If value of an optional argument does not matter, set it to {}

        Returns
        -------
        bool
            True if the tool call matches the expected name and args, False otherwise
        """
        if tool_call["name"] != expected_name:
            error_msg = f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
            return False

        # Check that all required arguments are present and have the expected values
        for arg_name, arg_value in expected_args.items():
            if arg_name in tool_call["args"]:
                if tool_call["args"][arg_name] != arg_value:
                    error_msg = f"Incorrect value {tool_call['args'][arg_name]} for argument '{arg_name}' in tool call {expected_name}."
                    self.logger.error(error_msg)
                    self.result.errors.append(error_msg)
                    return False
            else:
                error_msg = f"Required argument '{arg_name}' missing in tool call {expected_name}."
                self.logger.error(error_msg)
                self.result.errors.append(error_msg)
                return False

        # Check that no unexpected arguments are present (except for optional ones)
        for arg_name, arg_value in tool_call["args"].items():
            if arg_name not in expected_args:
                # If this argument is not required, check if it's an allowed optional argument
                if not expected_optional_args or arg_name not in expected_optional_args:
                    error_msg = f"Unexpected argument '{arg_name}' found in tool call {expected_name}."
                    self.logger.error(error_msg)
                    self.result.errors.append(error_msg)
                    return False
                # If optional argument has expected value, check if the value is correct
                elif expected_optional_args[arg_name]:
                    if expected_optional_args[arg_name] != arg_value:
                        error_msg = f"Optional argument '{arg_name}' has incorrect value '{arg_value}' in tool call {expected_name}."
                        self.logger.error(error_msg)
                        self.result.errors.append(error_msg)
                        return False

        return True

    def _check_multiple_tool_calls(
        self, message: AIMessage, expected_tool_calls: list[dict[str, Any]]
    ) -> bool:
        """Helper method to check multiple tool calls in a single AIMessage.

        Parameters
        ----------
        message : AIMessage
            The AIMessage to check
        expected_tool_calls : list[dict[str, Any]]
            A list of dictionaries, each containing expected 'name', 'args', and optional 'optional_args' for a tool call

        Returns
        -------
        bool
            True if all tool calls match expected patterns, False otherwise
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

                expected_name = expected["name"]
                expected_args = expected["args"]
                expected_optional_args = expected.get("optional_args", {})

                if self._check_tool_call(
                    tool_call=tool_call,
                    expected_name=expected_name,
                    expected_args=expected_args,
                    expected_optional_args=expected_optional_args,
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
        """Helper method to check number of tool calls in a single AIMessage.

        Parameters
        ----------
        message : AIMessage
            The AIMessage to check
        expected_num : int
            The expected number of tool calls

        Returns
        -------
        bool
            True if the number of tool calls in the message matches the expected number, False otherwise
        """
        if len(message.tool_calls) != expected_num:
            error_msg = f"Expected number of tool calls should be {expected_num}, but got {len(message.tool_calls)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
            return False
        return True


class ROS2ToolCallingAgentTask(ToolCallingAgentTask, ABC):
    """Abstract class for ROS2 related tasks for tool calling agent.

    Parameters
    ----------
    logger : loggers_type | None
        Logger for the task.
    """

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger)

    def _is_ai_message_requesting_get_ros2_topics_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 topics names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_topics_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_topics_names_and_types",
            expected_args={},
        ):
            return False
        return True
