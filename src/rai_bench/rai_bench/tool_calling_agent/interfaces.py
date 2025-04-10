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
import queue
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Dict, List, Literal, Tuple

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from langchain_core.tools import BaseTool

loggers_type = logging.Logger


class Result:
    def __init__(self):
        # bool for every validator
        self.passed: List[bool] = []
        # list for every validator
        self.errors: List[List[str]] = [[]]

    @property
    def score(self) -> float:
        """
        Counted as number of validators
        passed divided by numer of all validators
        """
        if self.passed:
            return sum(self.passed) / len(self.passed)
        else:
            return 0.0


class SubTaskValidationError(Exception):
    pass


class SubTask(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def validate(self, tool_call: ToolCall) -> bool:
        pass

    @abstractmethod
    def dump(self) -> Dict[str, Any]:
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
            raise SubTaskValidationError(
                f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}"
            )

        # Check that all required arguments are present and have the expected values
        for arg_name, arg_value in expected_args.items():
            if arg_name in tool_call["args"]:
                if tool_call["args"][arg_name] != arg_value:
                    SubTaskValidationError(
                        f"Expected argument '{arg_name}' should have value '{arg_value}', but got '{tool_call['args'][arg_name]}'"
                    )
            else:
                SubTaskValidationError(
                    f"Required argument '{arg_name}' missing in tool call {expected_name}."
                )

        # Check that no unexpected arguments are present (except for optional ones)
        for arg_name, arg_value in tool_call["args"].items():
            if arg_name not in expected_args:
                # If this argument is not required, check if it's an allowed optional argument
                if not expected_optional_args or arg_name not in expected_optional_args:
                    SubTaskValidationError(
                        f"Unexpected argument '{arg_name}' found in tool call {expected_name}."
                    )
                # If optional argument has expected value, check if the value is correct
                elif expected_optional_args[arg_name]:
                    if expected_optional_args[arg_name] != arg_value:
                        SubTaskValidationError(
                            f"Optional argument '{arg_name}' has incorrect value '{arg_value}' in tool call {expected_name}."
                        )
        return True


class Validator(ABC):
    def __init__(
        self, subtasks: List[SubTask], logger: loggers_type | None = None
    ) -> None:
        self.subtasks = subtasks
        self.errors_queue: queue.Queue[str] = Queue()
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def log_error(self, msg: str):
        self.logger.error(msg)
        self.errors_queue.put(msg)

    def dump(self) -> List[Dict[str, Any]]:
        return [subt.dump() for subt in self.subtasks]

    def get_all_errors(self):
        """Get all errors from queue"""
        errors: List[str] = []
        while not self.errors_queue.empty():
            errors.append(self.errors_queue.get())
        return errors

    @abstractmethod
    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        pass


class Task(ABC):
    complexity: Literal["easy", "medium", "hard"]
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    def __init__(
        self,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        """

        Parameters
        ----------
        validators : List[Validator]
            Every validator can be treated as single step of validation.
        extra_tool_calls : int
            How many extra tool calls agent can make to still pass test
        """
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.validators = validators
        self.extra_tool_calls = extra_tool_calls
        self.result = Result()

    def set_logger(self, logger: loggers_type):
        self.logger = logger
        for validator in self.validators:
            validator.logger = logger

    def get_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extracts all tool calls from the response, flattened across all AI messages."""
        tool_calls: List[ToolCall] = []
        for msg in response["messages"]:
            if isinstance(msg, AIMessage):
                tool_calls.extend(msg.tool_calls)
        return tool_calls

    def dump_validators(self) -> List[List[Dict[str, Any]]]:
        return [val.dump() for val in self.validators]

    @property
    @abstractmethod
    def available_tools(self) -> List[BaseTool]:
        """List of tool available for the agent"""
        pass

    @property
    def required_calls(self):
        """Minimal number of calls required to complete task"""
        total = 0
        for val in self.validators:
            total += len(val.subtasks)
        return total

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

    def validate(self, tool_calls: List[ToolCall]):
        self.logger.debug(
            f"required_calls: {self.required_calls}, extra_calls {self.extra_tool_calls}"
        )
        remaining_tool_calls = tool_calls[
            : self.required_calls + self.extra_tool_calls
        ].copy()
        self.logger.debug(f"Tool calls to validate: {remaining_tool_calls}")

        done_properly = 0
        for validator in self.validators:
            if_success, remaining_tool_calls = validator.validate(
                tool_calls=remaining_tool_calls
            )
            if if_success:
                done_properly += 1
                self.result.passed.append(True)
            else:
                self.result.passed.append(False)
                # get all errors from queue
                self.result.errors.append(validator.get_all_errors())

    # def _check_multiple_tool_calls(
    #     self, tool_calls: List[ToolCall], expected_tool_calls: list[dict[str, Any]]
    # ) -> bool:
    #     matched_calls = [False] * len(expected_tool_calls)
    #     error_occurs = False

    #     for tool_call in tool_calls:
    #         found_match = False

    #         for i, expected in enumerate(expected_tool_calls):
    #             if matched_calls[i]:
    #                 continue

    #             expected_name = expected["name"]
    #             expected_args = expected["args"]
    #             expected_optional_args = expected.get("optional_args", {})

    #             if self._check_tool_call(
    #                 tool_call=tool_call,
    #                 expected_name=expected_name,
    #                 expected_args=expected_args,
    #                 expected_optional_args=expected_optional_args,
    #             ):
    #                 matched_calls[i] = True
    #                 found_match = True
    #                 break

    #         if not found_match:
    #             self.log_error(
    #                 msg=f"Tool call {tool_call['name']} with args {tool_call['args']} does not match any expected call"
    #             )
    #             error_occurs = True

    #     return not error_occurs

    # def _is_ai_message_requesting_get_ros2_topics_and_types(
    #     self, tool_call: ToolCall
    # ) -> bool:
    #     if not self._check_tool_call(
    #         tool_call=tool_call,
    #         expected_name="get_ros2_topics_names_and_types",
    #         expected_args={},
    #     ):
    #         return False
    #     return True
