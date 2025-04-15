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
        # NOTE (jm) a bit weird logic with the expected_optional_args, take a look if can be done differently
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
            Optional arguments dictionary that can be present but don't need to be.
            Values define the expected type (use type or tuple of types, or None to accept any type)

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
                    raise SubTaskValidationError(
                        f"Expected argument '{arg_name}' should have value '{arg_value}', but got '{tool_call['args'][arg_name]}'"
                    )
            else:
                raise SubTaskValidationError(
                    f"Required argument '{arg_name}' missing in tool call {expected_name}."
                )

        # Check that no unexpected arguments are present (except for optional ones)
        for arg_name, arg_value in tool_call["args"].items():
            if arg_name not in expected_args:
                # If this argument is not required, check if it's an allowed optional argument
                if not expected_optional_args or arg_name not in expected_optional_args:
                    raise SubTaskValidationError(
                        f"Unexpected argument '{arg_name}' found in tool call {expected_name}."
                    )
                # If optional argument has an expected type, check if the value matches that type
                elif expected_optional_args[arg_name] is not None:
                    expected_type = expected_optional_args[arg_name]
                    if not isinstance(arg_value, expected_type):
                        raise SubTaskValidationError(
                            f"Optional argument '{arg_name}' has incorrect type. Expected {expected_type.__name__}, but got {type(arg_value).__name__} in tool call {expected_name}."
                        )
        return True

    def _check_topic_tool_call_field(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_topic: str,
        expected_message_type: str,
        field_path: str,
        expected_value: Any,
    ) -> bool:
        """
        Verifies a tool call for a topic publishing operation.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call dictionary containing keys such as "name" and "args".
        expected_name : str
            The expected tool call name (e.g., "publish_ros2_message").
        expected_topic : str
            The expected topic name in the tool call's arguments.
        expected_message_type : str
            The expected message type (e.g., "rai_interfaces/msg/HRIMessage").
        field_path : str
            Dot-separated path to the field inside the message (e.g., "header.frame_id").
        expected_value : Any
            The expected value at the given field path.

        Returns
        -------
        bool
            True if all conditions are met; False otherwise.
        """
        # Check tool call name.
        if tool_call.get("name") != expected_name:
            raise SubTaskValidationError(
                f"Expected tool call name '{expected_name}', but got '{tool_call.get('name')}'."
            )

        args = tool_call.get("args", {})

        # Check topic.
        if args.get("topic") != expected_topic:
            raise SubTaskValidationError(
                f"Expected topic '{expected_topic}', but got '{args.get('topic')}'."
            )

        # Check message type.
        if args.get("message_type") != expected_message_type:
            raise SubTaskValidationError(
                f"Expected message type '{expected_message_type}', but got '{args.get('message_type')}'."
            )

        # Traverse the message field.
        message = args.get("message")
        if message is None:
            raise SubTaskValidationError(
                "Tool call does not contain a 'message' argument."
            )

        keys = field_path.split(".")
        value: Any = message
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise SubTaskValidationError(
                    f"Field path '{field_path}' not found in the message."
                )

        if value != expected_value:
            raise SubTaskValidationError(
                f"Expected value for field '{field_path}' is '{expected_value}', but got '{value}'."
            )

        return True

    def _check_service_tool_call_field(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_service: str,
        expected_service_type: str,
        field_path: str,
        expected_value: Any,
    ) -> bool:
        """
        Verifies a tool call for a service call.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call dictionary containing keys such as "name" and "args".
        expected_name : str
            The expected tool call name (e.g., "call_ros2_service").
        expected_service : str
            The expected service name in the tool call's arguments.
        expected_message_type : str
            The expected message type.
        field_path : str
            Dot-separated path to the field inside the message.
        expected_value : Any
            The expected value at the given field path.

        Returns
        -------
        bool
            True if all conditions are met; False otherwise.
        """
        if tool_call.get("name") != expected_name:
            raise SubTaskValidationError(
                f"Expected tool call name '{expected_name}', but got '{tool_call.get('name')}'."
            )

        args = tool_call.get("args", {})

        # Check service.
        if args.get("service_name") != expected_service:
            raise SubTaskValidationError(
                f"Expected service '{expected_service}', but got '{args.get('service')}'."
            )

        # Check message type.
        if args.get("service_type") != expected_service_type:
            raise SubTaskValidationError(
                f"Expected message type '{expected_service_type}', but got '{args.get('service_type')}'."
            )

        service_args = args.get("service_args")
        if service_args is None:
            raise SubTaskValidationError(
                "Tool call does not contain a 'service_args' argument."
            )

        if field_path == "":
            if service_args == {}:
                return True
            else:
                raise SubTaskValidationError(
                    f"Expected empty service_args, but got: {service_args}"
                )

        keys = field_path.split(".")
        value: Any = service_args
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise SubTaskValidationError(
                    f"Field path '{field_path}' not found in the message."
                )

        if value != expected_value:
            raise SubTaskValidationError(
                f"Expected value for field '{field_path}' is '{expected_value}', but got '{value}'."
            )

        return True

    def _check_action_tool_call_field(
        self,
        tool_call: ToolCall,
        expected_name: str,
        expected_action: str,
        expected_action_type: str,
        field_path: str,
        expected_value: Any,
    ) -> bool:
        """
        Verifies a tool call for an action call.

        Parameters
        ----------
        tool_call : ToolCall
            The tool call dictionary containing keys such as "name" and "args".
        expected_name : str
            The expected tool call name (e.g., "call_ros2_action").
        expected_action_name : str
            The expected action name in the tool call's arguments.
        expected_action_type : str
            The expected action type.
        field_path : str
            Dot-separated path to the field inside the goal.
        expected_value : Any
            The expected value at the given field path.

        Returns
        -------
        bool
            True if all conditions are met; False otherwise.
        """
        if tool_call.get("name") != expected_name:
            raise SubTaskValidationError(
                f"Expected tool call name '{expected_name}', but got '{tool_call.get('name')}'."
            )

        args = tool_call.get("args", {})

        # Check action name
        if args.get("action_name") != expected_action:
            raise SubTaskValidationError(
                f"Expected action name '{expected_action}', but got '{args.get('action_name')}'."
            )

        # Check action type
        if args.get("action_type") != expected_action_type:
            raise SubTaskValidationError(
                f"Expected action type '{expected_action_type}', but got '{args.get('action_type')}'."
            )

        action_args = args.get("action_args")
        if action_args is None:
            raise SubTaskValidationError(
                "Tool call does not contain an 'action_args' argument."
            )

        if field_path == "":
            if action_args == {}:
                return True
            else:
                raise SubTaskValidationError(
                    f"Expected empty action_args, but got: {action_args}"
                )

        # Traverse nested keys
        keys = field_path.split(".")
        value: Any = action_args
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise SubTaskValidationError(
                    f"Field path '{field_path}' not found in the action_args."
                )

        if value != expected_value:
            raise SubTaskValidationError(
                f"Expected value for field '{field_path}' is '{expected_value}', but got '{value}'."
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

    def validation_error(self, msg: str):
        """
        Call when error results in fail of the validator.
        """
        self.logger.error(msg)
        self.errors_queue.put(msg)

    def dump(self) -> List[Dict[str, Any]]:
        return [subt.dump() for subt in self.subtasks]

    def get_all_validation_errors(self):
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

    def fail_rest_of_validators(self):
        """
        All remaining validators are marked as failed.
        Can be used when validation cannot be continued for whatever reason
        """
        if len(self.result.passed) < len(self.validators):
            self.result.passed.extend(
                [False] * (len(self.validators) - len(self.result.passed))
            )

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
                self.result.errors.append(validator.get_all_validation_errors())
