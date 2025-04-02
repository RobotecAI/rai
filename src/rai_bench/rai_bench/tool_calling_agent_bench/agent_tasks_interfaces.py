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
from typing import Any, Dict, List, Literal, Sequence

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

    complexity: Literal["easy", "medium", "hard"]
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

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
            self.log_error(
                msg=f"Expected tool call name should be '{expected_name}', but got {tool_call['name']}"
            )
            return False

        # Check that all required arguments are present and have the expected values
        for arg_name, arg_value in expected_args.items():
            if arg_name in tool_call["args"]:
                if tool_call["args"][arg_name] != arg_value:
                    self.log_error(
                        msg=f"Expected argument '{arg_name}' should have value '{arg_value}', but got '{tool_call['args'][arg_name]}'"
                    )
                    return False
            else:
                self.log_error(
                    msg=f"Required argument '{arg_name}' missing in tool call {expected_name}."
                )
                return False

        # Check that no unexpected arguments are present (except for optional ones)
        for arg_name, arg_value in tool_call["args"].items():
            if arg_name not in expected_args:
                # If this argument is not required, check if it's an allowed optional argument
                if not expected_optional_args or arg_name not in expected_optional_args:
                    self.log_error(
                        msg=f"Unexpected argument '{arg_name}' found in tool call {expected_name}."
                    )
                    return False
                # If optional argument has expected value, check if the value is correct
                elif expected_optional_args[arg_name]:
                    if expected_optional_args[arg_name] != arg_value:
                        self.log_error(
                            msg=f"Optional argument '{arg_name}' has incorrect value '{arg_value}' in tool call {expected_name}."
                        )
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
                self.log_error(
                    msg=f"Tool call {tool_call['name']} with args {tool_call['args']} does not match any expected call"
                )
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
            self.log_error(
                msg=f"Expected number of tool calls should be {expected_num}, but got {len(message.tool_calls)}"
            )
            return False
        return True

    def log_error(self, msg: str):
        self.logger.error(msg)
        self.result.errors.append(msg)


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

    def _is_ai_message_requesting_get_ros2_services_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 service names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_service_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_services_names_and_types",
            expected_args={},
        ):
            return False
        return True

    def _is_ai_message_requesting_get_ros2_actions_and_types(
        self, ai_message: AIMessage
    ) -> bool:
        """Helper method to check if the given AIMessage is calling the exactly one tool that gets ROS2 actions names and types correctly.

        Parameters
        ----------
        ai_message : AIMessage
            The AIMessage to check

        Returns
        -------
        bool
            True if the ai_message is requesting get_ros2_actions_names_and_types correctly, False otherwise
        """
        if not self._check_tool_calls_num_in_ai_message(ai_message, expected_num=1):
            return False

        tool_call: ToolCall = ai_message.tool_calls[0]
        if not self._check_tool_call(
            tool_call=tool_call,
            expected_name="get_ros2_actions_names_and_types",
            expected_args={},
        ):
            return False
        return True


class CustomInterfacesTopicTask(ROS2ToolCallingAgentTask, ABC):
    TOPICS_AND_TYPES: Dict[str, str] = {
        # sample topics
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
        # custom topics
        "/to_human": "rai_interfaces/msg/HRIMessage",
        "/send_audio": "rai_interfaces/msg/AudioMessage",
        "/send_detections": "rai_interfaces/msg/RAIDetectionArray",
    }
    topic_strings = [
        f"topic: {topic}\ntype: {msg_type}\n"
        for topic, msg_type in TOPICS_AND_TYPES.items()
    ]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        # self.expected_message_type = TOPICS_AND_TYPES[self.expected_topic]

    # def get_system_prompt(self) -> str:
    #     return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    @abstractmethod
    def expected_topic(self) -> str:
        pass

    @property
    @abstractmethod
    def expected_message(self) -> Dict[str, Any]:
        pass

    @property
    def expected_message_type(self) -> str:
        return self.TOPICS_AND_TYPES[self.expected_topic]

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the topics names and types to recognize what type of message to_human topic has
        2. The tool that retrieves interfaces to check HRIMessage type
        3. The tool to publish message with proper topic, message type and content

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": self.expected_message_type},
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="publish_ros2_message",
                    expected_args={
                        "topic": self.expected_topic,
                        "message": self.expected_message,
                        "message_type": self.expected_message_type,
                    },
                )
        if not self.result.errors:
            self.result.success = True


class CustomInterfacesServiceTask(ROS2ToolCallingAgentTask):
    SERVICES_AND_TYPES = {
        # sample interfaces
        "/load_map": "moveit_msgs/srv/LoadMap",
        "/query_planner_interface": "moveit_msgs/srv/QueryPlannerInterfaces",
        # custom interfaces
        "/manipulator_move_to": "rai_interfaces/srv/ManipulatorMoveTo",
        "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
        "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
        "/get_log_digest": "rai_interfaces/srv/StringList",
        "/rai_whoami_documentation_service": "rai_interfaces/srv/VectorStoreRetrieval",
        "rai/whatisee/get": "rai_interfaces/srv/WhatISee",
    }
    service_strings = [
        f"service: {service}\ntype: {msg_type}\n"
        for service, msg_type in SERVICES_AND_TYPES.items()
    ]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

    @property
    @abstractmethod
    def expected_service(self) -> str:
        pass

    @property
    @abstractmethod
    def expected_message(self) -> Dict[str, Any]:
        pass

    @property
    def expected_service_type(self) -> str:
        return self.SERVICES_AND_TYPES[self.expected_service]

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the topics names and types to recognize what type of message to_human topic has
        2. The tool that retrieves interfaces to check HRIMessage type
        3. The tool to publish message with proper topic, message type and content

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_services_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": self.expected_service_type},
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="call_ros2_service",
                    expected_args={
                        "topic": self.expected_service,
                        "message": self.expected_message,
                        "message_type": self.expected_service_type,
                    },
                )
        if not self.result.errors:
            self.result.success = True


class CustomInterfacesActionTask(ROS2ToolCallingAgentTask):
    ACTIONS_AND_TYPES = {
        # custom actions
        "/perform_task": "rai_interfaces/action/Task",
        # some sample actions
        # "/execute_trajectory": "moveit_msgs/action/ExecuteTrajectory",
        # "/move_action": "moveit_msgs/action/MoveGroup",
        # "/follow_joint_trajectory": "control_msgs/action/FollowJointTrajectory",
        # "/gripper_cmd": "control_msgs/action/GripperCommand",
    }

    action_strings = [
        f"action: {action}\ntype: {msg_type}\n"
        for action, msg_type in ACTIONS_AND_TYPES.items()
    ]

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

    @property
    @abstractmethod
    def expected_action(self) -> str:
        pass

    @property
    @abstractmethod
    def expected_message(self) -> Dict[str, Any]:
        pass

    @property
    def expected_action_type(self) -> str:
        return self.ACTIONS_AND_TYPES[self.expected_action]

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the topics names and types to recognize what type of message to_human topic has
        2. The tool that retrieves interfaces to check HRIMessage type
        3. The tool to publish message with proper topic, message type and content

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        self.logger.debug(ai_messages)
        if len(ai_messages) != 4:
            self.log_error(
                msg=f"Expected exactly 4 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_actions_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )
        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_message_interface",
                    expected_args={"msg_type": self.expected_action_type},
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="start_ros2_action",
                    expected_args={
                        "topic": self.expected_action,
                        "message": self.expected_message,
                        "message_type": self.expected_action_type,
                    },
                )
        if not self.result.errors:
            self.result.success = True
