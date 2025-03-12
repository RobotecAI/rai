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

from rai_bench.agent_bench.mocked_tools import (
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
)

loggers_type = logging.Logger


class Result(BaseModel):
    success: bool = False
    errors: list[str] = []


class AgentTask(ABC):
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


class ROS2AgentTask(AgentTask):
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


class GetROS2TopicsTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /display_contacts\ntype: visualization_msgs/msg/MarkerArray\n",
                    "topic: /display_planned_path\ntype: moveit_msgs/msg/DisplayTrajectory\n",
                    "topic: /execute_trajectory/_action/feedback\ntype: moveit_msgs/action/ExecuteTrajectory_FeedbackMessage\n",
                    "topic: /execute_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /joint_states\ntype: sensor_msgs/msg/JointState\n",
                    "topic: /monitored_planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /motion_plan_request\ntype: moveit_msgs/msg/MotionPlanRequest\n",
                    "topic: /move_action/_action/feedback\ntype: moveit_msgs/action/MoveGroup_FeedbackMessage\n",
                    "topic: /move_action/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/feedback\ntype: control_msgs/action/FollowJointTrajectory_FeedbackMessage\n",
                    "topic: /panda_arm_controller/follow_joint_trajectory/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/feedback\ntype: control_msgs/action/GripperCommand_FeedbackMessage\n",
                    "topic: /panda_hand_controller/gripper_cmd/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
                    "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
                    "topic: /planning_scene\ntype: moveit_msgs/msg/PlanningScene\n",
                    "topic: /planning_scene_world\ntype: moveit_msgs/msg/PlanningSceneWorld\n",
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /robot_description_semantic\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
                ]
            )
        ]

    def get_prompt(self) -> str:
        return "Get the names and types of all ROS2 topics"

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            error_msg = "No AI messages found in the response."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0])

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            error_msg = f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if not self.result.errors:
            self.result.success = True


class GetROS2RGBCameraTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
                ]
            ),
            MockGetROS2ImageTool(),
        ]

    def get_prompt(self) -> str:
        return "Get the RGB image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and not self._is_ai_message_requesting_get_ros2_topics_and_types(
            ai_messages[0]
        ):
            error_msg = (
                "First AI message did not request ROS2 topics and types correctly."
            )
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_image",
                    expected_args={"topic": "/camera_image_color"},
                )
        if not self.result.errors:
            self.result.success = True


class GetROS2DepthCameraTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                ]
            ),
            MockGetROS2ImageTool(),
        ]

    def get_prompt(self) -> str:
        return "Get the depth image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and not self._is_ai_message_requesting_get_ros2_topics_and_types(
            ai_messages[0]
        ):
            error_msg = (
                "First AI message did not request ROS2 topics and types correctly."
            )
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="get_ros2_image",
                    expected_args={"topic": "/camera_image_depth"},
                )
        if not self.result.errors:
            self.result.success = True


class GetAllROS2RGBCamerasTask(ROS2AgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
                    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
                    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
                ]
            ),
            MockGetROS2ImageTool(),
        ]

    def get_prompt(self) -> str:
        return "Get RGB images from all of the available cameras."

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and not self._is_ai_message_requesting_get_ros2_topics_and_types(
            ai_messages[0]
        ):
            error_msg = (
                "First AI message did not request ROS2 topics and types correctly."
            )
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if len(ai_messages) > 1:
            expected_tool_calls: list[dict[str, Any]] = [
                {"name": "get_ros2_image", "args": {"topic": "/camera_image_color"}},
                {"name": "get_ros2_image", "args": {"topic": "/color_image5"}},
            ]

            self._check_multiple_tool_calls(
                message=ai_messages[1], expected_tool_calls=expected_tool_calls
            )
        if not self.result.errors:
            self.result.success = True
