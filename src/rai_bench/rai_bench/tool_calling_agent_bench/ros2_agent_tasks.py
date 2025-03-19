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

import copy
import logging
from itertools import permutations
from typing import Any, Dict, List, Sequence

import inflect
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from rai.tools.ros.manipulation import MoveToPointToolInput

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    ROS2ToolCallingAgentTask,
)
from rai_bench.tool_calling_agent_bench.mocked_tools import (
    MockGetObjectPositionsTool,
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockMoveToPointTool,
    MockReceiveROS2MessageTool,
)

loggers_type = logging.Logger


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


class GetROS2TopicsTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
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

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "Get the names and types of all ROS2 topics"

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request only the tool that retrieves the ROS2 topics names and types
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
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


class GetROS2TopicsTask2(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
        self.expected_tools = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                ]
            )
        ]

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "What is in the ROS2 network?"

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request only the tool that retrieves the ROS2 topics names and types
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
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


class GetROS2RGBCameraTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
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

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "Get the RGB image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the RGB image topic
        2. The tool that retrieves the RGB image from the /camera_image_color topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
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
                    expected_args={"topic": "/camera_image_color"},
                )
        if not self.result.errors:
            self.result.success = True


class GetROS2DepthCameraTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
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

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "Get the depth image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the depth image topic.
        2. The tool that retrieves the RGB image from the /camera_image_depth topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
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


class GetAllROS2RGBCamerasTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
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

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the topics with RGB images.
        2. The tool that retrieves the RGB images - from the /camera_image_color and from the /color_image5 topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
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


class GetAllROS2DepthCamerasTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
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
        return "Get depth images from all of the available cameras."

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the topics with depth images.
        2. The tool that retrieves the depth images - from the /camera_image_depth and from the /depth_image5 topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                error_msg = (
                    "First AI message did not request ROS2 topics and types correctly."
                )
                self.logger.error(error_msg)
                self.result.errors.append(error_msg)

        if len(ai_messages) > 1:
            expected_tool_calls: list[dict[str, Any]] = [
                {"name": "get_ros2_image", "args": {"topic": "/camera_image_depth"}},
                {"name": "get_ros2_image", "args": {"topic": "/depth_image5"}},
            ]

            self._check_multiple_tool_calls(
                message=ai_messages[1], expected_tool_calls=expected_tool_calls
            )
        if not self.result.errors:
            self.result.success = True


class GetROS2MessageTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
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
            MockReceiveROS2MessageTool(),
        ]

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "Get RGB image."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the RGB image topic
        2. The tool that retrieves the RGB image from the /camera_image_color topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/camera_image_color"},
                )
        if not self.result.errors:
            self.result.success = True


class GetRobotDescriptionTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        self.complexity = "easy"
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
                ]
            ),
            MockReceiveROS2MessageTool(),
        ]

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "Give me description of the robot."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the topic with the robot description
        2. The tool that retrieves the message from the /robot_description topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/robot_description"},
                )
        if not self.result.errors:
            self.result.success = True


class GetPointcloudTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        self.complexity = "easy"
        super().__init__(logger=logger)
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
                    "topic: /trajectory_execution_event\ntype: std_msgs/msg/String\n",
                ]
            ),
            MockReceiveROS2MessageTool(),
        ]

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return "Get the pointcloud."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the topic with the pointcloud
        2. The tool that retrieves the message from the /pointcloud topic
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            error_msg = f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/pointcloud"},
                )
        if not self.result.errors:
            self.result.success = True


class MoveToPointTask(ROS2ToolCallingAgentTask):
    def __init__(
        self, args: Dict[str, Any], logger: loggers_type | None = None
    ) -> None:
        super().__init__(logger=logger)
        self.complexity = "easy"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                ]
            ),
            MockReceiveROS2MessageTool(),
            MockMoveToPointTool(manipulator_frame="base_link"),
        ]
        self.args = MoveToPointToolInput(**args)

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping the user to manipulate the robotic arm. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        return f"Move the arm to a point x={self.args.x}, y={self.args.y}, z={self.args.z} to {self.args.task} an object."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request the tool that moves the arm to a point specified in the prompt with requested task (grab or drop)"
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            error_msg = "No AI messages found in the response."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        else:
            total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
            if total_tool_calls != 1:
                error_msg = f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
                self.logger.error(error_msg)
                self.result.errors.append(error_msg)
            else:
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args={
                        "x": self.args.x,
                        "y": self.args.y,
                        "z": self.args.z,
                        "task": self.args.task,
                    },
                )

        if not self.result.errors:
            self.result.success = True


class GetObjectPositionsTask(ROS2ToolCallingAgentTask):
    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        logger: loggers_type | None = None,
    ) -> None:
        """
        Args:
            objects (Dict[str, List[dict[str, float]]): dictionary containing the object types and their positions. Object type should be passed as singular.
            logger (loggers_type | None, optional): Defaults to None.
        Examples:
            objects = {
                "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                "cube": [(0.7, 0.8, 0.9)],
            }
        """
        super().__init__(logger=logger)
        self.complexity = "easy"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /pointcloud\ntype: sensor_msgs/msg/PointCloud2\n",
                    "topic: /robot_description\ntype: std_msgs/msg/String\n",
                    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
                    "topic: /tf\ntype: tf2_msgs/msg/TFMessage\n",
                ]
            ),
            MockReceiveROS2MessageTool(),
            MockGetObjectPositionsTool(mock_objects=objects),
        ]
        self.objects = objects

    def get_system_prompt(self) -> str:
        return """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

    def get_prompt(self) -> str:
        """Generates a prompt based on the objects provided in the task. If there is more than one object, the object in the prompt will be pluralized.
        Returns:
            str: Formatted prompt for the task
        """
        inflector = inflect.engine()
        object_counts = {obj: len(positions) for obj, positions in self.objects.items()}
        formatted_objects = [
            inflector.plural(obj) if count > 1 else obj
            for obj, count in object_counts.items()
        ]
        if len(formatted_objects) > 1:
            objects_list = (
                ", ".join(formatted_objects[:-1]) + f", and {formatted_objects[-1]}"
            )
        else:
            objects_list = formatted_objects[0]
        return f"Get the {objects_list} positions."

    def verify_tool_calls(self, response: dict[str, Any]):
        """
        It is expected that the agent will request the tool for each object type to get its positions.
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            error_msg = "No AI messages found in the response."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        else:
            ai_message = ai_messages[0]
            self._check_multiple_tool_calls(
                message=ai_message,
                expected_tool_calls=[
                    {
                        "name": "get_object_positions",
                        "args": {"object_name": object_type},
                    }
                    for object_type in self.objects
                ],
            )
        if not self.result.errors:
            self.result.success = True


class GrabExistingObjectTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[dict[str, float]]): dictionary containing the object types and their positions. Object type should be passed as singular.
        object_to_grab (str): object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
        logger (loggers_type | None, optional): Defaults to None.
    Examples:
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        object_to_grab = "cube"
    """

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.complexity = "medium"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                ]
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
            MockGetROS2ImageTool(),
        ]
        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        """

    def get_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool get_object_positions to get the position of the object to grab.
        2. The tool move_to_point to move to the position of the object to grab.
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 3
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                obj_to_grab: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_grab.update({"task": "grab"})
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_grab,
                )
        if not self.result.errors:
            self.result.success = True


class GrabNotExistingObjectTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[dict[str, float]]): dictionary containing the object types and their positions. Object type should be passed as singular.
        object_to_grab (str): object to grab. Object type should be passed as singular. Object to be grabbed should NOT be defined in the objects argument.
        logger (loggers_type | None, optional): Defaults to None.
    Examples:
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        object_to_grab = "apple"
    """

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.complexity = "medium"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                    "topic: /color_camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
                ]
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
            MockGetROS2ImageTool(),
        ]
        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        """

    def get_prompt(self) -> str:
        return f"Grab {self.object_to_grab}."

    def _verify_args(self):
        if self.object_to_grab in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is present in defined objects: {self.objects} but should not be."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request the tool get_object_positions to get the position of the object to grab.
        It is expected that no positions are returned and agent will not request any more tool.
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 2
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if not self.result.errors:
            self.result.success = True


class MoveExistingObjectLeftTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[dict[str, float]]): dictionary containing the object types and their positions. Object type should be passed as singular.
        object_to_grab (str): object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
        logger (loggers_type | None, optional): Defaults to None.
    Examples:
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        object_to_grab = "cube"
    """

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.complexity = "medium"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
                    "topic: /collision_object\ntype: moveit_msgs/msg/CollisionObject\n",
                ]
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
            MockGetROS2ImageTool(),
        ]
        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        Coordinates are in meters.
        """

    def get_prompt(self) -> str:
        return f"Move {self.object_to_grab} 20 cm to the left."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                obj_to_grab: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_grab.update({"task": "grab"})
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_grab,
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                obj_to_drop: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_drop.update({"task": "drop"})
                obj_to_drop["y"] = obj_to_drop["y"] - 0.2
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_drop,
                )

        if not self.result.errors:
            self.result.success = True


class MoveExistingObjectFrontTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[dict[str, float]]): dictionary containing the object types and their positions. Object type should be passed as singular.
        object_to_grab (str): object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
        logger (loggers_type | None, optional): Defaults to None.
    Examples:
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        object_to_grab = "cube"
    """

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.complexity = "medium"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                ]
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
            MockGetROS2ImageTool(),
        ]
        self.objects = objects
        self.object_to_grab = object_to_grab
        self._verify_args()

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        Coordinates are in meters.
        """

    def get_prompt(self) -> str:
        return f"Move {self.object_to_grab} 60 cm to the front."

    def _verify_args(self):
        if self.object_to_grab not in self.objects:
            error_message = f"Requested object to grab {self.object_to_grab} is not present in defined objects: {self.objects}."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="get_object_positions",
                    expected_args={"object_name": self.object_to_grab},
                )

        if len(ai_messages) > 1:
            if self._check_tool_calls_num_in_ai_message(ai_messages[1], expected_num=1):
                obj_to_grab: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_grab.update({"task": "grab"})
                self._check_tool_call(
                    tool_call=ai_messages[1].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_grab,
                )

        if len(ai_messages) > 2:
            if self._check_tool_calls_num_in_ai_message(ai_messages[2], expected_num=1):
                obj_to_drop: dict[str, Any] = copy.deepcopy(
                    self.objects[self.object_to_grab][0]
                )
                obj_to_drop.update({"task": "drop"})
                obj_to_drop["x"] = obj_to_drop["x"] + 0.6
                self._check_tool_call(
                    tool_call=ai_messages[2].tool_calls[0],
                    expected_name="move_to_point",
                    expected_args=obj_to_drop,
                )

        if not self.result.errors:
            self.result.success = True


class SwapObjectsTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[dict[str, float]]): dictionary containing the object types and their positions. Object type should be passed as singular.
        objects_to_swap (str): objects to be swapped. Object type should be passed as singular. Objects to be swapped should be defined in the objects argument with only one instance (one position).
        logger (loggers_type | None, optional): Defaults to None.
    Examples:
        objects = {
            "banana": [(0.1, 0.2, 0.1)],
            "cube": [(0.7, 0.8, 0.1)],
            "apple": [(0.3, 0.4, 0.1), (0.5, 0.6, 0.1)],

        }
        objects_to_swap = ["cube", "banana"]
    """

    def __init__(
        self,
        objects: Dict[str, List[Dict[str, float]]],
        objects_to_swap: List[str],
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.complexity = "hard"
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=[
                    "topic: /attached_collision_object\ntype: moveit_msgs/msg/AttachedCollisionObject\n",
                    "topic: /camera_image_color\ntype: sensor_msgs/msg/Image\n",
                    "topic: /camera_image_depth\ntype: sensor_msgs/msg/Image\n",
                ]
            ),
            MockGetObjectPositionsTool(
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                mock_objects=objects,
            ),
            MockMoveToPointTool(manipulator_frame="panda_link0"),
            MockGetROS2ImageTool(),
        ]
        self.objects = objects
        self.objects_to_swap = objects_to_swap
        self._verify_args()

    def _verify_args(self):
        for obj in self.objects_to_swap:
            if obj not in self.objects:
                error_message = f"Requested object to swap {obj} is not present in defined objects: {self.objects}."
                self.result.errors.append(error_message)
                raise TaskParametrizationError(error_message)
            if len(self.objects[obj]) != 1:
                error_message = f"Number of positions for object to swap ({obj}) should be equal to 1."
                self.result.errors.append(error_message)
                raise TaskParametrizationError(error_message)
        if len(self.objects_to_swap) != 2:
            error_message = f"Number of requested objects to swap {len(self.objects_to_swap)} should be equal to 2."
            self.result.errors.append(error_message)
            raise TaskParametrizationError(error_message)

    def get_system_prompt(self) -> str:
        return """
        You are a robotic arm with interfaces to detect and manipulate objects in physical environment.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up).
        Coordinates are in meters.
        """

    def get_prompt(self) -> str:
        return f"Move {self.objects_to_swap[0]} to the initial position of {self.objects_to_swap[1]}, and move {self.objects_to_swap[1]} to the initial position of {self.objects_to_swap[0]}."

    def verify_tool_calls(self, response: Dict[str, Any]):
        """
        It is expected that the agent will request:
        1. get_object_positions for both objects to be swapped
        2. move_to_point for one object to some temporary position to make place to second object
        3. move_to_point for the second object to the position of the first object
        4. move_to_point for the first object to the position of the second object
        """

        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        actual_tool_calls = [
            tool_call for msg in ai_messages for tool_call in msg.tool_calls
        ]

        expected_num_tool_calls = 8
        if len(actual_tool_calls) < expected_num_tool_calls:
            error_msg = f"Expected at least {expected_num_tool_calls} tool calls, but got {len(actual_tool_calls)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
            return None

        obj1, obj2 = self.objects_to_swap
        obj1_pos, obj2_pos = self.objects[obj1][0], self.objects[obj2][0]

        # find a temporary position if exists
        move_to_point_args: Sequence[Dict[str, Any]] = [
            call["args"]
            for call in actual_tool_calls
            if call["name"] == "move_to_point"
        ]
        positions = copy.deepcopy(move_to_point_args)
        for arg in positions:
            arg.pop("task")
        temp_position = None
        for position in positions:
            if position != obj1_pos and position != obj2_pos:
                temp_position = position
                break

        if temp_position is None:
            error_msg = "No temporary position found."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)
        else:
            get_position_permutations = list(
                permutations(
                    [
                        {"name": "get_object_positions", "args": {"object_name": obj1}},
                        {"name": "get_object_positions", "args": {"object_name": obj2}},
                    ]
                )
            )

            obj_moves_options: List[List[dict[str, Any]]] = [
                [
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "grab"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "drop"},
                    },
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "grab"}},
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "drop"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "grab"},
                    },
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "drop"}},
                ],
                [
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "grab"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "drop"},
                    },
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "grab"}},
                    {"name": "move_to_point", "args": {**obj2_pos, "task": "drop"}},
                    {
                        "name": "move_to_point",
                        "args": {**temp_position, "task": "grab"},
                    },
                    {"name": "move_to_point", "args": {**obj1_pos, "task": "drop"}},
                ],
            ]

            valid_sequences: List[List[dict[str, Any]]] = []
            for get_positions in get_position_permutations:
                for obj_moves in obj_moves_options:
                    valid_sequences.append(list(get_positions) + obj_moves)

            if not any(
                self._matches_sequence(actual_tool_calls, seq)
                for seq in valid_sequences
            ):
                error_msg = (
                    "The tool calls are in an invalid sequence for object swapping."
                )
                self.logger.error(error_msg)
                self.result.errors.append(error_msg)

        if not self.result.errors:
            self.result.success = True

    def _matches_sequence(
        self,
        actual_tool_calls_seq: Sequence[ToolCall],
        expected_tool_calls_seq: Sequence[dict[str, Any]],
    ) -> bool:
        """
        Helper method to check if actual tool calls sequence match expected tool calls in terms of sequence and arguments.
        Parameters
        ----------
        actual_tool_calls_seq : Sequence[ToolCall]
            Sequence of tool calls requested by agent.
        expected_tool_calls_seq : Sequence[dict[str, Any]]
            Sequence of expected tool calls.

        Returns
        -------
        bool
            True if actual tool calls sequence matches expected tool calls sequence, False otherwise
        """
        if len(actual_tool_calls_seq) < len(expected_tool_calls_seq):
            return False
        it = iter(actual_tool_calls_seq)
        return all(
            any(call["name"] == e["name"] and call["args"] == e["args"] for call in it)
            for e in expected_tool_calls_seq
        )
