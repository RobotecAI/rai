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
from typing import Any, Dict, List, Tuple

import inflect
from langchain_core.messages import AIMessage
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


class GetROS2TopicsTask2(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
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


class GetROS2RGBCameraTask(ROS2ToolCallingAgentTask):
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


class GetROS2DepthCameraTask(ROS2ToolCallingAgentTask):
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


class GetAllROS2RGBCamerasTask(ROS2ToolCallingAgentTask):
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


class GetAllROS2DepthCamerasTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/camera_image_color"},
                )
        if not self.result.errors:
            self.result.success = True


class GetRobotDescriptionTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/robot_description"},
                )
        if not self.result.errors:
            self.result.success = True


class GetPointcloudTask(ROS2ToolCallingAgentTask):
    def __init__(self, logger: loggers_type | None = None) -> None:
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
        ai_messages: List[AIMessage] = [
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
        objects: Dict[str, List[Tuple[float, float, float]]],
        logger: loggers_type | None = None,
    ) -> None:
        """
        Args:
            objects (Dict[str, List[Tuple[float, float, float]]]): dictionary containing the object types and their positions. Object type should be passed as singular.
            logger (loggers_type | None, optional): Defaults to None.
        Examples:
            objects = {
                "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                "cube": [(0.7, 0.8, 0.9)],
            }
        """
        super().__init__(logger=logger)
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
        ai_messages: List[AIMessage] = [
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
        objects (Dict[str, List[Tuple[float, float, float]]]): dictionary containing the object types and their positions. Object type should be passed as singular.
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
        objects: Dict[str, List[Tuple[float, float, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
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
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 3
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and self._check_tool_calls_num_in_ai_message(
            ai_messages[0], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[0].tool_calls[0],
                expected_name="get_object_positions",
                expected_args={"object_name": self.object_to_grab},
            )

        if len(ai_messages) > 1 and self._check_tool_calls_num_in_ai_message(
            ai_messages[1], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[1].tool_calls[0],
                expected_name="move_to_point",
                expected_args=self._object_position_and_task_to_dict(
                    object_position=self.objects[self.object_to_grab][0], task="grab"
                ),
            )
        if not self.result.errors:
            self.result.success = True

    def _object_position_and_task_to_dict(
        self, object_position: Tuple[float, float, float], task: str
    ) -> Dict[str, Any]:
        return {
            "x": object_position[0],
            "y": object_position[1],
            "z": object_position[2],
            "task": task,
        }


class GrabNotExistingObjectTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[Tuple[float, float, float]]]): dictionary containing the object types and their positions. Object type should be passed as singular.
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
        objects: Dict[str, List[Tuple[float, float, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
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
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 2
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and self._check_tool_calls_num_in_ai_message(
            ai_messages[0], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[0].tool_calls[0],
                expected_name="get_object_positions",
                expected_args={"object_name": self.object_to_grab},
            )

        if not self.result.errors:
            self.result.success = True

    def _object_position_and_task_to_dict(
        self, object_position: Tuple[float, float, float], task: str
    ) -> Dict[str, Any]:
        return {
            "x": object_position[0],
            "y": object_position[1],
            "z": object_position[2],
            "task": task,
        }


class MoveExistingObjectLeftTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[Tuple[float, float, float]]]): dictionary containing the object types and their positions. Object type should be passed as singular.
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
        objects: Dict[str, List[Tuple[float, float, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
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
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and self._check_tool_calls_num_in_ai_message(
            ai_messages[0], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[0].tool_calls[0],
                expected_name="get_object_positions",
                expected_args={"object_name": self.object_to_grab},
            )

        if len(ai_messages) > 1 and self._check_tool_calls_num_in_ai_message(
            ai_messages[1], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[1].tool_calls[0],
                expected_name="move_to_point",
                expected_args=self._object_position_and_task_to_dict(
                    object_position=self.objects[self.object_to_grab][0], task="grab"
                ),
            )

        if len(ai_messages) > 2 and self._check_tool_calls_num_in_ai_message(
            ai_messages[2], expected_num=1
        ):
            object_position_and_task = self._object_position_and_task_to_dict(
                object_position=self.objects[self.object_to_grab][0], task="drop"
            )
            object_position_and_task["y"] = object_position_and_task["y"] - 0.2
            self._check_tool_call(
                tool_call=ai_messages[2].tool_calls[0],
                expected_name="move_to_point",
                expected_args=object_position_and_task,
            )

        if not self.result.errors:
            self.result.success = True

    def _object_position_and_task_to_dict(
        self, object_position: Tuple[float, float, float], task: str
    ) -> Dict[str, Any]:
        return {
            "x": object_position[0],
            "y": object_position[1],
            "z": object_position[2],
            "task": task,
        }


class MoveExistingObjectFrontTask(ROS2ToolCallingAgentTask):
    """
    Args:
        objects (Dict[str, List[Tuple[float, float, float]]]): dictionary containing the object types and their positions. Object type should be passed as singular.
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
        objects: Dict[str, List[Tuple[float, float, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
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
        ai_messages: List[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            error_msg = f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            self.logger.error(error_msg)
            self.result.errors.append(error_msg)

        if ai_messages and self._check_tool_calls_num_in_ai_message(
            ai_messages[0], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[0].tool_calls[0],
                expected_name="get_object_positions",
                expected_args={"object_name": self.object_to_grab},
            )

        if len(ai_messages) > 1 and self._check_tool_calls_num_in_ai_message(
            ai_messages[1], expected_num=1
        ):
            self._check_tool_call(
                tool_call=ai_messages[1].tool_calls[0],
                expected_name="move_to_point",
                expected_args=self._object_position_and_task_to_dict(
                    object_position=self.objects[self.object_to_grab][0], task="grab"
                ),
            )

        if len(ai_messages) > 2 and self._check_tool_calls_num_in_ai_message(
            ai_messages[2], expected_num=1
        ):
            object_position_and_task = self._object_position_and_task_to_dict(
                object_position=self.objects[self.object_to_grab][0], task="drop"
            )
            object_position_and_task["x"] = object_position_and_task["x"] + 0.6
            self._check_tool_call(
                tool_call=ai_messages[2].tool_calls[0],
                expected_name="move_to_point",
                expected_args=object_position_and_task,
            )

        if not self.result.errors:
            self.result.success = True

    def _object_position_and_task_to_dict(
        self, object_position: Tuple[float, float, float], task: str
    ) -> Dict[str, Any]:
        return {
            "x": object_position[0],
            "y": object_position[1],
            "z": object_position[2],
            "task": task,
        }
