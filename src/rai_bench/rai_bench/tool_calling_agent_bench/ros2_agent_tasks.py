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
        self.objects_types = objects

    def get_prompt(self) -> str:
        """Generates a prompt based on the objects provided in the task. If there is more than one object, the object in the prompt will be pluralized.
        Returns:
            str: Formatted prompt for the task
        """
        inflector = inflect.engine()
        object_counts = {
            obj: len(positions) for obj, positions in self.objects_types.items()
        }
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
                    for object_type in self.objects_types
                ],
            )
        if not self.result.errors:
            self.result.success = True
