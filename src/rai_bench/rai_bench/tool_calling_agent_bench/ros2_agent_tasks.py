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
    CustomInterfacesServiceTask,
    CustomInterfacesTopicTask,
    ROS2ToolCallingAgentTask,
)
from rai_bench.tool_calling_agent_bench.messages.base import (
    BoundingBox2D,
    Detection2D,
    Header,
    Orientation,
    Point2D,
    Pose,
    Pose2D,
    PoseStamped,
    Position,
    Time,
)
from rai_bench.tool_calling_agent_bench.messages.topics import (
    Image,
    RAIDetectionArray,
)
from rai_bench.tool_calling_agent_bench.mocked_tools import (
    MockGetObjectPositionsTool,
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockMoveToPointTool,
    MockReceiveROS2MessageTool,
)

loggers_type = logging.Logger


PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT = """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
Be proactive and use the tools to answer questions.

Example of tool calls:
- get_ros2_message_interface, args: {'msg_type': 'geometry_msgs/msg/Twist'}
- publish_ros2_message, args: {'topic': '/cmd_vel', 'message_type': 'geometry_msgs/msg/Twist', 'message': {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}}

- get_ros2_message_interface, args: {'msg_type': 'turtlesim/srv/TeleportAbsolute'}
- publish_ros2_message, args: {'topic': '/turtle1/teleport_absolute', 'message_type': 'turtlesim/srv/TeleportAbsolute', 'message': {x: 5.0, y: 2.0, theta: 1.57}}
"""


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


class GetROS2TopicsTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

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
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the names and types of all ROS2 topics"

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request only the tool that retrieves the ROS2 topics names and types

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")

        self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0])

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            self.log_error(
                msg=f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            )

        if not self.result.errors:
            self.result.success = True


class GetROS2TopicsTask2(ROS2ToolCallingAgentTask):
    complexity = "easy"

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
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "What is in the ROS2 network?"

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request only the tool that retrieves the ROS2 topics names and types

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")

        self._is_ai_message_requesting_get_ros2_topics_and_types(ai_messages[0])

        total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
        if total_tool_calls != 1:
            self.log_error(
                msg=f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
            )

        if not self.result.errors:
            self.result.success = True


class GetROS2RGBCameraTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the RGB image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the RGB image topic
        2. The tool that retrieves the RGB image from the /camera_image_color topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
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
                    expected_name="get_ros2_image",
                    expected_args={"topic": "/camera_image_color"},
                    expected_optional_args={"timeout_sec": None},
                )
        if not self.result.errors:
            self.result.success = True


class GetROS2DepthCameraTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the depth image from the camera."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the depth image topic.
        2. The tool that retrieves the depth image from the /camera_image_depth topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
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
                    expected_name="get_ros2_image",
                    expected_args={"topic": "/camera_image_depth"},
                    expected_optional_args={"timeout_sec": None},
                )
        if not self.result.errors:
            self.result.success = True


class GetAllROS2RGBCamerasTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/color_image5": "sensor_msgs/msg/Image",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_prompt(self) -> str:
        return "Get RGB images from all of the available cameras."

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the topics with RGB images.
        2. The tool that retrieves the RGB images - from the /camera_image_color and from the /color_image5 topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            expected_tool_calls: list[dict[str, Any]] = [
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/camera_image_color"},
                    "optional_args": {"timeout_sec": None},
                },
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/color_image5"},
                    "optional_args": {"timeout_sec": None},
                },
            ]
            self._check_multiple_tool_calls(
                message=ai_messages[1], expected_tool_calls=expected_tool_calls
            )
        if not self.result.errors:
            self.result.success = True


class GetAllROS2DepthCamerasTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/color_image5": "sensor_msgs/msg/Image",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)
        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]
        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetROS2ImageTool(available_topics=list(self.topics_and_types.keys())),
        ]

    def get_prompt(self) -> str:
        return "Get depth images from all of the available cameras."

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topic names and types to identify the topics with depth images.
        2. The tool that retrieves the depth images - from the /camera_image_depth and from the /depth_image5 topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
            )
        if ai_messages:
            if not self._is_ai_message_requesting_get_ros2_topics_and_types(
                ai_messages[0]
            ):
                self.log_error(
                    msg="First AI message did not request ROS2 topics and types correctly."
                )

        if len(ai_messages) > 1:
            expected_tool_calls: list[dict[str, Any]] = [
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/camera_image_depth"},
                    "optional_args": {"timeout_sec": None},
                },
                {
                    "name": "get_ros2_image",
                    "args": {"topic": "/depth_image5"},
                    "optional_args": {"timeout_sec": None},
                },
            ]
            self._check_multiple_tool_calls(
                message=ai_messages[1], expected_tool_calls=expected_tool_calls
            )
        if not self.result.errors:
            self.result.success = True


class GetROS2MessageTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
        "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
        "/depth_image5": "sensor_msgs/msg/Image",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockReceiveROS2MessageTool(
                available_topics=list(self.topics_and_types.keys())
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get RGB image."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the RGB image topic.
        2. The tool that retrieves the RGB image from the /camera_image_color topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/camera_image_color"},
                )

        if not self.result.errors:
            self.result.success = True


class GetRobotDescriptionTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/trajectory_execution_event": "std_msgs/msg/String",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockReceiveROS2MessageTool(
                available_topics=list(self.topics_and_types.keys())
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Give me description of the robot."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the topic with the robot description
        2. The tool that retrieves the message from the /robot_description topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/robot_description"},
                )

        if not self.result.errors:
            self.result.success = True


class GetPointcloudTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/trajectory_execution_event": "std_msgs/msg/String",
    }

    def __init__(self, logger: loggers_type | None = None) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockReceiveROS2MessageTool(
                available_topics=list(self.topics_and_types.keys())
            ),
        ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    def get_prompt(self) -> str:
        return "Get the pointcloud."

    def verify_tool_calls(self, response: dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool that retrieves the ROS2 topics names and types to recognize the topic with the pointcloud
        2. The tool that retrieves the message from the /pointcloud topic

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if len(ai_messages) < 3:
            self.log_error(
                msg=f"Expected at least 3 AI messages, but got {len(ai_messages)}."
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
                    expected_name="receive_ros2_message",
                    expected_args={"topic": "/pointcloud"},
                )

        if not self.result.errors:
            self.result.success = True


class MoveToPointTask(ROS2ToolCallingAgentTask):
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
    }

    def __init__(
        self, args: Dict[str, Any], logger: loggers_type | None = None
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
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
        """It is expected that the agent will request the tool that moves the arm to a point specified in the prompt with requested task (grab or drop)"

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")
        else:
            total_tool_calls = sum(len(message.tool_calls) for message in ai_messages)
            if total_tool_calls != 1:
                self.log_error(
                    msg=f"Total number of tool calls across all AI messages should be 1, but got {total_tool_calls}."
                )
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
    complexity = "easy"

    topics_and_types: Dict[str, str] = {
        "/pointcloud": "sensor_msgs/msg/PointCloud2",
        "/robot_description": "std_msgs/msg/String",
        "/rosout": "rcl_interfaces/msg/Log",
        "/tf": "tf2_msgs/msg/TFMessage",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        logger: loggers_type | None = None,
    ) -> None:
        """Task to get the positions of the objects

        Parameters
        ----------
        objects : Dict[str, List[dict[str, float]]]
            Dictionary containing the object types and their positions. Object type should be passed as singular.
        logger : loggers_type | None, optional
            Logger, by default None

        Examples
        --------
        objects = {
            "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
            "cube": [(0.7, 0.8, 0.9)],
        }
        """
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
            ),
            MockGetObjectPositionsTool(mock_objects=objects),
        ]

        self.objects = objects

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

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
        """It is expected that the agent will request the tool for each object type to get its positions.

        Parameters
        ----------
        response : dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if not ai_messages:
            self.log_error(msg="No AI messages found in the response.")
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
    complexity = "medium"

    """Task to grab an object

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "cube"
    """
    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
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
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. The tool get_object_positions to get the position of the object to grab.
        2. The tool move_to_point to move to the position of the object to grab.

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 3
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

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
    """Task to grab an object that does not exist

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should NOT be defined in the objects argument.
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "apple"
    """

    complexity = "medium"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
        "/color_camera_info": "sensor_msgs/msg/CameraInfo",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
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
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request the tool get_object_positions to get the position of the object to grab.
        It is expected that no positions are returned and agent will not request any more tool.

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 2
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

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
    """Task to move an existing object to the left.

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        "cube": [(0.7, 0.8, 0.9)],
    }
    object_to_grab = "cube"
    """

    complexity = "medium"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
        "/clock": "rosgraph_msgs/msg/Clock",
        "/collision_object": "moveit_msgs/msg/CollisionObject",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
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
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. get_object_positions for the object to grab
        2. move_to_point for the object to grab with the coordinates of the object to grab specified in the task
        3. move_to_point for the the same object but with the task set to "drop" and y coordinate smaller by 0.6

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

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
    """Task to move an existing object to the front

    Parameters
    ----------
    objects : Dict[str, List[dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    object_to_grab : str
        Object to grab. Object type should be passed as singular. Object to be grabbed should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None
    """

    complexity = "medium"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
    }

    def __init__(
        self,
        objects: Dict[str, List[dict[str, float]]],
        object_to_grab: str,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
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
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

        if len(self.objects[self.object_to_grab]) > 1:
            error_message = f"Requested object to grab {self.object_to_grab} has more than one position in defined objects: {self.objects[self.object_to_grab]}."
            self.log_error(msg=error_message)
            raise TaskParametrizationError(error_message)

    def verify_tool_calls(self, response: Dict[str, Any]):
        """It is expected that the agent will request:
        1. get_object_positions for the object to grab
        2. move_to_point for the object to grab with the coordinates of the object to grab specified in the task
        3. move_to_point for the the same object but with the task set to "drop" and x coordinate bigger by 0.6

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
        """
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        expected_num_ai_messages = 4
        if len(ai_messages) != expected_num_ai_messages:
            self.log_error(
                msg=f"Expected {expected_num_ai_messages} AI messages, but got {len(ai_messages)}."
            )

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
    """Task to swap objects

    Parameters
    ----------
    objects : Dict[str, List[Dict[str, float]]]
        Dictionary containing the object types and their positions. Object type should be passed as singular.
    objects_to_swap : List[str]
        Objects to be swapped. Object type should be passed as singular. Objects to be swapped should be defined in the objects argument with only one instance (one position).
    logger : loggers_type | None, optional
        Logger, by default None

    Examples
    --------
    objects = {
        "banana": [(0.1, 0.2, 0.1)],
        "cube": [(0.7, 0.8, 0.1)],
        "apple": [(0.3, 0.4, 0.1), (0.5, 0.6, 0.1)],

    }
    objects_to_swap = ["cube", "banana"]
    """

    complexity = "hard"

    topics_and_types: Dict[str, str] = {
        "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
        "/camera_image_color": "sensor_msgs/msg/Image",
        "/camera_image_depth": "sensor_msgs/msg/Image",
    }

    def __init__(
        self,
        objects: Dict[str, List[Dict[str, float]]],
        objects_to_swap: List[str],
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)

        topic_strings = [
            f"topic: {topic}\ntype: {msg_type}\n"
            for topic, msg_type in self.topics_and_types.items()
        ]

        self.expected_tools: List[BaseTool] = [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=topic_strings
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
        ]

        self.objects = objects
        self.objects_to_swap = objects_to_swap
        self._verify_args()

    def _verify_args(self):
        for obj in self.objects_to_swap:
            if obj not in self.objects:
                error_message = f"Requested object to swap {obj} is not present in defined objects: {self.objects}."
                self.log_error(msg=error_message)
                raise TaskParametrizationError(error_message)
            if len(self.objects[obj]) != 1:
                error_message = f"Number of positions for object to swap ({obj}) should be equal to 1."
                self.log_error(msg=error_message)
                raise TaskParametrizationError(error_message)
        if len(self.objects_to_swap) != 2:
            error_message = f"Number of requested objects to swap {len(self.objects_to_swap)} should be equal to 2."
            self.log_error(msg=error_message)
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
        """It is expected that the agent will request:
        1. get_object_positions for both objects to be swapped
        2. move_to_point for one object to some temporary position to make place to second object
        3. move_to_point for the second object to the position of the first object
        4. move_to_point for the first object to the position of the second object

        Parameters
        ----------
        response : Dict[str, Any]
            The response from the agent
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
            self.log_error(
                msg=f"Expected at least {expected_num_tool_calls} tool calls, but got {len(actual_tool_calls)}."
            )
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
            self.log_error(msg="No temporary position found.")
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
                self.log_error(
                    msg="The tool calls are in an invalid sequence for object swapping."
                )

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


class PublishROS2HRIMessageTask3ExtraCalls(CustomInterfacesTopicTask):
    complexity = "easy"
    expected_text = "Hello!"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_topic(self) -> str:
        return "/to_human"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:

        for call in tool_calls:
            if self._check_topic_tool_call_field(
                tool_call=call,
                expected_name="publish_ros2_message",
                expected_topic=self.expected_topic,
                expected_message_type=self.expected_message_type,
                field_path="text",
                expected_value=self.expected_text,
            ):
                return True

        self.log_error(f"No valid call to {self.expected_topic} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to publish a message to the topic '{self.expected_topic}' with the text value: '{self.expected_text}'.\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.expected_topic}'.\n"
            "3. Retrieve the full message interface definition for that type.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Publish the message to '{self.expected_topic}' using the correct message type and interface.\n"
        )


class PublishROS2HRIMessageTask1ExtraCall(PublishROS2HRIMessageTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class PublishROS2HRIMessageTask0ExtraCalls(PublishROS2HRIMessageTask3ExtraCalls):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class PublishROS2AudioMessageTask3ExtraCalls(CustomInterfacesTopicTask):
    complexity = "easy"
    expected_audio: List[int] = [123, 456, 789]
    expected_sample_rate: int = 44100
    expected_channels: int = 2

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_topic(self) -> str:
        return "/send_audio"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if (
                self._check_topic_tool_call_field(
                    tool_call=call,
                    expected_name="publish_ros2_message",
                    expected_topic=self.expected_topic,
                    expected_message_type=self.expected_message_type,
                    field_path="audio",
                    expected_value=self.expected_audio,
                )
                and self._check_topic_tool_call_field(
                    tool_call=call,
                    expected_name="publish_ros2_message",
                    expected_topic=self.expected_topic,
                    expected_message_type=self.expected_message_type,
                    field_path="sample_rate",
                    expected_value=self.expected_sample_rate,
                )
                and self._check_topic_tool_call_field(
                    tool_call=call,
                    expected_name="publish_ros2_message",
                    expected_topic=self.expected_topic,
                    expected_message_type=self.expected_message_type,
                    field_path="channels",
                    expected_value=self.expected_channels,
                )
            ):
                return True

        self.log_error(f"No valid call to {self.expected_topic} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to publish a message to the topic '{self.expected_topic}' with audio samples {self.expected_audio}, "
            f"sample rate {self.expected_sample_rate}, and {self.expected_channels} channels.\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.expected_topic}'.\n"
            "3. Retrieve the full message interface definition for that type.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Publish the message to '{self.expected_topic}' using the correct message type and interface.\n"
        )


class PublishROS2AudioMessageTask1ExtraCall(PublishROS2AudioMessageTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class PublishROS2AudioMessageTask0ExtraCalls(PublishROS2AudioMessageTask3ExtraCalls):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class PublishROS2DetectionArrayTask3ExtraCalls(CustomInterfacesTopicTask):
    complexity = "easy"

    expected_detection_classes: List[str] = ["person", "car"]
    expected_detections: List[Detection2D] = [
        Detection2D(
            bbox=BoundingBox2D(
                center=Pose2D(position=Point2D(x=320.0, y=240.0), theta=0.0),
                size_x=50.0,
                size_y=50.0,
            )
        )
    ]

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_topic(self) -> str:
        return "/send_detections"

    @property
    def expected_message(self) -> RAIDetectionArray:
        return RAIDetectionArray(
            detections=self.expected_detections,
            detection_classes=self.expected_detection_classes,
        )

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if self._check_topic_tool_call_field(
                tool_call=call,
                expected_name="publish_ros2_message",
                expected_topic=self.expected_topic,
                expected_message_type=self.expected_message_type,
                field_path="detection_classes",
                expected_value=self.expected_detection_classes,
            ):
                return True

        self.log_error(f"No valid call to {self.expected_topic} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to publish a detection message to the topic '{self.expected_topic}' with one detection:\n"
            f"{self.expected_detections[0].model_dump()} and detection classes {self.expected_detection_classes}.\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.expected_topic}'.\n"
            "3. Retrieve the full message interface definition for that type.\n"
            "4. Construct the message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Publish the message to '{self.expected_topic}' using the correct message type and interface.\n"
        )


class PublishROS2DetectionArrayTask1ExtraCall(PublishROS2DetectionArrayTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class PublishROS2DetectionArrayTask0ExtraCalls(
    PublishROS2DetectionArrayTask3ExtraCalls
):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class CallROS2ManipulatorMoveToServiceTask3ExtraCalls(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_initial_gripper_state = True
    expected_final_gripper_state = False
    expected_target_pose: PoseStamped = PoseStamped(
        pose=Pose(
            position=Position(x=1.0, y=2.0, z=3.0),
            orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/manipulator_move_to"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if (
                self._check_service_tool_call_field(
                    tool_call=call,
                    expected_name="call_ros2_service",
                    expected_service=self.expected_service,
                    expected_service_type=self.expected_service_type,
                    field_path="initial_gripper_state",
                    expected_value=self.expected_initial_gripper_state,
                )
                and self._check_service_tool_call_field(
                    tool_call=call,
                    expected_name="call_ros2_service",
                    expected_service=self.expected_service,
                    expected_service_type=self.expected_service_type,
                    field_path="final_gripper_state",
                    expected_value=self.expected_final_gripper_state,
                )
                and self._check_service_tool_call_field(
                    tool_call=call,
                    expected_name="call_ros2_service",
                    expected_service=self.expected_service,
                    expected_service_type=self.expected_service_type,
                    field_path="target_pose",
                    expected_value=self.expected_target_pose.model_dump(),
                )
            ):
                return True

        self.log_error(f"No valid call to {self.expected_service} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.expected_service}' with a target_pose: "
            f"{self.expected_target_pose.model_dump()} and gripper states (initial: {self.expected_initial_gripper_state}, final: {self.expected_final_gripper_state}).\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.expected_service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.expected_service}' using the correct message type and interface.\n"
        )


class CallROS2ManipulatorMoveToServiceTask1ExtraCall(
    CallROS2ManipulatorMoveToServiceTask3ExtraCalls
):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class CallROS2ManipulatorMoveToServiceTask0ExtraCalls(
    CallROS2ManipulatorMoveToServiceTask3ExtraCalls
):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class CallGroundedSAMSegmentTask3ExtraCalls(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_detections: RAIDetectionArray = RAIDetectionArray(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id="camera_frame"),
        detections=[],
    )

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/grounded_sam_segment"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if self._check_service_tool_call_field(
                tool_call=call,
                expected_name="call_ros2_service",
                expected_service=self.expected_service,
                expected_service_type=self.expected_service_type,
                field_path="detections",
                expected_value=self.expected_detections.model_dump(),
            ):
                return True
        self.log_error(f"No valid call to {self.expected_service} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.expected_service}' with detections: {self.expected_detections.model_dump()}\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.expected_service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.expected_service}' using the correct message type and interface.\n"
        )


class CallGroundedSAMSegmentTask1ExtraCall(CallGroundedSAMSegmentTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class CallGroundedSAMSegmentTask0ExtraCalls(CallGroundedSAMSegmentTask3ExtraCalls):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class CallGroundingDinoClassifyTask3ExtraCalls(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_classes: str = "bottle, book, chair"
    expected_box_threshold: float = 0.4
    expected_text_threshold: float = 0.25

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def extra_calls(self) -> int:
        return 3

    @property
    def expected_service(self) -> str:
        return "/grounding_dino_classify"

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if (
                self._check_service_tool_call_field(
                    call,
                    "call_ros2_service",
                    self.expected_service,
                    self.expected_service_type,
                    "classes",
                    self.expected_classes,
                )
                and self._check_service_tool_call_field(
                    call,
                    "call_ros2_service",
                    self.expected_service,
                    self.expected_service_type,
                    "box_threshold",
                    self.expected_box_threshold,
                )
                and self._check_service_tool_call_field(
                    call,
                    "call_ros2_service",
                    self.expected_service,
                    self.expected_service_type,
                    "text_threshold",
                    self.expected_text_threshold,
                )
            ):
                return True

        self.log_error(f"No valid call to {self.expected_service} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.expected_service}' with classes: '{self.expected_classes}', "
            f"box_threshold: {self.expected_box_threshold}, text_threshold: {self.expected_text_threshold}, "
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.expected_service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.expected_service}' using the correct message type and interface.\n"
        )


class CallGroundingDinoClassifyTask1ExtraCall(CallGroundingDinoClassifyTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class CallGroundingDinoClassifyTask0ExtraCalls(
    CallGroundingDinoClassifyTask3ExtraCalls
):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class CallGetLogDigestTask3ExtraCalls(CustomInterfacesServiceTask):
    complexity = "easy"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/get_log_digest"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if self._check_service_tool_call_field(
                call,
                "call_ros2_service",
                self.expected_service,
                self.expected_service_type,
                field_path="",  # empty request
                expected_value="",
            ):
                return True

        self.log_error(f"No valid call to {self.expected_service} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.expected_service}' with an empty request.\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.expected_service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.expected_service}' using the correct message type and interface.\n"
        )


class CallGetLogDigestTask1ExtraCall(CallGetLogDigestTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class CallGetLogDigestTask0ExtraCalls(CallGetLogDigestTask3ExtraCalls):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class CallVectorStoreRetrievalTask3ExtraCalls(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_query: str = "What is the purpose of this robot?"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/rai_whoami_documentation_service"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if self._check_service_tool_call_field(
                call,
                "call_ros2_service",
                self.expected_service,
                self.expected_service_type,
                "query",
                self.expected_query,
            ):
                return True

        self.log_error(f"No valid call to {self.expected_service} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.expected_service}' with the query: '{self.expected_query}'.\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.expected_service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.expected_service}' using the correct message type and interface.\n"
        )


class CallVectorStoreRetrievalTask1ExtraCall(CallVectorStoreRetrievalTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class CallVectorStoreRetrievalTask0ExtraCalls(CallVectorStoreRetrievalTask3ExtraCalls):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


class CallWhatISeeTask3ExtraCalls(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_observations: List[str] = ["table", "cup", "notebook"]
    expected_perception_source: str = "front_camera"

    expected_image: Image = Image(
        header=Header(frame_id="camera_frame"),
        height=480,
        width=640,
    )

    expected_pose: Pose = Pose(
        position=Position(x=1.0, y=2.0, z=0.5),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "rai/whatisee/get"

    @property
    def extra_calls(self) -> int:
        return 3

    def verify_message_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        for call in tool_calls:
            if self._check_service_tool_call_field(
                call,
                "call_ros2_service",
                self.expected_service,
                self.expected_service_type,
                field_path="",  # empty request
                expected_value="",
            ):
                return True

        self.log_error(f"No valid call to {self.expected_service} found.")
        return False

    def get_prompt(self) -> str:
        return (
            f"You need to call the service '{self.expected_service}' with observations: {self.expected_observations}, "
            f"source: '{self.expected_perception_source}', an image from 'camera_frame' (640x480, RGB), "
            f"and a pose: {self.expected_pose.model_dump()}.\n"
            "Before calling, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 services and their types.\n"
            f"2. Find the service type for '{self.expected_service}'.\n"
            "3. Retrieve the full message interface definition for that service.\n"
            "4. Construct the request message filling only the fields you are instructed to. Rest of the fields will have default values.\n"
            f"5. Call the service '{self.expected_service}' using the correct message type and interface.\n"
        )


class CallWhatISeeTask1ExtraCall(CallWhatISeeTask3ExtraCalls):
    complexity = "medium"

    @property
    def extra_calls(self) -> int:
        return 1


class CallWhatISeeTask0ExtraCalls(CallWhatISeeTask3ExtraCalls):
    complexity = "hard"

    @property
    def extra_calls(self) -> int:
        return 0


# class CallROS2CustomActionTask(CustomInterfacesActionTask):
#     complexity = "easy"

#     expected_task = "Where are you?"
#     expected_description = ""
#     expected_priority = "10"

#     def get_system_prompt(self) -> str:
#         return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

#     @property
#     def expected_action(self) -> str:
#         return "/perform_task"

#     @property
#     def expected_message(self) -> Dict[str, Any]:
#         expected = DEFAULT_MESSAGES[self.expected_action_type].copy()
#         expected["goal"]["task"] = self.expected_task
#         expected["goal"]["description"] = self.expected_description
#         expected["goal"]["priority "] = self.expected_priority
#         return expected


ROBOT_NAVIGATION_SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate.

    <rule> As a first step check transforms by getting 1 message from /tf topic </rule>
    <rule> use /cmd_vel topic very carefully. Obstacle detection works only with nav2 stack, so be careful when it is not used. </rule>>
    <rule> be patient with running ros2 actions. usually the take some time to run. </rule>
    <rule> Always check your transform before and after you perform ros2 actions, so that you can verify if it worked. </rule>

    Navigation tips:
    - it's good to start finding objects by rotating, then navigating to some diverse location with occasional rotations. Remember to frequency detect objects.
    - for driving forward/backward or to some coordinates, ros2 actions are better.
    - for driving for some specific time or in specific manner (like shaper or turns) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your current position, then very accurately estimate the goal pose.
    - it is good to verify using given information if the robot is not stuck
    - navigation actions sometimes fail. Their output can be read from rosout. You can also tell if they partially worked by checking the robot position and rotation.
    - before using any ros2 interfaces, always make sure to check you are using the right interface
    - processing camera image takes 5-10s. Take it into account that if the robot is moving, the information can be outdated. Handle it by good planning of your movements.
    - you are encouraged to use wait tool in between checking the status of actions
    - to find some object navigate around and check the surrounding area
    - when the goal is accomplished please make sure to cancel running actions
    - when you reach the navigation goal - double check if you reached it by checking the current position
    - if you detect collision, please stop operation

    - you will be given your camera image description. Based on this information you can reason about positions of objects.
    - be careful and aboid obstacles

    Here are the corners of your environment:
    (-2.76,9.04, 0.0),
    (4.62, 9.07, 0.0),
    (-2.79, -3.83, 0.0),
    (4.59, -3.81, 0.0)

    This is location of places:
    Kitchen:
    (2.06, -0.23, 0.0),
    (2.07, -1.43, 0.0),
    (-2.44, -0.38, 0.0),
    (-2.56, -1.47, 0.0)

    # Living room:
    (-2.49, 1.87, 0.0),
    (-2.50, 5.49, 0.0),
    (0.79, 5.73, 0.0),
    (0.92, 1.01, 0.0)

    Before starting anything, make sure to load available topics, services and actions.
    """
NAVIGATION_SERVICES_AND_TYPES: Dict[str, str] = {
    "/assisted_teleop/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/assisted_teleop/_action/get_result": "nav2_msgs/action/AssistedTeleop_GetResult",
    "/assisted_teleop/_action/send_goal": "nav2_msgs/action/AssistedTeleop_SendGoal",
    "/backup/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/backup/_action/get_result": "nav2_msgs/action/BackUp_GetResult",
    "/backup/_action/send_goal": "nav2_msgs/action/BackUp_SendGoal",
    "/behavior_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/behavior_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/behavior_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/behavior_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/behavior_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/behavior_server/get_state": "lifecycle_msgs/srv/GetState",
    "/behavior_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/behavior_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/behavior_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator/change_state": "lifecycle_msgs/srv/ChangeState",
    "/bt_navigator/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/bt_navigator/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator/get_state": "lifecycle_msgs/srv/GetState",
    "/bt_navigator/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_through_poses_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_to_pose_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/compute_path_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_through_poses/_action/get_result": "nav2_msgs/action/ComputePathThroughPoses_GetResult",
    "/compute_path_through_poses/_action/send_goal": "nav2_msgs/action/ComputePathThroughPoses_SendGoal",
    "/compute_path_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_to_pose/_action/get_result": "nav2_msgs/action/ComputePathToPose_GetResult",
    "/compute_path_to_pose/_action/send_goal": "nav2_msgs/action/ComputePathToPose_SendGoal",
    "/controller_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/controller_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/controller_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/controller_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/controller_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/controller_server/get_state": "lifecycle_msgs/srv/GetState",
    "/controller_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/controller_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/controller_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/drive_on_heading/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/drive_on_heading/_action/get_result": "nav2_msgs/action/DriveOnHeading_GetResult",
    "/drive_on_heading/_action/send_goal": "nav2_msgs/action/DriveOnHeading_SendGoal",
    "/follow_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_path/_action/get_result": "nav2_msgs/action/FollowPath_GetResult",
    "/follow_path/_action/send_goal": "nav2_msgs/action/FollowPath_SendGoal",
    "/follow_waypoints/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_waypoints/_action/get_result": "nav2_msgs/action/FollowWaypoints_GetResult",
    "/follow_waypoints/_action/send_goal": "nav2_msgs/action/FollowWaypoints_SendGoal",
    "/global_costmap/clear_around_global_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/global_costmap/clear_entirely_global_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/global_costmap/clear_except_global_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/global_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/global_costmap/global_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/global_costmap/global_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/global_costmap/global_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/global_costmap/global_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/global_costmap/global_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/global_costmap/global_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/global_costmap/global_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/global_costmap/global_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/global_costmap/global_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounded_sam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounded_sam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounded_sam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounded_sam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounded_sam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounded_sam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    "/grounding_dino/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounding_dino/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounding_dino/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounding_dino/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounding_dino/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounding_dino/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    "/is_path_valid": "nav2_msgs/srv/IsPathValid",
    "/launch_ros_138640/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/launch_ros_138640/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/launch_ros_138640/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/launch_ros_138640/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/launch_ros_138640/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/launch_ros_138640/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_navigation/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_navigation/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_navigation/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_navigation/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_navigation/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_navigation/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_navigation/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_navigation/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_slam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_slam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_slam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_slam/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_slam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_slam/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_slam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_slam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/local_costmap/clear_around_local_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/local_costmap/clear_entirely_local_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/local_costmap/clear_except_local_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/local_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/local_costmap/local_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/local_costmap/local_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/local_costmap/local_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/local_costmap/local_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/local_costmap/local_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/local_costmap/local_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/local_costmap/local_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/local_costmap/local_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/local_costmap/local_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/map_saver/change_state": "lifecycle_msgs/srv/ChangeState",
    "/map_saver/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/map_saver/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/map_saver/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/map_saver/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/map_saver/get_state": "lifecycle_msgs/srv/GetState",
    "/map_saver/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/map_saver/save_map": "nav2_msgs/srv/SaveMap",
    "/map_saver/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/map_saver/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/nav2_container/_container/list_nodes": "composition_interfaces/srv/ListNodes",
    "/nav2_container/_container/load_node": "composition_interfaces/srv/LoadNode",
    "/nav2_container/_container/unload_node": "composition_interfaces/srv/UnloadNode",
    "/navigate_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_through_poses/_action/get_result": "nav2_msgs/action/NavigateThroughPoses_GetResult",
    "/navigate_through_poses/_action/send_goal": "nav2_msgs/action/NavigateThroughPoses_SendGoal",
    "/navigate_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_to_pose/_action/get_result": "nav2_msgs/action/NavigateToPose_GetResult",
    "/navigate_to_pose/_action/send_goal": "nav2_msgs/action/NavigateToPose_SendGoal",
    "/o3de_ros2_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/o3de_ros2_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/o3de_ros2_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/o3de_ros2_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/o3de_ros2_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/o3de_ros2_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/planner_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/planner_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/planner_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/planner_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/planner_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/planner_server/get_state": "lifecycle_msgs/srv/GetState",
    "/planner_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/planner_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/planner_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/rai_ros2_ari_connector_b6ed00ab6356/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/clear_changes": "slam_toolbox/srv/Clear",
    "/slam_toolbox/clear_queue": "slam_toolbox/srv/ClearQueue",
    "/slam_toolbox/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/slam_toolbox/deserialize_map": "slam_toolbox/srv/DeserializePoseGraph",
    "/slam_toolbox/dynamic_map": "nav_msgs/srv/GetMap",
    "/slam_toolbox/get_interactive_markers": "visualization_msgs/srv/GetInteractiveMarkers",
    "/slam_toolbox/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/slam_toolbox/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/slam_toolbox/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/slam_toolbox/manual_loop_closure": "slam_toolbox/srv/LoopClosure",
    "/slam_toolbox/pause_new_measurements": "slam_toolbox/srv/Pause",
    "/slam_toolbox/save_map": "slam_toolbox/srv/SaveMap",
    "/slam_toolbox/serialize_map": "slam_toolbox/srv/SerializePoseGraph",
    "/slam_toolbox/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/slam_toolbox/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/toggle_interactive_mode": "slam_toolbox/srv/ToggleInteractive",
    "/smooth_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/smooth_path/_action/get_result": "nav2_msgs/action/SmoothPath_GetResult",
    "/smooth_path/_action/send_goal": "nav2_msgs/action/SmoothPath_SendGoal",
    "/smoother_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/smoother_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/smoother_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/smoother_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/smoother_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/smoother_server/get_state": "lifecycle_msgs/srv/GetState",
    "/smoother_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/smoother_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/smoother_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/spin/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/spin/_action/get_result": "nav2_msgs/action/Spin_GetResult",
    "/spin/_action/send_goal": "nav2_msgs/action/Spin_SendGoal",
    "/tf2_frames": "tf2_msgs/srv/FrameGraph",
    "/velocity_smoother/change_state": "lifecycle_msgs/srv/ChangeState",
    "/velocity_smoother/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/velocity_smoother/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/velocity_smoother/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/velocity_smoother/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/velocity_smoother/get_state": "lifecycle_msgs/srv/GetState",
    "/velocity_smoother/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/velocity_smoother/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/velocity_smoother/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/wait/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/wait/_action/get_result": "nav2_msgs/action/Wait_GetResult",
    "/wait/_action/send_goal": "nav2_msgs/action/Wait_SendGoal",
    "/waypoint_follower/change_state": "lifecycle_msgs/srv/ChangeState",
    "/waypoint_follower/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/waypoint_follower/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/waypoint_follower/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/waypoint_follower/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/waypoint_follower/get_state": "lifecycle_msgs/srv/GetState",
    "/waypoint_follower/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/waypoint_follower/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/waypoint_follower/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
}


# class NavigateToPointTask(ROS2ToolCallingAgentTask):
#     complexity = "medium"
#     actions_and_types: Dict[str, str] = {
#         "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
#         "/backup": "nav2_msgs/action/BackUp",
#         "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
#         "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
#         "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
#         "/follow_path": "nav2_msgs/action/FollowPath",
#         "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
#         "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
#         "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
#         "/smooth_path": "nav2_msgs/action/SmoothPath",
#         "/spin": "nav2_msgs/action/Spin",
#         "/wait": "nav2_msgs/action/Wait",
#     }
#     services_and_types: Dict[str, str] = NAVIGATION_SERVICES_AND_TYPES
#     interfaces: Dict[str, Dict[str, Any]] = {
#         "nav2_msgs/action/NavigateToPose": {
#             "goal": {
#                 "pose": {
#                     "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
#                     "pose": {
#                         "position": {"x": 0.0, "y": 0.0, "z": 0.0},
#                         "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
#                     },
#                 },
#                 "behavior_tree": "",
#             },
#             "result": {"result": {}},
#             "feedback": {
#                 "current_pose": {
#                     "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
#                     "pose": {
#                         "position": {"x": 0.0, "y": 0.0, "z": 0.0},
#                         "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
#                     },
#                 },
#                 "navigation_time": {"sec": 0, "nanosec": 0},
#                 "estimated_time_remaining": {"sec": 0, "nanosec": 0},
#                 "number_of_recoveries": 0,
#                 "distance_remaining": 0.0,
#             },
#         },
#     }
#     action_models: List[type[ActionBaseModel]] = [NavigateToPoseAction]

#     def __init__(self, logger: loggers_type | None = None) -> None:
#         super().__init__(logger=logger)
#         action_strings = [
#             f"action: {action}\ntype: {act_type}\n"
#             for action, act_type in self.actions_and_types.items()
#         ]
#         service_strings = [
#             f"service: {service}\ntype: {srv_type}\n"
#             for service, srv_type in self.services_and_types.items()
#         ]
#         interface_strings = {
#             msg_type: json.dumps(interface)
#             for msg_type, interface in self.interfaces.items()
#         }

#         self.expected_tools: List[BaseTool] = [
#             MockGetROS2ActionsNamesAndTypesTool(
#                 mock_actions_names_and_types=action_strings
#             ),
#             MockStartROS2ActionTool(
#                 available_actions=list(self.actions_and_types.keys()),
#                 available_action_types=list(self.actions_and_types.values()),
#                 available_action_models=self.action_models,
#             ),
#             MockGetROS2ActionFeedbackTool(),
#             MockGetROS2ActionResultTool(),
#             MockGetROS2ServicesNamesAndTypesTool(
#                 mock_service_names_and_types=service_strings
#             ),
#             MockGetROS2MessageInterfaceTool(mock_interfaces=interface_strings),
#         ]

#     def get_system_prompt(self) -> str:
#         return ROBOT_NAVIGATION_SYSTEM_PROMPT

#     def get_prompt(self) -> str:
#         return (
#             "Call action /perform_task with the provided goal values: "
#             "{priority: 10, description: '', task: 'Where are you?'}"
#         )


# class SpinAroundTask(ROS2ToolCallingAgentTask):
#     complexity = "medium"
#     interfaces: Dict[str, Dict[str, Any]] = {
#         "nav2_msgs/action/Spin": {
#             "goal": {"target_yaw": 0.0, "time_allowance": {"sec": 0, "nanosec": 0}},
#             "result": {"total_elapsed_time": {"sec": 0, "nanosec": 0}},
#             "feedback": {"angular_distance_traveled": 0.0},
#         }
#     }
#     actions_and_types: Dict[str, str] = {
#         "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
#         "/backup": "nav2_msgs/action/BackUp",
#         "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
#         "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
#         "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
#         "/follow_path": "nav2_msgs/action/FollowPath",
#         "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
#         "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
#         "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
#         "/smooth_path": "nav2_msgs/action/SmoothPath",
#         "/spin": "nav2_msgs/action/Spin",
#         "/wait": "nav2_msgs/action/Wait",
#     }
#     action_models: List[type[ActionBaseModel]] = [SpinAction]

#     def __init__(self, logger: loggers_type | None = None) -> None:
#         super().__init__(logger=logger)
#         action_strings = [
#             f"action: {action}\ntype: {act_type}\n"
#             for action, act_type in self.actions_and_types.items()
#         ]
#         self.expected_tools: List[BaseTool] = [
#             MockGetROS2ActionsNamesAndTypesTool(
#                 mock_actions_names_and_types=action_strings
#             ),
#             MockStartROS2ActionTool(
#                 available_actions=list(self.actions_and_types.keys()),
#                 available_action_types=list(self.actions_and_types.values()),
#                 available_action_models=self.action_models,
#             ),
#             MockGetROS2ActionFeedbackTool(),
#             MockGetROS2ActionResultTool(),
#         ]

#     def get_system_prompt(self) -> str:
#         return ROBOT_NAVIGATION_SYSTEM_PROMPT

#     def get_prompt(self) -> str:
#         return "Spin around by 3 radians."

#     def verify_tool_calls(self, response: dict[str, Any]):
#
# messages = response["messages"]
# ai_messages: Sequence[AIMessage] = [
#     message for message in messages if isinstance(message, AIMessage)
# ]
# tool_calls = [
#     tool_call for message in ai_messages for tool_call in message.tool_calls
# ]
# expected_tool_calls: list[dict[str, Any]] = [
#     {"name": "get_ros2_actions_names_and_types", "args": {}},
#     {
#         "name": "start_ros2_action",
#         "args": {
#             "action_name": "/spin",
#             "action_type": "nav2_msgs/action/Spin",
#             "action_args": {"target_yaw": 3},
#         },
#         "optional_args": {
#             "action_args": {
#                 "time_allowance": {"sec": ANY_VALUE, "nanosec": ANY_VALUE}
#             }
#         },
#     },
#     {"name": "get_ros2_action_feedback", "args": {"action_id": ANY_VALUE}},
#     {"name": "get_ros2_action_result", "args": {"action_id": ANY_VALUE}},
# ]
# self._check_multiple_tool_calls_from_list(
#     tool_calls=tool_calls, expected_tool_calls=expected_tool_calls
# )
# if not self.result.errors:
#     self.result.success = True
