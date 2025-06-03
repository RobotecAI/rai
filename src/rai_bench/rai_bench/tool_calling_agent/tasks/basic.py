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
from abc import ABC
from typing import List

from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.mocked_ros2_interfaces import COMMON_TOPICS_AND_TYPES
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockReceiveROS2MessageTool,
)

loggers_type = logging.Logger
PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT = """You are a ROS 2 expert that want to solve tasks. You have access to various tools that allow you to query the ROS 2 system.
Be proactive and use the tools to answer questions."""

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
    + """
Example of tool calls:
- get_ros2_message_interface, args: {'msg_type': 'geometry_msgs/msg/Twist'}
- publish_ros2_message, args: {'topic': '/cmd_vel', 'message_type': 'geometry_msgs/msg/Twist', 'message': {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}}"""
)

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
    + """
- get_ros2_topics_names_and_types, args: {}
- get_ros2_image, args: {'topic': '/camera/image_raw', 'timeout_sec': 10}
- publish_ros2_message, args: {'topic': '/turtle1/teleport_absolute', 'message_type': 'turtlesim/srv/TeleportAbsolute', 'message': {x: 5.0, y: 2.0, theta: 1.57}}"""
)

TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {topic_type}\n"
    for topic, topic_type in COMMON_TOPICS_AND_TYPES.items()
]


class BasicTask(Task, ABC):
    type = "basic"

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetROS2ImageTool(available_topics=list(COMMON_TOPICS_AND_TYPES.keys())),
            MockReceiveROS2MessageTool(
                available_topics=list(COMMON_TOPICS_AND_TYPES.keys())
            ),
        ]

    @property
    def optional_tool_calls_number(self) -> int:
        # Listing topics before getting any message
        return 1

    def get_system_prompt(self) -> str:
        if self.n_shots == 0:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
        elif self.n_shots == 2:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
        else:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT


class GetROS2TopicsTask(BasicTask):
    complexity = "easy"

    @property
    def optional_tool_calls_number(self) -> int:
        return 0

    def get_prompt(self) -> str:
        base_prompt = "Get all topics"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} names and types of all ROS2"
        else:
            return f"{base_prompt} ROS2 with their names and message types. Use the topics tool to list what's available in the system."


class GetROS2RGBCameraTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        base_prompt = "Get RGB camera image"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} from the camera topic"
        else:
            return f"{base_prompt}. Get the RGB color image from the camera. First check what camera topics are available, then capture the image from the RGB camera topic."


class GetROS2DepthCameraTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        base_prompt = "Get depth camera image"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} from the camera sensor"
        else:
            return f"{base_prompt}. Get the depth image from the camera. First check what camera topics are available, then capture the image from the depth camera topic."


class GetPointcloudTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        base_prompt = "Get the pointcloud"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} data from the topic"
        else:
            return f"{base_prompt} data. First check available topics to find the pointcloud topic, then receive the pointcloud message."


class GetRobotDescriptionTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        base_prompt = "Get robot description"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} of the robot from the topic"
        else:
            return f"{base_prompt}. First list available topics to find the robot_description topic, then receive the robot description message."


class GetAllROS2CamerasTask(BasicTask):
    complexity = "medium"

    def get_prompt(self) -> str:
        base_prompt = "Get all camera images"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} from all available cameras in the system, both RGB and depth"
        else:
            return f"{base_prompt} from all available camera topics in the ROS2 system. This includes both RGB color images and depth images. You should first explore what camera topics are available."


class CheckRobotHealthTask(BasicTask):
    complexity = "medium"

    def get_prompt(self) -> str:
        base_prompt = "Check robot health status"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} by listing topics and using receive_ros2_message"
        else:
            return f"{base_prompt}.  First list available topics, then all receive_ros2_message on diagnostics, joint_states and rosout."


class AssessSensorDataQualityTask(BasicTask):
    complexity = "hard"

    def get_prompt(self) -> str:
        base_prompt = "Assess sensor data quality"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} by listing topics and using receive_ros2_message"
        else:
            return f"{base_prompt}. First list available topics to find the robot_description topic, then receive scan, point cloud, camera images, camera infos and odometry."
