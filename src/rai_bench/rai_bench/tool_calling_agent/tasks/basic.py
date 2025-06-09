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
from rai_bench.tool_calling_agent.mocked_ros2_interfaces import (
    COMMON_INTERFACES,
    COMMON_SERVICES_AND_TYPES,
    COMMON_TOPICS_AND_TYPES,
)
from rai_bench.tool_calling_agent.mocked_tools import (
    MockCallROS2ServiceTool,
    MockGetROS2ImageTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
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
- get_ros2_topics_names_and_types, args: {}
- receive_ros2_message, args: {'topic': '/cmd_vel', 'timeout_sec': 10}"""
)

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
    + """
- get_ros2_image, args: {'topic': '/camera/image_raw', 'timeout_sec': 10}
- publish_ros2_message, args: {'topic': '/turtle1/teleport_absolute', 'message_type': 'turtlesim/srv/TeleportAbsolute', 'message': {x: 5.0, y: 2.0, theta: 1.57}}
- receive_ros2_message, args: {'topic': '/cmd_vel', 'timeout_sec': 10}"""
)

TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {topic_type}\n"
    for topic, topic_type in COMMON_TOPICS_AND_TYPES.items()
]

SERVICE_STRINGS = [
    f"service: {service}\ntype: {msg_type}\n"
    for service, msg_type in COMMON_SERVICES_AND_TYPES.items()
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
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=SERVICE_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=COMMON_INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(COMMON_SERVICES_AND_TYPES.keys()),
                available_service_types=list(COMMON_SERVICES_AND_TYPES.values()),
                available_service_models={},
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

    def get_base_prompt(self) -> str:
        return "Get all topics"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} available in the ROS2 system with their names and message types. "
                "You can discover what topics are currently active."
            )


class GetROS2RGBCameraTask(BasicTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return "Get RGB camera image."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can explore available camera topics and capture the RGB color image."
            )


class GetROS2DepthCameraTask(BasicTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return "Get depth camera image."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can explore available camera topics and capture the depth image data."
            )


class GetPointcloudTask(BasicTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return "Get the pointcloud data."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover available sensor topics and receive the pointcloud information."
            )


class GetRobotDescriptionTask(BasicTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return "Get robot description."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} You can explore the system "
                "to find robot description data."
            )


class GetAllROS2CamerasTask(BasicTask):
    complexity = "medium"

    def get_base_prompt(self) -> str:
        return "Get all camera images"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} from all available camera sources in the system. "
                "This includes both RGB color images and depth images. "
                "You can discover what camera topics are available and capture images from each."
            )


#### calling services ####
class GetROS2ServicesTask(BasicTask):
    complexity = "easy"

    @property
    def optional_tool_calls_number(self) -> int:
        return 0

    def get_base_prompt(self) -> str:
        return "Get all services"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} available in the ROS2 system"
        else:
            return (
                f"{self.get_base_prompt()} available in the ROS2 system with their names and service types. "
                "You can discover what services are currently available in the system."
            )


class ListRobotParametersTask(BasicTask):
    complexity = "easy"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services
        return 1

    def get_base_prompt(self) -> str:
        return "List robot state publisher parameters"

    def get_prompt(self) -> str:
        base_prompt = "List robot state publisher parameters"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} available for configuration."
        else:
            return (
                f"{self.get_base_prompt()} available for configuration. "
                "You can explore available services to find the appropriate service."
            )


class GetSpecificParameterTask(BasicTask):
    complexity = "easy"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interfaces
        return 2

    def get_base_prompt(self) -> str:
        return "Get robot `publish_frequency` parameter"

    def get_prompt(self) -> str:
        base_prompt = "Get robot publish frequency parameter"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} value from the robot state publisher."
        else:
            return (
                f"{self.get_base_prompt()} value from the robot state publisher. "
                "You can explore available services and their interfaces to find "
                "the appropriate service and retrieve the publish_frequency parameter value."
            )


class SetRobotParameterTask(BasicTask):
    complexity = "medium"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services, get interfaces
        return 2

    def get_base_prompt(self) -> str:
        return "Set robot state parameter `publish frequency` to 30.0 Hz"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} using parameter service. "
                "You can explore available services to find the appropriate service, "
                "check its interface and set the publish_frequency parameter to 30.0."
            )


class CheckSpawnableEntitiesTask(BasicTask):
    complexity = "easy"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services
        return 1

    def get_base_prompt(self) -> str:
        return "Check available spawnable entities"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} in the simulation environment. "
                "You can explore available services to find the appropriate "
                "service and see what entities can be spawned."
            )


class SpawnEntityTask(BasicTask):
    complexity = "medium"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services, get interfaces
        return 2

    def get_base_prompt(self) -> str:
        return "Spawn a red_cube entity"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} in the simulation environment"
        else:
            return (
                f"{self.get_base_prompt()} in the simulation environment. "
                "You can explore available services to find the spawn_entity service, "
                "check its interface, and add a box with appropriate SDF/XML description."
            )
