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
from typing import List, Tuple

from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
    Validator,
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

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT = """You are a ROS 2 expert that want to solve tasks. You have access to various tools that allow you to query the ROS 2 system.
Be proactive and use the tools to answer questions."""

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
    + """
Example of tool calls:
- name: get_ros2_topics_names_and_types, args: {}
- name: get_ros2_service_interface, args: {"service_type": "tf2_msgs/srv/LookupTransform"}"""
)

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
    + """
- name: get_ros2_image, args: {'topic': '/camera/image_raw', 'timeout_sec': 10}
- name: receive_ros2_message, args: {'topic': '/cmd_vel', 'timeout_sec': 10}
- name: call_ros2_service, args: {
        "service_name": "/execute_trajectory",
        "service_type": "moveit_msgs/srv/ExecuteKnownTrajectory",
        "service_args": {
            "trajectory": {
                "joint_trajectory": {
                    "header": {"frame_id": "base_link"},
                    "joint_names": ["joint1", "joint2"],
                    "points": [{
                        "positions": [0.0, 1.57],
                        "time_from_start": {"sec": 2, "nanosec": 0}
                    }]
                }
            },
            "wait_for_execution": True
        }
    }
"""
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
                "You can list available camera topics and capture the RGB color image."
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
                "You can list available camera topics and capture the depth image data."
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
                "You can list available topics to find appropriate topic and receive the pointcloud information from it."
            )


class GetRobotDescriptionTask(BasicTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return "Get robot description."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()}"
        else:
            return (
                f"{self.get_base_prompt()}. You can list available topics to find appropriate topic "
                "and receive robot description data from it."
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
                "You can list what camera topics are available and capture images from each."
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
                "You can list what services are currently available in the system."
            )


class ListRobotParametersTask(BasicTask):
    complexity = "easy"

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
                "You can list available services to find the appropriate service and receive the parameters from it"
            )


class GetSpecificParameterTask(BasicTask):
    complexity = "easy"

    def __init__(
        self,
        parameter: str,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators, task_args, logger)
        self.parameter = parameter

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interfaces
        return 2

    def get_base_prompt(self) -> str:
        return f"Get robot `{self.parameter}` parameter"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} value from the robot state publisher."
        else:
            return (
                f"{self.get_base_prompt()} value from the robot state publisher. "
                "You can list available services to find the appropriate service, "
                f"check its type's interface and retrieve the {self.parameter} parameter value."
            )


class SetRobotParameterTask(BasicTask):
    complexity = "medium"

    def __init__(
        self,
        value: float,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators, task_args, logger)
        self.value = value

    @property
    def optional_tool_calls_number(self) -> int:
        # list services, get interfaces
        return 2

    def get_base_prompt(self) -> str:
        return f"Set robot state parameter `publish frequency` to {self.value} Hz"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} using parameter service. "
                "You can list available services to find the appropriate service, "
                f"check its type's interface and set the publish_frequency parameter to {self.value}."
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
                "You can list available services to find the appropriate "
                "service and see what entities can be spawned."
            )


class SpawnEntityTask(BasicTask):
    complexity = "medium"

    def __init__(
        self,
        entity: str,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators, task_args, logger)
        self.entity = entity

    @property
    def optional_tool_calls_number(self) -> int:
        # list services, get interface
        return 2

    def get_base_prompt(self) -> str:
        return f"Spawn a {self.entity} entity"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} in the simulation environment"
        else:
            return (
                f"{self.get_base_prompt()} in the simulation environment. "
                "You can list available services to find the appropriate service, "
                f"check its type's interface and add a {self.entity} with any name and SDF/XML description."
            )


class ConfigureVisionPipelineTask(BasicTask):
    complexity = "hard"

    def __init__(
        self,
        sam_confidence_threshold: float,
        dino_confidence_threshold: float,
        fps: int,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators, task_args, logger)
        self.sam_confidence_threshold = sam_confidence_threshold
        self.dino_confidence_threshold = dino_confidence_threshold
        self.fps = fps

    @property
    def optional_tool_calls_number(self) -> int:
        return 2  # list services, get interface

    def get_base_prompt(self) -> str:
        return (
            f"Configure AI vision pipeline: set grounded_sam `confidence_threshold` "
            f"to {self.sam_confidence_threshold}, grounding_dino `confidence_threshold` "
            f"to {self.dino_confidence_threshold}, o3de_ros2_node `fps` to {self.fps}"
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} using parameter services."
        else:
            return (
                f"{self.get_base_prompt()} using parameter services. "
                "You can list parameter services to find appropriate services "
                "check their type's interface and set appropriate parameters."
            )


class RespawnEntitiesTask(BasicTask):
    complexity = "hard"

    def __init__(
        self,
        names: List[str],
        coords: List[Tuple[float, float, float]],
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators, task_args, logger)
        self.names = names
        self.coords = coords

    @property
    def optional_tool_calls_number(self) -> int:
        return 3  # list services, get interfaces of spawn and despawn

    def get_base_prompt(self) -> str:
        names_str = ", ".join(self.names)
        positions: List[str] = []
        for coord in self.coords:
            positions.append(f"({coord[0]}, {coord[1]}, {coord[2]})")
        positions_str = ", ".join(positions)

        return (
            f"Reconfigure simulation: remove old `cube` entities named {names_str}, "
            f"then respawn them at positions [{positions_str}]"
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} using entity management services. "
        else:
            return (
                f"{self.get_base_prompt()} using entity management services. "
                "You can list services to find appropriate services, check their type's interface "
                "and use them to delete old and spawn new `cube` entities with specific names and positions."
            )
