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
from typing import Any, List

from langchain_core.tools import BaseTool
from rai.types import (
    BoundingBox2D,
    Detection2D,
    Header,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Time,
)
from rai.types.rai_interfaces import (
    RAIDetectionArray,
)

from rai_bench.tool_calling_agent.interfaces import Task, TaskArgs, Validator
from rai_bench.tool_calling_agent.mocked_ros2_interfaces import (
    COMMON_INTERFACES,
    COMMON_SERVICES_AND_TYPES,
    COMMON_TOPIC_MODELS,
    COMMON_TOPICS_AND_TYPES,
    CUSTOM_INTERFACES,
    CUSTOM_SERVICE_MODELS,
    CUSTOM_SERVICES_AND_TYPES,
    CUSTOM_TOPIC_MODELS,
    CUSTOM_TOPICS_AND_TYPES,
)
from rai_bench.tool_calling_agent.mocked_tools import (
    MockCallROS2ServiceTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockPublishROS2MessageTool,
)

loggers_type = logging.Logger
INTERFACES = COMMON_INTERFACES | CUSTOM_INTERFACES

TOPICS_AND_TYPES = COMMON_TOPICS_AND_TYPES | CUSTOM_TOPICS_AND_TYPES
TOPIC_MODELS = COMMON_TOPIC_MODELS | CUSTOM_TOPIC_MODELS

SERVICES_AND_TYPES = COMMON_SERVICES_AND_TYPES | CUSTOM_SERVICES_AND_TYPES
TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {msg_type}\n"
    for topic, msg_type in TOPICS_AND_TYPES.items()
]


SERVICE_STRINGS = [
    f"service: {service}\ntype: {msg_type}\n"
    for service, msg_type in SERVICES_AND_TYPES.items()
]


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
- get_ros2_message_interface, args: {'msg_type': 'rai_interfaces/msg/HRIMessage'}
- call_ros2_service, args: {'service': '/grounding_dino_classify', 'service_type': 'rai_interfaces/srv/RAIGroundingDino', 'request': {'classes': 'bottle, book', 'box_threshold': 0.4, 'text_threshold': 0.25}}"""
)


class CustomInterfaceTask(Task, ABC):
    type = "custom_interface"

    @property
    def optional_tool_calls_number(self) -> int:
        # list topics
        # get interface is not optional
        return 1

    def get_system_prompt(self) -> str:
        if self.n_shots == 0:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
        elif self.n_shots == 2:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
        else:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT


class CustomInterfacesTopicTask(CustomInterfaceTask, ABC):
    def __init__(
        self, topic: str, validators: List[Validator], task_args: TaskArgs
    ) -> None:
        super().__init__(validators=validators, task_args=task_args)
        self.topic = topic

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockPublishROS2MessageTool(
                available_topics=list(TOPICS_AND_TYPES.keys()),
                available_message_types=list(TOPICS_AND_TYPES.values()),
                available_topic_models=TOPIC_MODELS,
            ),
        ]


class CustomInterfacesServiceTask(CustomInterfaceTask, ABC):
    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args)
        self.service = service
        self.service_args = service_args

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=SERVICE_STRINGS
            ),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockCallROS2ServiceTool(
                available_services=list(SERVICES_AND_TYPES.keys()),
                available_service_types=list(SERVICES_AND_TYPES.values()),
                available_service_models=CUSTOM_SERVICE_MODELS,
            ),
        ]


class PublishROS2HRIMessageTextTask(CustomInterfacesTopicTask):
    complexity = "easy"

    def __init__(
        self,
        topic: str,
        validators: List[Validator],
        task_args: TaskArgs,
        text: str = "Hello!",
    ) -> None:
        super().__init__(topic, validators=validators, task_args=task_args)
        self.text = text

    def get_base_prompt(self) -> str:
        return f"Publish message to topic '{self.topic}' with text: '{self.text}'."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover available topics, examine the message interface "
                f"structure, and publish an HRI message containing the text '{self.text}'."
            )


class PublishROS2AudioMessageTask(CustomInterfacesTopicTask):
    complexity = "easy"

    def __init__(
        self,
        topic: str,
        validators: List[Validator],
        task_args: TaskArgs,
        audio: List[int] = [123, 456, 789],
        sample_rate: int = 44100,
        channels: int = 2,
    ) -> None:
        super().__init__(topic, validators=validators, task_args=task_args)
        self.expected_audio = audio
        self.expected_sample_rate = sample_rate
        self.expected_channels = channels

    def get_base_prompt(self) -> str:
        return (
            f"Publish audio message to topic '{self.topic}' with samples "
            f"{self.expected_audio}, sample rate {self.expected_sample_rate}, "
            f"channels {self.expected_channels}."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can explore available audio topics, examine the message "
                f"interface, and publish audio data with samples={self.expected_audio}, "
                f"sample_rate={self.expected_sample_rate}, and channels={self.expected_channels}."
            )


class PublishROS2DetectionArrayTask(CustomInterfacesTopicTask):
    complexity = "easy"

    def __init__(
        self,
        topic: str,
        validators: List[Validator],
        task_args: TaskArgs,
        detection_classes: List[str] = ["person", "car"],
        bbox_center_x: float = 320.0,
        bbox_center_y: float = 320.0,
        bbox_size_x: float = 50.0,
        bbox_size_y: float = 50.0,
    ) -> None:
        super().__init__(topic, validators=validators, task_args=task_args)
        self.expected_detection_classes = detection_classes
        self.expected_detections = [
            Detection2D(
                bbox=BoundingBox2D(
                    center=Pose2D(x=bbox_center_x, y=bbox_center_y, theta=0.0),
                    size_x=bbox_size_x,
                    size_y=bbox_size_y,
                )
            )
        ]

    def get_base_prompt(self) -> str:
        bbox_center = self.expected_detections[0].bbox.center
        bbox_size = self.expected_detections[0].bbox
        return (
            f"Publish detection array to topic '{self.topic}' with classes "
            f"{self.expected_detection_classes} and bbox center "
            f"({bbox_center.x}, {bbox_center.y}) size {bbox_size.size_x}x{bbox_size.size_y}."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            bbox_center = self.expected_detections[0].bbox.center
            bbox_size = self.expected_detections[0].bbox
            return (
                f"{self.get_base_prompt()} "
                "You can explore available detection topics, examine the message "
                f"interface, and publish detection data with classes={self.expected_detection_classes} "
                f"and bounding box at center ({bbox_center.x}, {bbox_center.y}) "
                f"with size_x={bbox_size.size_x}, size_y={bbox_size.size_y}."
            )


class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
        target_x: float = 1.0,
        target_y: float = 2.0,
        target_z: float = 3.0,
        initial_gripper_state: bool = True,
        final_gripper_state: bool = False,
        frame_id: str = "base_link",
    ) -> None:
        super().__init__(
            service, service_args, validators=validators, task_args=task_args
        )
        self.expected_initial_gripper_state = initial_gripper_state
        self.expected_final_gripper_state = final_gripper_state
        self.expected_target_pose = PoseStamped(
            header=Header(frame_id=frame_id),
            pose=Pose(
                position=Point(x=target_x, y=target_y, z=target_z),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

    def get_base_prompt(self) -> str:
        pos = self.expected_target_pose.pose.position
        return (
            f"Call service '{self.service}' to move manipulator to pose "
            f"({pos.x}, {pos.y}, {pos.z}) with initial_gripper={self.expected_initial_gripper_state}, "
            f"final_gripper={self.expected_final_gripper_state}."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            pos = self.expected_target_pose.pose.position
            return (
                f"{self.get_base_prompt()} "
                "You can discover available manipulation services, examine the service "
                f"interface, and call the service with target_pose position (x={pos.x}, "
                f"y={pos.y}, z={pos.z}), initial_gripper_state={self.expected_initial_gripper_state}, "
                f"and final_gripper_state={self.expected_final_gripper_state}."
            )


class CallGroundedSAMSegmentTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
        frame_id: str = "camera_frame",
    ) -> None:
        super().__init__(
            service, service_args, validators=validators, task_args=task_args
        )
        self.expected_detections = RAIDetectionArray(
            header=Header(stamp=Time(sec=0, nanosec=0), frame_id=frame_id),
            detections=[],
        )

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' for image segmentation."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            frame_id = self.expected_detections.header.frame_id
            return (
                f"{self.get_base_prompt()} "
                "You can discover available AI vision services, examine the service "
                f"interface, and call the segmentation service with detections array "
                f"(empty detections, header frame_id='{frame_id}') and source image."
            )


class CallGroundingDinoClassify(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
        classes: str = "bottle, book, chair",
        box_threshold: float = 0.4,
        text_threshold: float = 0.25,
    ) -> None:
        super().__init__(
            service, service_args, validators=validators, task_args=task_args
        )
        self.expected_classes = classes
        self.expected_box_threshold = box_threshold
        self.expected_text_threshold = text_threshold

    def get_base_prompt(self) -> str:
        return (
            f"Call service '{self.service}' for object classification with classes "
            f"'{self.expected_classes}', box_threshold {self.expected_box_threshold}, "
            f"text_threshold {self.expected_text_threshold}."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover available AI detection services, examine the service "
                f"interface, and call the classification service with classes='{self.expected_classes}', "
                f"box_threshold={self.expected_box_threshold}, and text_threshold={self.expected_text_threshold}."
            )


class CallGetLogDigestTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
    ) -> None:
        super().__init__(
            service, service_args, validators=validators, task_args=task_args
        )

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' to get log digest."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover available logging services, examine the service "
                "interface, and call the service with an empty request to retrieve "
                "system log information."
            )


class CallVectorStoreRetrievalTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
        query: str = "What is the purpose of this robot?",
    ) -> None:
        super().__init__(
            service, service_args, validators=validators, task_args=task_args
        )
        self.expected_query = query

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' with query '{self.expected_query}'"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover available knowledge services, examine the service "
                f"interface, and call the retrieval service with query='{self.expected_query}' "
                "to search the robot's knowledge base."
            )


class CallWhatISeeTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        service: str,
        service_args: dict[str, Any],
        validators: List[Validator],
        task_args: TaskArgs,
    ) -> None:
        super().__init__(
            service, service_args, validators=validators, task_args=task_args
        )

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' to get visual observations."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} "
                "You can discover available vision services, examine the service "
                "interface, and call the service with an empty request to get "
                "visual observations and camera pose information."
            )
