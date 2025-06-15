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
from typing import Any, List, Tuple

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


class CustomInterfacesServiceTask(CustomInterfaceTask, ABC):
    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, logger=logger)
        self.moderate_sufix = " using service interface."
        self.descriptive_sufix = (
            ". Examine the message interface, and publish "
            "detection data with appropriate arguments."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return self.get_base_prompt() + self.moderate_sufix
        else:
            return self.get_base_prompt() + self.descriptive_sufix


class PublishROS2HRIMessageTextTask(CustomInterfaceTask):
    complexity = "easy"
    topic = "/to_human"

    def __init__(
        self,
        text: str,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, logger=logger)
        self.text = text

    def get_base_prompt(self) -> str:
        return f"Publish message to topic '{self.topic}' with text '{self.text}'"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return f"{self.get_base_prompt()} using HRI message interface with text='{self.text}'"
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} using HRI message interface. Set the text field to '{self.text}'"
        else:
            return (
                f"{self.get_base_prompt()}. "
                "Examine the message interface "
                f"structure, and publish an HRI message with appropriate arguments."
            )


class PublishROS2AudioMessageTask(CustomInterfaceTask):
    complexity = "easy"
    topic = "/audio_message"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        audio: List[int] = [123, 456, 789],
        sample_rate: int = 44100,
        channels: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, logger=logger)
        self.expected_audio = audio
        self.expected_sample_rate = sample_rate
        self.expected_channels = channels

    def get_base_prompt(self) -> str:
        return (
            f"Publish audio message to topic '{self.topic}' with samples "
            f"{self.expected_audio}, sample rate {self.expected_sample_rate} and "
            f"channels {self.expected_channels}"
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()}. "
                f"Examine the message interface, and publish audio data with appropriate arguments."
            )


class PublishROS2DetectionArrayTask(CustomInterfaceTask):
    complexity = "medium"
    topic = "/detection_array"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        detection_classes: List[str],
        bbox_centers: List[Tuple[float, float]],
        bbox_sizes: List[Tuple[float, float]],
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, logger=logger)
        if not (len(detection_classes) == len(bbox_centers) == len(bbox_sizes)):
            raise ValueError("detection_classes, bbox_centers, and bbox_sizes must have the same length")
        
        self.expected_detection_classes = detection_classes
        self.expected_bbox_centers = bbox_centers
        self.expected_bbox_sizes = bbox_sizes

  
    def get_base_prompt(self) -> str:
        detection_summaries:List[str] = []
        for _, (cls, center, size) in enumerate(zip(
            self.expected_detection_classes, 
            self.expected_bbox_centers, 
            self.expected_bbox_sizes
        )):
            detection_summaries.append(
                f"{cls} with bbox at center({center[0]}, {center[1]}) and size {size[0]}x{size[1]}"
            )
        
        return (
            f"Publish detection array to topic '{self.topic}' with {len(self.expected_detection_classes)} detections: "
            f"{'; '.join(detection_summaries)}"
        )
    
    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} using service interface."
        else:
            return (
                f"{self.get_base_prompt()}. Examine the message interface"
                "and publish detection data with appropriate arguments."
            )


class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesServiceTask):
    complexity = "medium"
    service = "/manipulator_move_to"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        target_x: float = 1.0,
        target_y: float = 2.0,
        target_z: float = 3.0,
        initial_gripper_state: bool = True,
        final_gripper_state: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger
        )
        self.expected_initial_gripper_state = initial_gripper_state
        self.expected_final_gripper_state = final_gripper_state
        self.expected_target_pose = PoseStamped(
            pose=Pose(
                position=Point(x=target_x, y=target_y, z=target_z),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

    def get_base_prompt(self) -> str:
        pos = self.expected_target_pose.pose.position
        return (
            f"Call service '{self.service}' to move manipulator to pose "
            f"({pos.x}, {pos.y}, {pos.z}) with initial gripper state {self.expected_initial_gripper_state} "
            f"and final gripper state {self.expected_final_gripper_state}"
        )


class CallGroundedSAMSegmentTask(CustomInterfacesServiceTask):
    complexity = "medium"
    service = "/grounded_sam_segment"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        detection_classes: List[str],
        bbox_centers: List[Tuple[float, float]],
        bbox_sizes: List[Tuple[float, float]],
        scores: List[float],
        positions_3d: List[Tuple[float, float, float]],
        image_width: int,
        image_height: int,
        image_encoding: str,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger
        )
        
        # Validate list lengths
        if not (len(detection_classes) == len(bbox_centers) == len(bbox_sizes) == 
                len(scores) == len(positions_3d)):
            raise ValueError("All detection parameter lists must have the same length")
        
        self.detection_classes = detection_classes
        self.bbox_centers = bbox_centers
        self.bbox_sizes = bbox_sizes
        self.scores = scores
        self.positions_3d = positions_3d
        self.image_width = image_width
        self.image_height = image_height
        self.image_encoding = image_encoding
        

    def get_base_prompt(self) -> str:
        detection_summary: List[str] = []
        for cls, center, size, score, pos3d in zip(
            self.detection_classes, self.bbox_centers, self.bbox_sizes, self.scores, self.positions_3d
        ):
            detection_summary.append(
                f"{cls} with score {score} at 3D position ({pos3d[0]}, {pos3d[1]}, {pos3d[2]}) "
                f"bbox ({center[0]}, {center[1]}) size {size[0]}x{size[1]}"
            )
        
        return (
            f"Call service '{self.service}' for image segmentation with {len(self.detection_classes)} detections: "
            f"{', '.join(detection_summary)} on {self.image_width}x{self.image_height} {self.image_encoding} image"
        )

class CallGroundingDinoClassify(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/grounding_dino_classify"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        classes: str = "bottle, book, chair",
        box_threshold: float = 0.4,
        text_threshold: float = 0.25,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger
        )
        self.expected_classes = classes
        self.expected_box_threshold = box_threshold
        self.expected_text_threshold = text_threshold

    def get_base_prompt(self) -> str:
        return (
            f"Call service '{self.service}' for object classification with classes "
            f"'{self.expected_classes}', box_threshold {self.expected_box_threshold} and "
            f"text_threshold {self.expected_text_threshold}"
        )


class CallGetLogDigestTask(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/get_log_digest"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger
        )

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' to get log digest."


class CallVectorStoreRetrievalTask(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/rai_whoami_documentation_service"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        query: str = "What is the purpose of this robot?",
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger
        )
        self.expected_query = query

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' with query '{self.expected_query}'"


class CallWhatISeeTask(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/rai_whatisee_get"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            validators=validators, task_args=task_args, logger=logger
        )

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' to get visual observations"
