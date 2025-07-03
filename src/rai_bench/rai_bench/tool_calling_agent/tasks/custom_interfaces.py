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
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool

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
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
    CheckServiceFieldsToolCallSubTask,
    CheckTopicFieldsToolCallSubTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

HRI_TOPIC = "/to_human"
AUDIO_TOPIC = "/audio_output"
DETECTIONS_TOPIC = "/detections"
MANIPULATOR_SERVICE = "/manipulator_move_to"
GROUNDED_SAM_SERVICE = "/grounded_sam_segment"
GROUNDING_DINO_SERVICE = "/grounding_dino_classify"
LOG_DIGEST_SERVICE = "/get_log_digest"
VECTOR_STORE_SERVICE = "/rai_whoami_documentation_service"
WHAT_I_SEE_SERVICE = "/rai_whatisee_get"

HRI_MESSAGE_TYPE = "rai_interfaces/msg/HRIMessage"
AUDIO_MESSAGE_TYPE = "rai_interfaces/msg/AudioMessage"
DETECTION_ARRAY_MESSAGE_TYPE = "rai_interfaces/msg/RAIDetectionArray"
MANIPULATOR_SERVICE_TYPE = "rai_interfaces/srv/ManipulatorMoveTo"
GROUNDED_SAM_SERVICE_TYPE = "rai_interfaces/srv/RAIGroundedSam"
GROUNDING_DINO_SERVICE_TYPE = "rai_interfaces/srv/RAIGroundingDino"
STRING_LIST_SERVICE_TYPE = "rai_interfaces/srv/StringList"
VECTOR_STORE_SERVICE_TYPE = "rai_interfaces/srv/VectorStoreRetrieval"
WHAT_I_SEE_SERVICE_TYPE = "rai_interfaces/srv/WhatISee"


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
- get_ros2_services_names_and_types, args: {}
- get_ros2_message_interface, args: {'msg_type': 'moveit_msgs/srv/ExecuteKnownTrajectory'}
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
    }"""
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
    """
    Base class for tasks that involve calling SINGLE service with custom interface.
    """

    descriptive_sufix = (
        " Examine the required service interface, and call  "
        "it with appropriate arguments."
    )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return self.get_base_prompt() + self.descriptive_sufix


class CustomInterfacesServicesTask(CustomInterfacesServiceTask, ABC):
    """
    Base class for tasks that involve calling MULITPLE services with custom interface.
    """

    descriptive_sufix = (
        " Examine the required services interfaces, and call  "
        "them with appropriate arguments."
    )


class PublishROS2HRIMessageTextTask(CustomInterfaceTask):
    complexity = "easy"
    topic = "/to_human"

    def __init__(
        self,
        text: str,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.text = text
        if validators is None:
            # Default validator for this task
            get_HRIMessage_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": HRI_MESSAGE_TYPE},
            )
            pub_HRIMessage_text_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=HRI_TOPIC,
                expected_message_type=HRI_MESSAGE_TYPE,
                expected_fields={"text": text},
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_HRIMessage_interface_subtask,
                        pub_HRIMessage_text_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return f"Publish message to topic '{self.topic}' with text '{self.text}'."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return f"{self.get_base_prompt()}."
        else:
            return (
                f"{self.get_base_prompt()}"
                " Examine the message interface "
                f"structure, and publish an HRI message with appropriate arguments."
            )


class PublishROS2AudioMessageTask(CustomInterfaceTask):
    complexity = "easy"
    topic = "/audio_message"

    def __init__(
        self,
        audio: List[int],
        sample_rate: int,
        channels: int,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.channels = channels
        if validators is None:
            # Default validator for this task
            get_audio_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": AUDIO_MESSAGE_TYPE},
            )
            pub_audio_message_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=AUDIO_TOPIC,
                expected_message_type=AUDIO_MESSAGE_TYPE,
                expected_fields={
                    "samples": audio,
                    "sample_rate": sample_rate,
                    "channels": channels,
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_audio_interface_subtask,
                        pub_audio_message_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return (
            f"Publish audio message to topic '{self.topic}' with samples "
            f"{self.audio}, sample rate {self.sample_rate} and "
            f"channels {self.channels}."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()}"
                f" Examine the message interface, and publish audio data with appropriate arguments."
            )


class PublishROS2DetectionArrayTask(CustomInterfaceTask):
    complexity = "medium"
    topic = "/detection_array"

    def __init__(
        self,
        task_args: TaskArgs,
        detection_classes: List[str],
        bbox_centers: List[Tuple[float, float]],
        bbox_sizes: List[Tuple[float, float]],
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not (len(detection_classes) == len(bbox_centers) == len(bbox_sizes)):
            raise ValueError(
                "detection_classes, bbox_centers, and bbox_sizes must have the same length"
            )

        self.expected_detection_classes = detection_classes
        self.expected_bbox_centers = bbox_centers
        self.expected_bbox_sizes = bbox_sizes

        if validators is None:
            # Create default validator based on the detection parameters
            get_detection_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": "rai_interfaces/msg/RAIDetectionArray"},
            )

            # Build expected fields dynamically based on detection parameters
            expected_fields: Dict[str, Any] = {}
            for i, (cls, center, size) in enumerate(
                zip(detection_classes, bbox_centers, bbox_sizes)
            ):
                expected_fields.update(
                    {
                        f"detections.{i}.results.0.hypothesis.class_id": cls,
                        f"detections.{i}.bbox.center.x": center[0],
                        f"detections.{i}.bbox.center.y": center[1],
                        f"detections.{i}.bbox.size_x": size[0],
                        f"detections.{i}.bbox.size_y": size[1],
                    }
                )

            pub_detection_array_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic="/detections",
                expected_message_type="rai_interfaces/msg/RAIDetectionArray",
                expected_fields=expected_fields,
            )

            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_detection_interface_subtask,
                        pub_detection_array_subtask,
                    ]
                )
            ]

        super().__init__(validators=validators, task_args=task_args, logger=logger)

    def get_base_prompt(self) -> str:
        detection_summaries: List[str] = []
        for _, (cls, center, size) in enumerate(
            zip(
                self.expected_detection_classes,
                self.expected_bbox_centers,
                self.expected_bbox_sizes,
            )
        ):
            detection_summaries.append(
                f"{cls} with bbox at center({center[0]}, {center[1]}) and size {size[0]}x{size[1]}"
            )

        return (
            f"Publish detection array to topic '{self.topic}' with {len(self.expected_detection_classes)} detections: "
            f"{'; '.join(detection_summaries)}."
        )

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        else:
            return (
                f"{self.get_base_prompt()} Examine the message interface "
                "and publish detection data with appropriate arguments."
            )


class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesServiceTask):
    complexity = "medium"
    service = "/manipulator_move_to"

    def __init__(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        initial_gripper_state: bool,
        final_gripper_state: bool,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.initial_gripper_state = initial_gripper_state
        self.final_gripper_state = final_gripper_state
        if validators is None:
            # Default validator for this task
            get_manipulator_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": MANIPULATOR_SERVICE_TYPE},
            )
            call_manipulator_service_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=MANIPULATOR_SERVICE,
                expected_service_type=MANIPULATOR_SERVICE_TYPE,
                expected_fields={
                    "target_pose.pose.position.x": target_x,
                    "target_pose.pose.position.y": target_y,
                    "target_pose.pose.position.z": target_z,
                    "initial_gripper_state": initial_gripper_state,
                    "final_gripper_state": final_gripper_state,
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_manipulator_interface_subtask,
                        call_manipulator_service_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return (
            f"Call service '{self.service}' to move manipulator to pose "
            f"({self.target_x}, {self.target_y}, {self.target_z}) with initial gripper state {self.initial_gripper_state} "
            f"and final gripper state {self.final_gripper_state}."
        )


class CallGroundedSAMSegmentTask(CustomInterfacesServiceTask):
    complexity = "medium"
    service = "/grounded_sam_segment"

    def __init__(
        self,
        task_args: TaskArgs,
        detection_classes: List[str],
        bbox_centers: List[Tuple[float, float]],
        bbox_sizes: List[Tuple[float, float]],
        scores: List[float],
        positions_3d: List[Tuple[float, float, float]],
        image_width: int,
        image_height: int,
        image_encoding: str,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        # Validate list lengths
        if not (
            len(detection_classes)
            == len(bbox_centers)
            == len(bbox_sizes)
            == len(scores)
            == len(positions_3d)
        ):
            raise ValueError("All detection parameter lists must have the same length")

        self.detection_classes = detection_classes
        self.bbox_centers = bbox_centers
        self.bbox_sizes = bbox_sizes
        self.scores = scores
        self.positions_3d = positions_3d
        self.image_width = image_width
        self.image_height = image_height
        self.image_encoding = image_encoding

        if validators is None:
            # Create default validator based on the detection parameters
            get_grounded_sam_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": "rai_interfaces/srv/RAIGroundedSam"},
            )

            # Build expected fields dynamically based on detection parameters
            expected_fields: Dict[str, Any] = {}
            for i, (cls, center, size, score, pos3d) in enumerate(
                zip(detection_classes, bbox_centers, bbox_sizes, scores, positions_3d)
            ):
                expected_fields.update(
                    {
                        f"detections.detections.{i}.results.0.hypothesis.class_id": cls,
                        f"detections.detections.{i}.results.0.hypothesis.score": score,
                        f"detections.detections.{i}.results.0.pose.pose.position.x": pos3d[
                            0
                        ],
                        f"detections.detections.{i}.results.0.pose.pose.position.y": pos3d[
                            1
                        ],
                        f"detections.detections.{i}.results.0.pose.pose.position.z": pos3d[
                            2
                        ],
                        f"detections.detections.{i}.bbox.center.x": center[0],
                        f"detections.detections.{i}.bbox.center.y": center[1],
                        f"detections.detections.{i}.bbox.size_x": size[0],
                        f"detections.detections.{i}.bbox.size_y": size[1],
                    }
                )

            # Add image fields
            expected_fields.update(
                {
                    "source_img.width": image_width,
                    "source_img.height": image_height,
                    "source_img.encoding": image_encoding,
                }
            )

            call_grounded_sam_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service="/grounded_sam_segment",
                expected_service_type="rai_interfaces/srv/RAIGroundedSam",
                expected_fields=expected_fields,
            )

            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_grounded_sam_interface_subtask,
                        call_grounded_sam_subtask,
                    ]
                )
            ]

        super().__init__(validators=validators, task_args=task_args, logger=logger)

    def get_base_prompt(self) -> str:
        detection_summary: List[str] = []
        for cls, center, size, score, pos3d in zip(
            self.detection_classes,
            self.bbox_centers,
            self.bbox_sizes,
            self.scores,
            self.positions_3d,
        ):
            detection_summary.append(
                f"{cls} with score {score} at 3D position ({pos3d[0]}, {pos3d[1]}, {pos3d[2]}) "
                f"bbox ({center[0]}, {center[1]}) size {size[0]}x{size[1]}"
            )

        return (
            f"Call service '{self.service}' for image segmentation with {len(self.detection_classes)} detections: "
            f"{', '.join(detection_summary)} on {self.image_width}x{self.image_height} {self.image_encoding} image."
        )


class CallGroundingDinoClassify(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/grounding_dino_classify"

    def __init__(
        self,
        classes: str,
        box_threshold: float,
        text_threshold: float,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        if validators is None:
            # Default validator for this task
            get_grounding_dino_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": GROUNDING_DINO_SERVICE_TYPE},
            )
            call_grounding_dino_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDING_DINO_SERVICE,
                expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
                expected_fields={
                    "classes": classes,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_grounding_dino_interface_subtask,
                        call_grounding_dino_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return (
            f"Call service '{self.service}' for object classification with classes "
            f"'{self.classes}', box_threshold {self.box_threshold} and "
            f"text_threshold {self.text_threshold}."
        )


class CallGetLogDigestTask(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/get_log_digest"

    def __init__(
        self,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if validators is None:
            # Default validator for this task
            get_log_digest_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": STRING_LIST_SERVICE_TYPE},
            )
            call_log_digest_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=LOG_DIGEST_SERVICE,
                expected_service_type=STRING_LIST_SERVICE_TYPE,
                expected_fields={"": {}},
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_log_digest_interface_subtask,
                        call_log_digest_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' to get log digest."


class CallVectorStoreRetrievalTask(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/rai_whoami_documentation_service"

    def __init__(
        self,
        query: str,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.query = query
        if validators is None:
            # Default validator for this task
            get_vector_store_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": VECTOR_STORE_SERVICE_TYPE},
            )
            call_vector_store_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=VECTOR_STORE_SERVICE,
                expected_service_type=VECTOR_STORE_SERVICE_TYPE,
                expected_fields={
                    "query": query,
                },
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_vector_store_interface_subtask,
                        call_vector_store_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' with query '{self.query}'."


class CallWhatISeeTask(CustomInterfacesServiceTask):
    complexity = "easy"
    service = "/rai_whatisee_get"

    def __init__(
        self,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if validators is None:
            # Default validator for this task
            get_what_i_see_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": WHAT_I_SEE_SERVICE_TYPE},
            )
            call_what_i_see_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=WHAT_I_SEE_SERVICE,
                expected_service_type=WHAT_I_SEE_SERVICE_TYPE,
                expected_fields={"": {}},
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        get_what_i_see_interface_subtask,
                        call_what_i_see_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    def get_base_prompt(self) -> str:
        return f"Call service '{self.service}' to get visual observations."


class CompleteObjectInteractionTask(CustomInterfacesServicesTask):
    complexity = "hard"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interface for all required services
        return 5

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        # Grounding DINO parameters
        target_classes: str = "bottle",
        box_threshold: float = 0.35,
        text_threshold: float = 0.2,
        # Grounded SAM parameters
        detection_classes: List[str] = ["bottle"],
        bbox_centers: List[Tuple[float, float]] = [(320.0, 240.0)],
        bbox_sizes: List[Tuple[float, float]] = [(80.0, 120.0)],
        scores: List[float] = [0.87],
        positions_3d: List[Tuple[float, float, float]] = [(1.2, 0.0, 0.5)],
        image_width: int = 640,
        image_height: int = 480,
        image_encoding: str = "rgb8",
        # Manipulator parameters
        target_x: float = 1.2,
        target_y: float = 0.0,
        target_z: float = 0.5,
        initial_gripper: bool = True,
        final_gripper: bool = False,
        # HRI parameters
        interaction_message: str = "Initiating object interaction sequence with detected bottle",
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, logger=logger)
        self.target_classes = target_classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.detection_classes = detection_classes
        self.bbox_centers = bbox_centers
        self.bbox_sizes = bbox_sizes
        self.scores = scores
        self.positions_3d = positions_3d
        self.image_width = image_width
        self.image_height = image_height
        self.image_encoding = image_encoding
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.initial_gripper = initial_gripper
        self.final_gripper = final_gripper
        self.interaction_message = interaction_message

    def get_base_prompt(self) -> str:
        return f"""Perform complete object interaction workflow with predetermined parameters:
            1) Call service '/grounding_dino_classify' to classify '{self.target_classes}' with box_threshold={self.box_threshold}, text_threshold={self.text_threshold}
            2) Call service '/grounded_sam_segment' to segment {self.detection_classes[0]} at bbox({self.bbox_centers[0][0]}, {self.bbox_centers[0][1]})
            on {self.image_width}x{self.image_height} {self.image_encoding} image
            3) Call service '/manipulator_move_to' to move to position ({self.target_x}, {self.target_y}, {self.target_z})
            with gripper {self.initial_gripper}→{self.final_gripper}
            4) Publish to topic '/to_human' HRI message: '{self.interaction_message}'
            """


class MultiModalSceneDocumentationTask(CustomInterfacesServiceTask):
    complexity = "hard"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        # Detection array parameters
        scene_objects: List[str] = ["person", "laptop", "coffee_cup"],
        bbox_centers: List[Tuple[float, float]] = [
            (160.0, 200.0),
            (400.0, 300.0),
            (520.0, 180.0),
        ],
        bbox_sizes: List[Tuple[float, float]] = [
            (80.0, 160.0),
            (200.0, 120.0),
            (60.0, 80.0),
        ],
        # Knowledge retrieval parameters
        scene_analysis_query: str = "What safety protocols apply when humans and robots share workspace?",
        # HRI reporting parameters
        documentation_report: str = "Scene Documentation Complete: Recorded 3 objects with audio markers and safety analysis",
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(validators=validators, task_args=task_args, logger=logger)
        self.scene_objects = scene_objects
        self.bbox_centers = bbox_centers
        self.bbox_sizes = bbox_sizes
        self.scene_analysis_query = scene_analysis_query
        self.documentation_report = documentation_report

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interface for all required services
        return 5

    def get_base_prompt(self) -> str:
        object_summary = ", ".join(
            [
                f"{obj} at ({center[0]}, {center[1]}) and size {size[0]}x{size[1]}"
                for obj, center, size in zip(
                    self.scene_objects, self.bbox_centers, self.bbox_sizes
                )
            ]
        )

        return f"""Perform comprehensive scene documentation using multiple services:
            1) Call service '/rai_whatisee_get' to get visual observations
            2) Publish to topic '/detections' detection array with {len(self.scene_objects)} objects: {object_summary}
            3) Call service '/rai_whoami_documentation_service' to query: '{self.scene_analysis_query}'
            4) Publish to topic '/to_human' final HRI report: '{self.documentation_report}'
            """


class EmergencyResponseProtocolTask(CustomInterfacesServiceTask):
    complexity = "hard"

    def __init__(
        self,
        classes: str,
        box_threshold: float,
        text_threshold: float,
        detection_classes: List[str],
        bbox_centers: List[Tuple[float, float]],
        bbox_sizes: List[Tuple[float, float]],
        scores: List[float],
        positions_3d: List[Tuple[float, float, float]],
        image_width: int,
        image_height: int,
        image_encoding: str,
        audio_samples: List[int],
        sample_rate: int,
        channels: int,
        message: str,
        task_args: TaskArgs,
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.detection_classes = detection_classes
        self.bbox_centers = bbox_centers
        self.bbox_sizes = bbox_sizes
        self.scores = scores
        self.positions_3d = positions_3d
        self.image_width = image_width
        self.image_height = image_height
        self.image_encoding = image_encoding
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.channels = channels
        self.message = message

        if validators is None:
            # Default validator for this task
            call_grounding_dino_emergency_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDING_DINO_SERVICE,
                expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
                expected_fields={
                    "classes": classes,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                },
            )
            call_grounded_sam_emergency_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDED_SAM_SERVICE,
                expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
                expected_fields={
                    "detections.detections.0.results.0.hypothesis.class_id": detection_classes[
                        0
                    ],
                    "detections.detections.0.results.0.hypothesis.score": scores[0],
                    "detections.detections.0.results.0.pose.pose.position.x": positions_3d[
                        0
                    ][0],
                    "detections.detections.0.results.0.pose.pose.position.y": positions_3d[
                        0
                    ][1],
                    "detections.detections.0.results.0.pose.pose.position.z": positions_3d[
                        0
                    ][2],
                    "detections.detections.0.bbox.center.x": bbox_centers[0][0],
                    "detections.detections.0.bbox.center.y": bbox_centers[0][1],
                    "detections.detections.0.bbox.size_x": bbox_sizes[0][0],
                    "detections.detections.0.bbox.size_y": bbox_sizes[0][1],
                    "source_img.width": image_width,
                    "source_img.height": image_height,
                    "source_img.encoding": image_encoding,
                },
            )
            pub_audio_emergency_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=AUDIO_TOPIC,
                expected_message_type=AUDIO_MESSAGE_TYPE,
                expected_fields={
                    "samples": audio_samples,
                    "sample_rate": sample_rate,
                    "channels": channels,
                },
            )
            pub_hri_emergency_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=HRI_TOPIC,
                expected_message_type=HRI_MESSAGE_TYPE,
                expected_fields={"text": message},
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        call_grounding_dino_emergency_subtask,
                        call_grounded_sam_emergency_subtask,
                        pub_audio_emergency_subtask,
                        pub_hri_emergency_subtask,
                    ]
                )
            ]
        super().__init__(validators, task_args, logger)

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interface for all required services
        return 5

    def get_base_prompt(self) -> str:
        return f"""Execute emergency response protocol with predetermined parameters:
            1) Call service '/grounding_dino_classify' to detect emergency: '{self.classes}' with high thresholds box={self.box_threshold}, text={self.text_threshold}
            2) Call service '/grounded_sam_segment' to segment person at ({self.bbox_centers[0][0]}, {self.bbox_centers[0][1]})
            on {self.image_width}x{self.image_height} {self.image_encoding} image
            3) Publish to topic '/audio_output' emergency alert: {self.audio_samples} at {self.sample_rate}Hz
            4) Publish to topic '/to_human' emergency message: '{self.message}'
            """
