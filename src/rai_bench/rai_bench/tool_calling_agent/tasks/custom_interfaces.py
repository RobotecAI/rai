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
AUDIO_TOPIC = "/audio_message"
DETECTIONS_TOPIC = "/detection_array"
MANIPULATOR_SERVICE = "/manipulator_move_to"
GROUNDED_SAM_SERVICE = "/grounded_sam_segment"
GROUNDING_DINO_SERVICE = "/grounding_dino_classify"
LOG_DIGEST_SERVICE = "/get_log_digest"
VECTOR_STORE_SERVICE = "/rai_whoami_documentation_service"

HRI_MESSAGE_TYPE = "rai_interfaces/msg/HRIMessage"
AUDIO_MESSAGE_TYPE = "rai_interfaces/msg/AudioMessage"
DETECTION_ARRAY_MESSAGE_TYPE = "rai_interfaces/msg/RAIDetectionArray"
MANIPULATOR_SERVICE_TYPE = "rai_interfaces/srv/ManipulatorMoveTo"
GROUNDED_SAM_SERVICE_TYPE = "rai_interfaces/srv/RAIGroundedSam"
GROUNDING_DINO_SERVICE_TYPE = "rai_interfaces/srv/RAIGroundingDino"
STRING_LIST_SERVICE_TYPE = "rai_interfaces/srv/StringList"
VECTOR_STORE_SERVICE_TYPE = "rai_interfaces/srv/VectorStoreRetrieval"

STANDARD_IMAGE_WIDTH = 640
STANDARD_IMAGE_HEIGHT = 480
STANDARD_IMAGE_ENCODING = "rgb8"

PERSON_BBOX_CENTER = (320.0, 320.0)
PERSON_BBOX_SIZE = (50.0, 50.0)
PERSON_SCORE = 0.85
PERSON_POSITION_3D = (2.0, 0.0, 0.0)

BOTTLE_BBOX_CENTER = (320.0, 240.0)
BOTTLE_BBOX_SIZE = (80.0, 120.0)
BOTTLE_SCORE = 0.87
BOTTLE_POSITION_3D = (1.2, 0.0, 0.5)

DETECTION_DEFAULTS: Dict[str, Any] = {
    "person": {
        "bbox_center": PERSON_BBOX_CENTER,
        "bbox_size": PERSON_BBOX_SIZE,
        "score": PERSON_SCORE,
        "position_3d": PERSON_POSITION_3D,
    },
    "bottle": {
        "bbox_center": BOTTLE_BBOX_CENTER,
        "bbox_size": BOTTLE_BBOX_SIZE,
        "score": BOTTLE_SCORE,
        "position_3d": BOTTLE_POSITION_3D,
    },
}

DEFAULT_BOX_THRESHOLD: float = 0.4
DEFAULT_TEXT_THRESHOLD: float = 0.25


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
    """Custom Interface Tasks are designed around out custom interfaces in RAI
    In these tasks we want to evaulate how well agent can understand these interfaes
    and fill them as requested
    """

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
        " Examine the required service interface and call  "
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
        " Examine the required services interfaces and call  "
        "them with appropriate arguments."
    )


#### NOTE(jmatejcz) Tasks come in 2 versions:
#### - basic one that gives required topic or service and field values directly,
#### they verify if model can handle custom interface
#### - indirect, where prompts are more natural and indirect,
#### they verify if model can on top understand and use the topic or service in suitable situation


class PublishROS2HRIMessageTextTask(CustomInterfaceTask):
    complexity = "easy"

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
        return f"Publish message to topic '{HRI_TOPIC}' with text '{self.text}'."

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return f"{self.get_base_prompt()}."
        else:
            return (
                f"{self.get_base_prompt()}"
                " Examine the message interface "
                f"structure, and publish an HRI message with appropriate arguments."
            )


class PublishROS2HRIMessageTextTaskIndirect(PublishROS2HRIMessageTextTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return f"Publish message '{self.text}' to human."


class PublishROS2AudioMessageTask(CustomInterfaceTask):
    complexity = "easy"

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
            f"Publish audio message to topic '{AUDIO_TOPIC}' with samples "
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


class PublishROS2AudioMessageTaskIndirect(PublishROS2AudioMessageTask):
    complexity = "easy"

    def get_base_prompt(self) -> str:
        return (
            f"Publish audio with samples {self.audio}, "
            f"sample rate {self.sample_rate} and {self.channels} channels."
        )


class PublishROS2DetectionArrayTask(CustomInterfaceTask):
    complexity = "medium"

    def __init__(
        self,
        task_args: TaskArgs,
        detection_classes: List[str],
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.detection_classes = detection_classes

        self.bbox_centers: List[Tuple[float, float]] = []
        self.bbox_sizes: List[Tuple[float, float]] = []
        for obj in detection_classes:
            if obj not in DETECTION_DEFAULTS:
                # use existing values
                obj = "person"
            defaults = DETECTION_DEFAULTS[obj]
            self.bbox_centers.append(defaults["bbox_center"])
            self.bbox_sizes.append(defaults["bbox_size"])

        if validators is None:
            get_detection_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": DETECTION_ARRAY_MESSAGE_TYPE},
            )

            expected_fields: Dict[str, Any] = {}
            for i, obj in enumerate(detection_classes):
                expected_fields.update(
                    {
                        f"detections.{i}.results.0.hypothesis.class_id": obj,
                        f"detections.{i}.bbox.center.x": self.bbox_centers[i][0],
                        f"detections.{i}.bbox.center.y": self.bbox_centers[i][1],
                        f"detections.{i}.bbox.size_x": self.bbox_sizes[i][0],
                        f"detections.{i}.bbox.size_y": self.bbox_sizes[i][1],
                    }
                )

            pub_detection_array_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=DETECTIONS_TOPIC,
                expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
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
                self.detection_classes,
                self.bbox_centers,
                self.bbox_sizes,
            )
        ):
            detection_summaries.append(
                f"{cls} with bbox at center({center[0]}, {center[1]}) and size {size[0]}x{size[1]}"
            )

        return (
            f"Publish detection array to topic '{DETECTIONS_TOPIC}' with {len(self.detection_classes)} detections: "
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


class PublishROS2DetectionArrayTaskIndirect(PublishROS2DetectionArrayTask):
    complexity = "hard"

    def get_base_prompt(self) -> str:
        detection_summaries: List[str] = []
        for _, (cls, center, size) in enumerate(
            zip(
                self.detection_classes,
                self.bbox_centers,
                self.bbox_sizes,
            )
        ):
            detection_summaries.append(
                f"{cls} at position ({center[0]}, {center[1]}) with size {size[0]}x{size[1]}"
            )

        return (
            f"Report detected objects in the scene: {'; '.join(detection_summaries)}."
        )


class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesServiceTask):
    complexity = "medium"

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
            f"Call service '{MANIPULATOR_SERVICE}' to move manipulator to pose "
            f"({self.target_x}, {self.target_y}, {self.target_z}) with initial gripper state {self.initial_gripper_state} "
            f"and final gripper state {self.final_gripper_state}."
        )


class CallROS2ManipulatorMoveToServiceTaskIndirect(
    CallROS2ManipulatorMoveToServiceTask
):
    complexity = "medium"

    def get_base_prompt(self) -> str:
        init_gripper_action = "close" if self.final_gripper_state else "open"
        final_gripper_action = "close" if self.final_gripper_state else "open"
        return (
            f"Move robot arm to position ({self.target_x}, {self.target_y}, {self.target_z}) "
            f"At the start keep griper {init_gripper_action}, at the end {final_gripper_action}. "
        )


class CallGroundedSAMSegmentTask(CustomInterfacesServiceTask):
    complexity = "medium"

    def __init__(
        self,
        task_args: TaskArgs,
        detection_classes: List[str],
        validators: Optional[List[Validator]] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.detection_classes = detection_classes

        # Get default parameters for each detection class
        self.bbox_centers: List[Tuple[float, float]] = []
        self.bbox_sizes: List[Tuple[float, float]] = []
        self.scores: List[Tuple[float, float]] = []
        self.positions_3d: List[Tuple[float, float]] = []

        for obj in detection_classes:
            if obj not in DETECTION_DEFAULTS:
                # use existing values
                obj = "person"
            defaults = DETECTION_DEFAULTS[obj]
            self.bbox_centers.append(defaults["bbox_center"])
            self.bbox_sizes.append(defaults["bbox_size"])
            self.scores.append(defaults["score"])
            self.positions_3d.append(defaults["position_3d"])

        if validators is None:
            get_grounded_sam_interface_subtask = CheckArgsToolCallSubTask(
                expected_tool_name="get_ros2_message_interface",
                expected_args={"msg_type": GROUNDED_SAM_SERVICE_TYPE},
            )

            expected_fields: Dict[str, Any] = {}
            for i, obj in enumerate(detection_classes):
                defaults = DETECTION_DEFAULTS[obj]
                expected_fields.update(
                    {
                        f"detections.detections.{i}.results.0.hypothesis.class_id": obj,
                        f"detections.detections.{i}.results.0.hypothesis.score": defaults[
                            "score"
                        ],
                        f"detections.detections.{i}.results.0.pose.pose.position.x": defaults[
                            "position_3d"
                        ][0],
                        f"detections.detections.{i}.results.0.pose.pose.position.y": defaults[
                            "position_3d"
                        ][1],
                        f"detections.detections.{i}.results.0.pose.pose.position.z": defaults[
                            "position_3d"
                        ][2],
                        f"detections.detections.{i}.bbox.center.x": defaults[
                            "bbox_center"
                        ][0],
                        f"detections.detections.{i}.bbox.center.y": defaults[
                            "bbox_center"
                        ][1],
                        f"detections.detections.{i}.bbox.size_x": defaults["bbox_size"][
                            0
                        ],
                        f"detections.detections.{i}.bbox.size_y": defaults["bbox_size"][
                            1
                        ],
                    }
                )

            expected_fields.update(
                {
                    "source_img.width": STANDARD_IMAGE_WIDTH,
                    "source_img.height": STANDARD_IMAGE_HEIGHT,
                    "source_img.encoding": STANDARD_IMAGE_ENCODING,
                }
            )

            call_grounded_sam_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDED_SAM_SERVICE,
                expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
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
        for cls in self.detection_classes:
            defaults = DETECTION_DEFAULTS[cls]
            center = defaults["bbox_center"]
            size = defaults["bbox_size"]
            score = defaults["score"]
            pos3d = defaults["position_3d"]
            detection_summary.append(
                f"{cls} with score {score} at 3D position ({pos3d[0]}, {pos3d[1]}, {pos3d[2]}) "
                f"bbox ({center[0]}, {center[1]}) size {size[0]}x{size[1]}"
            )

        return (
            f"Call service '{GROUNDED_SAM_SERVICE}' for image segmentation with {len(self.detection_classes)} detections: "
            f"{', '.join(detection_summary)} on {STANDARD_IMAGE_WIDTH}x{STANDARD_IMAGE_HEIGHT} {STANDARD_IMAGE_ENCODING} image."
        )


class CallGroundedSAMSegmentTaskIndirect(CallGroundedSAMSegmentTask):
    complexity = "medium"

    def get_base_prompt(self) -> str:
        detection_summary: List[str] = []
        for cls in self.detection_classes:
            defaults = DETECTION_DEFAULTS[cls]
            center = defaults["bbox_center"]
            size = defaults["bbox_size"]
            score = defaults["score"]
            pos3d = defaults["position_3d"]
            detection_summary.append(
                f"{cls} with score {score} at 3D position ({pos3d[0]}, {pos3d[1]}, {pos3d[2]}) "
                f"bbox ({center[0]}, {center[1]}) size {size[0]}x{size[1]}"
            )

        return (
            f"Segment detected objects: "
            f"{', '.join(detection_summary)}."
            f"on {STANDARD_IMAGE_WIDTH}x{STANDARD_IMAGE_HEIGHT} {STANDARD_IMAGE_ENCODING} image."
        )


class CallGroundingDinoClassify(CustomInterfacesServiceTask):
    complexity = "easy"

    def __init__(
        self,
        task_args: TaskArgs,
        classes: str,  # Comma-separated string of classes like "person, bottle"
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.classes = classes

        if validators is None:
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
                    "box_threshold": DEFAULT_BOX_THRESHOLD,
                    "text_threshold": DEFAULT_TEXT_THRESHOLD,
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
            f"Call service '{GROUNDING_DINO_SERVICE}' for object classification with classes "
            f"'{self.classes}', box_threshold {DEFAULT_BOX_THRESHOLD} and "
            f"text_threshold {DEFAULT_TEXT_THRESHOLD}."
        )


class CallGroundingDinoClassifyIndirect(CallGroundingDinoClassify):
    complexity = "medium"

    def get_base_prompt(self) -> str:
        return (
            f"Identify these objects in the scene: {self.classes}. "
            f"box_threshold should be {DEFAULT_BOX_THRESHOLD} and "
            f"text_threshold should be {DEFAULT_TEXT_THRESHOLD}."
        )


class CallGetLogDigestTask(CustomInterfacesServiceTask):
    complexity = "easy"

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
        return f"Call service '{LOG_DIGEST_SERVICE}' to get log digest."


class CallGetLogDigestTaskIndirect(CallGetLogDigestTask):
    complexity = "medium"

    def get_base_prompt(self) -> str:
        return "Get a summary of recent system logs."


class CallVectorStoreRetrievalTask(CustomInterfacesServiceTask):
    complexity = "easy"

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
        return f"Call service '{VECTOR_STORE_SERVICE}' with query '{self.query}'."


class CallVectorStoreRetrievalTaskIndirect(CallVectorStoreRetrievalTask):
    complexity = "medium"

    def get_base_prompt(self) -> str:
        return f"Search the documentation for information about: {self.query}."


class CompleteObjectInteractionTask(CustomInterfacesServicesTask):
    complexity = "hard"

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interface for all required services
        return 5

    def __init__(
        self,
        task_args: TaskArgs,
        target_class: str = "bottle",
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.target_class = target_class

        # Get default parameters for the target class
        obj_key = target_class if target_class in DETECTION_DEFAULTS else "person"
        defaults = DETECTION_DEFAULTS[obj_key]

        if validators is None:
            self.initial_gripper = False
            self.final_gripper = True
            self.interaction_message = (
                f"Initiating object interaction sequence with detected {target_class}"
            )

            call_grounding_dino_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDING_DINO_SERVICE,
                expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
                expected_fields={
                    "classes": target_class,
                    "box_threshold": DEFAULT_BOX_THRESHOLD,
                    "text_threshold": DEFAULT_BOX_THRESHOLD,
                },
            )

            call_grounded_sam_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDED_SAM_SERVICE,
                expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
                expected_fields={
                    "detections.detections.0.results.0.hypothesis.class_id": target_class,
                    "detections.detections.0.results.0.hypothesis.score": defaults[
                        "score"
                    ],
                    "detections.detections.0.results.0.pose.pose.position.x": defaults[
                        "position_3d"
                    ][0],
                    "detections.detections.0.results.0.pose.pose.position.y": defaults[
                        "position_3d"
                    ][1],
                    "detections.detections.0.results.0.pose.pose.position.z": defaults[
                        "position_3d"
                    ][2],
                    "detections.detections.0.bbox.center.x": defaults["bbox_center"][0],
                    "detections.detections.0.bbox.center.y": defaults["bbox_center"][1],
                    "detections.detections.0.bbox.size_x": defaults["bbox_size"][0],
                    "detections.detections.0.bbox.size_y": defaults["bbox_size"][1],
                    "source_img.width": STANDARD_IMAGE_WIDTH,
                    "source_img.height": STANDARD_IMAGE_HEIGHT,
                    "source_img.encoding": STANDARD_IMAGE_ENCODING,
                },
            )

            call_manipulator_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=MANIPULATOR_SERVICE,
                expected_service_type=MANIPULATOR_SERVICE_TYPE,
                expected_fields={
                    "target_pose.pose.position.x": defaults["position_3d"][0],
                    "target_pose.pose.position.y": defaults["position_3d"][1],
                    "target_pose.pose.position.z": defaults["position_3d"][2],
                    "initial_gripper_state": self.initial_gripper,
                    "final_gripper_state": self.final_gripper,
                },
            )

            pub_hri_interaction_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=HRI_TOPIC,
                expected_message_type=HRI_MESSAGE_TYPE,
                expected_fields={"text": self.interaction_message},
            )

            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        call_grounding_dino_subtask,
                        call_grounded_sam_subtask,
                        call_manipulator_subtask,
                        pub_hri_interaction_subtask,
                    ]
                )
            ]

        super().__init__(validators=validators, task_args=task_args, logger=logger)

    def get_base_prompt(self) -> str:
        obj_key = (
            self.target_class if self.target_class in DETECTION_DEFAULTS else "person"
        )
        defaults = DETECTION_DEFAULTS[obj_key]
        return f"""Perform complete object interaction workflow with {self.target_class}:
            1) Call service '{GROUNDING_DINO_SERVICE}' to classify '{self.target_class}' with box_threshold={DEFAULT_BOX_THRESHOLD}, text_threshold={DEFAULT_TEXT_THRESHOLD}
            2) Call service '{GROUNDED_SAM_SERVICE}' to segment {self.target_class} at bbox({defaults["bbox_center"][0]}, {defaults["bbox_center"][1]})
            on {STANDARD_IMAGE_WIDTH}x{STANDARD_IMAGE_HEIGHT} {STANDARD_IMAGE_ENCODING} image
            3) Call service '{MANIPULATOR_SERVICE}' to move to position ({defaults["position_3d"][0]}, {defaults["position_3d"][1]}, {defaults["position_3d"][2]})
            with initial gripper state {self.initial_gripper} and final griper state {self.final_gripper}
            4) Publish to topic '{HRI_TOPIC}' HRI message: '{self.interaction_message}'
            """


class CompleteObjectInteractionTaskIndirect(CompleteObjectInteractionTask):
    complexity = "hard"

    def get_base_prompt(self) -> str:
        obj_key = (
            self.target_class if self.target_class in DETECTION_DEFAULTS else "person"
        )
        defaults = DETECTION_DEFAULTS[obj_key]
        return f"""Find and pick up the {self.target_class} in front of you by following these steps:
            1) First, look for and identify the {self.target_class} in the scene
            2) Then, get a segmentation of the {self.target_class} to understand its exact location
            3) Move the robot arm to reach the {self.target_class} at position ({defaults["position_3d"][0]}, {defaults["position_3d"][1]}, {defaults["position_3d"][2]}) and grab it
            4) Finally, inform human with '{self.interaction_message}` message that you've completed the task.
            """


class MultiModalSceneDocumentationTask(CustomInterfacesServiceTask):
    complexity = "hard"
    safety_query = "What safety protocols apply when humans and robots share workspace?"
    doc_report = "Scene Documentation Complete: Recorded objects with safety analysis"

    def __init__(
        self,
        task_args: TaskArgs,
        objects: List[str] = ["person", "bottle"],
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.objects = objects

        self.bbox_centers: List[Tuple[float, float]] = []
        self.bbox_sizes: List[Tuple[float, float]] = []

        for obj in objects:
            obj_key = obj if obj in DETECTION_DEFAULTS else "person"
            defaults = DETECTION_DEFAULTS[obj_key]
            self.bbox_centers.append(defaults["bbox_center"])
            self.bbox_sizes.append(defaults["bbox_size"])

        if validators is None:
            expected_fields: Dict[str, Any] = {}
            for i, obj in enumerate(objects):
                obj_key = obj if obj in DETECTION_DEFAULTS else "person"
                defaults = DETECTION_DEFAULTS[obj_key]
                expected_fields.update(
                    {
                        f"detections.{i}.results.0.hypothesis.class_id": obj,
                        f"detections.{i}.bbox.center.x": defaults["bbox_center"][0],
                        f"detections.{i}.bbox.center.y": defaults["bbox_center"][1],
                        f"detections.{i}.bbox.size_x": defaults["bbox_size"][0],
                        f"detections.{i}.bbox.size_y": defaults["bbox_size"][1],
                    }
                )

            pub_detection_array_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=DETECTIONS_TOPIC,
                expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
                expected_fields=expected_fields,
            )

            call_vector_store_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=VECTOR_STORE_SERVICE,
                expected_service_type=VECTOR_STORE_SERVICE_TYPE,
                expected_fields={
                    "query": self.safety_query,
                },
            )

            pub_hri_documentation_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=HRI_TOPIC,
                expected_message_type=HRI_MESSAGE_TYPE,
                expected_fields={"text": self.doc_report},
            )

            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        pub_detection_array_subtask,
                        call_vector_store_subtask,
                        pub_hri_documentation_subtask,
                    ]
                )
            ]

        super().__init__(validators=validators, task_args=task_args, logger=logger)

    @property
    def optional_tool_calls_number(self) -> int:
        # list services and get interface for all required services
        return 5

    def get_base_prompt(self) -> str:
        object_summary = ", ".join(
            [
                f"{obj} at ({center[0]}, {center[1]}) and size {size[0]}x{size[1]}"
                for obj, center, size in zip(
                    self.objects, self.bbox_centers, self.bbox_sizes
                )
            ]
        )

        return f"""Perform comprehensive scene documentation using multiple services:
            1) Publish to topic '{DETECTIONS_TOPIC}' detection array with {len(self.objects)} objects: {object_summary}
            2) Call service '{VECTOR_STORE_SERVICE}' to query: '{self.safety_query}'
            3) Publish to topic '{HRI_TOPIC}' final HRI report: '{self.doc_report}'
            """


class MultiModalSceneDocumentationTaskIndirect(MultiModalSceneDocumentationTask):
    complexity = "hard"

    def get_base_prompt(self) -> str:
        object_summary = ", ".join(
            [
                f"{obj} at position ({center[0]}, {center[1]})"
                for obj, center in zip(self.objects, self.bbox_centers)
            ]
        )
        return f"""Document the current scene and ensure safety protocols are followed:
            1) Record all detected objects in the scene: {object_summary}
            2) Look up safety guidelines for human-robot workspace sharing with query: `{self.safety_query}`
            3) Provide a final documentation report confirming scene analysis and safety compliance with report: `{self.doc_report}`
            """


class EmergencyResponseProtocolTask(CustomInterfacesServiceTask):
    complexity = "hard"
    emergence_audio_samples = [880, 880, 880, 1760]
    emergence_sample_rate = 8000
    emergency_channels = 1
    emergency_message = "Person detected, alarm started!"

    def __init__(
        self,
        task_args: TaskArgs,
        target_class: str = "person",
        validators: Optional[List[Validator]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.target_class = target_class
        if validators is None:
            call_grounding_dino_emergency_subtask = CheckServiceFieldsToolCallSubTask(
                expected_tool_name="call_ros2_service",
                expected_service=GROUNDING_DINO_SERVICE,
                expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
                expected_fields={
                    "classes": target_class,
                    "box_threshold": DEFAULT_BOX_THRESHOLD,
                    "text_threshold": DEFAULT_TEXT_THRESHOLD,
                },
            )

            pub_audio_emergency_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=AUDIO_TOPIC,
                expected_message_type=AUDIO_MESSAGE_TYPE,
                expected_fields={
                    "samples": self.emergence_audio_samples,
                    "sample_rate": self.emergence_sample_rate,
                    "channels": self.emergency_channels,
                },
            )
            pub_hri_emergency_subtask = CheckTopicFieldsToolCallSubTask(
                expected_tool_name="publish_ros2_message",
                expected_topic=HRI_TOPIC,
                expected_message_type=HRI_MESSAGE_TYPE,
                expected_fields={"text": self.emergency_message},
            )
            validators = [
                OrderedCallsValidator(
                    subtasks=[
                        call_grounding_dino_emergency_subtask,
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
        return f"""Execute emergency response protocol for {self.target_class} detection:
           1) Call service '{GROUNDING_DINO_SERVICE}' to detect emergency: '{self.target_class}' with high thresholds box={DEFAULT_BOX_THRESHOLD}, text={DEFAULT_TEXT_THRESHOLD}
           2) Publish to topic '{AUDIO_TOPIC}' emergency alert: {self.emergence_audio_samples} at {self.emergence_sample_rate}Hz
           3) Publish to topic '{HRI_TOPIC}' emergency message: '{self.emergency_message}'
           """


class EmergencyResponseProtocolTaskIndirect(EmergencyResponseProtocolTask):
    complexity = "hard"

    def get_base_prompt(self) -> str:
        return f"""Execute emergency response protocol when {self.target_class} is detected:
           1) Detect any {self.target_class} presence.
           2) Sound an emergency audio alert with samples {self.emergence_audio_samples} at {self.emergence_sample_rate}Hz
           3) Send emergency notification: '{self.emergency_message}'
           """
