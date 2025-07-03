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

from typing import List, Literal

from rai_bench.tool_calling_agent.interfaces import (
    Task,
    TaskArgs,
)
from rai_bench.tool_calling_agent.subtasks import (
    CheckArgsToolCallSubTask,
    CheckServiceFieldsToolCallSubTask,
    CheckTopicFieldsToolCallSubTask,
)
from rai_bench.tool_calling_agent.tasks.custom_interfaces import (
    DETECTION_ARRAY_MESSAGE_TYPE,
    DETECTIONS_TOPIC,
    GROUNDED_SAM_SERVICE,
    GROUNDED_SAM_SERVICE_TYPE,
    GROUNDING_DINO_SERVICE,
    GROUNDING_DINO_SERVICE_TYPE,
    HRI_MESSAGE_TYPE,
    HRI_TOPIC,
    MANIPULATOR_SERVICE,
    MANIPULATOR_SERVICE_TYPE,
    CallGetLogDigestTask,
    CallGroundedSAMSegmentTask,
    CallGroundingDinoClassify,
    CallROS2ManipulatorMoveToServiceTask,
    CallVectorStoreRetrievalTask,
    CallWhatISeeTask,
    CompleteObjectInteractionTask,
    EmergencyResponseProtocolTask,
    PublishROS2AudioMessageTask,
    PublishROS2DetectionArrayTask,
    PublishROS2HRIMessageTextTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

PERSON_CLASS = "person"
BOTTLE_CLASS = "bottle"

HELLO_TEXT = "Hello!"
BOTTLE_INTERACTION_MESSAGE = (
    "Initiating object interaction sequence with detected bottle"
)
EMERGENCY_MESSAGE = "Person detected, alarm started."

BASIC_AUDIO_SAMPLES = [123, 456, 789]
BASIC_SAMPLE_RATE = 44100
BASIC_CHANNELS = 2
EMERGENCY_AUDIO_SAMPLES = [880, 880, 880, 1760]
EMERGENCY_SAMPLE_RATE = 8000
EMERGENCY_CHANNELS = 1

BOX_THRESHOLD_1 = 0.4
TEXT_THRESHOLD_1 = 0.25
BOTTLE_BOX_THRESHOLD = 0.35
BOTTLE_TEXT_THRESHOLD = 0.2
EMERGENCY_BOX_THRESHOLD = 0.9
EMERGENCY_TEXT_THRESHOLD = 0.8

PERSON_BBOX_CENTER = (320.0, 320.0)
PERSON_BBOX_SIZE = (50.0, 50.0)
BOTTLE_BBOX_CENTER = (320.0, 240.0)
BOTTLE_BBOX_SIZE = (80.0, 120.0)
EMERGENCY_BBOX_CENTER = (320.0, 240.0)
EMERGENCY_BBOX_SIZE = (100.0, 180.0)

BOTTLE_SCORE = 0.85
BOTTLE_ENHANCED_SCORE = 0.87
EMERGENCY_SCORE = 0.95

STANDARD_TARGET_POSITION = (1.0, 2.0, 3.0)
BOTTLE_POSITION_3D = (1.2, 0.0, 0.5)
EMERGENCY_POSITION_3D = (2.0, 0.0, 0.0)

STANDARD_IMAGE_WIDTH = 640
STANDARD_IMAGE_HEIGHT = 480
STANDARD_IMAGE_ENCODING = "rgb8"
HD_IMAGE_WIDTH = 1280
HD_IMAGE_HEIGHT = 720
HD_IMAGE_ENCODING = "bgr8"

# Query Strings
ROBOT_PURPOSE_QUERY = "What is the purpose of this robot?"
GROUNDING_DINO_CLASSES = "person, bottle"

get_detection_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": DETECTION_ARRAY_MESSAGE_TYPE},
)

pub_detection_array_subtask_person = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=DETECTIONS_TOPIC,
    expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
    expected_fields={
        "detections.0.results.0.hypothesis.class_id": PERSON_CLASS,
        "detections.0.bbox.center.x": PERSON_BBOX_CENTER[0],
        "detections.0.bbox.center.y": PERSON_BBOX_CENTER[1],
        "detections.0.bbox.size_x": PERSON_BBOX_SIZE[0],
        "detections.0.bbox.size_y": PERSON_BBOX_SIZE[1],
    },
)

pub_detection_array_subtask_bottle = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=DETECTIONS_TOPIC,
    expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
    expected_fields={
        "detections.0.results.0.hypothesis.class_id": BOTTLE_CLASS,
        "detections.0.bbox.center.x": BOTTLE_BBOX_CENTER[0],
        "detections.0.bbox.center.y": BOTTLE_BBOX_CENTER[1],
        "detections.0.bbox.size_x": BOTTLE_BBOX_SIZE[0],
        "detections.0.bbox.size_y": BOTTLE_BBOX_SIZE[1],
    },
)

# Subtasks for CallGroundedSAMSegmentTask - these are used with different parameters
get_grounded_sam_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": GROUNDED_SAM_SERVICE_TYPE},
)

call_grounded_sam_subtask_bottle = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SERVICE,
    expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": BOTTLE_CLASS,
        "detections.detections.0.results.0.hypothesis.score": BOTTLE_SCORE,
        "detections.detections.0.results.0.pose.pose.position.x": BOTTLE_POSITION_3D[0],
        "detections.detections.0.results.0.pose.pose.position.y": BOTTLE_POSITION_3D[1],
        "detections.detections.0.results.0.pose.pose.position.z": BOTTLE_POSITION_3D[2],
        "detections.detections.0.bbox.center.x": BOTTLE_BBOX_CENTER[0],
        "detections.detections.0.bbox.center.y": BOTTLE_BBOX_CENTER[1],
        "detections.detections.0.bbox.size_x": BOTTLE_BBOX_SIZE[0],
        "detections.detections.0.bbox.size_y": BOTTLE_BBOX_SIZE[1],
        "source_img.width": STANDARD_IMAGE_WIDTH,
        "source_img.height": STANDARD_IMAGE_HEIGHT,
        "source_img.encoding": STANDARD_IMAGE_ENCODING,
    },
)

# CompleteObjectInteractionTask subtasks
call_grounding_dino_bottle_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SERVICE,
    expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
    expected_fields={
        "classes": BOTTLE_CLASS,
        "box_threshold": BOTTLE_BOX_THRESHOLD,
        "text_threshold": BOTTLE_TEXT_THRESHOLD,
    },
)

call_grounded_sam_bottle_enhanced_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SERVICE,
    expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": BOTTLE_CLASS,
        "detections.detections.0.results.0.hypothesis.score": BOTTLE_ENHANCED_SCORE,
        "detections.detections.0.results.0.pose.pose.position.x": BOTTLE_POSITION_3D[0],
        "detections.detections.0.results.0.pose.pose.position.y": BOTTLE_POSITION_3D[1],
        "detections.detections.0.results.0.pose.pose.position.z": BOTTLE_POSITION_3D[2],
        "detections.detections.0.bbox.center.x": BOTTLE_BBOX_CENTER[0],
        "detections.detections.0.bbox.center.y": BOTTLE_BBOX_CENTER[1],
        "detections.detections.0.bbox.size_x": BOTTLE_BBOX_SIZE[0],
        "detections.detections.0.bbox.size_y": BOTTLE_BBOX_SIZE[1],
        "source_img.width": STANDARD_IMAGE_WIDTH,
        "source_img.height": STANDARD_IMAGE_HEIGHT,
        "source_img.encoding": STANDARD_IMAGE_ENCODING,
    },
)

call_manipulator_bottle_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=MANIPULATOR_SERVICE,
    expected_service_type=MANIPULATOR_SERVICE_TYPE,
    expected_fields={
        "target_pose.pose.position.x": BOTTLE_POSITION_3D[0],
        "target_pose.pose.position.y": BOTTLE_POSITION_3D[1],
        "target_pose.pose.position.z": BOTTLE_POSITION_3D[2],
        "initial_gripper_state": True,
        "final_gripper_state": False,
    },
)

pub_hri_bottle_interaction_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": BOTTLE_INTERACTION_MESSAGE},
)

# Validators for tasks that need specific parameter combinations
get_interface_publish_detection_ord_val_person = OrderedCallsValidator(
    subtasks=[
        get_detection_interface_subtask,
        pub_detection_array_subtask_person,
    ]
)

get_interface_publish_detection_ord_val_bottle = OrderedCallsValidator(
    subtasks=[
        get_detection_interface_subtask,
        pub_detection_array_subtask_bottle,
    ]
)

get_interface_call_grounded_sam_ord_val_bottle = OrderedCallsValidator(
    subtasks=[
        get_grounded_sam_interface_subtask,
        call_grounded_sam_subtask_bottle,
    ]
)

complete_object_interaction_bottle_validator = OrderedCallsValidator(
    subtasks=[
        call_grounding_dino_bottle_subtask,
        call_grounded_sam_bottle_enhanced_subtask,
        call_manipulator_bottle_subtask,
        pub_hri_bottle_interaction_subtask,
    ]
)


def get_custom_interfaces_tasks(
    extra_tool_calls: List[int] = [0],
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"],
    n_shots: List[Literal[0, 2, 5]] = [0, 2, 5],
) -> List[Task]:
    """Get predefined custom_interfaces tasks.

    Parameters
    ----------
    Parameters match :class:`~src.rai_bench.rai_bench.test_models.ToolCallingAgentBenchmarkConfig`.
    See the class documentation for parameter descriptions.

    Returns
    -------
    Returned list match :func:`~src.rai_bench.rai_bench.tool_calling_agent.predefined.tasks.get_tasks`.
    """
    tasks: List[Task] = []

    for extra_calls in extra_tool_calls:
        for detail in prompt_detail:
            for shots in n_shots:
                task_args = TaskArgs(
                    extra_tool_calls=extra_calls,
                    prompt_detail=detail,
                    examples_in_system_prompt=shots,
                )

                tasks.append(
                    PublishROS2HRIMessageTextTask(
                        task_args=task_args,
                        text=HELLO_TEXT,
                    ),
                )
                tasks.append(
                    PublishROS2AudioMessageTask(
                        task_args=task_args,
                        audio=BASIC_AUDIO_SAMPLES,
                        sample_rate=BASIC_SAMPLE_RATE,
                        channels=BASIC_CHANNELS,
                    )
                )

                tasks.append(
                    PublishROS2DetectionArrayTask(
                        validators=[get_interface_publish_detection_ord_val_person],
                        task_args=task_args,
                        detection_classes=[PERSON_CLASS],
                        bbox_centers=[PERSON_BBOX_CENTER],
                        bbox_sizes=[PERSON_BBOX_SIZE],
                    )
                )
                tasks.append(
                    PublishROS2DetectionArrayTask(
                        validators=[get_interface_publish_detection_ord_val_bottle],
                        task_args=task_args,
                        detection_classes=[BOTTLE_CLASS],
                        bbox_centers=[BOTTLE_BBOX_CENTER],
                        bbox_sizes=[BOTTLE_BBOX_SIZE],
                    )
                )

                tasks.append(
                    CallROS2ManipulatorMoveToServiceTask(
                        task_args=task_args,
                        target_x=STANDARD_TARGET_POSITION[0],
                        target_y=STANDARD_TARGET_POSITION[1],
                        target_z=STANDARD_TARGET_POSITION[2],
                        initial_gripper_state=True,
                        final_gripper_state=False,
                    )
                )

                tasks.append(
                    CallGroundedSAMSegmentTask(
                        validators=[get_interface_call_grounded_sam_ord_val_bottle],
                        task_args=task_args,
                        detection_classes=[BOTTLE_CLASS],
                        bbox_centers=[BOTTLE_BBOX_CENTER],
                        bbox_sizes=[BOTTLE_BBOX_SIZE],
                        scores=[BOTTLE_SCORE],
                        positions_3d=[BOTTLE_POSITION_3D],
                        image_width=STANDARD_IMAGE_WIDTH,
                        image_height=STANDARD_IMAGE_HEIGHT,
                        image_encoding=STANDARD_IMAGE_ENCODING,
                    )
                )

                tasks.append(
                    CallGroundingDinoClassify(
                        task_args=task_args,
                        classes=GROUNDING_DINO_CLASSES,
                        box_threshold=BOX_THRESHOLD_1,
                        text_threshold=TEXT_THRESHOLD_1,
                    )
                )
                tasks.append(
                    CallGetLogDigestTask(
                        task_args=task_args,
                    )
                )
                tasks.append(
                    CallVectorStoreRetrievalTask(
                        task_args=task_args,
                        query=ROBOT_PURPOSE_QUERY,
                    )
                )
                tasks.append(
                    CallWhatISeeTask(
                        task_args=task_args,
                    )
                )

                tasks.append(
                    CompleteObjectInteractionTask(
                        validators=[complete_object_interaction_bottle_validator],
                        task_args=task_args,
                        target_classes=BOTTLE_CLASS,
                        box_threshold=BOTTLE_BOX_THRESHOLD,
                        text_threshold=BOTTLE_TEXT_THRESHOLD,
                        detection_classes=[BOTTLE_CLASS],
                        bbox_centers=[BOTTLE_BBOX_CENTER],
                        bbox_sizes=[BOTTLE_BBOX_SIZE],
                        scores=[BOTTLE_ENHANCED_SCORE],
                        positions_3d=[BOTTLE_POSITION_3D],
                        image_width=STANDARD_IMAGE_WIDTH,
                        image_height=STANDARD_IMAGE_HEIGHT,
                        image_encoding=STANDARD_IMAGE_ENCODING,
                        target_x=BOTTLE_POSITION_3D[0],
                        target_y=BOTTLE_POSITION_3D[1],
                        target_z=BOTTLE_POSITION_3D[2],
                        initial_gripper=True,
                        final_gripper=False,
                        interaction_message=BOTTLE_INTERACTION_MESSAGE,
                    )
                )

                tasks.append(
                    EmergencyResponseProtocolTask(
                        task_args=task_args,
                        classes=PERSON_CLASS,
                        box_threshold=EMERGENCY_BOX_THRESHOLD,
                        text_threshold=EMERGENCY_TEXT_THRESHOLD,
                        detection_classes=[PERSON_CLASS],
                        bbox_centers=[EMERGENCY_BBOX_CENTER],
                        bbox_sizes=[EMERGENCY_BBOX_SIZE],
                        scores=[EMERGENCY_SCORE],
                        positions_3d=[EMERGENCY_POSITION_3D],
                        image_width=HD_IMAGE_WIDTH,
                        image_height=HD_IMAGE_HEIGHT,
                        image_encoding=STANDARD_IMAGE_ENCODING,
                        audio_samples=EMERGENCY_AUDIO_SAMPLES,
                        sample_rate=EMERGENCY_SAMPLE_RATE,
                        channels=EMERGENCY_CHANNELS,
                        message=EMERGENCY_MESSAGE,
                    )
                )

    return tasks
