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
    CallGetLogDigestTask,
    CallGroundedSAMSegmentTask,
    CallGroundingDinoClassify,
    CallROS2ManipulatorMoveToServiceTask,
    CallVectorStoreRetrievalTask,
    CallWhatISeeTask,
    PublishROS2AudioMessageTask,
    PublishROS2DetectionArrayTask,
    PublishROS2HRIMessageTextTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

########## SUBTASKS #################################################################
pub_HRIMessage_text_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic="/to_human",
    expected_message_type="rai_interfaces/msg/HRIMessage",
    expected_fields={"text": "Hello!"},
)
get_HRIMessage_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/msg/HRIMessage"},
)

pub_audio_message_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic="/audio_output",
    expected_message_type="rai_interfaces/msg/AudioMessage",
    expected_fields={"samples": [123, 456, 789], "sample_rate": 44100, "channels": 2},
)
get_audio_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/msg/AudioMessage"},
)

pub_detection_array_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic="/detections",
    expected_message_type="rai_interfaces/msg/RAIDetectionArray",
    expected_fields={
        "detections.0.results.0.hypothesis.class_id": "person",
        "detections.0.bbox.center.x": 320.0,
        "detections.0.bbox.center.y": 320.0,
        "detections.0.bbox.size_x": 50.0,
        "detections.0.bbox.size_y": 50.0,
    },
)
get_detection_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/msg/RAIDetectionArray"},
)

call_manipulator_service_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/manipulator_move_to",
    expected_service_type="rai_interfaces/srv/ManipulatorMoveTo",
    expected_fields={
        "target_pose.pose.position.x": 1.0,
        "target_pose.pose.position.y": 2.0,
        "target_pose.pose.position.z": 3.0,
        "initial_gripper_state": True,
        "final_gripper_state": False,
    },
)
get_manipulator_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/srv/ManipulatorMoveTo"},
)

call_grounded_sam_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/grounded_sam_segment",
    expected_service_type="rai_interfaces/srv/RAIGroundedSam",
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": "bottle",
        "detections.detections.0.results.0.hypothesis.score": 0.85,
        "detections.detections.0.results.0.pose.pose.position.x": 1.2,
        "detections.detections.0.results.0.pose.pose.position.y": 0.0,
        "detections.detections.0.results.0.pose.pose.position.z": 0.5,
        "detections.detections.0.bbox.center.x": 320.0,
        "detections.detections.0.bbox.center.y": 240.0,
        "detections.detections.0.bbox.size_x": 80.0,
        "detections.detections.0.bbox.size_y": 120.0,
        "source_img.width": 640,
        "source_img.height": 480,
        "source_img.encoding": "rgb8",
    },
)
get_grounded_sam_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/srv/GroundedSAMSegment"},
)

call_grounding_dino_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/grounding_dino_classify",
    expected_service_type="rai_interfaces/srv/RAIGroundingDino",
    expected_fields={
        "classes": "bottle",
        "box_threshold": 0.4,
        "text_threshold": 0.25,
    },
)
get_grounding_dino_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/srv/RAIGroundingDino"},
)

call_log_digest_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/get_log_digest",
    expected_service_type="rai_interfaces/srv/StringList",
    expected_fields={},
)
get_log_digest_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/srv/StringList"},
)

call_vector_store_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/rai_whoami_documentation_service",
    expected_service_type="rai_interfaces/srv/VectorStoreRetrieval",
    expected_fields={
        "query": "What is the purpose of this robot?",
    },
)
get_vector_store_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/srv/VectorStoreRetrieval"},
)

call_what_i_see_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service="/rai_whatisee_get",
    expected_service_type="rai_interfaces/srv/WhatISee",
    expected_fields={},
)
get_what_i_see_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": "rai_interfaces/srv/WhatISee"},
)


######### VALIDATORS #########################################################################################

get_interface_publish_ord_val = OrderedCallsValidator(
    subtasks=[
        get_HRIMessage_interface_subtask,
        pub_HRIMessage_text_subtask,
    ]
)
get_interface_publish_audio_ord_val = OrderedCallsValidator(
    subtasks=[
        get_audio_interface_subtask,
        pub_audio_message_subtask,
    ]
)


get_interface_publish_detection_ord_val = OrderedCallsValidator(
    subtasks=[
        get_detection_interface_subtask,
        pub_detection_array_subtask,
    ]
)


get_interface_call_manipulator_ord_val = OrderedCallsValidator(
    subtasks=[
        get_manipulator_interface_subtask,
        call_manipulator_service_subtask,
    ]
)


get_interface_call_grounded_sam_ord_val = OrderedCallsValidator(
    subtasks=[
        get_grounded_sam_interface_subtask,
        call_grounded_sam_subtask,
    ]
)


get_interface_call_grounding_dino_ord_val = OrderedCallsValidator(
    subtasks=[
        get_grounding_dino_interface_subtask,
        call_grounding_dino_subtask,
    ]
)


get_interface_call_log_digest_ord_val = OrderedCallsValidator(
    subtasks=[
        get_log_digest_interface_subtask,
        call_log_digest_subtask,
    ]
)


get_interface_call_vector_store_ord_val = OrderedCallsValidator(
    subtasks=[
        get_vector_store_interface_subtask,
        call_vector_store_subtask,
    ]
)


get_interface_call_what_i_see_ord_val = OrderedCallsValidator(
    subtasks=[
        get_what_i_see_interface_subtask,
        call_what_i_see_subtask,
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
                        validators=[
                            get_interface_publish_ord_val,
                        ],
                        task_args=task_args,
                        text="Hello!",
                    ),
                )
                tasks.append(
                    PublishROS2AudioMessageTask(
                        validators=[get_interface_publish_audio_ord_val],
                        task_args=task_args,
                        audio=[123, 456, 789],
                        sample_rate=44100,
                        channels=2,
                    )
                )
                tasks.append(
                    PublishROS2DetectionArrayTask(
                        validators=[get_interface_publish_detection_ord_val],
                        task_args=task_args,
                        detection_classes=["person"],
                        bbox_centers=[(320.0,320.0)],
                        bbox_sizes=[(50.0, 50.0)],
                    )
                )
                tasks.append(
                    CallROS2ManipulatorMoveToServiceTask(
                        validators=[get_interface_call_manipulator_ord_val],
                        task_args=task_args,
                        target_x=1.0,
                        target_y=2.0,
                        target_z=3.0,
                        initial_gripper_state=True,
                        final_gripper_state=False,
                    )
                )
                tasks.append(
                    CallGroundedSAMSegmentTask(
                        validators=[get_interface_call_grounded_sam_ord_val],
                        task_args=task_args,
                        detection_classes=["bottle"],  # Single detection to match subtask
                        bbox_centers=[(320.0, 240.0)],
                        bbox_sizes=[(80.0, 120.0)],
                        scores=[0.85],
                        positions_3d=[(1.2, 0.0, 0.5)],
                        image_width=640,
                        image_height=480,
                        image_encoding="rgb8",
                    )
                )
                tasks.append(
                    CallGroundingDinoClassify(
                        validators=[get_interface_call_grounding_dino_ord_val],
                        task_args=task_args,
                        classes="bottle, book, chair",
                        box_threshold=0.4,
                        text_threshold=0.25,
                    )
                )
                tasks.append(
                    CallGetLogDigestTask(
                        validators=[get_interface_call_log_digest_ord_val],
                        task_args=task_args,
                    )
                )
                tasks.append(
                    CallVectorStoreRetrievalTask(
                        validators=[get_interface_call_vector_store_ord_val],
                        task_args=task_args,
                        query="What is the purpose of this robot?",
                    )
                )
                tasks.append(
                    CallWhatISeeTask(
                        validators=[get_interface_call_what_i_see_ord_val],
                        task_args=task_args,
                    )
                )

    return tasks
