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
    CompleteObjectInteractionTask,
    EmergencyResponseProtocolTask,
    MultiModalSceneDocumentationTask,
    PublishROS2AudioMessageTask,
    PublishROS2DetectionArrayTask,
    PublishROS2HRIMessageTextTask,
)
from rai_bench.tool_calling_agent.validators import (
    OrderedCallsValidator,
)

# ROS2 Topic and Service Names
HRI_TOPIC = "/to_human"
AUDIO_TOPIC = "/audio_output"
DETECTIONS_TOPIC = "/detections"
MANIPULATOR_SERVICE = "/manipulator_move_to"
GROUNDED_SAM_SERVICE = "/grounded_sam_segment"
GROUNDING_DINO_SERVICE = "/grounding_dino_classify"
LOG_DIGEST_SERVICE = "/get_log_digest"
VECTOR_STORE_SERVICE = "/rai_whoami_documentation_service"
WHAT_I_SEE_SERVICE = "/rai_whatisee_get"

# ROS2 Message and Service Types
HRI_MESSAGE_TYPE = "rai_interfaces/msg/HRIMessage"
AUDIO_MESSAGE_TYPE = "rai_interfaces/msg/AudioMessage"
DETECTION_ARRAY_MESSAGE_TYPE = "rai_interfaces/msg/RAIDetectionArray"
MANIPULATOR_SERVICE_TYPE = "rai_interfaces/srv/ManipulatorMoveTo"
GROUNDED_SAM_SERVICE_TYPE = "rai_interfaces/srv/RAIGroundedSam"
GROUNDING_DINO_SERVICE_TYPE = "rai_interfaces/srv/RAIGroundingDino"
STRING_LIST_SERVICE_TYPE = "rai_interfaces/srv/StringList"
VECTOR_STORE_SERVICE_TYPE = "rai_interfaces/srv/VectorStoreRetrieval"
WHAT_I_SEE_SERVICE_TYPE = "rai_interfaces/srv/WhatISee"

# Detection Classes
PERSON_CLASS = "person"
BOTTLE_CLASS = "bottle"
CUP_CLASS = "cup"
CAR_CLASS = "car"
BOOK_CLASS = "book"
LAPTOP_CLASS = "laptop"
COFFEE_CUP_CLASS = "coffee_cup"
MICROWAVE_CLASS = "microwave"
KNIFE_CLASS = "knife"

# Text Messages
HELLO_TEXT = "Hello!"
BOTTLE_INTERACTION_MESSAGE = (
    "Initiating object interaction sequence with detected bottle"
)
CUP_INTERACTION_MESSAGE = "Successfully completed cup grasping sequence"
OFFICE_DOCUMENTATION_REPORT = "Scene Documentation Complete: Recorded 3 objects with audio markers and safety analysis"
KITCHEN_DOCUMENTATION_REPORT = (
    "Kitchen environment documented: 3 objects identified with safety protocols applied"
)
EMERGENCY_MESSAGE = "Person detected, alarm started."
INTRUDER_MESSAGE = (
    "ALERT: Unauthorized person detected in secure area. Security protocol activated."
)

# Audio Parameters
BASIC_AUDIO_SAMPLES = [123, 456, 789]
BASIC_SAMPLE_RATE = 44100
BASIC_CHANNELS = 2
EMERGENCY_AUDIO_SAMPLES = [880, 880, 880, 1760]
EMERGENCY_SAMPLE_RATE = 8000
EMERGENCY_CHANNELS = 1
INTRUDER_AUDIO_SAMPLES = [1000, 500, 1000, 500, 1000]
INTRUDER_SAMPLE_RATE = 16000
INTRUDER_CHANNELS = 2

# Threshold Parameters
BOX_THRESHOLD_1 = 0.4
TEXT_THRESHOLD_1 = 0.25
BOTTLE_BOX_THRESHOLD = 0.35
BOTTLE_TEXT_THRESHOLD = 0.2
CUP_BOX_THRESHOLD = 0.4
CUP_TEXT_THRESHOLD = 0.25
EMERGENCY_BOX_THRESHOLD = 0.9
EMERGENCY_TEXT_THRESHOLD = 0.8
INTRUDER_BOX_THRESHOLD = 0.85
INTRUDER_TEXT_THRESHOLD = 0.75

# Bounding Box Parameters
PERSON_BBOX_CENTER = (320.0, 320.0)
PERSON_BBOX_SIZE = (50.0, 50.0)
CAR_BBOX_CENTER = (640.0, 480.0)
CAR_BBOX_SIZE = (120.0, 80.0)
BOTTLE_BBOX_CENTER = (320.0, 240.0)
BOTTLE_BBOX_SIZE = (80.0, 120.0)
BOOK_BBOX_CENTER = (480.0, 360.0)
BOOK_BBOX_SIZE = (100.0, 150.0)
CUP_BBOX_CENTER = (400.0, 300.0)
CUP_BBOX_SIZE = (60.0, 80.0)
EMERGENCY_BBOX_CENTER = (320.0, 240.0)
EMERGENCY_BBOX_SIZE = (100.0, 180.0)
INTRUDER_BBOX_CENTER = (640.0, 360.0)
INTRUDER_BBOX_SIZE = (120.0, 200.0)

# Office Scene Detection Parameters
OFFICE_PERSON_BBOX_CENTER = (160.0, 200.0)
OFFICE_PERSON_BBOX_SIZE = (80.0, 160.0)
OFFICE_LAPTOP_BBOX_CENTER = (400.0, 300.0)
OFFICE_LAPTOP_BBOX_SIZE = (200.0, 120.0)
OFFICE_COFFEE_BBOX_CENTER = (520.0, 180.0)
OFFICE_COFFEE_BBOX_SIZE = (60.0, 80.0)

# Kitchen Scene Detection Parameters
KITCHEN_PERSON_BBOX_CENTER = (200.0, 250.0)
KITCHEN_PERSON_BBOX_SIZE = (90.0, 170.0)
KITCHEN_MICROWAVE_BBOX_CENTER = (500.0, 200.0)
KITCHEN_MICROWAVE_BBOX_SIZE = (150.0, 100.0)
KITCHEN_KNIFE_BBOX_CENTER = (350.0, 400.0)
KITCHEN_KNIFE_BBOX_SIZE = (80.0, 20.0)

# Score Parameters
BOTTLE_SCORE = 0.85
BOTTLE_ENHANCED_SCORE = 0.87
BOOK_SCORE = 0.91
CUP_SCORE = 0.92
EMERGENCY_SCORE = 0.95
INTRUDER_SCORE = 0.93

# 3D Position Parameters
STANDARD_TARGET_POSITION = (1.0, 2.0, 3.0)
ALTERNATIVE_TARGET_POSITION = (0.5, -1.5, 2.2)
BOTTLE_POSITION_3D = (1.2, 0.0, 0.5)
BOOK_POSITION_3D = (0.8, 0.3, 0.8)
CUP_POSITION_3D = (0.8, -0.3, 0.7)
EMERGENCY_POSITION_3D = (2.0, 0.0, 0.0)
INTRUDER_POSITION_3D = (1.5, 0.5, 0.0)

# Image Parameters
STANDARD_IMAGE_WIDTH = 640
STANDARD_IMAGE_HEIGHT = 480
STANDARD_IMAGE_ENCODING = "rgb8"
HD_IMAGE_WIDTH = 1280
HD_IMAGE_HEIGHT = 720
HD_IMAGE_ENCODING = "bgr8"
FULL_HD_IMAGE_WIDTH = 1920
FULL_HD_IMAGE_HEIGHT = 1080

# Query Strings
ROBOT_PURPOSE_QUERY = "What is the purpose of this robot?"
GROUNDING_DINO_CLASSES = "bottle, book, chair"
SAFETY_PROTOCOLS_QUERY = (
    "What safety protocols apply when humans and robots share workspace?"
)
KITCHEN_SAFETY_QUERY = (
    "What are the safety considerations for kitchen robot operations?"
)

########## SUBTASKS #################################################################
pub_HRIMessage_text_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": HELLO_TEXT},
)

get_HRIMessage_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": HRI_MESSAGE_TYPE},
)

pub_audio_message_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=AUDIO_TOPIC,
    expected_message_type=AUDIO_MESSAGE_TYPE,
    expected_fields={
        "samples": BASIC_AUDIO_SAMPLES,
        "sample_rate": BASIC_SAMPLE_RATE,
        "channels": BASIC_CHANNELS,
    },
)

get_audio_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": AUDIO_MESSAGE_TYPE},
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
pub_detection_array_subtask_car = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=DETECTIONS_TOPIC,
    expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
    expected_fields={
        "detections.0.results.0.hypothesis.class_id": CAR_CLASS,
        "detections.0.bbox.center.x": CAR_BBOX_CENTER[0],
        "detections.0.bbox.center.y": CAR_BBOX_CENTER[1],
        "detections.0.bbox.size_x": CAR_BBOX_SIZE[0],
        "detections.0.bbox.size_y": CAR_BBOX_SIZE[1],
    },
)

get_detection_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": DETECTION_ARRAY_MESSAGE_TYPE},
)
get_detection_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": DETECTION_ARRAY_MESSAGE_TYPE},
)

call_manipulator_service_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=MANIPULATOR_SERVICE,
    expected_service_type=MANIPULATOR_SERVICE_TYPE,
    expected_fields={
        "target_pose.pose.position.x": STANDARD_TARGET_POSITION[0],
        "target_pose.pose.position.y": STANDARD_TARGET_POSITION[1],
        "target_pose.pose.position.z": STANDARD_TARGET_POSITION[2],
        "initial_gripper_state": True,
        "final_gripper_state": False,
    },
)

get_manipulator_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": MANIPULATOR_SERVICE_TYPE},
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
call_grounded_sam_subtask_book = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SERVICE,
    expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": BOOK_CLASS,
        "detections.detections.0.results.0.hypothesis.score": BOOK_SCORE,
        "detections.detections.0.results.0.pose.pose.position.x": BOOK_POSITION_3D[0],
        "detections.detections.0.results.0.pose.pose.position.y": BOOK_POSITION_3D[1],
        "detections.detections.0.results.0.pose.pose.position.z": BOOK_POSITION_3D[2],
        "detections.detections.0.bbox.center.x": BOOK_BBOX_CENTER[0],
        "detections.detections.0.bbox.center.y": BOOK_BBOX_CENTER[1],
        "detections.detections.0.bbox.size_x": BOOK_BBOX_SIZE[0],
        "detections.detections.0.bbox.size_y": BOOK_BBOX_SIZE[1],
        "source_img.width": HD_IMAGE_WIDTH,
        "source_img.height": HD_IMAGE_HEIGHT,
        "source_img.encoding": HD_IMAGE_ENCODING,
    },
)


get_grounded_sam_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": GROUNDED_SAM_SERVICE_TYPE},
)

call_grounding_dino_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SERVICE,
    expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
    expected_fields={
        "classes": GROUNDING_DINO_CLASSES,
        "box_threshold": BOX_THRESHOLD_1,
        "text_threshold": TEXT_THRESHOLD_1,
    },
)

get_grounding_dino_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": GROUNDING_DINO_SERVICE_TYPE},
)

call_log_digest_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=LOG_DIGEST_SERVICE,
    expected_service_type=STRING_LIST_SERVICE_TYPE,
    expected_fields={"": {}},
)

get_log_digest_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": STRING_LIST_SERVICE_TYPE},
)

call_vector_store_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=VECTOR_STORE_SERVICE,
    expected_service_type=VECTOR_STORE_SERVICE_TYPE,
    expected_fields={
        "query": ROBOT_PURPOSE_QUERY,
    },
)

get_vector_store_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": VECTOR_STORE_SERVICE_TYPE},
)

call_what_i_see_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=WHAT_I_SEE_SERVICE,
    expected_service_type=WHAT_I_SEE_SERVICE_TYPE,
    expected_fields={"": {}},
)

get_what_i_see_interface_subtask = CheckArgsToolCallSubTask(
    expected_tool_name="get_ros2_message_interface",
    expected_args={"msg_type": WHAT_I_SEE_SERVICE_TYPE},
)

# New Task Subtasks - CompleteObjectInteractionTask
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

call_grounded_sam_bottle_subtask = CheckServiceFieldsToolCallSubTask(
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

call_grounding_dino_cup_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SERVICE,
    expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
    expected_fields={
        "classes": CUP_CLASS,
        "box_threshold": CUP_BOX_THRESHOLD,
        "text_threshold": CUP_TEXT_THRESHOLD,
    },
)

call_grounded_sam_cup_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SERVICE,
    expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": CUP_CLASS,
        "detections.detections.0.results.0.hypothesis.score": CUP_SCORE,
        "detections.detections.0.results.0.pose.pose.position.x": CUP_POSITION_3D[0],
        "detections.detections.0.results.0.pose.pose.position.y": CUP_POSITION_3D[1],
        "detections.detections.0.results.0.pose.pose.position.z": CUP_POSITION_3D[2],
        "detections.detections.0.bbox.center.x": CUP_BBOX_CENTER[0],
        "detections.detections.0.bbox.center.y": CUP_BBOX_CENTER[1],
        "detections.detections.0.bbox.size_x": CUP_BBOX_SIZE[0],
        "detections.detections.0.bbox.size_y": CUP_BBOX_SIZE[1],
        "source_img.width": HD_IMAGE_WIDTH,
        "source_img.height": HD_IMAGE_HEIGHT,
        "source_img.encoding": HD_IMAGE_ENCODING,
    },
)

call_manipulator_cup_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=MANIPULATOR_SERVICE,
    expected_service_type=MANIPULATOR_SERVICE_TYPE,
    expected_fields={
        "target_pose.pose.position.x": CUP_POSITION_3D[0],
        "target_pose.pose.position.y": CUP_POSITION_3D[1],
        "target_pose.pose.position.z": CUP_POSITION_3D[2],
        "initial_gripper_state": False,
        "final_gripper_state": True,
    },
)

pub_hri_cup_interaction_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": CUP_INTERACTION_MESSAGE},
)

# New Task Subtasks - MultiModalSceneDocumentationTask
pub_detection_office_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=DETECTIONS_TOPIC,
    expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
    expected_fields={
        "detections.0.results.0.hypothesis.class_id": PERSON_CLASS,
        "detections.0.bbox.center.x": OFFICE_PERSON_BBOX_CENTER[0],
        "detections.0.bbox.center.y": OFFICE_PERSON_BBOX_CENTER[1],
        "detections.0.bbox.size_x": OFFICE_PERSON_BBOX_SIZE[0],
        "detections.0.bbox.size_y": OFFICE_PERSON_BBOX_SIZE[1],
        "detections.1.results.0.hypothesis.class_id": LAPTOP_CLASS,
        "detections.1.bbox.center.x": OFFICE_LAPTOP_BBOX_CENTER[0],
        "detections.1.bbox.center.y": OFFICE_LAPTOP_BBOX_CENTER[1],
        "detections.1.bbox.size_x": OFFICE_LAPTOP_BBOX_SIZE[0],
        "detections.1.bbox.size_y": OFFICE_LAPTOP_BBOX_SIZE[1],
        "detections.2.results.0.hypothesis.class_id": COFFEE_CUP_CLASS,
        "detections.2.bbox.center.x": OFFICE_COFFEE_BBOX_CENTER[0],
        "detections.2.bbox.center.y": OFFICE_COFFEE_BBOX_CENTER[1],
        "detections.2.bbox.size_x": OFFICE_COFFEE_BBOX_SIZE[0],
        "detections.2.bbox.size_y": OFFICE_COFFEE_BBOX_SIZE[1],
    },
)

call_vector_store_safety_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=VECTOR_STORE_SERVICE,
    expected_service_type=VECTOR_STORE_SERVICE_TYPE,
    expected_fields={
        "query": SAFETY_PROTOCOLS_QUERY,
    },
)

pub_hri_office_documentation_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": OFFICE_DOCUMENTATION_REPORT},
)

pub_detection_kitchen_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=DETECTIONS_TOPIC,
    expected_message_type=DETECTION_ARRAY_MESSAGE_TYPE,
    expected_fields={
        "detections.0.results.0.hypothesis.class_id": PERSON_CLASS,
        "detections.0.bbox.center.x": KITCHEN_PERSON_BBOX_CENTER[0],
        "detections.0.bbox.center.y": KITCHEN_PERSON_BBOX_CENTER[1],
        "detections.0.bbox.size_x": KITCHEN_PERSON_BBOX_SIZE[0],
        "detections.0.bbox.size_y": KITCHEN_PERSON_BBOX_SIZE[1],
        "detections.1.results.0.hypothesis.class_id": MICROWAVE_CLASS,
        "detections.1.bbox.center.x": KITCHEN_MICROWAVE_BBOX_CENTER[0],
        "detections.1.bbox.center.y": KITCHEN_MICROWAVE_BBOX_CENTER[1],
        "detections.1.bbox.size_x": KITCHEN_MICROWAVE_BBOX_SIZE[0],
        "detections.1.bbox.size_y": KITCHEN_MICROWAVE_BBOX_SIZE[1],
        "detections.2.results.0.hypothesis.class_id": KNIFE_CLASS,
        "detections.2.bbox.center.x": KITCHEN_KNIFE_BBOX_CENTER[0],
        "detections.2.bbox.center.y": KITCHEN_KNIFE_BBOX_CENTER[1],
        "detections.2.bbox.size_x": KITCHEN_KNIFE_BBOX_SIZE[0],
        "detections.2.bbox.size_y": KITCHEN_KNIFE_BBOX_SIZE[1],
    },
)

call_vector_store_kitchen_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=VECTOR_STORE_SERVICE,
    expected_service_type=VECTOR_STORE_SERVICE_TYPE,
    expected_fields={
        "query": KITCHEN_SAFETY_QUERY,
    },
)

pub_hri_kitchen_documentation_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": KITCHEN_DOCUMENTATION_REPORT},
)

# New Task Subtasks - EmergencyResponseProtocolTask
call_grounding_dino_emergency_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SERVICE,
    expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
    expected_fields={
        "classes": PERSON_CLASS,
        "box_threshold": EMERGENCY_BOX_THRESHOLD,
        "text_threshold": EMERGENCY_TEXT_THRESHOLD,
    },
)

call_grounded_sam_emergency_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SERVICE,
    expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": PERSON_CLASS,
        "detections.detections.0.results.0.hypothesis.score": EMERGENCY_SCORE,
        "detections.detections.0.results.0.pose.pose.position.x": EMERGENCY_POSITION_3D[
            0
        ],
        "detections.detections.0.results.0.pose.pose.position.y": EMERGENCY_POSITION_3D[
            1
        ],
        "detections.detections.0.results.0.pose.pose.position.z": EMERGENCY_POSITION_3D[
            2
        ],
        "detections.detections.0.bbox.center.x": EMERGENCY_BBOX_CENTER[0],
        "detections.detections.0.bbox.center.y": EMERGENCY_BBOX_CENTER[1],
        "detections.detections.0.bbox.size_x": EMERGENCY_BBOX_SIZE[0],
        "detections.detections.0.bbox.size_y": EMERGENCY_BBOX_SIZE[1],
        "source_img.width": HD_IMAGE_WIDTH,
        "source_img.height": HD_IMAGE_HEIGHT,
        "source_img.encoding": STANDARD_IMAGE_ENCODING,
    },
)

pub_audio_emergency_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=AUDIO_TOPIC,
    expected_message_type=AUDIO_MESSAGE_TYPE,
    expected_fields={
        "samples": EMERGENCY_AUDIO_SAMPLES,
        "sample_rate": EMERGENCY_SAMPLE_RATE,
        "channels": EMERGENCY_CHANNELS,
    },
)

pub_hri_emergency_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": EMERGENCY_MESSAGE},
)

call_grounding_dino_intruder_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDING_DINO_SERVICE,
    expected_service_type=GROUNDING_DINO_SERVICE_TYPE,
    expected_fields={
        "classes": PERSON_CLASS,
        "box_threshold": INTRUDER_BOX_THRESHOLD,
        "text_threshold": INTRUDER_TEXT_THRESHOLD,
    },
)

call_grounded_sam_intruder_subtask = CheckServiceFieldsToolCallSubTask(
    expected_tool_name="call_ros2_service",
    expected_service=GROUNDED_SAM_SERVICE,
    expected_service_type=GROUNDED_SAM_SERVICE_TYPE,
    expected_fields={
        "detections.detections.0.results.0.hypothesis.class_id": PERSON_CLASS,
        "detections.detections.0.results.0.hypothesis.score": INTRUDER_SCORE,
        "detections.detections.0.results.0.pose.pose.position.x": INTRUDER_POSITION_3D[
            0
        ],
        "detections.detections.0.results.0.pose.pose.position.y": INTRUDER_POSITION_3D[
            1
        ],
        "detections.detections.0.results.0.pose.pose.position.z": INTRUDER_POSITION_3D[
            2
        ],
        "detections.detections.0.bbox.center.x": INTRUDER_BBOX_CENTER[0],
        "detections.detections.0.bbox.center.y": INTRUDER_BBOX_CENTER[1],
        "detections.detections.0.bbox.size_x": INTRUDER_BBOX_SIZE[0],
        "detections.detections.0.bbox.size_y": INTRUDER_BBOX_SIZE[1],
        "source_img.width": FULL_HD_IMAGE_WIDTH,
        "source_img.height": FULL_HD_IMAGE_HEIGHT,
        "source_img.encoding": HD_IMAGE_ENCODING,
    },
)

pub_audio_intruder_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=AUDIO_TOPIC,
    expected_message_type=AUDIO_MESSAGE_TYPE,
    expected_fields={
        "samples": INTRUDER_AUDIO_SAMPLES,
        "sample_rate": INTRUDER_SAMPLE_RATE,
        "channels": INTRUDER_CHANNELS,
    },
)

pub_hri_intruder_subtask = CheckTopicFieldsToolCallSubTask(
    expected_tool_name="publish_ros2_message",
    expected_topic=HRI_TOPIC,
    expected_message_type=HRI_MESSAGE_TYPE,
    expected_fields={"text": INTRUDER_MESSAGE},
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

get_interface_publish_detection_ord_val_person = OrderedCallsValidator(
    subtasks=[
        get_detection_interface_subtask,
        pub_detection_array_subtask_person,
    ]
)
get_interface_publish_detection_ord_val_car = OrderedCallsValidator(
    subtasks=[
        get_detection_interface_subtask,
        pub_detection_array_subtask_car,
    ]
)

get_interface_call_manipulator_ord_val = OrderedCallsValidator(
    subtasks=[
        get_manipulator_interface_subtask,
        call_manipulator_service_subtask,
    ]
)

get_interface_call_grounded_sam_ord_val_book = OrderedCallsValidator(
    subtasks=[
        get_grounded_sam_interface_subtask,
        call_grounded_sam_subtask_book,
    ]
)
get_interface_call_grounded_sam_ord_val_bottle = OrderedCallsValidator(
    subtasks=[
        get_grounded_sam_interface_subtask,
        call_grounded_sam_subtask_bottle,
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

# New Task Validators
complete_object_interaction_bottle_validator = OrderedCallsValidator(
    subtasks=[
        call_grounding_dino_bottle_subtask,
        call_grounded_sam_bottle_subtask,
        call_manipulator_bottle_subtask,
        pub_hri_bottle_interaction_subtask,
    ]
)

complete_object_interaction_cup_validator = OrderedCallsValidator(
    subtasks=[
        call_grounding_dino_cup_subtask,
        call_grounded_sam_cup_subtask,
        call_manipulator_cup_subtask,
        pub_hri_cup_interaction_subtask,
    ]
)

multimodal_scene_documentation_office_validator = OrderedCallsValidator(
    subtasks=[
        call_what_i_see_subtask,
        pub_detection_office_subtask,
        call_vector_store_safety_subtask,
        pub_hri_office_documentation_subtask,
    ]
)

multimodal_scene_documentation_kitchen_validator = OrderedCallsValidator(
    subtasks=[
        call_what_i_see_subtask,
        pub_detection_kitchen_subtask,
        call_vector_store_kitchen_subtask,
        pub_hri_kitchen_documentation_subtask,
    ]
)

emergency_response_protocol_validator = OrderedCallsValidator(
    subtasks=[
        call_grounding_dino_emergency_subtask,
        call_grounded_sam_emergency_subtask,
        pub_audio_emergency_subtask,
        pub_hri_emergency_subtask,
    ]
)

emergency_response_intruder_validator = OrderedCallsValidator(
    subtasks=[
        call_grounding_dino_intruder_subtask,
        call_grounded_sam_intruder_subtask,
        pub_audio_intruder_subtask,
        pub_hri_intruder_subtask,
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
                        validators=[get_interface_publish_ord_val],
                        task_args=task_args,
                        text=HELLO_TEXT,
                    ),
                )
                tasks.append(
                    PublishROS2AudioMessageTask(
                        validators=[get_interface_publish_audio_ord_val],
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
                        validators=[get_interface_publish_detection_ord_val_car],
                        task_args=task_args,
                        detection_classes=[CAR_CLASS],
                        bbox_centers=[CAR_BBOX_CENTER],
                        bbox_sizes=[CAR_BBOX_SIZE],
                    )
                )

                tasks.append(
                    CallROS2ManipulatorMoveToServiceTask(
                        validators=[get_interface_call_manipulator_ord_val],
                        task_args=task_args,
                        target_x=STANDARD_TARGET_POSITION[0],
                        target_y=STANDARD_TARGET_POSITION[1],
                        target_z=STANDARD_TARGET_POSITION[2],
                        initial_gripper_state=True,
                        final_gripper_state=False,
                    )
                )
                tasks.append(
                    CallROS2ManipulatorMoveToServiceTask(
                        validators=[get_interface_call_manipulator_ord_val],
                        task_args=task_args,
                        target_x=ALTERNATIVE_TARGET_POSITION[0],
                        target_y=ALTERNATIVE_TARGET_POSITION[1],
                        target_z=ALTERNATIVE_TARGET_POSITION[2],
                        initial_gripper_state=False,
                        final_gripper_state=True,
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
                    CallGroundedSAMSegmentTask(
                        validators=[get_interface_call_grounded_sam_ord_val_book],
                        task_args=task_args,
                        detection_classes=[BOOK_CLASS],
                        bbox_centers=[BOOK_BBOX_CENTER],
                        bbox_sizes=[BOOK_BBOX_SIZE],
                        scores=[BOOK_SCORE],
                        positions_3d=[BOOK_POSITION_3D],
                        image_width=HD_IMAGE_WIDTH,
                        image_height=HD_IMAGE_HEIGHT,
                        image_encoding=HD_IMAGE_ENCODING,
                    )
                )

                tasks.append(
                    CallGroundingDinoClassify(
                        validators=[get_interface_call_grounding_dino_ord_val],
                        task_args=task_args,
                        classes=GROUNDING_DINO_CLASSES,
                        box_threshold=BOX_THRESHOLD_1,
                        text_threshold=TEXT_THRESHOLD_1,
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
                        query=ROBOT_PURPOSE_QUERY,
                    )
                )
                tasks.append(
                    CallWhatISeeTask(
                        validators=[get_interface_call_what_i_see_ord_val],
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
                    CompleteObjectInteractionTask(
                        validators=[complete_object_interaction_cup_validator],
                        task_args=task_args,
                        target_classes=CUP_CLASS,
                        box_threshold=CUP_BOX_THRESHOLD,
                        text_threshold=CUP_TEXT_THRESHOLD,
                        detection_classes=[CUP_CLASS],
                        bbox_centers=[CUP_BBOX_CENTER],
                        bbox_sizes=[CUP_BBOX_SIZE],
                        scores=[CUP_SCORE],
                        positions_3d=[CUP_POSITION_3D],
                        image_width=HD_IMAGE_WIDTH,
                        image_height=HD_IMAGE_HEIGHT,
                        image_encoding=HD_IMAGE_ENCODING,
                        target_x=CUP_POSITION_3D[0],
                        target_y=CUP_POSITION_3D[1],
                        target_z=CUP_POSITION_3D[2],
                        initial_gripper=False,
                        final_gripper=True,
                        interaction_message=CUP_INTERACTION_MESSAGE,
                    )
                )

                tasks.append(
                    MultiModalSceneDocumentationTask(
                        validators=[multimodal_scene_documentation_office_validator],
                        task_args=task_args,
                        scene_objects=[PERSON_CLASS, LAPTOP_CLASS, COFFEE_CUP_CLASS],
                        bbox_centers=[
                            OFFICE_PERSON_BBOX_CENTER,
                            OFFICE_LAPTOP_BBOX_CENTER,
                            OFFICE_COFFEE_BBOX_CENTER,
                        ],
                        bbox_sizes=[
                            OFFICE_PERSON_BBOX_SIZE,
                            OFFICE_LAPTOP_BBOX_SIZE,
                            OFFICE_COFFEE_BBOX_SIZE,
                        ],
                        scene_analysis_query=SAFETY_PROTOCOLS_QUERY,
                        documentation_report=OFFICE_DOCUMENTATION_REPORT,
                    )
                )
                tasks.append(
                    MultiModalSceneDocumentationTask(
                        validators=[multimodal_scene_documentation_kitchen_validator],
                        task_args=task_args,
                        scene_objects=[PERSON_CLASS, MICROWAVE_CLASS, KNIFE_CLASS],
                        bbox_centers=[
                            KITCHEN_PERSON_BBOX_CENTER,
                            KITCHEN_MICROWAVE_BBOX_CENTER,
                            KITCHEN_KNIFE_BBOX_CENTER,
                        ],
                        bbox_sizes=[
                            KITCHEN_PERSON_BBOX_SIZE,
                            KITCHEN_MICROWAVE_BBOX_SIZE,
                            KITCHEN_KNIFE_BBOX_SIZE,
                        ],
                        scene_analysis_query=KITCHEN_SAFETY_QUERY,
                        documentation_report=KITCHEN_DOCUMENTATION_REPORT,
                    )
                )

                tasks.append(
                    EmergencyResponseProtocolTask(
                        validators=[emergency_response_protocol_validator],
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
                tasks.append(
                    EmergencyResponseProtocolTask(
                        validators=[emergency_response_intruder_validator],
                        task_args=task_args,
                        classes=PERSON_CLASS,
                        box_threshold=INTRUDER_BOX_THRESHOLD,
                        text_threshold=INTRUDER_TEXT_THRESHOLD,
                        detection_classes=[PERSON_CLASS],
                        bbox_centers=[INTRUDER_BBOX_CENTER],
                        bbox_sizes=[INTRUDER_BBOX_SIZE],
                        scores=[INTRUDER_SCORE],
                        positions_3d=[INTRUDER_POSITION_3D],
                        image_width=FULL_HD_IMAGE_WIDTH,
                        image_height=FULL_HD_IMAGE_HEIGHT,
                        image_encoding=HD_IMAGE_ENCODING,
                        audio_samples=INTRUDER_AUDIO_SAMPLES,
                        sample_rate=INTRUDER_SAMPLE_RATE,
                        channels=INTRUDER_CHANNELS,
                        message=INTRUDER_MESSAGE,
                    )
                )

    return tasks
