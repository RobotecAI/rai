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


import copy
from typing import Any, Dict, List

import pytest

from rai_bench.tool_calling_agent.interfaces import TaskArgs

# Import constants from predefined tasks
from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
    BASIC_AUDIO_SAMPLES,
    BASIC_CHANNELS,
    BASIC_SAMPLE_RATE,
    BOTTLE_CLASS,
    DEFAULT_SCENE_OBJECTS,
    GROUNDING_DINO_CLASSES,
    HRI_TEXT,
    PERSON_CLASS,
    ROBOT_PURPOSE_QUERY,
    STANDARD_TARGET_POSITION,
)
from rai_bench.tool_calling_agent.tasks.custom_interfaces import (
    AUDIO_MESSAGE_TYPE,
    AUDIO_TOPIC,
    BOTTLE_BBOX_CENTER,
    BOTTLE_BBOX_SIZE,
    BOTTLE_POSITION_3D,
    BOTTLE_SCORE,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    DETECTION_ARRAY_MESSAGE_TYPE,
    DETECTION_DEFAULTS,
    DETECTIONS_TOPIC,
    GROUNDED_SAM_SERVICE,
    GROUNDED_SAM_SERVICE_TYPE,
    GROUNDING_DINO_SERVICE,
    GROUNDING_DINO_SERVICE_TYPE,
    HRI_MESSAGE_TYPE,
    HRI_TOPIC,
    LOG_DIGEST_SERVICE,
    MANIPULATOR_SERVICE,
    MANIPULATOR_SERVICE_TYPE,
    STANDARD_IMAGE_ENCODING,
    STANDARD_IMAGE_HEIGHT,
    STANDARD_IMAGE_WIDTH,
    STRING_LIST_SERVICE_TYPE,
    VECTOR_STORE_SERVICE,
    VECTOR_STORE_SERVICE_TYPE,
    CallGetLogDigestTask,
    CallGroundedSAMSegmentTask,
    CallGroundingDinoClassify,
    CallROS2ManipulatorMoveToServiceTask,
    CallVectorStoreRetrievalTask,
    CompleteObjectInteractionTask,
    EmergencyResponseProtocolTask,
    MultiModalSceneDocumentationTask,
    PublishROS2AudioMessageTask,
    PublishROS2DetectionArrayTask,
    PublishROS2HRIMessageTextTask,
)


@pytest.fixture
def task_args() -> TaskArgs:
    return TaskArgs(
        extra_tool_calls=0,
        prompt_detail="brief",
        examples_in_system_prompt=0,
    )


class TestPublishROS2HRIMessageTextTask:
    """Test PublishROS2HRIMessageTextTask validation."""

    def test_publish_hri_message_valid(self, task_args: TaskArgs) -> None:
        """Test valid HRI message publication."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": HRI_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": HRI_TOPIC,
                    "message": {"text": HRI_TEXT},
                    "message_type": HRI_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            task_args=task_args,
            text=HRI_TEXT,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_publish_hri_message_wrong_text(self, task_args: TaskArgs) -> None:
        """Test HRI message with wrong text content."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": HRI_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": HRI_TOPIC,
                    "message": {"text": "Goodbye!"},  # Wrong text
                    "message_type": HRI_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            task_args=task_args,
            text=HRI_TEXT,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_hri_message_missing_interface_call(
        self, task_args: TaskArgs
    ) -> None:
        """Test missing interface call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": HRI_TOPIC,
                    "message": {"text": HRI_TEXT},
                    "message_type": HRI_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            task_args=task_args,
            text=HRI_TEXT,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_hri_message_too_much_calls(self, task_args: TaskArgs) -> None:
        """Test too many calls."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": HRI_MESSAGE_TYPE},
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": HRI_MESSAGE_TYPE},
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": HRI_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": HRI_TOPIC,
                    "message": {"text": HRI_TEXT},
                    "message_type": HRI_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            task_args=task_args,
            text=HRI_TEXT,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestPublishROS2AudioMessageTask:
    """Test PublishROS2AudioMessageTask validation."""

    def test_publish_audio_message_valid(self, task_args: TaskArgs) -> None:
        """Test valid audio message publication."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": AUDIO_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": AUDIO_TOPIC,
                    "message": {
                        "samples": BASIC_AUDIO_SAMPLES,
                        "sample_rate": BASIC_SAMPLE_RATE,
                        "channels": BASIC_CHANNELS,
                    },
                    "message_type": AUDIO_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2AudioMessageTask(
            task_args=task_args,
            audio=BASIC_AUDIO_SAMPLES,
            sample_rate=BASIC_SAMPLE_RATE,
            channels=BASIC_CHANNELS,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_publish_audio_message_wrong_param_value(self, task_args: TaskArgs) -> None:
        """Test audio message with wrong sample rate."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": AUDIO_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": AUDIO_TOPIC,
                    "message": {
                        "samples": BASIC_AUDIO_SAMPLES,
                        "sample_rate": 48000,  # Wrong sample rate
                        "channels": BASIC_CHANNELS,
                    },
                    "message_type": AUDIO_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2AudioMessageTask(
            task_args=task_args,
            audio=BASIC_AUDIO_SAMPLES,
            sample_rate=BASIC_SAMPLE_RATE,
            channels=BASIC_CHANNELS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_audio_message_missing_call(self, task_args: TaskArgs) -> None:
        """Test audio message with missing interface call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": AUDIO_TOPIC,
                    "message": {
                        "samples": BASIC_AUDIO_SAMPLES,
                        "sample_rate": BASIC_SAMPLE_RATE,
                        "channels": BASIC_CHANNELS,
                    },
                    "message_type": AUDIO_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2AudioMessageTask(
            task_args=task_args,
            audio=BASIC_AUDIO_SAMPLES,
            sample_rate=BASIC_SAMPLE_RATE,
            channels=BASIC_CHANNELS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestPublishROS2DetectionArrayTask:
    """Test PublishROS2DetectionArrayTask validation."""

    def valid_template(self, obj: str) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": DETECTION_ARRAY_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": DETECTIONS_TOPIC,
                    "message": {
                        "detections": [
                            {
                                "results": [{"hypothesis": {"class_id": obj}}],
                                "bbox": {
                                    "center": {
                                        "x": DETECTION_DEFAULTS[obj]["bbox_center"][0],
                                        "y": DETECTION_DEFAULTS[obj]["bbox_center"][1],
                                    },
                                    "size_x": DETECTION_DEFAULTS[obj]["bbox_size"][0],
                                    "size_y": DETECTION_DEFAULTS[obj]["bbox_size"][1],
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

    def valid_template_multiple_classes(
        self, classes: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate valid tool calls for multiple detection classes."""
        detections: list[Dict[str, Any]] = []
        for obj in classes:
            detections.append(
                {
                    "results": [{"hypothesis": {"class_id": obj}}],
                    "bbox": {
                        "center": {
                            "x": DETECTION_DEFAULTS[obj]["bbox_center"][0],
                            "y": DETECTION_DEFAULTS[obj]["bbox_center"][1],
                        },
                        "size_x": DETECTION_DEFAULTS[obj]["bbox_size"][0],
                        "size_y": DETECTION_DEFAULTS[obj]["bbox_size"][1],
                    },
                }
            )

        return [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": DETECTION_ARRAY_MESSAGE_TYPE},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": DETECTIONS_TOPIC,
                    "message": {"detections": detections},
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

    def test_publish_detection_array_person_valid(self, task_args: TaskArgs) -> None:
        """Test valid detection array publication with person."""

        task = PublishROS2DetectionArrayTask(
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
        )
        score = task.validate(self.valid_template(PERSON_CLASS))
        assert score == 1.0

    def test_publish_detection_array_multiple_classes_valid(
        self, task_args: TaskArgs
    ) -> None:
        """Test valid detection array publication with both bottle and person classes."""

        task = PublishROS2DetectionArrayTask(
            task_args=task_args,
            detection_classes=[BOTTLE_CLASS, PERSON_CLASS],
        )
        score = task.validate(
            self.valid_template_multiple_classes([BOTTLE_CLASS, PERSON_CLASS])
        )
        assert score == 1.0

    def test_publish_detection_array_wrong_class(self, task_args: TaskArgs) -> None:
        """Test detection array with wrong class."""
        tool_calls = copy.deepcopy(self.valid_template(PERSON_CLASS))

        # Modify the class_id to wrong value
        tool_calls[1]["args"]["message"]["detections"][0]["results"][0]["hypothesis"][
            "class_id"
        ] = BOTTLE_CLASS

        task = PublishROS2DetectionArrayTask(
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_detection_array_wrong_param_value(
        self, task_args: TaskArgs
    ) -> None:
        """Test detection array with wrong bounding box parameters."""
        tool_calls = copy.deepcopy(self.valid_template(PERSON_CLASS))

        # Modify the bbox center and size to wrong values
        tool_calls[1]["args"]["message"]["detections"][0]["bbox"]["center"]["x"] = 100.0
        tool_calls[1]["args"]["message"]["detections"][0]["bbox"]["center"]["y"] = 100.0
        tool_calls[1]["args"]["message"]["detections"][0]["bbox"]["size_x"] = 25.0
        tool_calls[1]["args"]["message"]["detections"][0]["bbox"]["size_y"] = 25.0

        task = PublishROS2DetectionArrayTask(
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_detection_array_missing_tool_call(
        self, task_args: TaskArgs
    ) -> None:
        """Test detection array with missing interface call."""
        tool_calls = copy.deepcopy(self.valid_template(PERSON_CLASS))

        # Remove the interface call
        tool_calls.pop(0)

        task = PublishROS2DetectionArrayTask(
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_detection_array_unknown_class_fallback(
        self, task_args: TaskArgs
    ) -> None:
        """Test detection array with unknown class falls back to person defaults."""
        tool_calls = copy.deepcopy(self.valid_template(PERSON_CLASS))

        # Modify the class_id to unknown class (should use person defaults)
        tool_calls[1]["args"]["message"]["detections"][0]["results"][0]["hypothesis"][
            "class_id"
        ] = "unknown_class"

        task = PublishROS2DetectionArrayTask(
            task_args=task_args,
            detection_classes=["unknown_class"],  # Uses person defaults
        )
        score = task.validate(tool_calls)
        assert score == 1.0


class TestCallGroundedSAMSegmentTask:
    """Test CallGroundedSAMSegmentTask validation."""

    VALID_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": GROUNDED_SAM_SERVICE_TYPE},
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": GROUNDED_SAM_SERVICE,
                "service_type": GROUNDED_SAM_SERVICE_TYPE,
                "service_args": {
                    "detections": {
                        "detections": [
                            {
                                "results": [
                                    {
                                        "hypothesis": {
                                            "class_id": BOTTLE_CLASS,
                                            "score": BOTTLE_SCORE,
                                        },
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": BOTTLE_POSITION_3D[0],
                                                    "y": BOTTLE_POSITION_3D[1],
                                                    "z": BOTTLE_POSITION_3D[2],
                                                }
                                            }
                                        },
                                    }
                                ],
                                "bbox": {
                                    "center": {
                                        "x": BOTTLE_BBOX_CENTER[0],
                                        "y": BOTTLE_BBOX_CENTER[1],
                                    },
                                    "size_x": BOTTLE_BBOX_SIZE[0],
                                    "size_y": BOTTLE_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "source_img": {
                        "width": STANDARD_IMAGE_WIDTH,
                        "height": STANDARD_IMAGE_HEIGHT,
                        "encoding": STANDARD_IMAGE_ENCODING,
                    },
                },
            },
        },
    ]

    def test_call_grounded_sam_bottle_valid(self, task_args: TaskArgs) -> None:
        """Test valid grounded SAM call with bottle."""
        task = CallGroundedSAMSegmentTask(
            task_args=task_args,
            detection_classes=[BOTTLE_CLASS],
        )
        score = task.validate(self.VALID_TOOL_CALLS_TEMPLATE)
        assert score == 1.0

    def test_call_grounded_sam_wrong_class(self, task_args: TaskArgs) -> None:
        """Test grounded SAM call with wrong class ID."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the class_id to wrong value
        tool_calls[1]["args"]["service_args"]["detections"]["detections"][0]["results"][
            0
        ]["hypothesis"]["class_id"] = PERSON_CLASS

        task = CallGroundedSAMSegmentTask(
            task_args=task_args,
            detection_classes=[BOTTLE_CLASS],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounded_sam_wrong_param_value(self, task_args: TaskArgs) -> None:
        """Test grounded SAM call with wrong parameter value."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the score to wrong value
        tool_calls[1]["args"]["service_args"]["detections"]["detections"][0]["results"][
            0
        ]["hypothesis"]["score"] = 0.95

        task = CallGroundedSAMSegmentTask(
            task_args=task_args,
            detection_classes=[BOTTLE_CLASS],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounded_sam_missing_tool_call(self, task_args: TaskArgs) -> None:
        """Test grounded SAM call missing the interface call."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Remove the interface call
        tool_calls.pop(0)

        task = CallGroundedSAMSegmentTask(
            task_args=task_args,
            detection_classes=[BOTTLE_CLASS],
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallGroundingDinoClassify:
    """Test CallGroundingDinoClassify validation."""

    def test_call_grounding_dino_valid(self, task_args: TaskArgs) -> None:
        """Test valid grounding DINO call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GROUNDING_DINO_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SERVICE,
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": GROUNDING_DINO_CLASSES,
                        "box_threshold": DEFAULT_BOX_THRESHOLD,
                        "text_threshold": DEFAULT_TEXT_THRESHOLD,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_grounding_dino_wrong_classes(self, task_args: TaskArgs) -> None:
        """Test grounding DINO call with wrong classes."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GROUNDING_DINO_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SERVICE,
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": "cat, dog",  # Wrong classes - expecting GROUNDING_DINO_CLASSES
                        "box_threshold": DEFAULT_BOX_THRESHOLD,
                        "text_threshold": DEFAULT_TEXT_THRESHOLD,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounding_dino_wrong_param_value(self, task_args: TaskArgs) -> None:
        """Test grounding DINO call with wrong parameter value."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GROUNDING_DINO_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SERVICE,
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": GROUNDING_DINO_CLASSES,
                        "box_threshold": 0.8,  # Wrong threshold - expecting DEFAULT_BOX_THRESHOLD
                        "text_threshold": DEFAULT_TEXT_THRESHOLD,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounding_dino_missing_tool_call(self, task_args: TaskArgs) -> None:
        """Test grounding DINO call missing the interface call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SERVICE,
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": GROUNDING_DINO_CLASSES,
                        "box_threshold": DEFAULT_BOX_THRESHOLD,
                        "text_threshold": DEFAULT_TEXT_THRESHOLD,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCompleteObjectInteractionTask:
    """Test CompleteObjectInteractionTask validation."""

    VALID_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": GROUNDING_DINO_SERVICE,
                "service_type": GROUNDING_DINO_SERVICE_TYPE,
                "service_args": {
                    "classes": "bottle",
                    "box_threshold": DEFAULT_BOX_THRESHOLD,
                    "text_threshold": DEFAULT_BOX_THRESHOLD,
                },
            },
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": GROUNDED_SAM_SERVICE,
                "service_type": GROUNDED_SAM_SERVICE_TYPE,
                "service_args": {
                    "detections": {
                        "detections": [
                            {
                                "results": [
                                    {
                                        "hypothesis": {
                                            "class_id": "bottle",
                                            "score": DETECTION_DEFAULTS["bottle"][
                                                "score"
                                            ],
                                        },
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": DETECTION_DEFAULTS["bottle"][
                                                        "position_3d"
                                                    ][0],
                                                    "y": DETECTION_DEFAULTS["bottle"][
                                                        "position_3d"
                                                    ][1],
                                                    "z": DETECTION_DEFAULTS["bottle"][
                                                        "position_3d"
                                                    ][2],
                                                }
                                            }
                                        },
                                    }
                                ],
                                "bbox": {
                                    "center": {
                                        "x": DETECTION_DEFAULTS["bottle"][
                                            "bbox_center"
                                        ][0],
                                        "y": DETECTION_DEFAULTS["bottle"][
                                            "bbox_center"
                                        ][1],
                                    },
                                    "size_x": DETECTION_DEFAULTS["bottle"]["bbox_size"][
                                        0
                                    ],
                                    "size_y": DETECTION_DEFAULTS["bottle"]["bbox_size"][
                                        1
                                    ],
                                },
                            }
                        ]
                    },
                    "source_img": {
                        "width": STANDARD_IMAGE_WIDTH,
                        "height": STANDARD_IMAGE_HEIGHT,
                        "encoding": STANDARD_IMAGE_ENCODING,
                    },
                },
            },
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": MANIPULATOR_SERVICE,
                "service_type": MANIPULATOR_SERVICE_TYPE,
                "service_args": {
                    "target_pose": {
                        "pose": {
                            "position": {
                                "x": DETECTION_DEFAULTS["bottle"]["position_3d"][0],
                                "y": DETECTION_DEFAULTS["bottle"]["position_3d"][1],
                                "z": DETECTION_DEFAULTS["bottle"]["position_3d"][2],
                            }
                        }
                    },
                    "initial_gripper_state": False,
                    "final_gripper_state": True,
                },
            },
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {
                    "text": "Initiating object interaction sequence with detected bottle"
                },
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    def test_complete_object_interaction_bottle_valid(
        self, task_args: TaskArgs
    ) -> None:
        """Test valid complete object interaction with bottle."""
        task = CompleteObjectInteractionTask(
            task_args=task_args,
            target_class="bottle",
        )
        score = task.validate(self.VALID_TOOL_CALLS_TEMPLATE)
        assert score == 1.0

    def test_complete_object_interaction_wrong_class(self, task_args: TaskArgs) -> None:
        """Test complete object interaction with wrong target class."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the classes to wrong value
        tool_calls[0]["args"]["service_args"]["classes"] = PERSON_CLASS

        task = CompleteObjectInteractionTask(
            task_args=task_args,
            target_class="bottle",
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_complete_object_interaction_wrong_param_value(
        self, task_args: TaskArgs
    ) -> None:
        """Test complete object interaction with wrong parameter value."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the box_threshold to wrong value
        tool_calls[0]["args"]["service_args"]["box_threshold"] = 0.8

        task = CompleteObjectInteractionTask(
            task_args=task_args,
            target_class="bottle",
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_complete_object_interaction_missing_tool_call(
        self, task_args: TaskArgs
    ) -> None:
        """Test complete object interaction missing a required tool call."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Remove the Grounded SAM service call (index 1)
        tool_calls.pop(1)

        task = CompleteObjectInteractionTask(
            task_args=task_args,
            target_class="bottle",
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallROS2ManipulatorMoveToServiceTask:
    """Test CallROS2ManipulatorMoveToServiceTask validation."""

    def test_call_manipulator_service_valid(self, task_args: TaskArgs) -> None:
        """Test valid manipulator service call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": MANIPULATOR_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": MANIPULATOR_SERVICE,
                    "service_type": MANIPULATOR_SERVICE_TYPE,
                    "service_args": {
                        "target_pose": {
                            "pose": {
                                "position": {
                                    "x": STANDARD_TARGET_POSITION[0],
                                    "y": STANDARD_TARGET_POSITION[1],
                                    "z": STANDARD_TARGET_POSITION[2],
                                }
                            }
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
        ]

        task = CallROS2ManipulatorMoveToServiceTask(
            task_args=task_args,
            target_x=STANDARD_TARGET_POSITION[0],
            target_y=STANDARD_TARGET_POSITION[1],
            target_z=STANDARD_TARGET_POSITION[2],
            initial_gripper_state=True,
            final_gripper_state=False,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_manipulator_service_wrong_position(self, task_args: TaskArgs) -> None:
        """Test manipulator service with wrong position values."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": MANIPULATOR_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": MANIPULATOR_SERVICE,
                    "service_type": MANIPULATOR_SERVICE_TYPE,
                    "service_args": {
                        "target_pose": {
                            "pose": {
                                "position": {
                                    "x": 2.0,
                                    "y": 3.0,
                                    "z": 4.0,
                                }  # Wrong position
                            }
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
        ]

        task = CallROS2ManipulatorMoveToServiceTask(
            task_args=task_args,
            target_x=STANDARD_TARGET_POSITION[0],
            target_y=STANDARD_TARGET_POSITION[1],
            target_z=STANDARD_TARGET_POSITION[2],
            initial_gripper_state=True,
            final_gripper_state=False,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_manipulator_service_wrong_message_type(
        self, task_args: TaskArgs
    ) -> None:
        """Test manipulator service with wrong message type."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "wrong_message_type"},  # Wrong message type
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": MANIPULATOR_SERVICE,
                    "service_type": MANIPULATOR_SERVICE_TYPE,
                    "service_args": {
                        "target_pose": {
                            "pose": {
                                "position": {
                                    "x": STANDARD_TARGET_POSITION[0],
                                    "y": STANDARD_TARGET_POSITION[1],
                                    "z": STANDARD_TARGET_POSITION[2],
                                }
                            }
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
        ]

        task = CallROS2ManipulatorMoveToServiceTask(
            task_args=task_args,
            target_x=STANDARD_TARGET_POSITION[0],
            target_y=STANDARD_TARGET_POSITION[1],
            target_z=STANDARD_TARGET_POSITION[2],
            initial_gripper_state=True,
            final_gripper_state=False,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_manipulator_service_wrong_tool_order(
        self, task_args: TaskArgs
    ) -> None:
        """Test manipulator service with wrong tool call order."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": MANIPULATOR_SERVICE,
                    "service_type": MANIPULATOR_SERVICE_TYPE,
                    "service_args": {
                        "target_pose": {
                            "pose": {
                                "position": {
                                    "x": STANDARD_TARGET_POSITION[0],
                                    "y": STANDARD_TARGET_POSITION[1],
                                    "z": STANDARD_TARGET_POSITION[2],
                                }
                            }
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": MANIPULATOR_SERVICE_TYPE},
            },
        ]

        task = CallROS2ManipulatorMoveToServiceTask(
            task_args=task_args,
            target_x=STANDARD_TARGET_POSITION[0],
            target_y=STANDARD_TARGET_POSITION[1],
            target_z=STANDARD_TARGET_POSITION[2],
            initial_gripper_state=True,
            final_gripper_state=False,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallGetLogDigestTask:
    """Test CallGetLogDigestTask validation."""

    def test_call_log_digest_valid(self, task_args: TaskArgs) -> None:
        """Test valid log digest service call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": STRING_LIST_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": LOG_DIGEST_SERVICE,
                    "service_type": STRING_LIST_SERVICE_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CallGetLogDigestTask(
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_log_digest_wrong_service_name(self, task_args: TaskArgs) -> None:
        """Test log digest with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": STRING_LIST_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_log_service",  # Wrong service name
                    "service_type": STRING_LIST_SERVICE_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CallGetLogDigestTask(
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_log_digest_wrong_message_type(self, task_args: TaskArgs) -> None:
        """Test log digest with wrong message type."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "wrong_message_type"},  # Wrong message type
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": LOG_DIGEST_SERVICE,
                    "service_type": STRING_LIST_SERVICE_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CallGetLogDigestTask(
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_log_digest_wrong_tool_order(self, task_args: TaskArgs) -> None:
        """Test log digest with wrong tool call order."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": LOG_DIGEST_SERVICE,
                    "service_type": STRING_LIST_SERVICE_TYPE,
                    "service_args": {},
                },
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": STRING_LIST_SERVICE_TYPE},
            },
        ]

        task = CallGetLogDigestTask(
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallVectorStoreRetrievalTask:
    """Test CallVectorStoreRetrievalTask validation."""

    def test_call_vector_store_valid(self, task_args: TaskArgs) -> None:
        """Test valid vector store service call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": VECTOR_STORE_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": VECTOR_STORE_SERVICE,
                    "service_type": VECTOR_STORE_SERVICE_TYPE,
                    "service_args": {
                        "query": "What is the purpose of this robot?",
                    },
                },
            },
        ]

        task = CallVectorStoreRetrievalTask(
            task_args=task_args,
            query="What is the purpose of this robot?",
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_vector_store_wrong_query(self, task_args: TaskArgs) -> None:
        """Test vector store with wrong query."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": VECTOR_STORE_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": VECTOR_STORE_SERVICE,
                    "service_type": VECTOR_STORE_SERVICE_TYPE,
                    "service_args": {
                        "query": "Wrong query text",  # Wrong query
                    },
                },
            },
        ]

        task = CallVectorStoreRetrievalTask(
            task_args=task_args,
            query=ROBOT_PURPOSE_QUERY,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_vector_store_wrong_service_name(self, task_args: TaskArgs) -> None:
        """Test vector store with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": VECTOR_STORE_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_vector_store_service",  # Wrong service name
                    "service_type": VECTOR_STORE_SERVICE_TYPE,
                    "service_args": {
                        "query": ROBOT_PURPOSE_QUERY,
                    },
                },
            },
        ]

        task = CallVectorStoreRetrievalTask(
            task_args=task_args,
            query=ROBOT_PURPOSE_QUERY,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_vector_store_wrong_tool_order(self, task_args: TaskArgs) -> None:
        """Test vector store with wrong tool call order."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": VECTOR_STORE_SERVICE,
                    "service_type": VECTOR_STORE_SERVICE_TYPE,
                    "service_args": {
                        "query": ROBOT_PURPOSE_QUERY,
                    },
                },
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": VECTOR_STORE_SERVICE_TYPE},
            },
        ]

        task = CallVectorStoreRetrievalTask(
            task_args=task_args,
            query=ROBOT_PURPOSE_QUERY,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestMultiModalSceneDocumentationTask:
    """Test MultiModalSceneDocumentationTask validation."""

    VALID_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": DETECTIONS_TOPIC,
                "message": {
                    "detections": [
                        {
                            "results": [{"hypothesis": {"class_id": PERSON_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": DETECTION_DEFAULTS["person"]["bbox_center"][0],
                                    "y": DETECTION_DEFAULTS["person"]["bbox_center"][1],
                                },
                                "size_x": DETECTION_DEFAULTS["person"]["bbox_size"][0],
                                "size_y": DETECTION_DEFAULTS["person"]["bbox_size"][1],
                            },
                        },
                        {
                            "results": [{"hypothesis": {"class_id": BOTTLE_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": DETECTION_DEFAULTS["bottle"]["bbox_center"][0],
                                    "y": DETECTION_DEFAULTS["bottle"]["bbox_center"][1],
                                },
                                "size_x": DETECTION_DEFAULTS["bottle"]["bbox_size"][0],
                                "size_y": DETECTION_DEFAULTS["bottle"]["bbox_size"][1],
                            },
                        },
                    ]
                },
                "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
            },
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": VECTOR_STORE_SERVICE,
                "service_type": VECTOR_STORE_SERVICE_TYPE,
                "service_args": {
                    "query": "What safety protocols apply when humans and robots share workspace?",
                },
            },
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {
                    "text": "Scene Documentation Complete: Recorded objects with safety analysis"
                },
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    def test_multimodal_scene_documentation_valid(self, task_args: TaskArgs) -> None:
        """Test valid multimodal scene documentation."""
        task = MultiModalSceneDocumentationTask(
            task_args=task_args,
            objects=DEFAULT_SCENE_OBJECTS,
        )
        score = task.validate(self.VALID_TOOL_CALLS_TEMPLATE)
        assert score == 1.0

    def test_multimodal_scene_documentation_wrong_class(
        self, task_args: TaskArgs
    ) -> None:
        """Test multimodal scene documentation with wrong object class."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the first detection class to wrong value
        tool_calls[0]["args"]["message"]["detections"][0]["results"][0]["hypothesis"][
            "class_id"
        ] = "unknown_object"

        task = MultiModalSceneDocumentationTask(
            task_args=task_args,
            objects=DEFAULT_SCENE_OBJECTS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_multimodal_scene_documentation_wrong_param_value(
        self, task_args: TaskArgs
    ) -> None:
        """Test multimodal scene documentation with wrong parameter value."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the safety query to wrong value
        tool_calls[1]["args"]["service_args"]["query"] = "What is the weather like?"

        task = MultiModalSceneDocumentationTask(
            task_args=task_args,
            objects=DEFAULT_SCENE_OBJECTS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_multimodal_scene_documentation_missing_tool_call(
        self, task_args: TaskArgs
    ) -> None:
        """Test multimodal scene documentation missing a required tool call."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Remove the vector store service call (index 2)
        tool_calls.pop(2)

        task = MultiModalSceneDocumentationTask(
            task_args=task_args,
            objects=DEFAULT_SCENE_OBJECTS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestEmergencyResponseProtocolTask:
    """Test EmergencyResponseProtocolTask validation."""

    VALID_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": GROUNDING_DINO_SERVICE,
                "service_type": GROUNDING_DINO_SERVICE_TYPE,
                "service_args": {
                    "classes": "person",
                    "box_threshold": DEFAULT_BOX_THRESHOLD,
                    "text_threshold": DEFAULT_TEXT_THRESHOLD,
                },
            },
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": GROUNDED_SAM_SERVICE,
                "service_type": GROUNDED_SAM_SERVICE_TYPE,
                "service_args": {
                    "detections": {
                        "detections": [
                            {
                                "results": [
                                    {
                                        "hypothesis": {
                                            "class_id": PERSON_CLASS,
                                            "score": DETECTION_DEFAULTS["person"][
                                                "score"
                                            ],
                                        },
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": DETECTION_DEFAULTS["person"][
                                                        "position_3d"
                                                    ][0],
                                                    "y": DETECTION_DEFAULTS["person"][
                                                        "position_3d"
                                                    ][1],
                                                    "z": DETECTION_DEFAULTS["person"][
                                                        "position_3d"
                                                    ][2],
                                                }
                                            }
                                        },
                                    }
                                ],
                                "bbox": {
                                    "center": {
                                        "x": DETECTION_DEFAULTS["person"][
                                            "bbox_center"
                                        ][0],
                                        "y": DETECTION_DEFAULTS["person"][
                                            "bbox_center"
                                        ][1],
                                    },
                                    "size_x": DETECTION_DEFAULTS["person"]["bbox_size"][
                                        0
                                    ],
                                    "size_y": DETECTION_DEFAULTS["person"]["bbox_size"][
                                        1
                                    ],
                                },
                            }
                        ]
                    },
                    "source_img": {
                        "width": STANDARD_IMAGE_WIDTH,
                        "height": STANDARD_IMAGE_HEIGHT,
                        "encoding": STANDARD_IMAGE_ENCODING,
                    },
                },
            },
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": AUDIO_TOPIC,
                "message": {
                    "samples": [880, 880, 880, 1760],
                    "sample_rate": 8000,
                    "channels": 1,
                },
                "message_type": AUDIO_MESSAGE_TYPE,
            },
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": "Person detected, alarm started!"},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    def test_emergency_response_protocol_valid(self, task_args: TaskArgs) -> None:
        """Test valid emergency response protocol."""
        task = EmergencyResponseProtocolTask(
            task_args=task_args,
            target_class=PERSON_CLASS,
        )
        score = task.validate(self.VALID_TOOL_CALLS_TEMPLATE)
        assert score == 1.0

    def test_emergency_response_wrong_class(self, task_args: TaskArgs) -> None:
        """Test emergency response with wrong target class."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the classes to wrong value
        tool_calls[0]["args"]["service_args"]["classes"] = BOTTLE_CLASS

        task = EmergencyResponseProtocolTask(
            task_args=task_args,
            target_class=PERSON_CLASS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_emergency_response_wrong_param_value(self, task_args: TaskArgs) -> None:
        """Test emergency response with wrong parameter value."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Modify the box_threshold to wrong value
        tool_calls[0]["args"]["service_args"]["box_threshold"] = 0.8

        task = EmergencyResponseProtocolTask(
            task_args=task_args,
            target_class=PERSON_CLASS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_emergency_response_missing_tool_call(self, task_args: TaskArgs) -> None:
        """Test emergency response missing a required tool call."""
        tool_calls = copy.deepcopy(self.VALID_TOOL_CALLS_TEMPLATE)

        # Remove the Grounded SAM service call (index 1)
        tool_calls.pop(1)

        task = EmergencyResponseProtocolTask(
            task_args=task_args,
            target_class=PERSON_CLASS,
        )
        score = task.validate(tool_calls)
        assert score == 0.0
