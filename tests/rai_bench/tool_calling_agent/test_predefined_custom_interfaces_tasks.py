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
from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
    AUDIO_MESSAGE_TYPE,
    AUDIO_TOPIC,
    BOOK_BBOX_CENTER,
    BOOK_BBOX_SIZE,
    BOOK_CLASS,
    BOOK_POSITION_3D,
    BOOK_SCORE,
    BOTTLE_BBOX_CENTER,
    BOTTLE_BBOX_SIZE,
    BOTTLE_BOX_THRESHOLD,
    BOTTLE_CLASS,
    BOTTLE_ENHANCED_SCORE,
    BOTTLE_INTERACTION_MESSAGE,
    BOTTLE_POSITION_3D,
    BOTTLE_SCORE,
    BOTTLE_TEXT_THRESHOLD,
    BOX_THRESHOLD_1,
    CAR_BBOX_CENTER,
    CAR_BBOX_SIZE,
    CAR_CLASS,
    COFFEE_CUP_CLASS,
    CUP_BBOX_CENTER,
    CUP_BBOX_SIZE,
    CUP_BOX_THRESHOLD,
    CUP_CLASS,
    CUP_INTERACTION_MESSAGE,
    CUP_POSITION_3D,
    CUP_SCORE,
    CUP_TEXT_THRESHOLD,
    DETECTION_ARRAY_MESSAGE_TYPE,
    DETECTIONS_TOPIC,
    EMERGENCY_AUDIO_SAMPLES,
    EMERGENCY_BBOX_CENTER,
    EMERGENCY_BBOX_SIZE,
    EMERGENCY_BOX_THRESHOLD,
    EMERGENCY_CHANNELS,
    EMERGENCY_MESSAGE,
    EMERGENCY_POSITION_3D,
    EMERGENCY_SAMPLE_RATE,
    EMERGENCY_SCORE,
    EMERGENCY_TEXT_THRESHOLD,
    FULL_HD_IMAGE_HEIGHT,
    FULL_HD_IMAGE_WIDTH,
    GROUNDED_SAM_SERVICE,
    GROUNDED_SAM_SERVICE_TYPE,
    GROUNDING_DINO_CLASSES,
    GROUNDING_DINO_SERVICE,
    GROUNDING_DINO_SERVICE_TYPE,
    HD_IMAGE_ENCODING,
    HD_IMAGE_HEIGHT,
    HD_IMAGE_WIDTH,
    HRI_MESSAGE_TYPE,
    HRI_TOPIC,
    INTRUDER_AUDIO_SAMPLES,
    INTRUDER_BBOX_CENTER,
    INTRUDER_BBOX_SIZE,
    INTRUDER_BOX_THRESHOLD,
    INTRUDER_CHANNELS,
    INTRUDER_MESSAGE,
    INTRUDER_POSITION_3D,
    INTRUDER_SAMPLE_RATE,
    INTRUDER_SCORE,
    INTRUDER_TEXT_THRESHOLD,
    KITCHEN_DOCUMENTATION_REPORT,
    KITCHEN_KNIFE_BBOX_CENTER,
    KITCHEN_KNIFE_BBOX_SIZE,
    KITCHEN_MICROWAVE_BBOX_CENTER,
    KITCHEN_MICROWAVE_BBOX_SIZE,
    KITCHEN_PERSON_BBOX_CENTER,
    KITCHEN_PERSON_BBOX_SIZE,
    KITCHEN_SAFETY_QUERY,
    KNIFE_CLASS,
    LAPTOP_CLASS,
    LOG_DIGEST_SERVICE,
    MANIPULATOR_SERVICE,
    MANIPULATOR_SERVICE_TYPE,
    MICROWAVE_CLASS,
    OFFICE_COFFEE_BBOX_CENTER,
    OFFICE_COFFEE_BBOX_SIZE,
    OFFICE_DOCUMENTATION_REPORT,
    OFFICE_LAPTOP_BBOX_CENTER,
    OFFICE_LAPTOP_BBOX_SIZE,
    OFFICE_PERSON_BBOX_CENTER,
    OFFICE_PERSON_BBOX_SIZE,
    PERSON_BBOX_CENTER,
    PERSON_BBOX_SIZE,
    PERSON_CLASS,
    ROBOT_PURPOSE_QUERY,
    SAFETY_PROTOCOLS_QUERY,
    STANDARD_IMAGE_ENCODING,
    STANDARD_IMAGE_HEIGHT,
    STANDARD_IMAGE_WIDTH,
    STRING_LIST_SERVICE_TYPE,
    TEXT_THRESHOLD_1,
    VECTOR_STORE_SERVICE,
    VECTOR_STORE_SERVICE_TYPE,
    WHAT_I_SEE_SERVICE,
    WHAT_I_SEE_SERVICE_TYPE,
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


@pytest.fixture
def task_args() -> TaskArgs:
    return TaskArgs(
        extra_tool_calls=0,
        prompt_detail="brief",
        examples_in_system_prompt=0,
    )


class TestPublishROS2HRIMessageTextTask:
    """Test PublishROS2HRIMessageTextTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_publish_ord_val,
    )

    def test_publish_hri_message_valid(self, task_args: TaskArgs) -> None:
        """Test valid HRI message publication."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/HRIMessage"},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": "/to_human",
                    "message": {"text": "Hello!"},
                    "message_type": "rai_interfaces/msg/HRIMessage",
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            validators=[self.get_interface_publish_ord_val],
            task_args=task_args,
            text="Hello!",
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_publish_hri_message_wrong_text(self, task_args: TaskArgs) -> None:
        """Test HRI message with wrong text content."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/HRIMessage"},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": "/to_human",
                    "message": {"text": "Goodbye!"},  # Wrong text
                    "message_type": "rai_interfaces/msg/HRIMessage",
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            validators=[self.get_interface_publish_ord_val],
            task_args=task_args,
            text="Hello!",
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
                    "topic": "/to_human",
                    "message": {"text": "Hello!"},
                    "message_type": "rai_interfaces/msg/HRIMessage",
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            validators=[self.get_interface_publish_ord_val],
            task_args=task_args,
            text="Hello!",
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_hri_message_too_much_calls(self, task_args: TaskArgs) -> None:
        """Test too many calls."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/HRIMessage"},
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/HRIMessage"},
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/HRIMessage"},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": "/to_human",
                    "message": {"text": "Hello!"},
                    "message_type": "rai_interfaces/msg/HRIMessage",
                },
            },
        ]

        task = PublishROS2HRIMessageTextTask(
            validators=[self.get_interface_publish_ord_val],
            task_args=task_args,
            text="Hello!",
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestPublishROS2AudioMessageTask:
    """Test PublishROS2AudioMessageTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_publish_audio_ord_val,
    )

    def test_publish_audio_message_valid(self, task_args: TaskArgs) -> None:
        """Test valid audio message publication."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/AudioMessage"},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": "/audio_output",
                    "message": {
                        "samples": [123, 456, 789],
                        "sample_rate": 44100,
                        "channels": 2,
                    },
                    "message_type": "rai_interfaces/msg/AudioMessage",
                },
            },
        ]

        task = PublishROS2AudioMessageTask(
            validators=[self.get_interface_publish_audio_ord_val],
            task_args=task_args,
            audio=[123, 456, 789],
            sample_rate=44100,
            channels=2,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_publish_audio_message_wrong_param_value(self, task_args: TaskArgs) -> None:
        """Test audio message with wrong sample rate."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/msg/AudioMessage"},
            },
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": "/audio_output",
                    "message": {
                        "samples": [123, 456, 789],
                        "sample_rate": 48000,  # Wrong sample rate
                        "channels": 2,
                    },
                    "message_type": "rai_interfaces/msg/AudioMessage",
                },
            },
        ]

        task = PublishROS2AudioMessageTask(
            validators=[self.get_interface_publish_audio_ord_val],
            task_args=task_args,
            audio=[123, 456, 789],
            sample_rate=44100,
            channels=2,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_audio_message_missing_call(self, task_args: TaskArgs) -> None:
        """Test audio message with wrong sample rate."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "publish_ros2_message",
                "args": {
                    "topic": "/audio_output",
                    "message": {
                        "samples": [123, 456, 789],
                        "sample_rate": 48000,  # Wrong sample rate
                        "channels": 2,
                    },
                    "message_type": "rai_interfaces/msg/AudioMessage",
                },
            },
        ]

        task = PublishROS2AudioMessageTask(
            validators=[self.get_interface_publish_audio_ord_val],
            task_args=task_args,
            audio=[123, 456, 789],
            sample_rate=44100,
            channels=2,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestPublishROS2DetectionArrayTask:
    """Test PublishROS2DetectionArrayTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_publish_detection_ord_val_car,
        get_interface_publish_detection_ord_val_person,
    )

    def test_publish_detection_array_person_valid(self, task_args: TaskArgs) -> None:
        """Test valid detection array publication with person."""
        tool_calls: List[Dict[str, Any]] = [
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
                                "results": [{"hypothesis": {"class_id": PERSON_CLASS}}],
                                "bbox": {
                                    "center": {
                                        "x": PERSON_BBOX_CENTER[0],
                                        "y": PERSON_BBOX_CENTER[1],
                                    },
                                    "size_x": PERSON_BBOX_SIZE[0],
                                    "size_y": PERSON_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2DetectionArrayTask(
            validators=[self.get_interface_publish_detection_ord_val_person],
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
            bbox_centers=[PERSON_BBOX_CENTER],
            bbox_sizes=[PERSON_BBOX_SIZE],
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_publish_detection_array_car_valid(self, task_args: TaskArgs) -> None:
        """Test valid detection array publication with car."""
        tool_calls: List[Dict[str, Any]] = [
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
                                "results": [{"hypothesis": {"class_id": CAR_CLASS}}],
                                "bbox": {
                                    "center": {
                                        "x": CAR_BBOX_CENTER[0],
                                        "y": CAR_BBOX_CENTER[1],
                                    },
                                    "size_x": CAR_BBOX_SIZE[0],
                                    "size_y": CAR_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2DetectionArrayTask(
            validators=[self.get_interface_publish_detection_ord_val_car],
            task_args=task_args,
            detection_classes=[CAR_CLASS],
            bbox_centers=[CAR_BBOX_CENTER],
            bbox_sizes=[CAR_BBOX_SIZE],
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_publish_detection_array_wrong_class(self, task_args: TaskArgs) -> None:
        """Test detection array with wrong class."""
        tool_calls: List[Dict[str, Any]] = [
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
                                "results": [
                                    {
                                        "hypothesis": {"class_id": CAR_CLASS}
                                    }  # Wrong class - expecting person
                                ],
                                "bbox": {
                                    "center": {
                                        "x": PERSON_BBOX_CENTER[0],
                                        "y": PERSON_BBOX_CENTER[1],
                                    },
                                    "size_x": PERSON_BBOX_SIZE[0],
                                    "size_y": PERSON_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2DetectionArrayTask(
            validators=[self.get_interface_publish_detection_ord_val_person],
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
            bbox_centers=[PERSON_BBOX_CENTER],
            bbox_sizes=[PERSON_BBOX_SIZE],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_detection_array_wrong_bbox(self, task_args: TaskArgs) -> None:
        """Test detection array with wrong bounding box."""
        tool_calls: List[Dict[str, Any]] = [
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
                                "results": [{"hypothesis": {"class_id": PERSON_CLASS}}],
                                "bbox": {
                                    "center": {
                                        "x": 100.0,
                                        "y": 100.0,
                                    },  # Wrong bbox center
                                    "size_x": 25.0,  # Wrong bbox size
                                    "size_y": 25.0,  # Wrong bbox size
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2DetectionArrayTask(
            validators=[self.get_interface_publish_detection_ord_val_person],
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
            bbox_centers=[PERSON_BBOX_CENTER],
            bbox_sizes=[PERSON_BBOX_SIZE],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_detection_array_wrong_message_type(
        self, task_args: TaskArgs
    ) -> None:
        """Test detection array with wrong message type."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "wrong_message_type"},  # Wrong message type
            },
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
                                        "x": PERSON_BBOX_CENTER[0],
                                        "y": PERSON_BBOX_CENTER[1],
                                    },
                                    "size_x": PERSON_BBOX_SIZE[0],
                                    "size_y": PERSON_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2DetectionArrayTask(
            validators=[self.get_interface_publish_detection_ord_val_person],
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
            bbox_centers=[PERSON_BBOX_CENTER],
            bbox_sizes=[PERSON_BBOX_SIZE],
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_publish_detection_array_missing_interface_call(
        self, task_args: TaskArgs
    ) -> None:
        """Test detection array with missing interface call."""
        tool_calls: List[Dict[str, Any]] = [
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
                                        "x": PERSON_BBOX_CENTER[0],
                                        "y": PERSON_BBOX_CENTER[1],
                                    },
                                    "size_x": PERSON_BBOX_SIZE[0],
                                    "size_y": PERSON_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
                },
            },
        ]

        task = PublishROS2DetectionArrayTask(
            validators=[self.get_interface_publish_detection_ord_val_person],
            task_args=task_args,
            detection_classes=[PERSON_CLASS],
            bbox_centers=[PERSON_BBOX_CENTER],
            bbox_sizes=[PERSON_BBOX_SIZE],
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallROS2ManipulatorMoveToServiceTask:
    """Test CallROS2ManipulatorMoveToServiceTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_call_manipulator_ord_val,
    )

    def test_call_manipulator_service_valid(self, task_args: TaskArgs) -> None:
        """Test valid manipulator service call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/srv/ManipulatorMoveTo"},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/manipulator_move_to",
                    "service_type": "rai_interfaces/srv/ManipulatorMoveTo",
                    "service_args": {
                        "target_pose": {
                            "pose": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}}
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
        ]

        task = CallROS2ManipulatorMoveToServiceTask(
            validators=[self.get_interface_call_manipulator_ord_val],
            task_args=task_args,
            target_x=1.0,
            target_y=2.0,
            target_z=3.0,
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
                "args": {"msg_type": "rai_interfaces/srv/ManipulatorMoveTo"},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/manipulator_move_to",
                    "service_type": "rai_interfaces/srv/ManipulatorMoveTo",
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
            validators=[self.get_interface_call_manipulator_ord_val],
            task_args=task_args,
            target_x=1.0,
            target_y=2.0,
            target_z=3.0,
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
                    "service_name": "/manipulator_move_to",
                    "service_type": "rai_interfaces/srv/ManipulatorMoveTo",
                    "service_args": {
                        "target_pose": {
                            "pose": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}}
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
        ]

        from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
            get_interface_call_manipulator_ord_val,
        )

        task = CallROS2ManipulatorMoveToServiceTask(
            validators=[get_interface_call_manipulator_ord_val],
            task_args=task_args,
            target_x=1.0,
            target_y=2.0,
            target_z=3.0,
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
                    "service_name": "/manipulator_move_to",
                    "service_type": "rai_interfaces/srv/ManipulatorMoveTo",
                    "service_args": {
                        "target_pose": {
                            "pose": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}}
                        },
                        "initial_gripper_state": True,
                        "final_gripper_state": False,
                    },
                },
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "rai_interfaces/srv/ManipulatorMoveTo"},
            },
        ]

        from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
            get_interface_call_manipulator_ord_val,
        )

        task = CallROS2ManipulatorMoveToServiceTask(
            validators=[get_interface_call_manipulator_ord_val],
            task_args=task_args,
            target_x=1.0,
            target_y=2.0,
            target_z=3.0,
            initial_gripper_state=True,
            final_gripper_state=False,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallGroundedSAMSegmentTask:
    """Test CallGroundedSAMSegmentTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_call_grounded_sam_ord_val_book,
        get_interface_call_grounded_sam_ord_val_bottle,
    )

    def test_call_grounded_sam_bottle_valid(self, task_args: TaskArgs) -> None:
        """Test valid Grounded SAM service call with bottle detection."""
        tool_calls: List[Dict[str, Any]] = [
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

        task = CallGroundedSAMSegmentTask(
            validators=[self.get_interface_call_grounded_sam_ord_val_bottle],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_grounded_sam_book_valid(self, task_args: TaskArgs) -> None:
        """Test valid Grounded SAM service call with book detection."""
        tool_calls: List[Dict[str, Any]] = [
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
                                                "class_id": BOOK_CLASS,
                                                "score": BOOK_SCORE,
                                            },
                                            "pose": {
                                                "pose": {
                                                    "position": {
                                                        "x": BOOK_POSITION_3D[0],
                                                        "y": BOOK_POSITION_3D[1],
                                                        "z": BOOK_POSITION_3D[2],
                                                    }
                                                }
                                            },
                                        }
                                    ],
                                    "bbox": {
                                        "center": {
                                            "x": BOOK_BBOX_CENTER[0],
                                            "y": BOOK_BBOX_CENTER[1],
                                        },
                                        "size_x": BOOK_BBOX_SIZE[0],
                                        "size_y": BOOK_BBOX_SIZE[1],
                                    },
                                }
                            ]
                        },
                        "source_img": {
                            "width": HD_IMAGE_WIDTH,
                            "height": HD_IMAGE_HEIGHT,
                            "encoding": HD_IMAGE_ENCODING,
                        },
                    },
                },
            },
        ]

        task = CallGroundedSAMSegmentTask(
            validators=[self.get_interface_call_grounded_sam_ord_val_book],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_grounded_sam_wrong_class(self, task_args: TaskArgs) -> None:
        """Test Grounded SAM service call with wrong detection class."""
        tool_calls: List[Dict[str, Any]] = [
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
                                                "class_id": BOOK_CLASS,  # Wrong class
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

        task = CallGroundedSAMSegmentTask(
            validators=[self.get_interface_call_grounded_sam_ord_val_bottle],
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
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounded_sam_wrong_tool_order(self, task_args: TaskArgs) -> None:
        """Test Grounded SAM service call with wrong tool order."""
        tool_calls: List[Dict[str, Any]] = [
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
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GROUNDED_SAM_SERVICE_TYPE},
            },
        ]

        task = CallGroundedSAMSegmentTask(
            validators=[self.get_interface_call_grounded_sam_ord_val_bottle],
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
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallGetLogDigestTask:
    """Test CallGetLogDigestTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_call_log_digest_ord_val,
    )

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
            validators=[self.get_interface_call_log_digest_ord_val],
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
            validators=[self.get_interface_call_log_digest_ord_val],
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
            validators=[self.get_interface_call_log_digest_ord_val],
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
            validators=[self.get_interface_call_log_digest_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallVectorStoreRetrievalTask:
    """Test CallVectorStoreRetrievalTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_call_vector_store_ord_val,
    )

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
                        "query": ROBOT_PURPOSE_QUERY,
                    },
                },
            },
        ]

        task = CallVectorStoreRetrievalTask(
            validators=[self.get_interface_call_vector_store_ord_val],
            task_args=task_args,
            query=ROBOT_PURPOSE_QUERY,
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
            validators=[self.get_interface_call_vector_store_ord_val],
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
            validators=[self.get_interface_call_vector_store_ord_val],
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
            validators=[self.get_interface_call_vector_store_ord_val],
            task_args=task_args,
            query=ROBOT_PURPOSE_QUERY,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallWhatISeeTask:
    """Test CallWhatISeeTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_call_what_i_see_ord_val,
    )

    def test_call_what_i_see_valid(self, task_args: TaskArgs) -> None:
        """Test valid WhatISee service call."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": WHAT_I_SEE_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": WHAT_I_SEE_SERVICE,
                    "service_type": WHAT_I_SEE_SERVICE_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CallWhatISeeTask(
            validators=[self.get_interface_call_what_i_see_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_what_i_see_wrong_service_name(self, task_args: TaskArgs) -> None:
        """Test WhatISee with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": WHAT_I_SEE_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_what_i_see_service",  # Wrong service name
                    "service_type": WHAT_I_SEE_SERVICE_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CallWhatISeeTask(
            validators=[self.get_interface_call_what_i_see_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_what_i_see_wrong_message_type(self, task_args: TaskArgs) -> None:
        """Test WhatISee with wrong message type."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "wrong_message_type"},  # Wrong message type
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": WHAT_I_SEE_SERVICE,
                    "service_type": WHAT_I_SEE_SERVICE_TYPE,
                    "service_args": {},
                },
            },
        ]

        task = CallWhatISeeTask(
            validators=[self.get_interface_call_what_i_see_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_what_i_see_wrong_tool_order(self, task_args: TaskArgs) -> None:
        """Test WhatISee with wrong tool call order."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": WHAT_I_SEE_SERVICE,
                    "service_type": WHAT_I_SEE_SERVICE_TYPE,
                    "service_args": {},
                },
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": WHAT_I_SEE_SERVICE_TYPE},
            },
        ]

        task = CallWhatISeeTask(
            validators=[self.get_interface_call_what_i_see_ord_val],
            task_args=task_args,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCallGroundingDinoClassify:
    """Test CallGroundingDinoClassify validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        get_interface_call_grounding_dino_ord_val,
    )

    def test_call_grounding_dino_valid(self, task_args: TaskArgs) -> None:
        """Test valid Grounding DINO service call."""
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
                        "box_threshold": BOX_THRESHOLD_1,
                        "text_threshold": TEXT_THRESHOLD_1,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            validators=[self.get_interface_call_grounding_dino_ord_val],
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD_1,
        )
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_call_grounding_dino_wrong_param_value(self, task_args: TaskArgs) -> None:
        """Test Grounding DINO with wrong text threshold."""
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
                        "box_threshold": BOX_THRESHOLD_1,
                        "text_threshold": 0.5,  # Wrong text threshold
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            validators=[self.get_interface_call_grounding_dino_ord_val],
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD_1,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounding_dino_wrong_service_name(self, task_args: TaskArgs) -> None:
        """Test Grounding DINO with wrong service name."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GROUNDING_DINO_SERVICE_TYPE},
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": "/wrong_grounding_dino_service",  # Wrong service name
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": GROUNDING_DINO_CLASSES,
                        "box_threshold": BOX_THRESHOLD_1,
                        "text_threshold": TEXT_THRESHOLD_1,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            validators=[self.get_interface_call_grounding_dino_ord_val],
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD_1,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounding_dino_wrong_message_type(self, task_args: TaskArgs) -> None:
        """Test Grounding DINO with wrong message type."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": "wrong_message_type"},  # Wrong message type
            },
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SERVICE,
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": GROUNDING_DINO_CLASSES,
                        "box_threshold": BOX_THRESHOLD_1,
                        "text_threshold": TEXT_THRESHOLD_1,
                    },
                },
            },
        ]

        task = CallGroundingDinoClassify(
            validators=[self.get_interface_call_grounding_dino_ord_val],
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD_1,
        )
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_call_grounding_dino_wrong_tool_order(self, task_args: TaskArgs) -> None:
        """Test Grounding DINO with wrong tool call order."""
        tool_calls: List[Dict[str, Any]] = [
            {
                "name": "call_ros2_service",
                "args": {
                    "service_name": GROUNDING_DINO_SERVICE,
                    "service_type": GROUNDING_DINO_SERVICE_TYPE,
                    "service_args": {
                        "classes": GROUNDING_DINO_CLASSES,
                        "box_threshold": BOX_THRESHOLD_1,
                        "text_threshold": TEXT_THRESHOLD_1,
                    },
                },
            },
            {
                "name": "get_ros2_message_interface",
                "args": {"msg_type": GROUNDING_DINO_SERVICE_TYPE},
            },
        ]

        task = CallGroundingDinoClassify(
            validators=[self.get_interface_call_grounding_dino_ord_val],
            task_args=task_args,
            classes=GROUNDING_DINO_CLASSES,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD_1,
        )
        score = task.validate(tool_calls)
        assert score == 0.0


class TestCompleteObjectInteractionTask:
    """Test CompleteObjectInteractionTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        complete_object_interaction_bottle_validator,
        complete_object_interaction_cup_validator,
    )

    BOTTLE_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
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
                    "classes": BOTTLE_CLASS,
                    "box_threshold": BOTTLE_BOX_THRESHOLD,
                    "text_threshold": BOTTLE_TEXT_THRESHOLD,
                },
            },
        },
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
                                            "score": BOTTLE_ENHANCED_SCORE,
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
                                "x": BOTTLE_POSITION_3D[0],
                                "y": BOTTLE_POSITION_3D[1],
                                "z": BOTTLE_POSITION_3D[2],
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
            "args": {"msg_type": HRI_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": BOTTLE_INTERACTION_MESSAGE},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]
    CUP_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
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
                    "classes": CUP_CLASS,
                    "box_threshold": CUP_BOX_THRESHOLD,
                    "text_threshold": CUP_TEXT_THRESHOLD,
                },
            },
        },
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
                                            "class_id": CUP_CLASS,
                                            "score": CUP_SCORE,
                                        },
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": CUP_POSITION_3D[0],
                                                    "y": CUP_POSITION_3D[1],
                                                    "z": CUP_POSITION_3D[2],
                                                }
                                            }
                                        },
                                    }
                                ],
                                "bbox": {
                                    "center": {
                                        "x": CUP_BBOX_CENTER[0],
                                        "y": CUP_BBOX_CENTER[1],
                                    },
                                    "size_x": CUP_BBOX_SIZE[0],
                                    "size_y": CUP_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "source_img": {
                        "width": HD_IMAGE_WIDTH,
                        "height": HD_IMAGE_HEIGHT,
                        "encoding": HD_IMAGE_ENCODING,
                    },
                },
            },
        },
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
                                "x": CUP_POSITION_3D[0],
                                "y": CUP_POSITION_3D[1],
                                "z": CUP_POSITION_3D[2],
                            }
                        }
                    },
                    "initial_gripper_state": False,
                    "final_gripper_state": True,
                },
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": HRI_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": CUP_INTERACTION_MESSAGE},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    def test_complete_object_interaction_bottle_valid(
        self, task_args: TaskArgs
    ) -> None:
        """Test valid complete object interaction with bottle."""
        tool_calls = copy.deepcopy(self.BOTTLE_TOOL_CALLS_TEMPLATE)

        task = CompleteObjectInteractionTask(
            validators=[self.complete_object_interaction_bottle_validator],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_complete_object_interaction_cup_valid(self, task_args: TaskArgs) -> None:
        """Test valid complete object interaction with cup."""
        tool_calls = copy.deepcopy(self.CUP_TOOL_CALLS_TEMPLATE)

        task = CompleteObjectInteractionTask(
            validators=[self.complete_object_interaction_cup_validator],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_complete_object_interaction_wrong_step_order(
        self, task_args: TaskArgs
    ) -> None:
        """Test complete object interaction with wrong step order."""
        tool_calls = copy.deepcopy(self.BOTTLE_TOOL_CALLS_TEMPLATE)
        # swap the tool calls
        tool_calls[1], tool_calls[3] = tool_calls[3], tool_calls[1]

        task = CompleteObjectInteractionTask(
            validators=[self.complete_object_interaction_bottle_validator],
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
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_complete_object_interaction_wrong_param(self, task_args: TaskArgs) -> None:
        """Test complete object interaction with wrong detection thresholds."""
        tool_calls = copy.deepcopy(self.BOTTLE_TOOL_CALLS_TEMPLATE)

        # Change thresholds to wrong values
        tool_calls[1]["args"]["service_args"]["box_threshold"] = 0.8  # Should be 0.35
        tool_calls[1]["args"]["service_args"]["text_threshold"] = 0.6  # Should be 0.2

        task = CompleteObjectInteractionTask(
            validators=[self.complete_object_interaction_bottle_validator],
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
        score = task.validate(tool_calls)
        assert score == 0.0


class TestMultiModalSceneDocumentationTask:
    """Test MultiModalSceneDocumentationTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        multimodal_scene_documentation_kitchen_validator,
        multimodal_scene_documentation_office_validator,
    )

    OFFICE_SCENE_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": WHAT_I_SEE_SERVICE_TYPE},
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": WHAT_I_SEE_SERVICE,
                "service_type": WHAT_I_SEE_SERVICE_TYPE,
                "service_args": {},
            },
        },
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
                            "results": [{"hypothesis": {"class_id": PERSON_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": OFFICE_PERSON_BBOX_CENTER[0],
                                    "y": OFFICE_PERSON_BBOX_CENTER[1],
                                },
                                "size_x": OFFICE_PERSON_BBOX_SIZE[0],
                                "size_y": OFFICE_PERSON_BBOX_SIZE[1],
                            },
                        },
                        {
                            "results": [{"hypothesis": {"class_id": LAPTOP_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": OFFICE_LAPTOP_BBOX_CENTER[0],
                                    "y": OFFICE_LAPTOP_BBOX_CENTER[1],
                                },
                                "size_x": OFFICE_LAPTOP_BBOX_SIZE[0],
                                "size_y": OFFICE_LAPTOP_BBOX_SIZE[1],
                            },
                        },
                        {
                            "results": [{"hypothesis": {"class_id": COFFEE_CUP_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": OFFICE_COFFEE_BBOX_CENTER[0],
                                    "y": OFFICE_COFFEE_BBOX_CENTER[1],
                                },
                                "size_x": OFFICE_COFFEE_BBOX_SIZE[0],
                                "size_y": OFFICE_COFFEE_BBOX_SIZE[1],
                            },
                        },
                    ]
                },
                "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
            },
        },
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
                    "query": SAFETY_PROTOCOLS_QUERY,
                },
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": HRI_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": OFFICE_DOCUMENTATION_REPORT},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    KITCHEN_SCENE_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": WHAT_I_SEE_SERVICE_TYPE},
        },
        {
            "name": "call_ros2_service",
            "args": {
                "service_name": WHAT_I_SEE_SERVICE,
                "service_type": WHAT_I_SEE_SERVICE_TYPE,
                "service_args": {},
            },
        },
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
                            "results": [{"hypothesis": {"class_id": PERSON_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": KITCHEN_PERSON_BBOX_CENTER[0],
                                    "y": KITCHEN_PERSON_BBOX_CENTER[1],
                                },
                                "size_x": KITCHEN_PERSON_BBOX_SIZE[0],
                                "size_y": KITCHEN_PERSON_BBOX_SIZE[1],
                            },
                        },
                        {
                            "results": [{"hypothesis": {"class_id": MICROWAVE_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": KITCHEN_MICROWAVE_BBOX_CENTER[0],
                                    "y": KITCHEN_MICROWAVE_BBOX_CENTER[1],
                                },
                                "size_x": KITCHEN_MICROWAVE_BBOX_SIZE[0],
                                "size_y": KITCHEN_MICROWAVE_BBOX_SIZE[1],
                            },
                        },
                        {
                            "results": [{"hypothesis": {"class_id": KNIFE_CLASS}}],
                            "bbox": {
                                "center": {
                                    "x": KITCHEN_KNIFE_BBOX_CENTER[0],
                                    "y": KITCHEN_KNIFE_BBOX_CENTER[1],
                                },
                                "size_x": KITCHEN_KNIFE_BBOX_SIZE[0],
                                "size_y": KITCHEN_KNIFE_BBOX_SIZE[1],
                            },
                        },
                    ]
                },
                "message_type": DETECTION_ARRAY_MESSAGE_TYPE,
            },
        },
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
                    "query": KITCHEN_SAFETY_QUERY,
                },
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": HRI_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": KITCHEN_DOCUMENTATION_REPORT},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    def test_multimodal_scene_documentation_office_valid(
        self, task_args: TaskArgs
    ) -> None:
        """Test valid multimodal scene documentation for office."""
        tool_calls = copy.deepcopy(self.OFFICE_SCENE_TOOL_CALLS_TEMPLATE)

        task = MultiModalSceneDocumentationTask(
            validators=[self.multimodal_scene_documentation_office_validator],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_multimodal_scene_documentation_kitchen_valid(
        self, task_args: TaskArgs
    ) -> None:
        """Test valid multimodal scene documentation for kitchen."""
        tool_calls = copy.deepcopy(self.KITCHEN_SCENE_TOOL_CALLS_TEMPLATE)

        task = MultiModalSceneDocumentationTask(
            validators=[self.multimodal_scene_documentation_kitchen_validator],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_multimodal_scene_documentation_wrong_step_order(
        self, task_args: TaskArgs
    ) -> None:
        """Test multimodal scene documentation with wrong step order."""
        tool_calls = copy.deepcopy(self.OFFICE_SCENE_TOOL_CALLS_TEMPLATE)

        # Swap WhatISee and Detection Array calls
        tool_calls[1], tool_calls[3] = tool_calls[3], tool_calls[1]

        task = MultiModalSceneDocumentationTask(
            validators=[self.multimodal_scene_documentation_office_validator],
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
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_multimodal_scene_documentation_wrong_param(
        self, task_args: TaskArgs
    ) -> None:
        """Test multimodal scene documentation with wrong parameters."""
        tool_calls = copy.deepcopy(self.OFFICE_SCENE_TOOL_CALLS_TEMPLATE)

        # Change safety query to wrong query
        tool_calls[5]["args"]["service_args"]["query"] = "What is the weather like?"

        task = MultiModalSceneDocumentationTask(
            validators=[self.multimodal_scene_documentation_office_validator],
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
        score = task.validate(tool_calls)
        assert score == 0.0


class TestEmergencyResponseProtocolTask:
    """Test EmergencyResponseProtocolTask validation."""

    from rai_bench.tool_calling_agent.predefined.custom_interfaces_tasks import (
        emergency_response_intruder_validator,
        emergency_response_protocol_validator,
    )

    EMERGENCY_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
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
                    "classes": PERSON_CLASS,
                    "box_threshold": EMERGENCY_BOX_THRESHOLD,
                    "text_threshold": EMERGENCY_TEXT_THRESHOLD,
                },
            },
        },
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
                                            "class_id": PERSON_CLASS,
                                            "score": EMERGENCY_SCORE,
                                        },
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": EMERGENCY_POSITION_3D[0],
                                                    "y": EMERGENCY_POSITION_3D[1],
                                                    "z": EMERGENCY_POSITION_3D[2],
                                                }
                                            }
                                        },
                                    }
                                ],
                                "bbox": {
                                    "center": {
                                        "x": EMERGENCY_BBOX_CENTER[0],
                                        "y": EMERGENCY_BBOX_CENTER[1],
                                    },
                                    "size_x": EMERGENCY_BBOX_SIZE[0],
                                    "size_y": EMERGENCY_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "source_img": {
                        "width": HD_IMAGE_WIDTH,
                        "height": HD_IMAGE_HEIGHT,
                        "encoding": STANDARD_IMAGE_ENCODING,
                    },
                },
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": AUDIO_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": AUDIO_TOPIC,
                "message": {
                    "samples": EMERGENCY_AUDIO_SAMPLES,
                    "sample_rate": EMERGENCY_SAMPLE_RATE,
                    "channels": EMERGENCY_CHANNELS,
                },
                "message_type": AUDIO_MESSAGE_TYPE,
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": HRI_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": EMERGENCY_MESSAGE},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    INTRUDER_TOOL_CALLS_TEMPLATE: List[Dict[str, Any]] = [
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
                    "classes": PERSON_CLASS,
                    "box_threshold": INTRUDER_BOX_THRESHOLD,
                    "text_threshold": INTRUDER_TEXT_THRESHOLD,
                },
            },
        },
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
                                            "class_id": PERSON_CLASS,
                                            "score": INTRUDER_SCORE,
                                        },
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": INTRUDER_POSITION_3D[0],
                                                    "y": INTRUDER_POSITION_3D[1],
                                                    "z": INTRUDER_POSITION_3D[2],
                                                }
                                            }
                                        },
                                    }
                                ],
                                "bbox": {
                                    "center": {
                                        "x": INTRUDER_BBOX_CENTER[0],
                                        "y": INTRUDER_BBOX_CENTER[1],
                                    },
                                    "size_x": INTRUDER_BBOX_SIZE[0],
                                    "size_y": INTRUDER_BBOX_SIZE[1],
                                },
                            }
                        ]
                    },
                    "source_img": {
                        "width": FULL_HD_IMAGE_WIDTH,
                        "height": FULL_HD_IMAGE_HEIGHT,
                        "encoding": HD_IMAGE_ENCODING,
                    },
                },
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": AUDIO_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": AUDIO_TOPIC,
                "message": {
                    "samples": INTRUDER_AUDIO_SAMPLES,
                    "sample_rate": INTRUDER_SAMPLE_RATE,
                    "channels": INTRUDER_CHANNELS,
                },
                "message_type": AUDIO_MESSAGE_TYPE,
            },
        },
        {
            "name": "get_ros2_message_interface",
            "args": {"msg_type": HRI_MESSAGE_TYPE},
        },
        {
            "name": "publish_ros2_message",
            "args": {
                "topic": HRI_TOPIC,
                "message": {"text": INTRUDER_MESSAGE},
                "message_type": HRI_MESSAGE_TYPE,
            },
        },
    ]

    def test_emergency_response_protocol_valid(self, task_args: TaskArgs) -> None:
        """Test valid emergency response protocol."""
        tool_calls = copy.deepcopy(self.EMERGENCY_TOOL_CALLS_TEMPLATE)

        task = EmergencyResponseProtocolTask(
            validators=[self.emergency_response_protocol_validator],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_emergency_response_intruder_valid(self, task_args: TaskArgs) -> None:
        """Test valid emergency response for intruder detection."""
        tool_calls = copy.deepcopy(self.INTRUDER_TOOL_CALLS_TEMPLATE)

        task = EmergencyResponseProtocolTask(
            validators=[self.emergency_response_intruder_validator],
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
        score = task.validate(tool_calls)
        assert score == 1.0

    def test_emergency_response_wrong_step_order(self, task_args: TaskArgs) -> None:
        """Test emergency response with wrong step order."""
        tool_calls = copy.deepcopy(self.EMERGENCY_TOOL_CALLS_TEMPLATE)

        # Swap Grounding DINO and Grounded SAM calls
        tool_calls[1], tool_calls[3] = tool_calls[3], tool_calls[1]

        task = EmergencyResponseProtocolTask(
            validators=[self.emergency_response_protocol_validator],
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
        score = task.validate(tool_calls)
        assert score == 0.0

    def test_emergency_response_wrong_param(self, task_args: TaskArgs) -> None:
        """Test emergency response with wrong parameters."""
        tool_calls = copy.deepcopy(self.EMERGENCY_TOOL_CALLS_TEMPLATE)

        # Change thresholds to wrong values
        tool_calls[1]["args"]["service_args"]["box_threshold"] = 0.5  # Should be 0.9
        tool_calls[1]["args"]["service_args"]["text_threshold"] = 0.3  # Should be 0.8

        task = EmergencyResponseProtocolTask(
            validators=[self.emergency_response_protocol_validator],
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
        score = task.validate(tool_calls)
        assert score == 0.0
