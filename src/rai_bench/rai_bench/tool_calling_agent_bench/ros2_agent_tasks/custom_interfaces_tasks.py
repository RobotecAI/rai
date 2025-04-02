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
from typing import Any, Dict, List

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    CustomInterfacesActionTask,
    CustomInterfacesServiceTask,
    CustomInterfacesTopicTask,
)

loggers_type = logging.Logger


PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT = """You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions.
                """

# only custom interfaces will be tested, so there no need for defualts for all of interfaces
DEFAULT_MESSAGES: Dict[str, Dict[str, Any]] = {
    "rai_interfaces/msg/HRIMessage": {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
        "text": "",
        "images": [
            {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "height": 0,
                "width": 0,
                "encoding": "",
                "is_bigendian": 0,
                "step": 0,
                "data": [],
            }
        ],
        "audios": [{"audio": [], "sample_rate": 0, "channels": 0}],
    },
    "rai_interfaces/msg/AudioMessage": {
        "audio": [],
        "sample_rate": 0,
        "channels": 0,
    },
    "rai_interfaces/msg/RAIDetectionArray": {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
        "detections": [
            {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "results": [
                    {
                        "hypothesis": {"class_id": "", "score": 0.0},
                        "pose": {
                            "pose": {
                                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
                            },
                            "covariance": {},
                        },
                    }
                ],
                "bbox": {
                    "center": {"position": {"x": 0.0, "y": 0.0}, "theta": 0.0},
                    "size_x": 0.0,
                    "size_y": 0.0,
                },
                "id": "",
            }
        ],
        "detection_classes": [],
    },
    "rai_interfaces/srv/ManipulatorMoveTo": {
        "request": {
            "initial_gripper_state": False,
            "final_gripper_state": False,
            "target_pose": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
                },
            },
        },
        "response": {"success": False},
    },
    "rai_interfaces/srv/RAIGroundedSam": {
        "request": {
            "detections": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "detections": [
                    {
                        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                        "results": [
                            {
                                "hypothesis": {"class_id": "", "score": 0.0},
                                "pose": {
                                    "pose": {
                                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "orientation": {
                                            "x": 0.0,
                                            "y": 0.0,
                                            "z": 0.0,
                                            "w": 0.0,
                                        },
                                    },
                                    "covariance": {},
                                },
                            }
                        ],
                        "bbox": {
                            "center": {"position": {"x": 0.0, "y": 0.0}, "theta": 0.0},
                            "size_x": 0.0,
                            "size_y": 0.0,
                        },
                        "id": "",
                    }
                ],
                "detection_classes": [],
            },
            "source_img": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "height": 0,
                "width": 0,
                "encoding": "",
                "is_bigendian": 0,
                "step": 0,
                "data": [],
            },
        },
        "response": {
            "masks": [
                {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "height": 0,
                    "width": 0,
                    "encoding": "",
                    "is_bigendian": 0,
                    "step": 0,
                    "data": [],
                }
            ]
        },
    },
    "rai_interfaces/srv/RAIGroundingDino": {
        "request": {
            "classes": "",
            "box_threshold": 0.0,
            "text_threshold": 0.0,
            "source_img": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "height": 0,
                "width": 0,
                "encoding": "",
                "is_bigendian": 0,
                "step": 0,
                "data": [],
            },
        },
        "response": {
            "detections": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "detections": [
                    {
                        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                        "results": [
                            {
                                "hypothesis": {"class_id": "", "score": 0.0},
                                "pose": {
                                    "pose": {
                                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "orientation": {
                                            "x": 0.0,
                                            "y": 0.0,
                                            "z": 0.0,
                                            "w": 0.0,
                                        },
                                    },
                                    "covariance": {},
                                },
                            }
                        ],
                        "bbox": {
                            "center": {"position": {"x": 0.0, "y": 0.0}, "theta": 0.0},
                            "size_x": 0.0,
                            "size_y": 0.0,
                        },
                        "id": "",
                    }
                ],
                "detection_classes": [],
            }
        },
    },
    "rai_interfaces/srv/StringList": {
        "request": {},
        "response": {"success": False, "string_list": []},
    },
    "rai_interfaces/srv/VectorStoreRetrieval": {
        "request": {"query": ""},
        "response": {"success": False, "message": "", "documents": [], "scores": []},
    },
    "rai_interfaces/srv/WhatISee": {
        "request": {},
        "response": {
            "observations": [],
            "perception_source": "",
            "image": {
                "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                "height": 0,
                "width": 0,
                "encoding": "",
                "is_bigendian": 0,
                "step": 0,
                "data": [],
            },
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
            },
        },
    },
    "rai_interfaces/action/Task": {
        "goal": {"task": "", "description": "", "priority": ""},
        "result": {"success": False, "report": ""},
        "feedback": {"current_status": ""},
    },
}


class PublishROS2HRIMessageTask(CustomInterfacesTopicTask):
    complexity = "easy"
    expected_text = "Hello!"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_topic(self) -> str:
        return "/to_human"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_message_type].copy()
        expected["text"] = self.expected_text
        return expected

    def get_prompt(self) -> str:
        return (
            f"You need to publish a message to the topic '{self.expected_topic}' with the text value: '{self.expected_text}'.\n\n"
            "Before publishing, follow these steps:\n"
            "1. Use the tool to retrieve the available ROS2 topics and their message types.\n"
            f"2. Find the message type for the topic '{self.expected_topic}'.\n"
            "3. Use the message type to get the full message interface definition.\n"
            f"4. Publish the message to '{self.expected_topic}' using the correct message type and interface.\n\n"
            "Make sure all required fields are correctly filled according to the interface."
        )


class PublishROS2AudioMessageTask(CustomInterfacesTopicTask):
    complexity = "easy"
    expected_audio = [123, 456, 789]
    expected_sample_rate = 44100
    expected_channels = 2

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_topic(self) -> str:
        return "/send_audio"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_message_type].copy()
        expected["audio"] = self.expected_audio
        expected["sample_rate"] = self.expected_sample_rate
        expected["channels"] = self.expected_channels
        return expected

    def get_prompt(self) -> str:
        return (
            f"Publish message to the {self.expected_topic} topic with audio samples [123, 456, 789], "
            "sample rate 44100, and 2 channels. Before publishing, check the message type "
            "of this topic and its interface."
        )


class PublishROS2DetectionArrayTask(CustomInterfacesTopicTask):
    complexity = "easy"

    expected_detection_classes: List[str] = ["person", "car"]
    expected_detections: List[Any] = [
        {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
            "results": [],
            "bbox": {
                "center": {"x": 320.0, "y": 240.0},
                "size": {"x": 50.0, "y": 50.0},
            },
        }
    ]
    expected_header: Dict[str, Any] = {
        "stamp": {"sec": 0, "nanosec": 0},
        "frame_id": "camera",
    }

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_topic(self) -> str:
        return "/send_detections"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_message_type].copy()
        expected["detections"] = {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
            "results": [],
            "bbox": {
                "center": {"x": 320.0, "y": 240.0},
                "size": {"x": 50.0, "y": 50.0},
            },
        }
        expected["detection_classes"] = ["person", "car"]
        return expected

    def get_prompt(self) -> str:
        return (
            "Publish a detection message to the /send_detections topic. The message should have a unchanged header,"
            f"one detection: {self.expected_detections}, and detection classes "
            f"{self.expected_detection_classes}. Before publishing, check the message type of this topic "
            "and its interface."
        )


# TODO (jm) apply parent classes and expected messages to service and action tasks
class CallROS2ManipulatorMoveToServiceTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_initial_gripper_state = True
    expected_final_gripper_state = False
    expected_target_pose: Dict[str, Dict[str, Any]] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "world"},
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/manipulator_move_to"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_service_type].copy()
        expected["detections"] = {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
            "results": [],
            "bbox": {
                "center": {"x": 320.0, "y": 240.0},
                "size": {"x": 50.0, "y": 50.0},
            },
        }
        expected["detection_classes"] = ["person", "car"]
        return expected

    def get_prompt(self) -> str:
        return (
            f"Call service {self.expected_service} with a target_pose: {self.expected_target_pose}. "
            "Before calling the service, check the service type and its interface."
        )


class CallGroundedSAMSegmentTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_detections: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "detections": [],
    }
    expected_source_img: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/grounded_sam_segment"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_service_type].copy()
        expected["detections"] = self.expected_detections
        expected["source_img"] = self.expected_source_img
        return expected

    def get_prompt(self) -> str:
        return (
            "Call the service /grounded_sam_segment using the RAIGroundedSam interface.\n"
            "Steps to follow:\n"
            "1. Look up the available ROS2 services and their types.\n"
            "2. Retrieve the message interface for the /grounded_sam_segment service.\n"
            "3. Use the interface to construct the request message with:\n"
            "   - detections from 'camera_frame'\n"
            "   - an RGB image of size 640x480\n"
            "4. Call the service with the populated message."
        )


class CallGroundingDinoClassifyTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_classes = "bottle, book, chair"
    expected_box_threshold = 0.4
    expected_text_threshold = 0.25
    expected_source_img: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/grounding_dino_classify"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_service_type].copy()
        expected["classes"] = self.expected_classes
        expected["box_threshold"] = self.expected_box_threshold
        expected["text_threshold"] = self.expected_text_threshold
        expected["source_img"] = self.expected_source_img
        return expected

    def get_prompt(self) -> str:
        return (
            "Call the service /grounding_dino_classify using the RAIGroundingDino interface.\n"
            "Follow these steps:\n"
            "1. Look up all available ROS2 services and their types.\n"
            "2. Retrieve the interface of /grounding_dino_classify.\n"
            "3. Create a message with:\n"
            f"   - classes: '{self.expected_classes}'\n"
            f"   - box_threshold: {self.expected_box_threshold}\n"
            f"   - text_threshold: {self.expected_text_threshold}\n"
            "   - source image from 'camera_frame' (640x480, RGB)\n"
            "4. Call the service with the populated request."
        )


class CallGetLogDigestTask(CustomInterfacesServiceTask):
    complexity = "easy"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/get_log_digest"

    @property
    def expected_message(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self) -> str:
        return (
            "Call the service /get_log_digest using the StringList interface.\n"
            "Steps:\n"
            "1. Look up available ROS2 services and their message types.\n"
            "2. Retrieve the interface for /get_log_digest.\n"
            "3. Use an empty request (no fields needed).\n"
            "4. Call the service and retrieve the response."
        )


class CallVectorStoreRetrievalTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_query = "What is the purpose of this robot?"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "/rai_whoami_documentation_service"

    @property
    def expected_message(self) -> Dict[str, Any]:
        return {"query": self.expected_query}

    def get_prompt(self) -> str:
        return (
            f"Call the service {self.expected_service} with the query: '{self.expected_query}'.\n"
            "Before calling:\n"
            "1. Retrieve the list of ROS2 services and their types.\n"
            "2. Fetch the message interface for this service.\n"
            "3. Create a request message using the query string.\n"
            "4. Call the service using the interface."
        )


class CallWhatISeeTask(CustomInterfacesServiceTask):
    complexity = "easy"

    expected_observations = ["table", "cup", "notebook"]
    expected_perception_source = "front_camera"
    expected_image: Dict[str, Any] = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera_frame"},
        "height": 480,
        "width": 640,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": 1920,
        "data": [],
    }
    expected_pose = {
        "position": {"x": 1.0, "y": 2.0, "z": 0.5},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_service(self) -> str:
        return "rai/whatisee/get"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_service_type].copy()
        expected["observations"] = self.expected_observations
        expected["perception_source"] = self.expected_perception_source
        expected["image"] = self.expected_image
        expected["pose"] = self.expected_pose
        return expected

    def get_prompt(self) -> str:
        return (
            f"Call the service {self.expected_service} using the WhatISee interface.\n"
            "Steps:\n"
            "1. Get available services and their types.\n"
            "2. Retrieve the message interface for the WhatISee service.\n"
            "3. Create the request using:\n"
            f"   - Observations: {self.expected_observations}\n"
            f"   - Source: '{self.expected_perception_source}'\n"
            "   - Image: 640x480 RGB from 'camera_frame'\n"
            f"   - Pose: {self.expected_pose}\n"
            "4. Call the service with this structured message."
        )


class CallROS2CustomActionTask(CustomInterfacesActionTask):
    complexity = "easy"

    expected_task = "Where are you?"
    expected_description = ""
    expected_priority = "10"

    def get_system_prompt(self) -> str:
        return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT

    @property
    def expected_action(self) -> str:
        return "/perform_task"

    @property
    def expected_message(self) -> Dict[str, Any]:
        expected = DEFAULT_MESSAGES[self.expected_action_type].copy()
        expected["goal"]["task"] = self.expected_task
        expected["goal"]["description"] = self.expected_description
        expected["goal"]["priority "] = self.expected_priority
        return expected

    def get_prompt(self) -> str:
        return (
            "Call action /perform_task with the provided goal values: "
            "{priority: 10, description: '', task: 'Where are you?'}"
        )
