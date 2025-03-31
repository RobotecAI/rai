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

import uuid
from threading import Lock
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.communication.ros2.messages import ROS2ARIMessage
from rai.messages import MultimodalArtifact, preprocess_image
from rai.tools.ros.manipulation import (
    GetGrabbingPointTool,
    GetObjectPositionsTool,
    MoveToPointTool,
)
from rai.tools.ros2 import (
    CallROS2ServiceTool,
    CancelROS2ActionTool,
    GetROS2ActionFeedbackTool,
    GetROS2ActionIDsTool,
    GetROS2ActionResultTool,
    GetROS2ImageTool,
    GetROS2MessageInterfaceTool,
    GetROS2TopicsNamesAndTypesTool,
    PublishROS2MessageTool,
    ReceiveROS2MessageTool,
    StartROS2ActionTool,
)


class MockGetROS2TopicsNamesAndTypesTool(GetROS2TopicsNamesAndTypesTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    mock_topics_names_and_types: list[str]

    def _run(self) -> str:
        """Mocked method that returns the mock topics and types instead of fetching from ROS2.

        Returns
        -------
        str
            Mocked output of 'get_ros2_topics_names_and_types' tool.
        """
        return "\n".join(self.mock_topics_names_and_types)


class MockGetROS2ImageTool(GetROS2ImageTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    available_topics: List[str]

    def _run(
        self, topic: str, timeout_sec: float = 1.0
    ) -> Tuple[str, MultimodalArtifact]:
        """Method that returns a mock black image if the passed topic is correct.

        Parameters
        ----------
        topic : str
            Topic with the image
        timeout_sec : float, optional
            Timeout in seconds, by default 1.0

        Returns
        -------
        Tuple[str, MultimodalArtifact]
            Message from the tool and the image.

        Raises
        ------
        ValueError
            If the passed topic is not correct.
        """
        if topic not in self.available_topics:
            raise ValueError(
                f"Topic {topic} is not available within {timeout_sec} seconds. Check if the topic exists."
            )
        image = self.generate_mock_image()
        return "Image received successfully", MultimodalArtifact(
            images=[preprocess_image(image)]
        )  # type: ignore

    @staticmethod
    def generate_mock_image() -> npt.NDArray[np.uint8]:
        """Generates a blank black image (480x640, RGB).

        Returns
        -------
        npt.NDArray[np.uint8]
            Blank black image
        """
        height, width = 480, 640
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)
        return blank_image


class MockReceiveROS2MessageTool(ReceiveROS2MessageTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    available_topics: List[str]

    def _run(self, topic: str) -> str:
        """Method that returns a mock message if the passed topic is correct.

        Parameters
        ----------
        topic : str
            Topic to receive the message from

        Returns
        -------
        str
            Message from the tool

        Raises
        ------
        ValueError
            If the passed topic is not correct.
        """
        if topic not in self.available_topics:
            raise ValueError(
                f"Topic {topic} is not available within 1.0 seconds. Check if the topic exists."
            )
        message: ROS2ARIMessage = MagicMock(spec=ROS2ARIMessage)
        message.payload = {"mock": "payload"}
        message.metadata = {"mock": "metadata"}
        return str({"payload": message.payload, "metadata": message.metadata})


class MockMoveToPointTool(MoveToPointTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)

    def _run(self, x: float, y: float, z: float, task: str) -> str:
        """Method that return a mock message with the end effector position.

        Parameters
        ----------
        x : float
        y : float
        z : float
        task : str
            Task to perform.

        Returns
        -------
        str
            Message from the tool
        """
        return f"End effector successfully positioned at coordinates ({x:.2f}, {y:.2f}, {z:.2f}). Note: The status of object interaction (grab/drop) is not confirmed by this movement."


class MockGetObjectPositionsTool(GetObjectPositionsTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)

    # Create mock instances for the arguments
    target_frame: str = MagicMock(spec=str)
    source_frame: str = MagicMock(spec=str)
    camera_topic: str = MagicMock(spec=str)
    depth_topic: str = MagicMock(spec=str)
    camera_info_topic: str = MagicMock(spec=str)
    get_grabbing_point_tool: GetGrabbingPointTool = MagicMock(spec=GetGrabbingPointTool)
    mock_objects: dict[str, List[dict[str, float]]]

    def _run(self, object_name: str) -> str:
        """Method that returns a mock message with the object positions if the object_name is present in the mock_objects dictionary.

        Parameters
        ----------
        object_name : str
            Name of the object to get the positions of

        Returns
        -------
        str
            Message from the tool
        """
        expected_positions = self.mock_objects.get(object_name, [])
        print(f"Expected positions: {expected_positions}")
        if len([expected_positions]) == 0:
            return f"No {object_name}s detected."
        else:
            return f"Centroids of detected {object_name}s in manipulator frame: {expected_positions} Sizes of the detected objects are unknown."


class MockPublishROS2MessageTool(PublishROS2MessageTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    available_topics: List[str]
    available_message_types: List[str]

    def _run(self, topic: str, message: Dict[str, Any], message_type: str) -> str:
        """
        Mocked method that simulates publihing to a topic and return a status string.

        Parameters
        ----------
        topic : str
            The name of the topic to which the message is published.
        message : Dict[str, Any]
            The content of the message as a dictionary.
        message_type : str
            The type of the message being published.

        """
        if topic not in self.available_topics:
            raise ValueError(
                f"Topic {topic} is not available within 1.0 seconds. Check if the topic exists."
            )
        if message_type not in self.available_message_types:
            raise TypeError(
                "Expected message one of message types: {}, got {}".format(
                    self.available_message_types, message_type
                )
            )
        return "Message published successfully"


class MockGetROS2MessageInterfaceTool(GetROS2MessageInterfaceTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    mock_interfaces: Dict[str, str]

    def _run(self, msg_type: str) -> str:
        """
        Mocked method that returns the interface definition for a given ROS2 message type.

        Parameters
        ----------
        msg_type : str
            The ROS2 message type for which to retrieve the interface definition.

        Returns
        -------
        str
            The mocked output of 'ros2 interface show' for the specified message type.
        """
        if msg_type in self.mock_interfaces:
            return self.mock_interfaces[msg_type]
        else:
            raise ImportError(f"Module {msg_type} not found.")


class MockCallROS2ServiceTool(CallROS2ServiceTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    available_services: List[str]
    available_service_types: List[str]

    def _run(
        self, service_name: str, service_type: str, service_args: Dict[str, Any]
    ) -> str:
        if service_name not in self.available_services:
            raise ValueError(
                f"Service {service_name} is not available within 1.0 seconds. Check if the service exists."
            )
        if service_type not in self.available_service_types:
            raise TypeError(
                "Expected one of service types: {}, got {}".format(
                    self.available_service_types, service_type
                )
            )
        response = ROS2ARIMessage(payload={"response": "success"})
        return str(
            {
                "payload": response.payload,
                "metadata": response.metadata,
            }
        )


class MockStartROS2ActionTool(StartROS2ActionTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    available_actions: List[str] = []
    available_action_types: List[str] = []

    def _run(
        self, action_name: str, action_type: str, action_args: Dict[str, Any]
    ) -> str:
        if action_name not in self.available_actions:
            raise ValueError(
                f"Action {action_name} is not available within 1.0 seconds. Check if the action exists."
            )
        if action_type not in self.available_action_types:
            raise TypeError(
                f"Expected one of action types: {self.available_action_types}, got {action_type}"
            )
        action_id = str(uuid.uuid4())
        response = "mock_action_response_" + action_id
        self.internal_action_id_mapping[response] = action_id
        return "Action started with ID: " + response


class MockCancelROS2ActionTool(CancelROS2ActionTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    available_action_ids: List[str] = []

    def _run(self, action_id: str) -> str:
        if action_id not in self.available_action_ids:
            raise ValueError(f"Action {action_id} is not available for cancellation.")
        return f"Action {action_id} cancelled"


class MockGetROS2ActionFeedbackTool(GetROS2ActionFeedbackTool):
    available_feedbacks: Dict[str, List[Any]] = {}
    internal_action_id_mapping: Dict[str, str] = {}
    action_feedbacks_store_lock: Lock = Lock()

    def _run(self, action_id: str) -> str:
        if action_id not in self.internal_action_id_mapping:
            raise KeyError(f"Action ID {action_id} not found in internal mapping.")
        external_id = self.internal_action_id_mapping[action_id]
        with self.action_feedbacks_store_lock:
            feedbacks = self.available_feedbacks.get(external_id, [])
            self.available_feedbacks[external_id] = []
        return str(feedbacks)


class MockGetROS2ActionResultTool(GetROS2ActionResultTool):
    available_results: Dict[str, Any] = {}
    internal_action_id_mapping: Dict[str, str] = {}
    action_results_store_lock: Lock = Lock()

    def _run(self, action_id: str) -> str:
        if action_id not in self.internal_action_id_mapping:
            raise KeyError(f"Action ID {action_id} not found in internal mapping.")
        external_id = self.internal_action_id_mapping[action_id]
        with self.action_results_store_lock:
            if external_id not in self.available_results:
                raise ValueError(f"No result available for action {action_id}")
            result = self.available_results[external_id]
        return str(result)


class MockGetROS2ActionIDsTool(GetROS2ActionIDsTool):
    internal_action_id_mapping: Dict[str, str] = {}

    def _run(self) -> str:
        return str(list(self.internal_action_id_mapping.keys()))
