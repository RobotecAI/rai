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
import uuid
from threading import Lock
from typing import Any, Dict, List, Tuple, Type
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError, computed_field
from rai.communication.ros2.api.conversion import import_message_from_str
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2.messages import ROS2Message
from rai.messages import MultimodalArtifact, preprocess_image
from rai.tools.ros2 import (
    CallROS2ServiceTool,
    CancelROS2ActionTool,
    GetObjectPositionsTool,
    GetROS2ActionFeedbackTool,
    GetROS2ActionIDsTool,
    GetROS2ActionResultTool,
    GetROS2ActionsNamesAndTypesTool,
    GetROS2ImageTool,
    GetROS2MessageInterfaceTool,
    GetROS2ServicesNamesAndTypesTool,
    GetROS2TopicsNamesAndTypesTool,
    MoveToPointTool,
    PublishROS2MessageTool,
    ReceiveROS2MessageTool,
    ROS2ActionToolkit,
    StartROS2ActionTool,
)
from rai.types import Point
from rai_perception.tools import (
    DistanceMeasurement,
    GetDistanceToObjectsTool,
    GetGrabbingPointTool,
)
from rosidl_runtime_py import set_message_fields


class MockGetROS2TopicsNamesAndTypesTool(GetROS2TopicsNamesAndTypesTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
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
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
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
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    available_topics: List[str]

    def _run(self, topic: str, timeout_sec: float = 1.0) -> str:
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
                f"Topic {topic} is not available within {timeout_sec} seconds. Check if the topic exists."
            )
        message: ROS2Message = MagicMock(spec=ROS2Message)
        message.payload = {"mock": "payload"}
        message.metadata = {"mock": "metadata"}
        return str({"payload": message.payload, "metadata": message.metadata})


class MockMoveToPointTool(MoveToPointTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)

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
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)

    # Create mock instances for the arguments
    target_frame: str = MagicMock(spec=str)
    source_frame: str = MagicMock(spec=str)
    camera_topic: str = MagicMock(spec=str)
    depth_topic: str = MagicMock(spec=str)
    camera_info_topic: str = MagicMock(spec=str)
    get_grabbing_point_tool: GetGrabbingPointTool = MagicMock(spec=GetGrabbingPointTool)
    mock_objects: dict[str, List[Point]]

    def _run(self, object_name: str) -> str:
        """Method that returns a mock message with the object positions
        if the object_name is present in the mock_objects dictionary.

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
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    available_topics: List[str]
    available_message_types: List[str]
    available_topic_models: Dict[str, Type[BaseModel]]

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

        model = self.available_topic_models[message_type]
        try:
            model.model_validate(message)
        except ValidationError as e:
            raise ValueError(f"Failed to populate fields: {e}")

        return "Message published successfully"


class MockGetROS2MessageInterfaceTool(GetROS2MessageInterfaceTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
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


class ServiceValidator:
    """
    Validator that is responsible for checking if given service type exists
    and if it is used correctly.
    Validator uses ROS 2 native types when available,
    falls back to Pydantic models of custom interfaces when not.
    """

    def __init__(self, custom_models: Dict[str, Type[BaseModel]]):
        self.custom_models = custom_models
        self.ros2_services_cache: Dict[str, Any] = {}

    def validate_with_ros2(self, service_type: str, args: Dict[str, Any]):
        """Validate using installed ROS2 packages services definition

        Parameters
        ----------
        service_type : str
        args : Dict[str, Any]
            Dictionary of arguments to validate against the service definition.

        Raises
        ------
        TypeError
            When service type does not exist in ROS2 installed packages
        """
        service_class = import_message_from_str(service_type)
        if not service_class:
            raise TypeError(f"Service type: {service_type} does not exist.")

        request = service_class.Request()
        # set message fields converts them to object so we need deepcopy to avoid it
        args_to_validate = copy.deepcopy(args)
        set_message_fields(request, args_to_validate)

    def validate_with_custom(self, service_type: str, args: Dict[str, Any]):
        """
        Validate using Pydantic model of custom messages.

        Parameters
        ----------
        service_type : str
        args : Dict[str, Any]
            Dictionary of arguments to validate against the Pydantic model.

        Raises
        ------
        ValueError
            If service_type is not found in custom_models or if Pydantic
            validation fails.
        """
        if service_type not in self.custom_models:
            raise ValueError(f"Service type: {service_type} is invalid custom type")

        model = self.custom_models[service_type]
        try:
            model.model_validate(args)
        except ValidationError as e:
            raise ValueError(f"Pydantic validation failed: {e}") from e

    def validate(self, service_type: str, args: Dict[str, Any]):
        """
        Try ROS 2 validation first, fall back to Pydantic models.

        Parameters
        ----------
        service_type : str
        args : Dict[str, Any]
            Dictionary of arguments to validate.
        """
        if service_type in self.custom_models:
            self.validate_with_custom(service_type, args)
        else:
            self.validate_with_ros2(service_type, args)


class MockCallROS2ServiceTool(CallROS2ServiceTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    available_services: List[str]
    available_service_types: List[str]
    available_service_models: Dict[str, Type[BaseModel]]

    @computed_field
    @property
    def models_validator(self) -> ServiceValidator:
        """computed field for instancinating ServiceValidator with available service models"""
        return ServiceValidator(self.available_service_models)

    def _run(
        self,
        service_name: str,
        service_type: str,
        service_args: Dict[str, Any] = {},
        timeout_sec: float = 1.0,
    ) -> str:
        """
        Execute the mocked ROS2 service call with validation of service type and its args.

        Parameters
        ----------
        service_name : str
            Name of the service to call
        service_type : str
            Type of the service
        service_args : Optional[Dict[str, Any]], optional
            Arguments for the service call, by default None
        """
        if service_name not in self.available_services:
            raise ValueError(
                f"Service {service_name} is not available within {timeout_sec} seconds. Check if the service exists."
            )
        if service_type not in self.available_service_types:
            raise TypeError(
                "Expected one of service types: {}, got {}".format(
                    self.available_service_types, service_type
                )
            )
        if not service_args:
            service_args = {}
        try:
            self.models_validator.validate(service_type, service_args)
            response = ROS2Message(payload={"response": "success"})
            return str(
                {
                    "payload": response.payload,
                    "metadata": response.metadata,
                }
            )
        except ValueError as e:
            raise ValueError(f"Failed to populate fields: {e}")


class MockGetROS2ServicesNamesAndTypesTool(GetROS2ServicesNamesAndTypesTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    mock_service_names_and_types: list[str]

    def _run(self) -> str:
        """Mocked method that returns the mock topics and types instead of fetching from ROS2.

        Returns
        -------
        str
            Mocked output of 'get_ros2_topics_names_and_types' tool.
        """
        return "\n".join(self.mock_service_names_and_types)


class MockGetROS2ActionsNamesAndTypesTool(GetROS2ActionsNamesAndTypesTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    mock_actions_names_and_types: list[str]

    def _run(self) -> str:
        """Mocked method that returns the mock topics and types instead of fetching from ROS2.

        Returns
        -------
        str
            Mocked output of 'get_ros2_topics_names_and_types' tool.
        """
        return "\n".join(self.mock_actions_names_and_types)


class MockStartROS2ActionTool(StartROS2ActionTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    available_actions: List[str] = []
    available_action_types: List[str] = []
    available_action_models: Dict[str, Type[BaseModel]]

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
        model = self.available_action_models[action_type]
        try:
            model.model_validate(action_args)
        except ValidationError as e:
            raise ValueError(f"Failed to populate fields: {e}")

        action_id = str(uuid.uuid4())
        response = action_id
        self.internal_action_id_mapping[response] = action_id
        return "Action started with ID: " + response


class MockCancelROS2ActionTool(CancelROS2ActionTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    available_action_ids: List[str] = []

    def _run(self, action_id: str) -> str:
        if action_id not in self.available_action_ids:
            raise ValueError(f"Action {action_id} is not available for cancellation.")
        return f"Action {action_id} cancelled"


class MockGetROS2ActionFeedbackTool(GetROS2ActionFeedbackTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
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


class MockActionsToolkit(ROS2ActionToolkit):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    mock_actions_names_and_types: list[str] = []
    available_actions: List[str] = []
    available_action_types: List[str] = []
    available_action_models: Dict[str, Type[BaseModel]] = {}

    def get_tools(self) -> List[BaseTool]:
        return [
            MockStartROS2ActionTool(
                available_actions=self.available_actions,
                feedback_callback=self._generic_feedback_callback,
                on_done_callback=self._generic_on_done_callback,
                available_action_types=self.available_action_types,
                available_action_models=self.available_action_models,
            ),
            MockCancelROS2ActionTool(),
            MockGetROS2ActionFeedbackTool(),
            MockGetROS2ActionResultTool(),
            MockGetROS2ActionIDsTool(),
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=self.mock_actions_names_and_types
            ),
        ]


class MockGetDistanceToObjectsTool(GetDistanceToObjectsTool):
    connector: ROS2Connector = MagicMock(spec=ROS2Connector)
    node: MagicMock = MagicMock()
    mock_distance_measurements: List[DistanceMeasurement] = []
    available_topics: List[str]

    def _run(self, camera_topic: str, depth_topic: str, object_names: list[str]):
        """Method that returns a mock message with the distance to the objects.

        Parameters
        ----------
        camera_topic : str
            Topic with the camera image
        depth_topic : str
            Topic with the depth image
        object_names : list[str]
            List of object names to get the distance to

        Returns
        -------
        str
            Message from the tool
        """
        if camera_topic not in self.available_topics:
            return f"Topic {camera_topic} is not available within 1.0 seconds. Check if the topic exists."
        if depth_topic not in self.available_topics:
            return f"Topic {depth_topic} is not available within 1.0 seconds. Check if the topic exists."
        measurement_string = ", ".join(
            [
                f"{measurement[0]}: {measurement[1]:.2f}m away"
                for measurement in self.mock_distance_measurements
                if measurement[0] in object_names
            ]
        )

        return f"I have detected the following items in the picture {measurement_string or 'no objects'}"
