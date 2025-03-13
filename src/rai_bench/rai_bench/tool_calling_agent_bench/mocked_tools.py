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

from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.communication.ros2.messages import ROS2ARIMessage
from rai.messages import MultimodalArtifact, preprocess_image
from rai.tools.ros.manipulation import (
    GetGrabbingPointTool,
    GetObjectPositionsTool,
    MoveToPointTool,
)
from rai.tools.ros2 import (
    GetROS2ImageTool,
    GetROS2TopicsNamesAndTypesTool,
    ReceiveROS2MessageTool,
)


class MockGetROS2TopicsNamesAndTypesTool(GetROS2TopicsNamesAndTypesTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)
    mock_topics_names_and_types: list[str]

    def _run(self) -> str:
        """Return the mock topics and types instead of fetching from ROS2."""
        return "\n".join(self.mock_topics_names_and_types)


class MockGetROS2ImageTool(GetROS2ImageTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)

    def _run(self, topic: str) -> Tuple[str, MultimodalArtifact]:
        image = self.generate_mock_image()
        return "Image received successfully", MultimodalArtifact(
            images=[preprocess_image(image)]
        )  # type: ignore

    @staticmethod
    def generate_mock_image():
        """Generate a blank black mock image (480x640, RGB)."""
        height, width = 480, 640
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)
        return blank_image


class MockReceiveROS2MessageTool(ReceiveROS2MessageTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)

    def _run(self, topic: str) -> str:
        message: ROS2ARIMessage = MagicMock(spec=ROS2ARIMessage)
        message.payload = {"mock": "payload"}
        message.metadata = {"mock": "metadata"}
        return str({"payload": message.payload, "metadata": message.metadata})


class MockMoveToPointTool(MoveToPointTool):
    connector: ROS2ARIConnector = MagicMock(spec=ROS2ARIConnector)

    def _run(self, x: float, y: float, z: float, task: str) -> str:
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
    mock_objects: dict[str, List[Tuple[float, float, float]]]

    def _run(self, object_name: str):
        expected_positions = self.mock_objects.get(object_name, [])
        print(f"Expected positions: {expected_positions}")
        if len([expected_positions]) == 0:
            return f"No {object_name}s detected."
        else:
            return f"Centroids of detected {object_name}s in manipulator frame: {expected_positions} Sizes of the detected objects are unknown."
