from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.messages import MultimodalArtifact, preprocess_image
from rai.tools.ros2 import GetROS2ImageTool, GetROS2TopicsNamesAndTypesTool


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
