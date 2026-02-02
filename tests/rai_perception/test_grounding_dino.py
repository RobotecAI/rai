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
# See the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import patch

from rai_perception.agents.grounding_dino import (
    GDINO_SERVICE_NAME,
    GroundingDinoAgent,
)
from rai_perception.vision_markup.boxer import Box
from sensor_msgs.msg import Image

from tests.rai_perception.conftest import patch_ros2_for_agent_tests
from tests.rai_perception.test_base_vision_agent import (
    cleanup_agent,
    create_valid_weights_file,
    get_weights_path,
)


def setup_mock_clock(agent):
    """Setup mock clock for agent tests.

    Uses the shared utility from tests.communication.ros2.helpers to handle
    ROS2 Humble vs Jazzy differences.
    """
    from tests.communication.ros2 import setup_mock_clock_for_agent

    setup_mock_clock_for_agent(agent, use_time_wrapper=True)


class MockGDBoxer:
    """Mock GDBoxer for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
        """Mock box detection."""
        box1 = Box((50.0, 50.0), 40.0, 40.0, classes[0], 0.9)
        box2 = Box((100.0, 100.0), 30.0, 30.0, classes[1], 0.8)
        return [box1, box2]


class TestGroundingDinoAgent:
    """Test cases for GroundingDinoAgent.

    Note: All tests patch ROS2Connector to prevent hanging. BaseVisionAgent.__init__
    creates a real ROS2Connector which requires ROS2 to be initialized, so we patch
    it to use a mock instead for unit testing.
    """

    def test_init(self, tmp_path, mock_connector):
        """Test GroundingDinoAgent initialization."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounding_dino.GDBoxer", MockGDBoxer),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            assert agent.WEIGHTS_URL is not None
            assert agent.WEIGHTS_FILENAME == "groundingdino_swint_ogc.pth"
            assert agent._boxer is not None

            cleanup_agent(agent)

    def test_init_default_path(self, mock_connector):
        """Test GroundingDinoAgent initialization with default path."""
        weights_path = (
            Path.home() / ".cache/rai/vision/weights/groundingdino_swint_ogc.pth"
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounding_dino.GDBoxer", MockGDBoxer),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundingDinoAgent(ros2_name="test")

            assert agent._boxer is not None

            cleanup_agent(agent)
            weights_path.unlink()

    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounding_dino.GDBoxer", MockGDBoxer),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            with patch.object(
                agent.ros2_connector, "create_service"
            ) as mock_create_service:
                agent.run()

                mock_create_service.assert_called_once()
                call_args = mock_create_service.call_args
                assert call_args[0][0] == GDINO_SERVICE_NAME
                assert call_args[0][1] == agent._classify_callback
                assert (
                    call_args[1]["service_type"]
                    == "rai_interfaces/srv/RAIGroundingDino"
                )

            cleanup_agent(agent)

    def test_classify_callback(self, tmp_path, mock_connector):
        """Test classify callback processes request correctly."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounding_dino.GDBoxer", MockGDBoxer),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            # Create mock request
            from rai_interfaces.srv import RAIGroundingDino

            request = RAIGroundingDino.Request()
            request.source_img = Image()
            request.classes = "dinosaur, dragon"
            request.box_threshold = 0.4
            request.text_threshold = 0.4

            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            # Call callback
            result = agent._classify_callback(request, response)

            # Verify response
            assert len(result.detections.detections) == 2
            assert result.detections.detection_classes == ["dinosaur", "dragon"]
            assert result is response

            cleanup_agent(agent)

    def test_classify_callback_empty_boxes(self, tmp_path, mock_connector):
        """Test classify callback with no detections."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        class EmptyBoxer:
            def __init__(self, weights_path):
                self.weights_path = weights_path

            def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
                return []

        with (
            patch("rai_perception.agents.grounding_dino.GDBoxer", EmptyBoxer),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = RAIGroundingDino.Request()
            request.source_img = Image()
            request.classes = "dinosaur"
            request.box_threshold = 0.4
            request.text_threshold = 0.4

            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            result = agent._classify_callback(request, response)

            assert len(result.detections.detections) == 0
            assert result.detections.detection_classes == ["dinosaur"]

            cleanup_agent(agent)
