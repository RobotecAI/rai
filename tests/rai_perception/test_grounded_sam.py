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

import numpy as np
from rai_perception.agents.grounded_sam import (
    GSAM_SERVICE_NAME,
    GroundedSamAgent,
)
from sensor_msgs.msg import Image

from rai_interfaces.srv import RAIGroundedSam
from tests.rai_perception.conftest import patch_ros2_for_agent_tests
from tests.rai_perception.test_base_vision_agent import (
    cleanup_agent,
    create_valid_weights_file,
    get_weights_path,
)


class MockGDSegmenter:
    """Mock GDSegmenter for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_segmentation(self, image, boxes):
        """Mock segmentation that returns simple masks."""
        # Return 2 masks for testing
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[10:50, 10:50] = 1.0
        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[60:90, 60:90] = 1.0
        return [mask1, mask2]


class TestGroundedSamAgent:
    """Test cases for GroundedSamAgent.

    Note: All tests patch ROS2Connector to prevent hanging. BaseVisionAgent.__init__
    creates a real ROS2Connector which requires ROS2 to be initialized, so we patch
    it to use a mock instead for unit testing.
    """

    def test_init(self, tmp_path, mock_connector):
        """Test GroundedSamAgent initialization."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounded_sam.GDSegmenter", MockGDSegmenter),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            assert agent.WEIGHTS_URL is not None
            assert agent.WEIGHTS_FILENAME == "sam2_hiera_large.pt"
            assert agent._segmenter is not None

            cleanup_agent(agent)

    def test_init_default_path(self, mock_connector):
        """Test GroundedSamAgent initialization with default path."""
        weights_path = Path.home() / ".cache/rai/vision/weights/sam2_hiera_large.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounded_sam.GDSegmenter", MockGDSegmenter),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundedSamAgent(ros2_name="test")

            assert agent._segmenter is not None

            cleanup_agent(agent)
            weights_path.unlink()

    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounded_sam.GDSegmenter", MockGDSegmenter),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            with patch.object(
                agent.ros2_connector, "create_service"
            ) as mock_create_service:
                agent.run()

                mock_create_service.assert_called_once_with(
                    service_name=GSAM_SERVICE_NAME,
                    on_request=agent._segment_callback,
                    service_type="rai_interfaces/srv/RAIGroundedSam",
                )

            cleanup_agent(agent)

    def test_segment_callback(self, tmp_path, mock_connector):
        """Test segment callback processes request correctly."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        with (
            patch("rai_perception.agents.grounded_sam.GDSegmenter", MockGDSegmenter),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            # Create mock request
            request = RAIGroundedSam.Request()
            request.source_img = Image()

            # Create mock detections
            from vision_msgs.msg import BoundingBox2D, Detection2D

            from rai_interfaces.msg import RAIDetectionArray

            detection1 = Detection2D()
            detection1.bbox = BoundingBox2D()
            detection1.bbox.center.position.x = 30.0
            detection1.bbox.center.position.y = 30.0
            detection1.bbox.size_x = 40.0
            detection1.bbox.size_y = 40.0

            detection2 = Detection2D()
            detection2.bbox = BoundingBox2D()
            detection2.bbox.center.position.x = 75.0
            detection2.bbox.center.position.y = 75.0
            detection2.bbox.size_x = 30.0
            detection2.bbox.size_y = 30.0

            request.detections = RAIDetectionArray()
            request.detections.detections = [detection1, detection2]

            response = RAIGroundedSam.Response()

            # Call callback
            result = agent._segment_callback(request, response)

            # Verify response contains masks
            assert len(result.masks) == 2
            assert result is response

            cleanup_agent(agent)

    def test_segment_callback_empty_detections(self, tmp_path, mock_connector):
        """Test segment callback with empty detections."""
        weights_path = get_weights_path(tmp_path)
        create_valid_weights_file(weights_path)

        class EmptySegmenter:
            def __init__(self, weights_path):
                self.weights_path = weights_path

            def get_segmentation(self, image, boxes):
                return []

        with (
            patch("rai_perception.agents.grounded_sam.GDSegmenter", EmptySegmenter),
            patch_ros2_for_agent_tests(mock_connector),
            patch(
                "rai_perception.agents.base_vision_agent.BaseVisionAgent._download_weights"
            ),
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            request = RAIGroundedSam.Request()
            request.source_img = Image()

            from rai_interfaces.msg import RAIDetectionArray

            request.detections = RAIDetectionArray()
            request.detections.detections = []

            response = RAIGroundedSam.Response()
            result = agent._segment_callback(request, response)

            assert len(result.masks) == 0

            cleanup_agent(agent)
