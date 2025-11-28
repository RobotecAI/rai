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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sensor_msgs.msg
from rai_perception.tools.gdino_tools import (
    BoundingBox,
    DetectionData,
    GetDetectionTool,
    GetDistanceToObjectsTool,
    GroundingDinoBaseTool,
)

from rai_interfaces.srv import RAIGroundingDino


# Create a concrete test subclass for GroundingDinoBaseTool
# Note: Name doesn't start with "Test" to avoid pytest collection
class ConcreteGroundingDinoBaseTool(GroundingDinoBaseTool):
    """Concrete implementation for testing GroundingDinoBaseTool."""

    def _run(self, *args, **kwargs):
        """Test implementation of _run."""
        return "test"


class TestGroundingDinoBaseTool:
    """Test cases for GroundingDinoBaseTool."""

    @pytest.fixture
    def base_tool(self, mock_connector):
        """Create a GroundingDinoBaseTool instance."""
        # Use model_construct to bypass Pydantic validation
        tool = ConcreteGroundingDinoBaseTool.model_construct(connector=mock_connector)
        return tool

    def test_base_tool_initialization(self, base_tool):
        """Test GroundingDinoBaseTool initialization."""
        assert base_tool.box_threshold == 0.35
        assert base_tool.text_threshold == 0.45
        assert base_tool.connector is not None

    def test_get_img_from_topic_success(self, base_tool, mock_connector):
        """Test get_img_from_topic with successful message."""
        image_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.return_value.payload = image_msg

        result = base_tool.get_img_from_topic("test_topic", timeout_sec=2)

        assert result == image_msg
        mock_connector.receive_message.assert_called_once_with(
            "test_topic", timeout_sec=2
        )

    def test_get_image_message_success(self, base_tool, mock_connector):
        """Test _get_image_message with valid image."""
        image_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.return_value.payload = image_msg

        result = base_tool._get_image_message("test_topic")

        assert result == image_msg

    def test_call_gdino_node(self, base_tool, mock_connector):
        """Test _call_gdino_node creates service call."""
        image_msg = sensor_msgs.msg.Image()
        mock_client = MagicMock()
        mock_client.wait_for_service.return_value = True
        mock_connector.node.create_client.return_value = mock_client

        future = base_tool._call_gdino_node(image_msg, ["dinosaur", "dragon"])

        assert future is not None
        mock_client.call_async.assert_called_once()

    def test_parse_detection_array(self, base_tool):
        """Test _parse_detection_array converts response correctly."""
        response = RAIGroundingDino.Response()

        from vision_msgs.msg import (
            BoundingBox2D,
            Detection2D,
            ObjectHypothesis,
            ObjectHypothesisWithPose,
        )

        from rai_interfaces.msg import RAIDetectionArray

        detection1 = Detection2D()
        detection1.bbox = BoundingBox2D()
        detection1.bbox.center.position.x = 50.0
        detection1.bbox.center.position.y = 50.0
        detection1.bbox.size_x = 40.0
        detection1.bbox.size_y = 40.0
        detection1.results = [ObjectHypothesisWithPose()]
        detection1.results[0].hypothesis = ObjectHypothesis()
        detection1.results[0].hypothesis.class_id = "dinosaur"
        detection1.results[0].hypothesis.score = 0.9

        detection2 = Detection2D()
        detection2.bbox = BoundingBox2D()
        detection2.bbox.center.position.x = 100.0
        detection2.bbox.center.position.y = 100.0
        detection2.bbox.size_x = 30.0
        detection2.bbox.size_y = 30.0
        detection2.results = [ObjectHypothesisWithPose()]
        detection2.results[0].hypothesis = ObjectHypothesis()
        detection2.results[0].hypothesis.class_id = "dragon"
        detection2.results[0].hypothesis.score = 0.8

        response.detections = RAIDetectionArray()
        response.detections.detections = [detection1, detection2]

        result = base_tool._parse_detection_array(response)

        assert len(result) == 2
        assert result[0].class_name == "dinosaur"
        assert result[0].confidence == 0.9
        assert result[0].bbox.x_center == 50.0
        assert result[1].class_name == "dragon"
        assert result[1].confidence == 0.8


class TestGetDetectionTool:
    """Test cases for GetDetectionTool."""

    @pytest.fixture
    def detection_tool(self, mock_connector):
        """Create a GetDetectionTool instance."""
        # Use model_construct to bypass Pydantic validation
        tool = GetDetectionTool.model_construct(connector=mock_connector)
        return tool

    def test_get_detection_tool_run_success(self, detection_tool, mock_connector):
        """Test GetDetectionTool._run with successful detection."""
        image_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.return_value.payload = image_msg

        mock_client = MagicMock()
        mock_client.wait_for_service.return_value = True
        mock_connector.node.create_client.return_value = mock_client

        mock_future = MagicMock()
        mock_client.call_async.return_value = mock_future

        response = RAIGroundingDino.Response()
        from vision_msgs.msg import (
            BoundingBox2D,
            Detection2D,
            ObjectHypothesis,
            ObjectHypothesisWithPose,
        )

        from rai_interfaces.msg import RAIDetectionArray

        detection = Detection2D()
        detection.bbox = BoundingBox2D()
        detection.bbox.center.position.x = 50.0
        detection.bbox.center.position.y = 50.0
        detection.results = [ObjectHypothesisWithPose()]
        detection.results[0].hypothesis = ObjectHypothesis()
        detection.results[0].hypothesis.class_id = "dinosaur"
        detection.results[0].hypothesis.score = 0.9

        response.detections = RAIDetectionArray()
        response.detections.detections = [detection]

        with patch(
            "rai_perception.tools.gdino_tools.get_future_result",
            return_value=response,
        ):
            result = detection_tool._run("test_topic", ["dinosaur", "dragon"])

            assert "detected" in result.lower()
            assert "dinosaur" in result


class TestGetDistanceToObjectsTool:
    """Test cases for GetDistanceToObjectsTool."""

    @pytest.fixture
    def distance_tool(self, mock_connector):
        """Create a GetDistanceToObjectsTool instance."""
        # Use model_construct to bypass Pydantic validation
        tool = GetDistanceToObjectsTool.model_construct(connector=mock_connector)
        return tool

    def test_get_distance_from_detections(self, distance_tool):
        """Test _get_distance_from_detections calculates distances."""
        # Create mock depth image data
        depth_arr = np.ones((200, 200), dtype=np.uint16) * 1000  # 1 meter in mm

        detection1 = DetectionData(
            class_name="dinosaur",
            confidence=0.9,
            bbox=BoundingBox(x_center=50.0, y_center=50.0, width=40.0, height=40.0),
        )

        with patch(
            "rai_perception.tools.gdino_tools.convert_ros_img_to_ndarray",
            return_value=depth_arr,
        ):
            measurements = distance_tool._get_distance_from_detections(
                MagicMock(), [detection1], sigma_threshold=1.0, conversion_ratio=0.001
            )

            assert len(measurements) == 1
            assert measurements[0][0] == "dinosaur"
            assert isinstance(measurements[0][1], (int, float))

    def test_get_distance_tool_run(self, distance_tool, mock_connector):
        """Test GetDistanceToObjectsTool._run."""
        image_msg = sensor_msgs.msg.Image()
        depth_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.side_effect = [
            MagicMock(payload=image_msg),
            MagicMock(payload=depth_msg),
        ]

        mock_client = MagicMock()
        mock_client.wait_for_service.return_value = True
        mock_connector.node.create_client.return_value = mock_client

        mock_connector.node.get_parameter.side_effect = [
            MagicMock(value=1.0),  # outlier_sigma_threshold
            MagicMock(value=0.001),  # conversion_ratio
        ]

        response = RAIGroundingDino.Response()
        from vision_msgs.msg import (
            BoundingBox2D,
            Detection2D,
            ObjectHypothesis,
            ObjectHypothesisWithPose,
        )

        from rai_interfaces.msg import RAIDetectionArray

        detection = Detection2D()
        detection.bbox = BoundingBox2D()
        detection.bbox.center.position.x = 50.0
        detection.bbox.center.position.y = 50.0
        detection.bbox.size_x = 40.0
        detection.bbox.size_y = 40.0
        detection.results = [ObjectHypothesisWithPose()]
        detection.results[0].hypothesis = ObjectHypothesis()
        detection.results[0].hypothesis.class_id = "dinosaur"
        detection.results[0].hypothesis.score = 0.9

        response.detections = RAIDetectionArray()
        response.detections.detections = [detection]

        depth_arr = np.ones((200, 200), dtype=np.uint16) * 1000

        with (
            patch(
                "rai_perception.tools.gdino_tools.get_future_result",
                return_value=response,
            ),
            patch(
                "rai_perception.tools.gdino_tools.convert_ros_img_to_ndarray",
                return_value=depth_arr,
            ),
        ):
            result = distance_tool._run("camera_topic", "depth_topic", ["dinosaur"])

            assert "detected" in result.lower()
            assert "dinosaur" in result
            assert "away" in result
