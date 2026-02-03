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
import rclpy
import sensor_msgs.msg
from rai.communication.ros2 import ROS2ServiceError
from rai_perception.tools.gdino_tools import (
    BoundingBox,
    DetectionData,
    GetDetectionTool,
    GetDistanceToObjectsTool,
    GroundingDinoBaseTool,
)
from rclpy.parameter import Parameter
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from rai_interfaces.msg import RAIDetectionArray
from rai_interfaces.srv import RAIGroundingDino


def _create_detection2d(
    x: float, y: float, size_x: float, size_y: float, class_id: str, score: float
):
    """Helper to create Detection2D message with class and score."""
    detection = Detection2D()
    detection.bbox = BoundingBox2D()
    detection.bbox.center.position.x = x
    detection.bbox.center.position.y = y
    detection.bbox.size_x = size_x
    detection.bbox.size_y = size_y
    detection.results = [ObjectHypothesisWithPose()]
    detection.results[0].hypothesis = ObjectHypothesis()
    detection.results[0].hypothesis.class_id = class_id
    detection.results[0].hypothesis.score = score
    return detection


def _create_detection_response(*detections):
    """Helper to create RAIGroundingDino.Response with detections."""
    response = RAIGroundingDino.Response()
    response.detections = RAIDetectionArray()
    response.detections.detections = list(detections)
    return response


def _setup_mock_service_client(mock_connector, available: bool = True):
    """Helper to set up mock service client."""
    mock_client = MagicMock()
    mock_client.wait_for_service.return_value = available
    mock_client.call_async.return_value = MagicMock()
    mock_connector.node.create_client.return_value = mock_client
    return mock_client


def _setup_distance_tool_params(mock_connector):
    """Helper to set up distance tool parameters."""
    mock_connector.node.set_parameters(
        [
            Parameter(
                "perception.distance_to_objects.outlier_sigma_threshold",
                rclpy.parameter.Parameter.Type.DOUBLE,
                1.0,
            ),
            Parameter(
                "perception.distance_to_objects.conversion_ratio",
                rclpy.parameter.Parameter.Type.DOUBLE,
                0.001,
            ),
        ]
    )


def _create_depth_array(shape=(200, 200), depth_mm: int = 1000):
    """Helper to create depth image array."""
    return np.ones(shape, dtype=np.uint16) * depth_mm


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

    def test_get_detection_service_name_from_param(self, base_tool, mock_connector):
        """Test _get_detection_service_name reads from ROS2 parameter."""
        mock_connector.node.set_parameters(
            [
                Parameter(
                    "/detection_tool/service_name",
                    rclpy.parameter.Parameter.Type.STRING,
                    "/custom/detection_service",
                )
            ]
        )

        service_name = base_tool._get_detection_service_name()
        assert service_name == "/custom/detection_service"

    def test_call_gdino_node(self, base_tool, mock_connector):
        """Test _call_gdino_node creates service call."""
        image_msg = sensor_msgs.msg.Image()
        mock_client = _setup_mock_service_client(mock_connector)

        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            mock_wait.return_value = None  # No exception means success

            future = base_tool._call_gdino_node(image_msg, ["dinosaur", "dragon"])

            assert future is not None
            mock_wait.assert_called_once()
            mock_connector.node.create_client.assert_called_once()
            mock_client.call_async.assert_called_once()

    def test_parse_detection_array(self, base_tool):
        """Test _parse_detection_array converts response correctly."""
        detection1 = _create_detection2d(50.0, 50.0, 40.0, 40.0, "dinosaur", 0.9)
        detection2 = _create_detection2d(100.0, 100.0, 30.0, 30.0, "dragon", 0.8)
        response = _create_detection_response(detection1, detection2)

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
        _setup_mock_service_client(mock_connector)

        detection = _create_detection2d(50.0, 50.0, 40.0, 40.0, "dinosaur", 0.9)
        response = _create_detection_response(detection)

        with (
            patch(
                "rai_perception.components.service_utils.wait_for_ros2_services"
            ) as mock_wait,
            patch(
                "rai_perception.tools.gdino_tools.get_future_result",
                return_value=response,
            ),
        ):
            mock_wait.return_value = None  # No exception means success

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

        _setup_mock_service_client(mock_connector)
        _setup_distance_tool_params(mock_connector)
        distance_tool._load_parameters()

        detection = _create_detection2d(50.0, 50.0, 40.0, 40.0, "dinosaur", 0.9)
        response = _create_detection_response(detection)
        depth_arr = _create_depth_array()

        with (
            patch(
                "rai_perception.components.service_utils.wait_for_ros2_services"
            ) as mock_wait,
            patch(
                "rai_perception.tools.gdino_tools.get_future_result",
                return_value=response,
            ),
            patch(
                "rai_perception.tools.gdino_tools.convert_ros_img_to_ndarray",
                return_value=depth_arr,
            ),
        ):
            mock_wait.return_value = None  # No exception means success

            result = distance_tool._run("camera_topic", "depth_topic", ["dinosaur"])

            assert "detected" in result.lower()
            assert "dinosaur" in result
            assert "away" in result

    def test_get_distance_tool_service_call_failure(
        self, distance_tool, mock_connector
    ):
        """Test GetDistanceToObjectsTool raises ROS2ServiceError when service call fails."""
        image_msg = sensor_msgs.msg.Image()
        depth_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.side_effect = [
            MagicMock(payload=image_msg),
            MagicMock(payload=depth_msg),
        ]

        _setup_mock_service_client(mock_connector)
        _setup_distance_tool_params(mock_connector)
        distance_tool._load_parameters()

        with (
            patch(
                "rai_perception.components.service_utils.wait_for_ros2_services"
            ) as mock_wait,
            patch(
                "rai_perception.tools.gdino_tools.get_future_result",
                return_value=None,
            ),
        ):
            mock_wait.return_value = None  # No exception means success

            with pytest.raises(ROS2ServiceError) as exc_info:
                distance_tool._run("camera_topic", "depth_topic", ["dinosaur"])

        assert exc_info.value.service_name == "/detection"
        assert exc_info.value.timeout_sec == 5.0
        assert exc_info.value.service_state == "call_failed"
        assert "Service call timed out" in exc_info.value.suggestion
