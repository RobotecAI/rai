# Copyright (C) 2025 Julia Jia
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

from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rai.communication.ros2 import ROS2Connector
from rai_perception.components.detection_publisher import DetectionPublisher
from sensor_msgs.msg import Image


@pytest.fixture(scope="module")
def ros2_context():
    """Initialize ROS2 context for testing."""
    rclpy.init()
    yield
    rclpy.shutdown()


def set_parameter(node, name: str, value, param_type):
    """Helper to set a single parameter on a node."""
    node.connector.node.set_parameters(
        [rclpy.parameter.Parameter(name, param_type, value)]
    )


@pytest.fixture
def detection_publisher(ros2_context):
    """Create a DetectionPublisher instance for testing.

    Uses single_threaded executor to avoid executor performance warnings
    in simple unit tests that don't need multi-threaded execution.
    Mocks the DINO service client and check_service_available to prevent warnings
    about unavailable service.
    """
    connector = ROS2Connector(
        node_name="detection_publisher", executor_type="single_threaded"
    )

    # Mock the service client to prevent warnings about unavailable service
    mock_client = MagicMock()
    mock_client.wait_for_service.return_value = True

    with (
        patch.object(connector.node, "create_client", return_value=mock_client),
        patch(
            "rai_perception.components.detection_publisher.check_service_available",
            return_value=True,
        ),
    ):
        node = DetectionPublisher(connector=connector)
        yield node
        node.connector.shutdown()


def test_detection_publisher_initialization(detection_publisher):
    """Test that DetectionPublisher initializes correctly."""
    assert detection_publisher is not None
    assert detection_publisher.connector.node.get_name() == "detection_publisher"
    assert detection_publisher.bridge is not None
    assert detection_publisher.last_image is None
    assert detection_publisher.last_depth_image is None
    assert detection_publisher.last_camera_info is None
    assert detection_publisher.last_detection_time == 0.0


def test_parse_detection_classes_basic(detection_publisher):
    """Test parsing detection classes with basic format."""
    set_parameter(
        detection_publisher,
        "default_class_threshold",
        0.3,
        rclpy.parameter.Parameter.Type.DOUBLE,
    )

    classes_str = "person, cup, bottle"
    class_names, class_thresholds = detection_publisher._parse_detection_classes(
        classes_str
    )

    assert len(class_names) == 3
    assert set(class_names) == {"person", "cup", "bottle"}
    assert all(class_thresholds[cls] == 0.3 for cls in class_names)


def test_parse_detection_classes_with_thresholds(detection_publisher):
    """Test parsing detection classes with explicit thresholds."""
    set_parameter(
        detection_publisher,
        "default_class_threshold",
        0.3,
        rclpy.parameter.Parameter.Type.DOUBLE,
    )

    classes_str = "person:0.7, cup, bottle:0.4"
    class_names, class_thresholds = detection_publisher._parse_detection_classes(
        classes_str
    )

    assert len(class_names) == 3
    assert set(class_names) == {"person", "cup", "bottle"}
    assert class_thresholds["person"] == 0.7
    assert class_thresholds["cup"] == 0.3
    assert class_thresholds["bottle"] == 0.4


def test_parse_detection_classes_empty_string(detection_publisher):
    """Test parsing empty detection classes string."""
    set_parameter(
        detection_publisher,
        "default_class_threshold",
        0.3,
        rclpy.parameter.Parameter.Type.DOUBLE,
    )

    class_names, class_thresholds = detection_publisher._parse_detection_classes("")

    assert len(class_names) == 0
    assert len(class_thresholds) == 0


def test_parse_detection_classes_invalid_threshold(detection_publisher):
    """Test parsing detection classes with invalid threshold falls back to default."""
    set_parameter(
        detection_publisher,
        "default_class_threshold",
        0.3,
        rclpy.parameter.Parameter.Type.DOUBLE,
    )

    classes_str = "person:invalid, cup"
    class_names, class_thresholds = detection_publisher._parse_detection_classes(
        classes_str
    )

    assert len(class_names) == 2
    assert "person" in class_names
    assert "cup" in class_names
    # Invalid threshold should fall back to default
    assert class_thresholds["person"] == 0.3
    assert class_thresholds["cup"] == 0.3


def test_image_callback_handles_service_error(ros2_context):
    """Test that image_callback catches ROS2ServiceError and logs warning without crashing."""
    connector = ROS2Connector(
        node_name="detection_publisher", executor_type="single_threaded"
    )

    # Mock the service client
    mock_client = MagicMock()
    mock_client.wait_for_service.return_value = True

    # Mock logger to verify warning is logged
    mock_logger = MagicMock()
    connector.node.get_logger = MagicMock(return_value=mock_logger)

    # Mock check_service_available to return False, which will cause ROS2ServiceError
    with (
        patch.object(connector.node, "create_client", return_value=mock_client),
        patch(
            "rai_perception.components.detection_publisher.check_service_available",
            return_value=False,
        ),
    ):
        detection_publisher = DetectionPublisher(connector=connector)

        # Set required parameters
        set_parameter(
            detection_publisher,
            "detection_interval",
            0.0,
            rclpy.parameter.Parameter.Type.DOUBLE,
        )
        set_parameter(
            detection_publisher,
            "dino_service",
            "/test_dino_service",
            rclpy.parameter.Parameter.Type.STRING,
        )

        # Create a mock Image message
        image_msg = Image()
        image_msg.header.frame_id = "test_frame"
        image_msg.header.stamp.sec = 0
        image_msg.header.stamp.nanosec = 0

        # Call image_callback - should not raise exception
        detection_publisher.image_callback(image_msg)

        # Verify warning was logged (called at least once - may be called during init and in callback)
        assert mock_logger.warning.call_count >= 1

        # Verify that the image_callback warning was logged with expected message
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        image_callback_warning = next(
            (msg for msg in warning_calls if "Failed to process image" in msg), None
        )
        assert image_callback_warning is not None, (
            "Expected warning from image_callback not found"
        )
        assert "/test_dino_service" in image_callback_warning
        assert "Failed to process image" in image_callback_warning

        detection_publisher.connector.shutdown()
