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

"""Shared test helpers for service and agent tests."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import rclpy
from rclpy.parameter import Parameter

from tests.rai_perception.conftest import (
    create_valid_weights_file,
    patch_ros2_for_service_tests,
)


def setup_service_parameters(mock_connector, model_name: str, service_name: str):
    """Setup ROS2 parameters for service tests.

    Args:
        mock_connector: Mock ROS2Connector instance
        model_name: Model name parameter value
        service_name: Service name parameter value
    """
    mock_connector.node.set_parameters(
        [
            Parameter(
                "model_name",
                rclpy.parameter.Parameter.Type.STRING,
                model_name,
            ),
            Parameter(
                "service_name",
                rclpy.parameter.Parameter.Type.STRING,
                service_name,
            ),
        ]
    )


@contextmanager
def patch_detection_service_dependencies(
    mock_connector, mock_boxer_class, weights_path
):
    """Context manager to patch all dependencies for DetectionService tests.

    Args:
        mock_connector: Mock ROS2Connector instance
        mock_boxer_class: Mock boxer class to use
        weights_path: Path to weights file
    """
    with (
        patch("rai_perception.algorithms.boxer.GDBoxer", mock_boxer_class),
        patch("rai_perception.models.detection.get_model") as mock_get_model,
        patch_ros2_for_service_tests(mock_connector),
        patch("rai_perception.services.base_vision_service.download_weights"),
        patch(
            "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
        ) as mock_load_model,
    ):
        from rai_perception.algorithms.boxer import GDBoxer

        mock_get_model.return_value = (GDBoxer, "config_path")
        mock_load_model.return_value = mock_boxer_class(weights_path)
        yield mock_get_model, mock_load_model


@contextmanager
def patch_segmentation_service_dependencies(
    mock_connector, mock_segmenter_class, weights_path
):
    """Context manager to patch all dependencies for SegmentationService tests.

    Args:
        mock_connector: Mock ROS2Connector instance
        mock_segmenter_class: Mock segmenter class to use
        weights_path: Path to weights file
    """
    with (
        patch("rai_perception.algorithms.segmenter.GDSegmenter", mock_segmenter_class),
        patch("rai_perception.models.segmentation.get_model") as mock_get_model,
        patch_ros2_for_service_tests(mock_connector),
        patch("rai_perception.services.base_vision_service.download_weights"),
        patch(
            "rai_perception.services.segmentation_service.SegmentationService._load_model_with_error_handling"
        ) as mock_load_model,
    ):
        from rai_perception.algorithms.segmenter import GDSegmenter

        mock_get_model.return_value = (GDSegmenter, "config_path")
        mock_load_model.return_value = mock_segmenter_class(weights_path)
        yield mock_get_model, mock_load_model


@contextmanager
def patch_detection_agent_dependencies(mock_connector, mock_boxer_class, weights_path):
    """Context manager to patch all dependencies for GroundingDinoAgent tests.

    Args:
        mock_connector: Mock ROS2Connector instance
        mock_boxer_class: Mock boxer class to use
        weights_path: Path to weights file
    """
    from tests.rai_perception.conftest import patch_ros2_for_agent_tests

    with (
        patch("rai_perception.algorithms.boxer.GDBoxer", mock_boxer_class),
        patch("rai_perception.models.detection.get_model") as mock_get_model,
        patch_ros2_for_agent_tests(mock_connector),
        patch("rai_perception.services.base_vision_service.download_weights"),
        patch(
            "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
        ) as mock_load_model,
    ):
        from rai_perception.algorithms.boxer import GDBoxer

        mock_get_model.return_value = (GDBoxer, "config_path")
        mock_load_model.return_value = mock_boxer_class(weights_path)
        yield mock_get_model, mock_load_model


@contextmanager
def patch_segmentation_agent_dependencies(
    mock_connector, mock_segmenter_class, weights_path
):
    """Context manager to patch all dependencies for GroundedSamAgent tests.

    Args:
        mock_connector: Mock ROS2Connector instance
        mock_segmenter_class: Mock segmenter class to use
        weights_path: Path to weights file
    """
    from tests.rai_perception.conftest import patch_ros2_for_agent_tests

    with (
        patch("rai_perception.algorithms.segmenter.GDSegmenter", mock_segmenter_class),
        patch("rai_perception.models.segmentation.get_model") as mock_get_model,
        patch_ros2_for_agent_tests(mock_connector),
        patch("rai_perception.services.base_vision_service.download_weights"),
        patch(
            "rai_perception.services.segmentation_service.SegmentationService._load_model_with_error_handling"
        ) as mock_load_model,
    ):
        from rai_perception.algorithms.segmenter import GDSegmenter

        mock_get_model.return_value = (GDSegmenter, "config_path")
        mock_load_model.return_value = mock_segmenter_class(weights_path)
        yield mock_get_model, mock_load_model


def get_detection_weights_path(tmp_path: Path) -> Path:
    """Get standard detection weights path for testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to detection weights file
    """
    weights_path = tmp_path / "vision" / "weights" / "groundingdino_swint_ogc.pth"
    create_valid_weights_file(weights_path)
    return weights_path


def get_segmentation_weights_path(tmp_path: Path) -> Path:
    """Get standard segmentation weights path for testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to segmentation weights file
    """
    weights_path = tmp_path / "vision" / "weights" / "sam2_hiera_large.pt"
    create_valid_weights_file(weights_path)
    return weights_path


def get_default_detection_weights_path() -> Path:
    """Get default detection weights path in home directory.

    Returns:
        Path to default detection weights file
    """
    return Path.home() / ".cache/rai/vision/weights/groundingdino_swint_ogc.pth"


def get_default_segmentation_weights_path() -> Path:
    """Get default segmentation weights path in home directory.

    Returns:
        Path to default segmentation weights file
    """
    return Path.home() / ".cache/rai/vision/weights/sam2_hiera_large.pt"


@contextmanager
def patch_detection_agent_dependencies_default_path(
    mock_connector, mock_boxer_class, weights_path
):
    """Context manager for detection agent tests with default path (no weights_root_path).

    Args:
        mock_connector: Mock ROS2Connector instance
        mock_boxer_class: Mock boxer class to use
        weights_path: Path to weights file
    """
    from tests.rai_perception.conftest import patch_ros2_for_agent_tests

    with (
        patch("rai_perception.algorithms.boxer.GDBoxer", mock_boxer_class),
        patch("rai_perception.models.detection.get_model") as mock_get_model,
        patch_ros2_for_agent_tests(mock_connector),
        patch("rai_perception.services.base_vision_service.download_weights"),
        patch(
            "rai_perception.services.detection_service.DetectionService._load_model_with_error_handling"
        ) as mock_load_model,
    ):
        from rai_perception.algorithms.boxer import GDBoxer

        mock_get_model.return_value = (GDBoxer, "config_path")
        mock_load_model.return_value = mock_boxer_class(weights_path)
        yield mock_get_model, mock_load_model


@contextmanager
def patch_segmentation_agent_dependencies_default_path(
    mock_connector, mock_segmenter_class, weights_path
):
    """Context manager for segmentation agent tests with default path (no weights_root_path).

    Args:
        mock_connector: Mock ROS2Connector instance
        mock_segmenter_class: Mock segmenter class to use
        weights_path: Path to weights file
    """
    from tests.rai_perception.conftest import patch_ros2_for_agent_tests

    with (
        patch("rai_perception.algorithms.segmenter.GDSegmenter", mock_segmenter_class),
        patch("rai_perception.models.segmentation.get_model") as mock_get_model,
        patch_ros2_for_agent_tests(mock_connector),
        patch("rai_perception.services.base_vision_service.download_weights"),
        patch(
            "rai_perception.services.segmentation_service.SegmentationService._load_model_with_error_handling"
        ) as mock_load_model,
    ):
        from rai_perception.algorithms.segmenter import GDSegmenter

        mock_get_model.return_value = (GDSegmenter, "config_path")
        mock_load_model.return_value = mock_segmenter_class(weights_path)
        yield mock_get_model, mock_load_model


def create_detection_request(
    classes: str = "dinosaur", box_threshold: float = 0.4, text_threshold: float = 0.4
):
    """Create a RAIGroundingDino request for testing.

    Args:
        classes: Comma-separated class names
        box_threshold: Box threshold value
        text_threshold: Text threshold value

    Returns:
        RAIGroundingDino.Request instance
    """
    from sensor_msgs.msg import Image

    from rai_interfaces.srv import RAIGroundingDino

    request = RAIGroundingDino.Request()
    request.source_img = Image()
    request.classes = classes
    request.box_threshold = box_threshold
    request.text_threshold = text_threshold
    return request


def create_segmentation_request(detections=None):
    """Create a RAIGroundedSam request for testing.

    Args:
        detections: List of Detection2D messages, or None for empty

    Returns:
        RAIGroundedSam.Request instance
    """
    from sensor_msgs.msg import Image

    from rai_interfaces.msg import RAIDetectionArray
    from rai_interfaces.srv import RAIGroundedSam

    request = RAIGroundedSam.Request()
    request.source_img = Image()
    request.detections = RAIDetectionArray()
    request.detections.detections = detections or []
    return request


def create_test_detection2d(
    center_x: float, center_y: float, size_x: float, size_y: float
):
    """Create a Detection2D message for testing.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        size_x: Width of bounding box
        size_y: Height of bounding box

    Returns:
        Detection2D message
    """
    from vision_msgs.msg import BoundingBox2D, Detection2D

    detection = Detection2D()
    detection.bbox = BoundingBox2D()
    detection.bbox.center.position.x = center_x
    detection.bbox.center.position.y = center_y
    detection.bbox.size_x = size_x
    detection.bbox.size_y = size_y
    return detection
