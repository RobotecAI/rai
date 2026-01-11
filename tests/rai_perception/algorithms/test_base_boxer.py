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

"""Base test class for GDBoxer - shared test logic for algorithms and vision_markup."""

import time
from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import rclpy
from sensor_msgs.msg import Image

from tests.rai_perception.algorithms.test_utils import (
    create_mock_image_array,
    create_test_weights_file,
)


class TestGDBoxerBase(ABC):
    """Base test class for GDBoxer - shared test logic."""

    @abstractmethod
    def get_boxer_class(self):
        """Return the GDBoxer class to test (from algorithms or vision_markup)."""
        pass

    @abstractmethod
    def get_patch_path(self, target):
        """Return the patch path for the given target (algorithms or vision_markup)."""
        pass

    def setup_method(self):
        """Initialize ROS2 before tests that use Time() or ROS2 messages."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test to prevent thread exceptions."""
        try:
            if rclpy.ok():
                time.sleep(0.1)
                rclpy.shutdown()
        except Exception:
            pass

    def test_gdboxer_initialization(self, tmp_path):
        """Test GDBoxer initialization with use_cuda=True."""
        weights_path = create_test_weights_file(tmp_path)
        GDBoxer = self.get_boxer_class()
        patch_path = self.get_patch_path("Model")

        with patch(patch_path) as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            boxer = GDBoxer(str(weights_path), use_cuda=True)

            assert boxer.weight_path == str(weights_path)
            assert hasattr(boxer, "model")
            assert boxer.model == mock_model_instance
            mock_model.assert_called_once()

    def test_gdboxer_initialization_use_cuda_false(self, tmp_path):
        """Test GDBoxer initialization with use_cuda=False."""
        weights_path = create_test_weights_file(tmp_path)
        GDBoxer = self.get_boxer_class()
        patch_path = self.get_patch_path("Model")
        logger_path = self.get_patch_path("logging.getLogger")

        with (
            patch(patch_path) as mock_model,
            patch(logger_path) as mock_get_logger,
        ):
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            boxer = GDBoxer(str(weights_path), use_cuda=False)

            assert boxer.device == "cpu"
            assert boxer.weight_path == str(weights_path)
            assert hasattr(boxer, "model")
            assert boxer.model == mock_model_instance
            mock_model.assert_called_once_with(
                boxer.cfg_path, boxer.weight_path, device="cpu"
            )
            mock_logger.warning.assert_not_called()

    def test_gdboxer_get_boxes(self, tmp_path):
        """Test GDBoxer get_boxes method."""
        weights_path = create_test_weights_file(tmp_path)
        GDBoxer = self.get_boxer_class()
        model_patch = self.get_patch_path("Model")
        bridge_patch = self.get_patch_path("CvBridge")

        with (
            patch(model_patch) as mock_model,
            patch(bridge_patch) as mock_bridge,
        ):
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            mock_predictions = MagicMock()
            mock_predictions.xyxy = [[10, 10, 50, 50], [60, 60, 90, 90]]
            mock_predictions.class_id = [0, 1]
            mock_predictions.confidence = [0.9, 0.8]
            mock_model_instance.predict_with_classes.return_value = mock_predictions

            mock_bridge_instance = MagicMock()
            mock_bridge.return_value = mock_bridge_instance
            mock_bridge_instance.imgmsg_to_cv2.return_value = create_mock_image_array()

            boxer = GDBoxer(str(weights_path), use_cuda=True)

            assert hasattr(boxer, "model")
            image_msg = Image()
            classes = ["dinosaur", "dragon"]
            boxes = boxer.get_boxes(image_msg, classes, 0.4, 0.4)

            assert len(boxes) == 2
            assert boxes[0].phrase == "dinosaur"
            assert boxes[0].confidence == 0.9
            assert boxes[1].phrase == "dragon"
            assert boxes[1].confidence == 0.8

    def test_gdboxer_get_boxes_empty(self, tmp_path):
        """Test GDBoxer get_boxes with no detections."""
        weights_path = create_test_weights_file(tmp_path)
        GDBoxer = self.get_boxer_class()
        model_patch = self.get_patch_path("Model")
        bridge_patch = self.get_patch_path("CvBridge")

        with (
            patch(model_patch) as mock_model,
            patch(bridge_patch) as mock_bridge,
        ):
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            mock_predictions = MagicMock()
            mock_predictions.xyxy = []
            mock_model_instance.predict_with_classes.return_value = mock_predictions

            mock_bridge_instance = MagicMock()
            mock_bridge.return_value = mock_bridge_instance
            mock_bridge_instance.imgmsg_to_cv2.return_value = create_mock_image_array()

            boxer = GDBoxer(str(weights_path), use_cuda=False)

            image_msg = Image()
            classes = ["dinosaur"]
            boxes = boxer.get_boxes(image_msg, classes, 0.4, 0.4)

            assert len(boxes) == 0
