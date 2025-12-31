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

import time
from unittest.mock import MagicMock, patch

import numpy as np
import rclpy
from rai_perception.vision_markup.boxer import Box, GDBoxer
from rclpy.time import Time
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D


class TestBox:
    """Test cases for Box class."""

    def test_box_initialization(self):
        """Test Box initialization."""
        box = Box((50.0, 50.0), 40.0, 40.0, "dinosaur", 0.9)

        assert box.center == (50.0, 50.0)
        assert box.size_x == 40.0
        assert box.size_y == 40.0
        assert box.phrase == "dinosaur"
        assert box.confidence == 0.9

    def test_box_to_detection_msg(self):
        """Test Box conversion to Detection2D message."""
        box = Box((50.0, 50.0), 40.0, 40.0, "dinosaur", 0.9)

        class_dict = {"dinosaur": 0, "dragon": 1}
        timestamp = Time()

        detection = box.to_detection_msg(class_dict, timestamp)

        assert isinstance(detection, Detection2D)
        assert detection.bbox.center.position.x == 50.0
        assert detection.bbox.center.position.y == 50.0
        assert detection.bbox.size_x == 40.0
        assert detection.bbox.size_y == 40.0
        assert detection.results[0].hypothesis.class_id == "dinosaur"
        assert detection.results[0].hypothesis.score == 0.9
        assert detection.header.stamp == timestamp.to_msg()


class TestGDBoxer:
    """Test cases for GDBoxer class."""

    def setup_method(self):
        """Initialize ROS2 before tests that use Time() or ROS2 messages."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test to prevent thread exceptions."""
        try:
            if rclpy.ok():
                # Give any executor threads a moment to finish before shutting down
                time.sleep(0.1)
                rclpy.shutdown()
        except Exception:
            # Ignore errors during shutdown - thread may have already been cleaned up
            pass

    def test_gdboxer_initialization(self, tmp_path):
        """Test GDBoxer initialization with use_cuda=True to verify model is always initialized."""
        weights_path = tmp_path / "weights.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"test")

        with patch("rai_perception.vision_markup.boxer.Model") as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            boxer = GDBoxer(str(weights_path), use_cuda=True)

            assert boxer.weight_path == str(weights_path)
            assert hasattr(boxer, "model")
            assert boxer.model == mock_model_instance
            mock_model.assert_called_once()

    def test_gdboxer_initialization_use_cuda_false(self, tmp_path):
        """Test GDBoxer initialization with use_cuda=False to verify CPU device selection."""
        weights_path = tmp_path / "weights.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"test")

        with (
            patch("rai_perception.vision_markup.boxer.Model") as mock_model,
            patch(
                "rai_perception.vision_markup.boxer.logging.getLogger"
            ) as mock_get_logger,
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
        """Test GDBoxer get_boxes method with use_cuda=True to verify model is initialized."""
        weights_path = tmp_path / "weights.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"test")

        with (
            patch("rai_perception.vision_markup.boxer.Model") as mock_model,
            patch("rai_perception.vision_markup.boxer.CvBridge") as mock_bridge,
        ):
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            # Mock predictions
            mock_predictions = MagicMock()
            mock_predictions.xyxy = [[10, 10, 50, 50], [60, 60, 90, 90]]
            mock_predictions.class_id = [0, 1]
            mock_predictions.confidence = [0.9, 0.8]
            mock_model_instance.predict_with_classes.return_value = mock_predictions

            # Mock bridge
            mock_bridge_instance = MagicMock()
            mock_bridge.return_value = mock_bridge_instance
            # Return a valid numpy array (BGR format) that cv2.cvtColor can process
            mock_bridge_instance.imgmsg_to_cv2.return_value = np.zeros(
                (100, 100, 3), dtype=np.uint8
            )

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
        weights_path = tmp_path / "weights.pth"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"test")

        with (
            patch("rai_perception.vision_markup.boxer.Model") as mock_model,
            patch("rai_perception.vision_markup.boxer.CvBridge") as mock_bridge,
        ):
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            mock_predictions = MagicMock()
            mock_predictions.xyxy = []
            mock_model_instance.predict_with_classes.return_value = mock_predictions

            mock_bridge_instance = MagicMock()
            mock_bridge.return_value = mock_bridge_instance
            # Return a valid numpy array (BGR format) that cv2.cvtColor can process
            mock_bridge_instance.imgmsg_to_cv2.return_value = np.zeros(
                (100, 100, 3), dtype=np.uint8
            )

            boxer = GDBoxer(str(weights_path), use_cuda=False)

            image_msg = Image()
            classes = ["dinosaur"]
            boxes = boxer.get_boxes(image_msg, classes, 0.4, 0.4)

            assert len(boxes) == 0
