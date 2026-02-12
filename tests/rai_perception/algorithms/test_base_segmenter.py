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

"""Base test class for GDSegmenter - shared test logic for algorithms and vision_markup."""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import rclpy
from sensor_msgs.msg import Image

from tests.rai_perception.algorithms.test_utils import (
    create_test_bbox,
    create_test_weights_file,
)


class TestGDSegmenterBase(ABC):
    """Base test class for GDSegmenter - shared test logic."""

    @abstractmethod
    def get_segmenter_class(self):
        """Return the GDSegmenter class to test (from algorithms or vision_markup)."""
        pass

    @abstractmethod
    def get_patch_path(self, target):
        """Return the patch path for the given target (algorithms or vision_markup)."""
        pass

    def setup_method(self):
        """Initialize ROS2 before tests that use ROS2 messages."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test."""
        try:
            if rclpy.ok():
                time.sleep(0.1)
                rclpy.shutdown()
        except Exception:
            pass

    @contextmanager
    def _patch_segmenter_dependencies(self):
        """Context manager to patch all GDSegmenter dependencies."""
        build_patch = self.get_patch_path("build_sam2")
        predictor_patch = self.get_patch_path("SAM2ImagePredictor")

        with (
            patch(build_patch) as mock_build,
            patch(predictor_patch) as mock_predictor_class,
        ):
            mock_model = MagicMock()
            mock_build.return_value = mock_model
            mock_predictor = MagicMock()
            mock_predictor_class.return_value = mock_predictor
            yield mock_build, mock_predictor

    def test_gdsegmenter_initialization(self, tmp_path):
        """Test GDSegmenter initialization with default config."""
        weights_path = create_test_weights_file(tmp_path, "weights.pt")
        GDSegmenter = self.get_segmenter_class()

        with self._patch_segmenter_dependencies() as (mock_build, _):
            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            assert segmenter.weight_path == str(weights_path)
            assert segmenter.device == "cpu"
            assert hasattr(segmenter, "sam2_model")
            assert hasattr(segmenter, "sam2_predictor")
            mock_build.assert_called_once()

    def test_gdsegmenter_initialization_with_config_path(self, tmp_path):
        """Test GDSegmenter initialization with config_path (ignored for SAM2)."""
        weights_path = create_test_weights_file(tmp_path, "weights.pt")
        config_path = tmp_path / "custom_config.yml"
        config_path.write_text("test: config")
        GDSegmenter = self.get_segmenter_class()

        with self._patch_segmenter_dependencies() as (mock_build, _):
            GDSegmenter(str(weights_path), config_path=str(config_path), use_cuda=False)

            mock_build.assert_called_once_with(
                "seg_config.yml", str(weights_path), device="cpu"
            )

    def test_gdsegmenter_get_segmentation(self, tmp_path):
        """Test GDSegmenter get_segmentation method."""
        weights_path = create_test_weights_file(tmp_path, "weights.pt")
        GDSegmenter = self.get_segmenter_class()
        convert_patch = self.get_patch_path("convert_ros_img_to_ndarray")

        with (
            self._patch_segmenter_dependencies() as (_, mock_predictor),
            patch(convert_patch) as mock_convert,
        ):
            mock_img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_convert.return_value = mock_img_array

            mask1 = np.zeros((100, 100), dtype=np.float32)
            mask1[10:50, 10:50] = 1.0
            mask2 = np.zeros((100, 100), dtype=np.float32)
            mask2[60:90, 60:90] = 1.0
            mock_predictor.predict.side_effect = [
                (mask1, None, None),
                (mask2, None, None),
            ]

            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            image_msg = Image()
            bbox1 = create_test_bbox(30.0, 30.0, 40.0, 40.0)
            bbox2 = create_test_bbox(75.0, 75.0, 30.0, 30.0)

            masks = segmenter.get_segmentation(image_msg, [bbox1, bbox2])

            assert len(masks) == 2
            assert isinstance(masks[0], np.ndarray)
            assert isinstance(masks[1], np.ndarray)
            mock_predictor.set_image.assert_called_once_with(mock_img_array)
            assert mock_predictor.predict.call_count == 2

    def test_gdsegmenter_get_segmentation_empty_bboxes(self, tmp_path):
        """Test GDSegmenter get_segmentation with empty bbox list."""
        weights_path = create_test_weights_file(tmp_path, "weights.pt")
        GDSegmenter = self.get_segmenter_class()
        convert_patch = self.get_patch_path("convert_ros_img_to_ndarray")

        with (
            self._patch_segmenter_dependencies() as (_, mock_predictor),
            patch(convert_patch) as mock_convert,
        ):
            mock_img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_convert.return_value = mock_img_array

            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            image_msg = Image()
            masks = segmenter.get_segmentation(image_msg, [])

            assert len(masks) == 0
            mock_predictor.set_image.assert_called_once()
            mock_predictor.predict.assert_not_called()

    def test_gdsegmenter_get_boxes_xyxy(self, tmp_path):
        """Test internal _get_boxes_xyxy conversion method."""
        weights_path = create_test_weights_file(tmp_path, "weights.pt")
        GDSegmenter = self.get_segmenter_class()

        with self._patch_segmenter_dependencies():
            segmenter = GDSegmenter(str(weights_path), use_cuda=False)

            bbox = create_test_bbox(50.0, 50.0, 40.0, 30.0)
            xyxy_boxes = segmenter._get_boxes_xyxy([bbox])

            assert len(xyxy_boxes) == 1
            expected = np.array([30.0, 35.0, 70.0, 65.0])
            np.testing.assert_array_almost_equal(xyxy_boxes[0], expected)
