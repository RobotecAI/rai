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

"""Unit tests for GDSegmenter algorithm."""

from rai_perception.algorithms.segmenter import GDSegmenter

from tests.rai_perception.algorithms.test_base_segmenter import TestGDSegmenterBase


class TestGDSegmenter(TestGDSegmenterBase):
    """Test cases for algorithms.segmenter.GDSegmenter class."""

    def get_segmenter_class(self):
        """Return the GDSegmenter class from algorithms."""
        return GDSegmenter

    def get_patch_path(self, target):
        """Return patch path for algorithms module."""
        patch_map = {
            "build_sam2": "rai_perception.algorithms.segmenter.build_sam2",
            "SAM2ImagePredictor": "rai_perception.algorithms.segmenter.SAM2ImagePredictor",
            "convert_ros_img_to_ndarray": "rai_perception.algorithms.segmenter.convert_ros_img_to_ndarray",
        }
        return patch_map[target]
