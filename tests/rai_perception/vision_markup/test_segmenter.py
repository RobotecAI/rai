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

"""Tests for vision_markup.segmenter - deprecated wrapper that delegates to algorithms."""

import warnings

from rai_perception.vision_markup.segmenter import GDSegmenter

from tests.rai_perception.algorithms.test_base_segmenter import TestGDSegmenterBase


class TestVisionMarkupGDSegmenter(TestGDSegmenterBase):
    """Test cases for vision_markup.segmenter.GDSegmenter - deprecated wrapper."""

    def get_segmenter_class(self):
        """Return the GDSegmenter class from vision_markup."""
        return GDSegmenter

    def get_patch_path(self, target):
        """Return patch path for algorithms module (delegation target)."""
        patch_map = {
            "build_sam2": "rai_perception.algorithms.segmenter.build_sam2",
            "SAM2ImagePredictor": "rai_perception.algorithms.segmenter.SAM2ImagePredictor",
            "convert_ros_img_to_ndarray": "rai_perception.algorithms.segmenter.convert_ros_img_to_ndarray",
        }
        return patch_map[target]

    def test_deprecation_warning(self, tmp_path):
        """Test that vision_markup.GDSegmenter emits deprecation warning."""
        from tests.rai_perception.algorithms.test_utils import create_test_weights_file

        weights_path = create_test_weights_file(tmp_path, "weights.pt")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Import directly from segmenter module to avoid module-level warning
            from rai_perception.vision_markup.segmenter import (
                GDSegmenter as SegmenterClass,
            )

            with self._patch_segmenter_dependencies():
                SegmenterClass(str(weights_path), use_cuda=False)

            # Filter to only class-level deprecation warnings
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "segmenter.GDSegmenter" in str(warning.message)
            ]
            assert len(deprecation_warnings) == 1
            assert "vision_markup" in str(deprecation_warnings[0].message)
            assert "algorithms" in str(deprecation_warnings[0].message)

    def test_inheritance(self, tmp_path):
        """Test that vision_markup.GDSegmenter is a subclass of algorithms.GDSegmenter."""
        from rai_perception.algorithms.segmenter import (
            GDSegmenter as AlgorithmsGDSegmenter,
        )

        from tests.rai_perception.algorithms.test_utils import create_test_weights_file

        weights_path = create_test_weights_file(tmp_path, "weights.pt")

        with self._patch_segmenter_dependencies():
            vision_markup_segmenter = GDSegmenter(str(weights_path), use_cuda=False)
            assert isinstance(vision_markup_segmenter, AlgorithmsGDSegmenter)
