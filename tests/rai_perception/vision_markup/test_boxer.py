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

"""Tests for vision_markup.boxer - deprecated wrapper that delegates to algorithms."""

import warnings
from unittest.mock import patch

from rai_perception.vision_markup.boxer import Box, GDBoxer

from tests.rai_perception.algorithms.test_base_boxer import TestGDBoxerBase


class TestVisionMarkupGDBoxer(TestGDBoxerBase):
    """Test cases for vision_markup.boxer.GDBoxer - deprecated wrapper."""

    def get_boxer_class(self):
        """Return the GDBoxer class from vision_markup."""
        return GDBoxer

    def get_patch_path(self, target):
        """Return patch path for algorithms module (delegation target)."""
        patch_map = {
            "Model": "rai_perception.algorithms.boxer.Model",
            "CvBridge": "rai_perception.algorithms.boxer.CvBridge",
            "logging.getLogger": "rai_perception.algorithms.boxer.logging.getLogger",
        }
        return patch_map[target]

    def test_deprecation_warning(self, tmp_path):
        """Test that vision_markup.GDBoxer emits deprecation warning."""
        from tests.rai_perception.algorithms.test_utils import create_test_weights_file

        weights_path = create_test_weights_file(tmp_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("rai_perception.algorithms.boxer.Model"):
                GDBoxer(str(weights_path), use_cuda=False)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "vision_markup" in str(w[0].message)
            assert "algorithms" in str(w[0].message)

    def test_inheritance(self, tmp_path):
        """Test that vision_markup.GDBoxer is a subclass of algorithms.GDBoxer."""
        from rai_perception.algorithms.boxer import GDBoxer as AlgorithmsGDBoxer

        from tests.rai_perception.algorithms.test_utils import create_test_weights_file

        weights_path = create_test_weights_file(tmp_path)

        with patch("rai_perception.algorithms.boxer.Model"):
            vision_markup_boxer = GDBoxer(str(weights_path), use_cuda=False)
            assert isinstance(vision_markup_boxer, AlgorithmsGDBoxer)


class TestVisionMarkupBox:
    """Test cases for vision_markup.boxer.Box class (re-exported from algorithms)."""

    def test_box_is_from_algorithms(self):
        """Test that Box is the same class from algorithms."""
        from rai_perception.algorithms.boxer import Box as AlgorithmsBox

        assert Box is AlgorithmsBox
