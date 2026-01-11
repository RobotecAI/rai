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

import pytest
from rai_perception.components.gripping_points import (
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
)
from rai_perception.components.perception_presets import (
    apply_preset,
    get_preset,
    list_presets,
)


class TestGetPreset:
    """Test cases for get_preset function."""

    def test_get_preset_valid(self):
        """Test getting valid preset."""
        preset = get_preset("default_grasp")

        assert "filter_config" in preset
        assert "estimator_config" in preset
        assert preset["filter_config"]["strategy"] == "aggressive_outlier_removal"

    def test_get_preset_invalid_name(self):
        """Test getting invalid preset name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_preset("invalid_preset")

        assert "Unknown preset" in str(exc_info.value)

    def test_get_preset_returns_copy(self):
        """Test that get_preset returns a copy, not the original."""
        preset1 = get_preset("default_grasp")
        preset2 = get_preset("default_grasp")

        preset1["filter_config"]["custom_key"] = "value"

        assert "custom_key" not in preset2["filter_config"]


class TestApplyPreset:
    """Test cases for apply_preset function."""

    def test_apply_preset_to_empty_configs(self):
        """Test applying preset to empty base configs."""
        filter_config, estimator_config = apply_preset("default_grasp")

        assert isinstance(filter_config, PointCloudFilterConfig)
        assert isinstance(estimator_config, GrippingPointEstimatorConfig)
        assert filter_config.strategy == "aggressive_outlier_removal"
        assert estimator_config.strategy == "centroid"

    def test_apply_preset_merges_with_base(self):
        """Test applying preset merges with base configs."""
        base_filter = PointCloudFilterConfig(strategy="density_based", min_points=50)
        base_estimator = GrippingPointEstimatorConfig(
            strategy="centroid", min_points=20
        )

        filter_config, estimator_config = apply_preset(
            "precise_grasp", base_filter, base_estimator
        )

        assert (
            filter_config.strategy == "aggressive_outlier_removal"
        )  # Preset overrides
        assert filter_config.min_points == 30  # Preset overrides
        assert estimator_config.strategy == "top_plane"  # Preset overrides
        assert estimator_config.min_points == 20  # Base preserved (not in preset)

    def test_apply_preset_invalid_name(self):
        """Test applying invalid preset name raises ValueError."""
        with pytest.raises(ValueError):
            apply_preset("invalid_preset")


class TestListPresets:
    """Test cases for list_presets function."""

    def test_list_presets_returns_all(self):
        """Test that list_presets returns a non-empty list with default_grasp."""
        presets = list_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0
        assert "default_grasp" in presets
