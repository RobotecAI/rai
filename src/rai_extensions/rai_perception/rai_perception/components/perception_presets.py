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

import copy
from typing import Any, Dict, Literal, Optional

from rai_perception.components.gripping_points import (
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
)

PresetName = Literal["default_grasp", "precise_grasp", "top_grasp"]
"""Preset names for gripping points extraction configuration.

Presets provide pre-configured combinations of filter and estimator strategies
optimized for different use cases. All presets use domain-oriented parameter names
that map to underlying algorithms internally.

- "default_grasp": Default preset matching tool defaults. Uses aggressive_outlier_removal
  filtering (5% outlier_fraction, maps to Isolation Forest) and centroid estimation.
  Good general-purpose option for most scenarios.

- "precise_grasp": High-quality preset with aggressive outlier filtering
  (aggressive_outlier_removal with 1% outlier_fraction, maps to Isolation Forest)
  and precise top-plane estimation (500 RANSAC iterations, 5mm distance threshold).
  Best for accurate grasping in clean environments.

- "top_grasp": Optimized for top-down grasping from above. Uses aggressive_outlier_removal
  filtering (maps to Isolation Forest) with top-plane estimation that focuses on the
  top 5% of Z-height points. Best when grasping objects from directly above.
"""


_PRESETS: Dict[PresetName, Dict[str, Any]] = {
    "default_grasp": {
        "filter_config": {
            # Domain-oriented strategy name: maps to Isolation Forest algorithm
            "strategy": "aggressive_outlier_removal",
            # Semantic parameter: maps to Isolation Forest contamination
            "outlier_fraction": 0.05,
            "min_points": 20,
        },
        "estimator_config": {
            "strategy": "centroid",
            "ransac_iterations": 200,
            "distance_threshold_m": 0.01,
            "min_points": 10,
        },
    },
    "precise_grasp": {
        "filter_config": {
            # Domain-oriented strategy name: maps to Isolation Forest algorithm
            "strategy": "aggressive_outlier_removal",
            # Semantic parameter: maps to Isolation Forest contamination
            "outlier_fraction": 0.01,
            "min_points": 30,
        },
        "estimator_config": {
            "strategy": "top_plane",
            "ransac_iterations": 500,
            "distance_threshold_m": 0.005,
        },
    },
    "top_grasp": {
        "filter_config": {
            # Domain-oriented strategy name: maps to Isolation Forest algorithm
            "strategy": "aggressive_outlier_removal",
            # Semantic parameter: maps to Isolation Forest contamination
            "outlier_fraction": 0.05,
        },
        "estimator_config": {
            "strategy": "top_plane",
            "top_percentile": 0.05,
            "ransac_iterations": 300,
        },
    },
}


def get_preset(preset_name: PresetName) -> Dict[str, any]:
    """Get preset configuration by name.

    Args:
        preset_name: Name of preset ("default_grasp", "precise_grasp", "top_grasp")

    Returns:
        Dictionary with filter_config and estimator_config (deep copy)

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in _PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(_PRESETS.keys())}"
        )
    return copy.deepcopy(_PRESETS[preset_name])


def apply_preset(
    preset_name: PresetName,
    base_filter_config: Optional[PointCloudFilterConfig] = None,
    base_estimator_config: Optional[GrippingPointEstimatorConfig] = None,
) -> tuple[PointCloudFilterConfig, GrippingPointEstimatorConfig]:
    """Apply preset to base configs, merging preset values with base configs.

    Args:
        preset_name: Name of preset to apply
        base_filter_config: Base filter config (optional)
        base_estimator_config: Base estimator config (optional)

    Returns:
        Tuple of (filter_config, estimator_config) with preset applied
    """
    preset = get_preset(preset_name)

    # Apply filter preset
    filter_dict = base_filter_config.model_dump() if base_filter_config else {}
    filter_dict.update(preset.get("filter_config", {}))
    filter_config = PointCloudFilterConfig(**filter_dict)

    # Apply estimator preset
    estimator_dict = base_estimator_config.model_dump() if base_estimator_config else {}
    estimator_dict.update(preset.get("estimator_config", {}))
    estimator_config = GrippingPointEstimatorConfig(**estimator_dict)

    return filter_config, estimator_config


def list_presets() -> list[PresetName]:
    """List all available preset names.

    Returns:
        List of preset names
    """
    return list(_PRESETS.keys())
