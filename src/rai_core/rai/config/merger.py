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

from typing import Any, Dict


def merge_configs(
    defaults: Dict[str, Any],
    ros2_params: Dict[str, Any],
    user_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge configuration dictionaries with precedence.

    Precedence order (highest to lowest):
    1. user_overrides
    2. ros2_params
    3. defaults

    Args:
        defaults: Default configuration values
        ros2_params: ROS2 parameter values
        user_overrides: User-provided override values

    Returns:
        Merged configuration dictionary
    """
    merged = defaults.copy()
    merged.update(ros2_params)
    merged.update(user_overrides)
    return merged


def merge_nested_configs(
    defaults: Dict[str, Any],
    ros2_params: Dict[str, Any],
    user_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge nested configuration dictionaries with precedence.

    Recursively merges nested dictionaries. Precedence order (highest to lowest):
    1. user_overrides
    2. ros2_params
    3. defaults

    Args:
        defaults: Default configuration values (may be nested)
        ros2_params: ROS2 parameter values (may be nested)
        user_overrides: User-provided override values (may be nested)

    Returns:
        Merged configuration dictionary
    """
    merged = defaults.copy()

    # Merge ROS2 params (recursive)
    for key, value in ros2_params.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_nested_configs(merged[key], value, {})
        else:
            merged[key] = value

    # Merge user overrides (recursive)
    for key, value in user_overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_nested_configs(merged[key], {}, value)
        else:
            merged[key] = value

    return merged
