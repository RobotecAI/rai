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
from rai.config.merger import merge_configs, merge_nested_configs


class TestMergeConfigs:
    """Test cases for merge_configs function."""

    def test_merge_configs_precedence(self):
        """Test that precedence order is correct: user > ROS2 > defaults."""
        defaults = {"key": "default"}
        ros2_params = {"key": "ros2"}
        user_overrides = {"key": "user"}

        result = merge_configs(defaults, ros2_params, user_overrides)

        assert result["key"] == "user"

    def test_merge_configs_empty_dicts(self):
        """Test merging with empty dictionaries."""
        defaults = {"key": "default"}
        ros2_params = {}
        user_overrides = {}

        result = merge_configs(defaults, ros2_params, user_overrides)

        assert result == {"key": "default"}

    @pytest.mark.parametrize(
        "defaults,ros2_params,user_overrides,expected",
        [
            ({}, {}, {}, {}),
            ({"a": 1}, {}, {}, {"a": 1}),
            ({}, {"a": 1}, {}, {"a": 1}),
            ({}, {}, {"a": 1}, {"a": 1}),
            ({"a": 1}, {"a": 2}, {}, {"a": 2}),
            ({"a": 1}, {}, {"a": 3}, {"a": 3}),
            ({"a": 1}, {"a": 2}, {"a": 3}, {"a": 3}),
        ],
    )
    def test_merge_configs_combinations(
        self, defaults, ros2_params, user_overrides, expected
    ):
        """Test various combinations of merge inputs."""
        result = merge_configs(defaults, ros2_params, user_overrides)
        assert result == expected


class TestMergeNestedConfigs:
    """Test cases for merge_nested_configs function."""

    def test_merge_nested_configs_precedence(self):
        """Test nested config precedence order."""
        defaults = {"section": {"key": "default"}}
        ros2_params = {"section": {"key": "ros2"}}
        user_overrides = {"section": {"key": "user"}}

        result = merge_nested_configs(defaults, ros2_params, user_overrides)

        assert result["section"]["key"] == "user"

    def test_merge_nested_configs_mixed_nested_flat(self):
        """Test merging with mixed nested and flat keys."""
        defaults = {"nested": {"key1": "default1"}, "flat": "default_flat"}
        ros2_params = {"nested": {"key2": "ros2_value"}, "flat": "ros2_flat"}
        user_overrides = {"nested": {"key3": "user_value"}}

        result = merge_nested_configs(defaults, ros2_params, user_overrides)

        assert result["nested"]["key1"] == "default1"
        assert result["nested"]["key2"] == "ros2_value"
        assert result["nested"]["key3"] == "user_value"
        assert result["flat"] == "ros2_flat"
