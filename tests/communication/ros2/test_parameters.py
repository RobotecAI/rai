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

from unittest.mock import MagicMock

import pytest
import rclpy
from rai.communication.ros2.exceptions import ROS2ParameterError
from rai.communication.ros2.parameters import _extract_param_value, get_param_value
from rclpy.parameter import Parameter, ParameterValue


@pytest.fixture
def ros2_node():
    """Create a ROS2 node for testing."""
    rclpy.init()
    try:
        node = rclpy.create_node("test_node")
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


class TestGetParamValue:
    """Test cases for get_param_value."""

    def test_returns_default_when_param_not_found(self, ros2_node):
        """Test returns default value when parameter not found."""
        value = get_param_value(ros2_node, "nonexistent_param", default="default_value")
        assert value == "default_value"

    def test_raises_error_when_no_default(self, ros2_node):
        """Test raises ROS2ParameterError when parameter not found and no default."""
        with pytest.raises(ROS2ParameterError) as exc_info:
            get_param_value(ros2_node, "nonexistent_param")

        assert exc_info.value.param_name == "nonexistent_param"
        assert "not found and no default provided" in exc_info.value.suggestion

    def test_returns_string_parameter(self, ros2_node):
        """Test returns string parameter value."""
        ros2_node.declare_parameter("test_string", "test_value")
        value = get_param_value(ros2_node, "test_string")
        assert value == "test_value"

    def test_returns_integer_parameter(self, ros2_node):
        """Test returns integer parameter value."""
        ros2_node.declare_parameter("test_int", 42)
        value = get_param_value(ros2_node, "test_int")
        assert value == 42

    def test_returns_double_parameter(self, ros2_node):
        """Test returns double parameter value."""
        ros2_node.declare_parameter("test_double", 3.14)
        value = get_param_value(ros2_node, "test_double")
        assert value == 3.14

    def test_returns_bool_parameter(self, ros2_node):
        """Test returns bool parameter value."""
        ros2_node.declare_parameter("test_bool", True)
        value = get_param_value(ros2_node, "test_bool")
        assert value is True

    @pytest.mark.parametrize(
        "param_name,param_value,expected",
        [
            ("byte_array", [1, 2, 3], [1, 2, 3]),
            ("bool_array", [True, False, True], [True, False, True]),
            ("int_array", [1, 2, 3], [1, 2, 3]),
            ("double_array", [1.1, 2.2, 3.3], [1.1, 2.2, 3.3]),
            ("string_array", ["a", "b", "c"], ["a", "b", "c"]),
        ],
    )
    def test_returns_array_parameters(
        self, ros2_node, param_name, param_value, expected
    ):
        """Test returns array parameter values."""
        ros2_node.declare_parameter(param_name, param_value)
        value = get_param_value(ros2_node, param_name)
        assert value == expected
        assert isinstance(value, list)

    def test_extract_param_value_unknown_type_returns_none(self):
        """Test _extract_param_value returns None for unknown parameter type."""
        mock_param = MagicMock(spec=Parameter)
        mock_param_value = MagicMock(spec=ParameterValue)
        mock_param.get_parameter_value.return_value = mock_param_value
        # Use an invalid type value that doesn't match any known ParameterType
        mock_param.type_ = MagicMock()
        mock_param.type_.value = 999  # Invalid type value

        result = _extract_param_value(mock_param)
        assert result is None
