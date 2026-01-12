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


from rai.communication.ros2.exceptions import ROS2ParameterError, ROS2ServiceError


class TestROS2ServiceError:
    """Test cases for ROS2ServiceError."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = ROS2ServiceError(
            service_name="/test_service",
            timeout_sec=5.0,
        )

        assert error.service_name == "/test_service"
        assert error.timeout_sec == 5.0
        assert error.service_state is None
        assert error.suggestion is None
        assert error.underlying_error is None
        assert "Service /test_service error: unavailable" in str(error)

    def test_with_all_fields(self):
        """Test error with all optional fields."""
        underlying = ValueError("Connection failed")
        error = ROS2ServiceError(
            service_name="/test_service",
            timeout_sec=3.0,
            service_state="unavailable",
            suggestion="Check if service is running",
            underlying_error=underlying,
        )

        assert error.service_name == "/test_service"
        assert error.timeout_sec == 3.0
        assert error.service_state == "unavailable"
        assert error.suggestion == "Check if service is running"
        assert error.underlying_error == underlying
        assert "Service /test_service error: unavailable" in str(error)
        assert "Check if service is running" in str(error)


class TestROS2ParameterError:
    """Test cases for ROS2ParameterError."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = ROS2ParameterError(
            param_name="test_param",
        )

        assert error.param_name == "test_param"
        assert error.expected_type is None
        assert error.expected_value is None
        assert error.suggestion is None
        assert error.default_value is None
        assert "Parameter test_param error: missing or invalid" in str(error)

    def test_with_all_fields(self):
        """Test error with all optional fields."""
        error = ROS2ParameterError(
            param_name="test_param",
            expected_type="string",
            expected_value="/test/service",
            suggestion="Set in launch file",
            default_value="/default/service",
        )

        assert error.param_name == "test_param"
        assert error.expected_type == "string"
        assert error.expected_value == "/test/service"
        assert error.suggestion == "Set in launch file"
        assert error.default_value == "/default/service"
        assert "Parameter test_param error: Set in launch file" in str(error)
        assert "(expected type: string)" in str(error)
