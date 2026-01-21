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

from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rai_perception.components.service_utils import (
    check_service_available,
    create_service_client,
    get_detection_service_name,
    get_segmentation_service_name,
)
from rclpy.parameter import Parameter


def _set_service_name_parameter(mock_connector, param_name: str, value: str):
    """Helper to set service name parameter."""
    mock_connector.node.set_parameters(
        [
            Parameter(
                param_name,
                rclpy.parameter.Parameter.Type.STRING,
                value,
            )
        ]
    )


class TestGetDetectionServiceName:
    """Test cases for get_detection_service_name."""

    def test_service_name_from_parameter(self, mock_connector):
        """Test reads service name from ROS2 parameter."""
        _set_service_name_parameter(
            mock_connector, "/detection_tool/service_name", "/custom/detection"
        )
        service_name = get_detection_service_name(mock_connector)
        assert service_name == "/custom/detection"


class TestGetSegmentationServiceName:
    """Test cases for get_segmentation_service_name."""

    def test_service_name_from_parameter(self, mock_connector):
        """Test reads service name from ROS2 parameter."""
        _set_service_name_parameter(
            mock_connector, "/segmentation_tool/service_name", "/custom/segmentation"
        )
        service_name = get_segmentation_service_name(mock_connector)
        assert service_name == "/custom/segmentation"


class TestCheckServiceAvailable:
    """Test cases for check_service_available."""

    def test_service_available(self, mock_connector):
        """Test returns True when service is available."""
        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            mock_wait.return_value = None  # No exception means success

            result = check_service_available(
                mock_connector, "/test_service", timeout_sec=0.1
            )

            assert result is True
            mock_wait.assert_called_once_with(
                mock_connector, ["/test_service"], timeout=0.1
            )

    def test_service_unavailable(self, mock_connector):
        """Test returns False when service is not available."""
        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            mock_wait.side_effect = TimeoutError("Service not available")

            result = check_service_available(
                mock_connector, "/test_service", timeout_sec=0.1
            )

            assert result is False
            mock_wait.assert_called_once_with(
                mock_connector, ["/test_service"], timeout=0.1
            )

    def test_invalid_timeout_returns_false(self, mock_connector):
        """Test returns False when ValueError is raised (invalid timeout)."""
        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            mock_wait.side_effect = ValueError("Invalid timeout")

            result = check_service_available(
                mock_connector, "/test_service", timeout_sec=-1.0
            )

            assert result is False

    def test_non_blocking_check_timeout_zero(self, mock_connector):
        """Test that timeout=0 performs non-blocking check without calling wait_for_ros2_services."""
        mock_connector.get_services_names_and_types.return_value = [
            ("/test_service", ["std_srvs/srv/Empty"]),
            ("/other_service", ["std_srvs/srv/Empty"]),
        ]

        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            result = check_service_available(
                mock_connector, "/test_service", timeout_sec=0.0
            )

            assert result is True
            # Should not call wait_for_ros2_services when timeout <= 0
            mock_wait.assert_not_called()

        # Test with negative timeout
        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            result = check_service_available(
                mock_connector, "/nonexistent_service", timeout_sec=-0.1
            )

            assert result is False
            # Should not call wait_for_ros2_services when timeout <= 0
            mock_wait.assert_not_called()


class TestCreateServiceClient:
    """Test cases for create_service_client."""

    @pytest.fixture
    def service_type(self):
        """Service type for testing."""
        from std_srvs.srv import Empty

        return Empty

    def test_service_available_immediately(self, mock_connector, service_type):
        """Test returns client when service is available immediately."""
        mock_client = MagicMock()
        mock_connector.node.create_client.return_value = mock_client

        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            mock_wait.return_value = None  # No exception means success

            result = create_service_client(
                mock_connector, service_type, "/test_service", timeout_sec=1.0
            )

            assert result == mock_client
            mock_wait.assert_called_once_with(
                mock_connector, ["/test_service"], timeout=0.0
            )
            mock_connector.node.create_client.assert_called_once_with(
                service_type, "/test_service"
            )

    def test_service_timeout_raises_error(self, mock_connector, service_type):
        """Test raises ROS2ServiceError when service times out."""
        from rai.communication.ros2 import ROS2ServiceError

        mock_connector.get_services_names_and_types.return_value = [
            ("/other_service", ["std_srvs/srv/Empty"])
        ]

        with patch(
            "rai_perception.components.service_utils.wait_for_ros2_services"
        ) as mock_wait:
            mock_wait.side_effect = TimeoutError("Service not available")

            with pytest.raises(ROS2ServiceError) as exc_info:
                create_service_client(
                    mock_connector,
                    service_type,
                    "/test_service",
                    timeout_sec=1.0,
                    max_wait_time=5.0,
                )

            assert exc_info.value.service_name == "/test_service"
            assert exc_info.value.timeout_sec == 5.0
            assert "Service not available after 5.0s" in exc_info.value.suggestion
