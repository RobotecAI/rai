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

    def _create_mock_client(self, available: bool):
        """Helper to create mock service client."""
        mock_client = MagicMock()
        mock_client.wait_for_service.return_value = available
        return mock_client

    def test_service_available(self, mock_connector):
        """Test returns True when service is available."""
        mock_client = self._create_mock_client(True)
        mock_connector.node.create_client.return_value = mock_client

        result = check_service_available(
            mock_connector, "/test_service", timeout_sec=0.1
        )

        assert result is True
        mock_client.wait_for_service.assert_called_once_with(timeout_sec=0.1)

    def test_service_unavailable(self, mock_connector):
        """Test returns False when service is not available."""
        mock_client = self._create_mock_client(False)
        mock_connector.node.create_client.return_value = mock_client

        result = check_service_available(
            mock_connector, "/test_service", timeout_sec=0.1
        )

        assert result is False

    def test_client_creation_failure(self, mock_connector):
        """Test returns False when client creation fails."""
        mock_connector.node.create_client.side_effect = Exception("Connection failed")

        result = check_service_available(
            mock_connector, "/test_service", timeout_sec=0.1
        )

        assert result is False
        mock_connector.node.get_logger().debug.assert_called()


class TestCreateServiceClient:
    """Test cases for create_service_client."""

    @pytest.fixture
    def service_type(self):
        """Service type for testing."""
        from std_srvs.srv import Empty

        return Empty

    def _create_mock_client(self, wait_results):
        """Helper to create mock client with wait_for_service results."""
        mock_client = MagicMock()
        if isinstance(wait_results, list):
            mock_client.wait_for_service.side_effect = wait_results
        else:
            mock_client.wait_for_service.return_value = wait_results
        return mock_client

    def test_service_available_immediately(self, mock_connector, service_type):
        """Test returns client when service is available immediately."""
        mock_client = self._create_mock_client(True)
        mock_connector.node.create_client.return_value = mock_client

        result = create_service_client(
            mock_connector, service_type, "/test_service", timeout_sec=1.0
        )

        assert result == mock_client
        mock_client.wait_for_service.assert_called_once_with(timeout_sec=1.0)

    def test_service_available_after_wait(self, mock_connector, service_type):
        """Test returns client when service becomes available after waiting."""
        mock_client = self._create_mock_client([False, True])
        mock_connector.node.create_client.return_value = mock_client

        result = create_service_client(
            mock_connector,
            service_type,
            "/test_service",
            timeout_sec=1.0,
            max_wait_time=0.0,
        )

        assert result == mock_client
        assert mock_client.wait_for_service.call_count == 2
