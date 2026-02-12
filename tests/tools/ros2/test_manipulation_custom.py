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

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

try:
    import rclpy  # noqa: F401
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

try:
    from rai_interfaces.srv import ManipulatorMoveTo
except ImportError:
    pytest.skip("rai_interfaces is not installed", allow_module_level=True)

from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import custom
from rai.tools.ros2.manipulation.custom import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
    MoveToPointTool,
    ResetArmTool,
)


def test_rai_perception_import_error_handling(caplog):
    """Test that ImportError for rai_perception.tools is handled gracefully with a warning."""
    # Remove the module from cache to allow re-import
    module_name = "rai.tools.ros2.manipulation.custom"
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Mock the import to raise ImportError for rai_perception.tools
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "rai_perception.tools":
            raise ImportError("No module named 'rai_perception'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with caplog.at_level(logging.WARNING):
            # Import should succeed despite the ImportError for rai_perception
            import rai.tools.ros2.manipulation.custom  # noqa: F401

            # Check that the warning was logged
            assert any(
                "rai-perception is not installed, GetGrabbingPointTool will not work"
                in record.message
                for record in caplog.records
            )


@pytest.fixture
def mock_connector():
    """Create a mock ROS2Connector for testing."""
    connector = MagicMock(spec=ROS2Connector)
    connector.node = MagicMock()
    connector.node.create_client = MagicMock()
    connector.node.get_logger.return_value = MagicMock()
    connector.logger = MagicMock()
    connector.get_services_names_and_types.return_value = [
        ("/service1", ["type1"]),
        ("/service2", ["type2"]),
    ]
    return connector


@pytest.fixture
def mock_service_client():
    """Create a mock service client."""
    client = MagicMock()
    client.wait_for_service.return_value = True
    future = MagicMock()
    client.call_async.return_value = future
    return client


@pytest.fixture
def success_response():
    """Create a successful ManipulatorMoveTo response."""
    response = ManipulatorMoveTo.Response()
    response.success = True
    return response


@pytest.fixture
def failure_response():
    """Create a failed ManipulatorMoveTo response."""
    response = ManipulatorMoveTo.Response()
    response.success = False
    return response


@pytest.fixture
def identity_transform():
    """Create an identity transform for testing."""
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.0
    transform.transform.translation.z = 0.0
    transform.transform.rotation.w = 1.0
    return transform


def setup_service_client(mock_connector, mock_service_client, available=True):
    """Helper to set up service client on connector."""
    mock_service_client.wait_for_service.return_value = available
    mock_connector.node.create_client.return_value = mock_service_client
    return mock_service_client


class TestMoveToPointTool:
    """Test cases for MoveToPointTool."""

    @pytest.fixture
    def tool(self, mock_connector):
        """Create a MoveToPointTool instance."""
        return MoveToPointTool(
            connector=mock_connector,
            manipulator_frame="base_link",
        )

    def test_service_not_available(self, tool, mock_connector, mock_service_client):
        """Test behavior when service is not available."""
        setup_service_client(mock_connector, mock_service_client, available=False)

        result = tool._run(x=0.1, y=0.2, z=0.3, task="grab")

        assert "not available" in result
        assert "/manipulator_move_to" in result

    def test_service_timeout(self, tool, mock_connector, mock_service_client):
        """Test behavior when service call times out."""
        setup_service_client(mock_connector, mock_service_client)
        # Use shorter timeout for faster test execution
        tool.timeout_sec = 0.5

        with patch.object(custom, "get_future_result", return_value=None):
            result = tool._run(x=0.1, y=0.2, z=0.3, task="grab")

        assert "timed out" in result
        assert all(coord in result for coord in ["0.10", "0.20", "0.30"])

    def test_grab_task_success(
        self, tool, mock_connector, mock_service_client, success_response
    ):
        """Test successful grab task."""
        setup_service_client(mock_connector, mock_service_client)

        with patch.object(custom, "get_future_result", return_value=success_response):
            result = tool._run(x=0.1, y=0.2, z=0.3, task="grab")

        assert "successfully positioned" in result
        assert all(coord in result for coord in ["0.10", "0.20", "0.30"])
        # Verify gripper states for grab
        call_args = mock_service_client.call_async.call_args[0][0]
        assert call_args.initial_gripper_state is True
        assert call_args.final_gripper_state is False

    def test_drop_task_success(
        self, tool, mock_connector, mock_service_client, success_response
    ):
        """Test successful drop task."""
        setup_service_client(mock_connector, mock_service_client)

        tool.additional_height = 0.05
        with patch.object(custom, "get_future_result", return_value=success_response):
            result = tool._run(x=0.1, y=0.2, z=0.3, task="drop")

        assert "successfully positioned" in result
        # Verify gripper states for drop
        call_args = mock_service_client.call_async.call_args[0][0]
        assert call_args.initial_gripper_state is False
        assert call_args.final_gripper_state is True
        # Verify additional height was added
        assert call_args.target_pose.pose.position.z == pytest.approx(
            0.3 + 0.05, abs=0.01
        )

    def test_service_failure(
        self, tool, mock_connector, mock_service_client, failure_response
    ):
        """Test behavior when service returns failure."""
        setup_service_client(mock_connector, mock_service_client)

        with patch.object(custom, "get_future_result", return_value=failure_response):
            result = tool._run(x=0.1, y=0.2, z=0.3, task="grab")

        assert "Failed to position" in result

    def test_calibration_applied(
        self, tool, mock_connector, mock_service_client, success_response
    ):
        """Test that calibration offsets are applied."""
        setup_service_client(mock_connector, mock_service_client)

        tool.calibration_x = 0.01
        tool.calibration_y = 0.02
        tool.calibration_z = 0.03

        with patch.object(custom, "get_future_result", return_value=success_response):
            tool._run(x=0.1, y=0.2, z=0.3, task="grab")

        call_args = mock_service_client.call_async.call_args[0][0]
        assert call_args.target_pose.pose.position.x == pytest.approx(0.11, abs=0.001)
        assert call_args.target_pose.pose.position.y == pytest.approx(0.22, abs=0.001)
        assert call_args.target_pose.pose.position.z == pytest.approx(0.33, abs=0.001)

    def test_min_z_enforced(
        self, tool, mock_connector, mock_service_client, success_response
    ):
        """Test that minimum z coordinate is enforced."""
        setup_service_client(mock_connector, mock_service_client)

        tool.min_z = 0.2

        with patch(
            "rai.tools.ros2.manipulation.custom.get_future_result",
            return_value=success_response,
        ):
            tool._run(x=0.1, y=0.2, z=0.1, task="grab")

        call_args = mock_service_client.call_async.call_args[0][0]
        assert call_args.target_pose.pose.position.z >= 0.2


class TestMoveObjectFromToTool:
    """Test cases for MoveObjectFromToTool."""

    @pytest.fixture
    def tool(self, mock_connector):
        """Create a MoveObjectFromToTool instance."""
        return MoveObjectFromToTool(
            connector=mock_connector,
            manipulator_frame="base_link",
        )

    def test_service_not_available(self, tool, mock_connector, mock_service_client):
        """Test behavior when service is not available."""
        setup_service_client(mock_connector, mock_service_client, available=False)

        result = tool._run(x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6)

        assert "not available" in result

    def test_first_move_timeout(self, tool, mock_connector, mock_service_client):
        """Test behavior when first move times out."""
        setup_service_client(mock_connector, mock_service_client)
        # Use shorter timeout for faster test execution
        tool.timeout_sec = 0.5

        with patch.object(custom, "get_future_result", return_value=None):
            result = tool._run(x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6)

        assert "timed out" in result
        assert "0.10" in result

    def test_second_move_timeout(
        self, tool, mock_connector, mock_service_client, success_response
    ):
        """Test behavior when second move times out."""
        setup_service_client(mock_connector, mock_service_client)
        # Use shorter timeout for faster test execution
        tool.timeout_sec = 0.5

        with patch.object(
            custom,
            "get_future_result",
            side_effect=[success_response, None],
        ):
            result = tool._run(x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6)

        assert "timed out" in result
        assert "0.40" in result

    def test_first_move_failure(
        self, tool, mock_connector, mock_service_client, failure_response
    ):
        """Test behavior when first move fails."""
        setup_service_client(mock_connector, mock_service_client)

        with patch.object(custom, "get_future_result", return_value=failure_response):
            result = tool._run(x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6)

        assert "Failed to position" in result
        assert "0.10" in result

    def test_success(self, tool, mock_connector, mock_service_client, success_response):
        """Test successful move from one point to another."""
        setup_service_client(mock_connector, mock_service_client)

        with (
            patch.object(
                custom,
                "get_future_result",
                side_effect=[success_response, success_response],
            ),
            patch.object(ResetArmTool, "_run", return_value="Arm successfully reset."),
        ):
            result = tool._run(x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6)

        assert "successfully positioned" in result
        assert all(coord in result for coord in ["0.40", "0.50", "0.60"])
        # Verify reset was called
        assert mock_service_client.call_async.call_count == 2

    def test_gripper_states(
        self, tool, mock_connector, mock_service_client, success_response
    ):
        """Test that gripper states are set correctly for grab and drop."""
        setup_service_client(mock_connector, mock_service_client)

        with (
            patch.object(
                custom,
                "get_future_result",
                side_effect=[success_response, success_response],
            ),
            patch.object(ResetArmTool, "_run", return_value="Arm successfully reset."),
        ):
            tool._run(x=0.1, y=0.2, z=0.3, x1=0.4, y1=0.5, z1=0.6)

        # First call: grab (open -> closed)
        first_call = mock_service_client.call_async.call_args_list[0][0][0]
        assert first_call.initial_gripper_state is True
        assert first_call.final_gripper_state is False

        # Second call: drop (closed -> open)
        second_call = mock_service_client.call_async.call_args_list[1][0][0]
        assert second_call.initial_gripper_state is False
        assert second_call.final_gripper_state is True


class TestResetArmTool:
    """Test cases for ResetArmTool."""

    @pytest.fixture
    def tool(self, mock_connector):
        """Create a ResetArmTool instance."""
        return ResetArmTool(
            connector=mock_connector,
            manipulator_frame="base_link",
        )

    def test_service_timeout(self, tool, mock_connector, mock_service_client):
        """Test behavior when service call times out."""
        setup_service_client(mock_connector, mock_service_client)
        # Note: ResetArmTool uses hardcoded 5.0s timeout, but patch should make it return immediately

        with patch.object(custom, "get_future_result", return_value=None):
            result = tool._run()

        assert "Failed to reset" in result

    def test_success(self, tool, mock_connector, mock_service_client, success_response):
        """Test successful arm reset."""
        setup_service_client(mock_connector, mock_service_client)

        with patch.object(custom, "get_future_result", return_value=success_response):
            result = tool._run()

        assert "Arm successfully reset" in result
        call_args = mock_service_client.call_async.call_args[0][0]
        assert call_args.target_pose.pose.position.x == 0.31
        assert call_args.target_pose.pose.position.y == 0.0
        assert call_args.target_pose.pose.position.z == 0.59

    def test_failure(self, tool, mock_connector, mock_service_client, failure_response):
        """Test behavior when reset fails."""
        setup_service_client(mock_connector, mock_service_client)

        with patch.object(custom, "get_future_result", return_value=failure_response):
            result = tool._run()

        assert "Failed to reset" in result


class TestGetObjectPositionsTool:
    """Test cases for GetObjectPositionsTool."""

    @pytest.fixture
    def mock_get_grabbing_point_tool(self):
        """Create a mock GetGrabbingPointTool."""
        tool = MagicMock()
        return tool

    @pytest.fixture
    def tool(self, mock_connector, mock_get_grabbing_point_tool):
        """Create a GetObjectPositionsTool instance."""
        # Use model_construct to bypass Pydantic validation for the mock tool
        return GetObjectPositionsTool.model_construct(
            connector=mock_connector,
            target_frame="base_link",
            source_frame="camera_link",
            camera_topic="/camera/rgb/image_raw",
            depth_topic="/camera/depth/image_raw",
            camera_info_topic="/camera/rgb/camera_info",
            get_grabbing_point_tool=mock_get_grabbing_point_tool,
        )

    def test_no_objects_detected(
        self, tool, mock_connector, mock_get_grabbing_point_tool, identity_transform
    ):
        """Test behavior when no objects are detected."""
        mock_get_grabbing_point_tool._run.return_value = []
        mock_connector.get_transform.return_value = identity_transform

        result = tool._run("cup")

        assert "No cups detected" in result

    def test_objects_detected(
        self, tool, mock_connector, mock_get_grabbing_point_tool, identity_transform
    ):
        """Test successful object detection and position calculation."""
        # Mock grabbing point tool to return camera frame positions
        mock_get_grabbing_point_tool._run.return_value = [
            ([0.1, 0.2, 0.3], None),  # (camera_pose, other_data)
            ([0.4, 0.5, 0.6], None),
        ]
        mock_connector.get_transform.return_value = identity_transform

        result = tool._run("cup")

        assert "Centroids of detected cups" in result
        assert "base_link" in result
        assert "Centroid" in result
        # Verify grabbing point tool was called with correct parameters
        mock_get_grabbing_point_tool._run.assert_called_once_with(
            camera_topic="/camera/rgb/image_raw",
            depth_topic="/camera/depth/image_raw",
            camera_info_topic="/camera/rgb/camera_info",
            object_name="cup",
        )
