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
# See the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rclpy
import sensor_msgs.msg
from rai_perception.tools.segmentation_tools import (
    GetGrabbingPointTool,
    GetSegmentationTool,
)
from rclpy.parameter import Parameter

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino


class TestGetSegmentationTool:
    """Test cases for GetSegmentationTool."""

    @pytest.fixture
    def segmentation_tool(self, mock_connector):
        """Create a GetSegmentationTool instance."""
        tool = GetSegmentationTool()
        tool.connector = mock_connector
        # Set actual float values since GetSegmentationTool uses Field annotations
        # but isn't a Pydantic model, so self.box_threshold would be a Field object
        tool.box_threshold = 0.35
        tool.text_threshold = 0.45
        return tool

    def test_get_image_message_success(self, segmentation_tool, mock_connector):
        """Test _get_image_message with valid image."""
        image_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.return_value.payload = image_msg

        result = segmentation_tool._get_image_message("test_topic")

        assert result == image_msg

    def test_get_detection_service_name_default(
        self, segmentation_tool, mock_connector
    ):
        """Test _get_detection_service_name returns default when parameter not set."""
        # Parameter not set, so will use default
        service_name = segmentation_tool._get_detection_service_name()
        assert service_name == "/detection"

    def test_get_segmentation_service_name_default(
        self, segmentation_tool, mock_connector
    ):
        """Test _get_segmentation_service_name returns default when parameter not set."""
        # Parameter not set, so will use default
        service_name = segmentation_tool._get_segmentation_service_name()
        assert service_name == "/segmentation"

    def test_call_gdino_node(self, segmentation_tool, mock_connector):
        """Test _call_gdino_node creates service call."""
        image_msg = sensor_msgs.msg.Image()
        mock_client = MagicMock()
        mock_client.wait_for_service.return_value = True
        mock_connector.node.create_client.return_value = mock_client

        # Parameter not set, so will use default service name
        future = segmentation_tool._call_gdino_node(image_msg, "dinosaur")

        assert future is not None
        mock_connector.node.create_client.assert_called_once()
        mock_client.wait_for_service.assert_called_once()
        mock_client.call_async.assert_called_once()

    def test_call_gsam_node(self, segmentation_tool, mock_connector):
        """Test _call_gsam_node creates service call."""
        image_msg = sensor_msgs.msg.Image()
        gdino_response = RAIGroundingDino.Response()
        from rai_interfaces.msg import RAIDetectionArray

        gdino_response.detections = RAIDetectionArray()

        mock_client = MagicMock()
        mock_client.wait_for_service.return_value = True
        mock_connector.node.create_client.return_value = mock_client

        # Parameter not set, so will use default service name
        future = segmentation_tool._call_gsam_node(image_msg, gdino_response)

        assert future is not None
        mock_connector.node.create_client.assert_called_once()
        mock_client.wait_for_service.assert_called_once()
        mock_client.call_async.assert_called_once()

    def test_run_success(self, segmentation_tool, mock_connector):
        """Test _run method with successful segmentation."""
        image_msg = sensor_msgs.msg.Image()
        mock_connector.receive_message.return_value.payload = image_msg

        mock_gdino_client = MagicMock()
        mock_gdino_client.wait_for_service.return_value = True
        mock_gsam_client = MagicMock()
        mock_gsam_client.wait_for_service.return_value = True

        def create_client_side_effect(service_type, service_name):
            if "GroundingDino" in str(service_type):
                return mock_gdino_client
            return mock_gsam_client

        mock_connector.node.create_client.side_effect = create_client_side_effect

        # Service name parameters not set, so will use defaults

        gdino_response = RAIGroundingDino.Response()
        from rai_interfaces.msg import RAIDetectionArray

        gdino_response.detections = RAIDetectionArray()

        gsam_response = RAIGroundedSam.Response()
        mask_msg1 = sensor_msgs.msg.Image()
        mask_msg1.encoding = "mono8"  # Set encoding to avoid cv_bridge errors
        mask_msg2 = sensor_msgs.msg.Image()
        mask_msg2.encoding = "mono8"  # Set encoding to avoid cv_bridge errors
        gsam_response.masks = [mask_msg1, mask_msg2]

        # Set ROS2 parameters
        mock_connector.node.set_parameters(
            [
                Parameter(
                    "conversion_ratio", rclpy.parameter.Parameter.Type.DOUBLE, 0.001
                ),
            ]
        )

        with (
            patch(
                "rai_perception.tools.segmentation_tools.get_future_result"
            ) as mock_get_result,
            patch(
                "rai_perception.tools.segmentation_tools.convert_ros_img_to_base64"
            ) as mock_convert,
        ):
            mock_get_result.side_effect = [gdino_response, gsam_response]
            mock_convert.side_effect = ["base64_1", "base64_2"]

            with patch("rclpy.ok", return_value=True):
                result_text, result_data = segmentation_tool._run(
                    "camera_topic", "dinosaur"
                )

                assert result_text == ""
                assert "segmentations" in result_data
                assert len(result_data["segmentations"]) == 2


class TestGetGrabbingPointTool:
    """Test cases for GetGrabbingPointTool."""

    @pytest.fixture
    def grabbing_tool(self, mock_connector):
        """Create a GetGrabbingPointTool instance."""
        # Use model_construct to bypass Pydantic validation for connector field
        tool = GetGrabbingPointTool.model_construct(connector=mock_connector)
        return tool

    def test_get_camera_info_message(self, grabbing_tool, mock_connector):
        """Test _get_camera_info_message."""
        camera_info = sensor_msgs.msg.CameraInfo()
        mock_connector.receive_message.return_value.payload = camera_info

        result = grabbing_tool._get_camera_info_message("camera_info_topic")

        assert result == camera_info

    def test_get_intrinsic_from_camera_info(self, grabbing_tool):
        """Test _get_intrinsic_from_camera_info extracts parameters."""
        camera_info = sensor_msgs.msg.CameraInfo()
        camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

        fx, fy, cx, cy = grabbing_tool._get_intrinsic_from_camera_info(camera_info)

        assert fx == 500.0
        assert fy == 500.0
        assert cx == 320.0
        assert cy == 240.0

    def test_process_mask(self, grabbing_tool):
        """Test _process_mask calculates centroid and rotation."""
        mask_msg = sensor_msgs.msg.Image()
        depth_msg = sensor_msgs.msg.Image()

        # Create mock mask (100x100 with a square region)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        # Create mock depth (1 meter = 1000mm)
        depth = np.ones((100, 100), dtype=np.uint16) * 1000

        intrinsic = (500.0, 500.0, 50.0, 50.0)  # fx, fy, cx, cy

        with patch(
            "rai_perception.tools.segmentation_tools.convert_ros_img_to_ndarray"
        ) as mock_convert:
            mock_convert.side_effect = [mask, depth]

            with patch("cv2.minAreaRect") as mock_min_area:
                mock_min_area.return_value = (
                    (50.0, 50.0),
                    (60.0, 60.0),
                    0.0,
                )  # center, dimensions, angle

                centroid, rotation = grabbing_tool._process_mask(
                    mask_msg, depth_msg, intrinsic, depth_to_meters_ratio=0.001
                )

                assert len(centroid) == 3
                assert isinstance(rotation, (int, float))

    def test_get_detection_service_name_default(self, grabbing_tool, mock_connector):
        """Test _get_detection_service_name returns default when parameter not set."""
        # Parameter not set, so will use default
        service_name = grabbing_tool._get_detection_service_name()
        assert service_name == "/detection"

    def test_get_segmentation_service_name_default(self, grabbing_tool, mock_connector):
        """Test _get_segmentation_service_name returns default when parameter not set."""
        # Parameter not set, so will use default
        service_name = grabbing_tool._get_segmentation_service_name()
        assert service_name == "/segmentation"

    def test_run(self, grabbing_tool, mock_connector):
        """Test GetGrabbingPointTool._run."""
        import rclpy
        from rclpy.parameter import Parameter

        image_msg = sensor_msgs.msg.Image()
        depth_msg = sensor_msgs.msg.Image()
        camera_info = sensor_msgs.msg.CameraInfo()
        camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

        mock_connector.receive_message.side_effect = [
            MagicMock(payload=image_msg),
            MagicMock(payload=depth_msg),
            MagicMock(payload=camera_info),
        ]

        mock_gdino_client = MagicMock()
        mock_gdino_client.wait_for_service.return_value = True
        mock_gsam_client = MagicMock()
        mock_gsam_client.wait_for_service.return_value = True

        def create_client_side_effect(service_type, service_name):
            if "GroundingDino" in str(service_type):
                return mock_gdino_client
            return mock_gsam_client

        mock_connector.node.create_client.side_effect = create_client_side_effect

        # Set conversion_ratio parameter (service name parameters will use defaults)
        mock_connector.node.set_parameters(
            [
                Parameter(
                    "conversion_ratio",
                    rclpy.parameter.Parameter.Type.DOUBLE,
                    0.001,
                )
            ]
        )

        gdino_response = RAIGroundingDino.Response()
        from rai_interfaces.msg import RAIDetectionArray

        gdino_response.detections = RAIDetectionArray()

        gsam_response = RAIGroundedSam.Response()
        mask_msg = sensor_msgs.msg.Image()
        mask_msg.encoding = "mono8"  # Set encoding to avoid cv_bridge errors
        gsam_response.masks = [mask_msg]

        # Set ROS2 parameters
        mock_connector.node.set_parameters(
            [
                Parameter(
                    "conversion_ratio", rclpy.parameter.Parameter.Type.DOUBLE, 0.001
                ),
            ]
        )

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        depth = np.ones((100, 100), dtype=np.uint16) * 1000

        call_count = [0]  # Use list to allow modification in nested function

        def convert_side_effect(msg):
            """Return appropriate array based on call order: first call is mask, second is depth."""
            call_count[0] += 1
            if call_count[0] == 1:
                return mask
            else:
                return depth

        # Patch convert_ros_img_to_base64 to avoid cv2 errors - GetGrabbingPointTool._run
        # calls this function which internally uses cv2.cvtColor on empty mock images
        with (
            patch(
                "rai_perception.tools.segmentation_tools.get_future_result"
            ) as mock_get_result,
            patch(
                "rai_perception.tools.segmentation_tools.convert_ros_img_to_ndarray",
                side_effect=convert_side_effect,
            ),
            patch(
                "rai_perception.tools.segmentation_tools.convert_ros_img_to_base64",
                return_value="mock_base64_string",
            ),
            patch("cv2.minAreaRect") as mock_min_area,
            patch(
                "rai.communication.ros2.api.convert_ros_img_to_ndarray",
                side_effect=convert_side_effect,
            ),
        ):
            mock_get_result.side_effect = [gdino_response, gsam_response]
            mock_min_area.return_value = ((50.0, 50.0), (60.0, 60.0), 0.0)

            result = grabbing_tool._run(
                "camera_topic", "depth_topic", "camera_info_topic", "dinosaur"
            )

            assert isinstance(result, list)
            assert len(result) == 1
            assert len(result[0]) == 2  # (centroid, rotation)
