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

from rai_perception.scripts.run_perception_services import main


class TestRunPerceptionServices:
    """Test cases for run_perception_services.main function."""

    def test_main_initializes_services(self):
        """Test that main function initializes both services."""
        with (
            patch("rai_perception.scripts.run_perception_services.rclpy") as mock_rclpy,
            patch(
                "rai_perception.scripts.run_perception_services.ROS2Connector"
            ) as mock_connector,
            patch(
                "rai_perception.scripts.run_perception_services.DetectionService"
            ) as mock_detection,
            patch(
                "rai_perception.scripts.run_perception_services.SegmentationService"
            ) as mock_segmentation,
            patch(
                "rai_perception.scripts.run_perception_services.wait_for_shutdown"
            ) as mock_wait,
        ):
            mock_connector_instance = MagicMock()
            mock_connector.return_value = mock_connector_instance
            mock_detection_instance = MagicMock()
            mock_segmentation_instance = MagicMock()
            mock_detection.return_value = mock_detection_instance
            mock_segmentation.return_value = mock_segmentation_instance

            main()

            mock_rclpy.init.assert_called_once()
            assert mock_connector.call_count == 2
            # Verify connectors are created with correct names
            connector_calls = mock_connector.call_args_list
            assert connector_calls[0][0][0] == "detection_service"
            assert connector_calls[1][0][0] == "segmentation_service"
            # Verify executor_type is set correctly
            assert connector_calls[0][1]["executor_type"] == "single_threaded"
            assert connector_calls[1][1]["executor_type"] == "single_threaded"
            mock_detection.assert_called_once()
            mock_segmentation.assert_called_once()
            mock_detection_instance.run.assert_called_once()
            mock_segmentation_instance.run.assert_called_once()
            mock_wait.assert_called_once_with(
                [mock_detection_instance, mock_segmentation_instance]
            )
            mock_rclpy.shutdown.assert_called_once()

    def test_main_calls_services_in_order(self):
        """Test that services are called in the correct order."""
        call_order = []

        def track_connector_init(*args, **kwargs):
            call_order.append("connector_init")
            return MagicMock()

        def track_detection_init(*args, **kwargs):
            call_order.append("detection_init")
            mock_instance = MagicMock()
            mock_instance.run.side_effect = lambda: call_order.append("detection_run")
            return mock_instance

        def track_segmentation_init(*args, **kwargs):
            call_order.append("segmentation_init")
            mock_instance = MagicMock()
            mock_instance.run.side_effect = lambda: call_order.append(
                "segmentation_run"
            )
            return mock_instance

        with (
            patch("rai_perception.scripts.run_perception_services.rclpy"),
            patch(
                "rai_perception.scripts.run_perception_services.ROS2Connector",
                side_effect=track_connector_init,
            ),
            patch(
                "rai_perception.scripts.run_perception_services.DetectionService",
                side_effect=track_detection_init,
            ),
            patch(
                "rai_perception.scripts.run_perception_services.SegmentationService",
                side_effect=track_segmentation_init,
            ),
            patch("rai_perception.scripts.run_perception_services.wait_for_shutdown"),
        ):
            main()

            # Verify order: connectors, init detection, init segmentation, run detection, run segmentation
            assert call_order[:4] == [
                "connector_init",
                "connector_init",
                "detection_init",
                "segmentation_init",
            ]
            assert "detection_run" in call_order
            assert "segmentation_run" in call_order

    def test_main_handles_shutdown(self):
        """Test that main properly shuts down ROS2."""
        with (
            patch("rai_perception.scripts.run_perception_services.rclpy") as mock_rclpy,
            patch(
                "rai_perception.scripts.run_perception_services.ROS2Connector"
            ) as mock_connector,
            patch(
                "rai_perception.scripts.run_perception_services.DetectionService"
            ) as mock_detection,
            patch(
                "rai_perception.scripts.run_perception_services.SegmentationService"
            ) as mock_segmentation,
            patch("rai_perception.scripts.run_perception_services.wait_for_shutdown"),
        ):
            mock_connector.return_value = MagicMock()
            mock_detection.return_value = MagicMock()
            mock_segmentation.return_value = MagicMock()

            main()

            mock_rclpy.init.assert_called_once()
            mock_rclpy.shutdown.assert_called_once()

    def test_main_creates_connectors_with_correct_parameters(self):
        """Test that connectors are created with correct node names and executor type."""
        with (
            patch("rai_perception.scripts.run_perception_services.rclpy"),
            patch(
                "rai_perception.scripts.run_perception_services.ROS2Connector"
            ) as mock_connector,
            patch("rai_perception.scripts.run_perception_services.DetectionService"),
            patch("rai_perception.scripts.run_perception_services.SegmentationService"),
            patch("rai_perception.scripts.run_perception_services.wait_for_shutdown"),
        ):
            mock_connector.return_value = MagicMock()

            main()

            # Verify both connectors are created
            assert mock_connector.call_count == 2

            # Verify first connector (detection) parameters
            first_call = mock_connector.call_args_list[0]
            assert first_call[0][0] == "detection_service"
            assert first_call[1]["executor_type"] == "single_threaded"

            # Verify second connector (segmentation) parameters
            second_call = mock_connector.call_args_list[1]
            assert second_call[0][0] == "segmentation_service"
            assert second_call[1]["executor_type"] == "single_threaded"

    def test_main_passes_connectors_to_services(self):
        """Test that connectors are passed to service constructors."""
        with (
            patch("rai_perception.scripts.run_perception_services.rclpy"),
            patch(
                "rai_perception.scripts.run_perception_services.ROS2Connector"
            ) as mock_connector,
            patch(
                "rai_perception.scripts.run_perception_services.DetectionService"
            ) as mock_detection,
            patch(
                "rai_perception.scripts.run_perception_services.SegmentationService"
            ) as mock_segmentation,
            patch("rai_perception.scripts.run_perception_services.wait_for_shutdown"),
        ):
            detection_connector = MagicMock()
            segmentation_connector = MagicMock()
            mock_connector.side_effect = [detection_connector, segmentation_connector]

            main()

            # Verify DetectionService is called with detection_connector
            mock_detection.assert_called_once()
            detection_call = mock_detection.call_args
            assert detection_call[1]["ros2_connector"] == detection_connector

            # Verify SegmentationService is called with segmentation_connector
            mock_segmentation.assert_called_once()
            segmentation_call = mock_segmentation.call_args
            assert segmentation_call[1]["ros2_connector"] == segmentation_connector
