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

"""Tests for SegmentationService.

Tests the model-agnostic segmentation service that uses the segmentation model registry.
"""

from unittest.mock import patch

import pytest
from rai_perception.services.segmentation_service import SegmentationService

from rai_interfaces.srv import RAIGroundedSam
from tests.rai_perception.test_helpers import (
    create_segmentation_request,
    create_test_detection2d,
    get_segmentation_weights_path,
    patch_segmentation_service_dependencies,
    setup_service_parameters,
)
from tests.rai_perception.test_mocks import EmptySegmenter, MockGDSegmenter

# Service name default changed from "grounded_sam_segment" to "/segmentation"
SEGMENTATION_SERVICE_NAME = "/segmentation"


class TestSegmentationService:
    """Tests for SegmentationService."""

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test initialization."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_service_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounded_sam", SEGMENTATION_SERVICE_NAME
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            assert instance._segmenter is not None
            instance.stop()

    @pytest.mark.timeout(10)
    def test_segment_callback_empty_detections(self, tmp_path, mock_connector):
        """Test segment callback with empty detections."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_service_dependencies(
            mock_connector, EmptySegmenter, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounded_sam", SEGMENTATION_SERVICE_NAME
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            request = create_segmentation_request()
            response = RAIGroundedSam.Response()
            result = instance._segment_callback(request, response)

            assert len(result.masks) == 0
            instance.stop()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_service_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounded_sam", SEGMENTATION_SERVICE_NAME
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            with patch.object(
                instance.ros2_connector, "create_service"
            ) as mock_create_service:
                instance.run()

                mock_create_service.assert_called_once()
                call_args = mock_create_service.call_args
                assert (
                    call_args[1].get("service_type")
                    == "rai_interfaces/srv/RAIGroundedSam"
                    or call_args[0][2] == "rai_interfaces/srv/RAIGroundedSam"
                )

            instance.stop()

    @pytest.mark.timeout(10)
    def test_segment_callback(self, tmp_path, mock_connector):
        """Test segment callback processes request correctly."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_service_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounded_sam", SEGMENTATION_SERVICE_NAME
            )

            instance = SegmentationService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            detection1 = create_test_detection2d(30.0, 30.0, 40.0, 40.0)
            detection2 = create_test_detection2d(75.0, 75.0, 30.0, 30.0)
            request = create_segmentation_request([detection1, detection2])
            response = RAIGroundedSam.Response()
            result = instance._segment_callback(request, response)

            assert len(result.masks) == 2
            assert result is response

            instance.stop()
