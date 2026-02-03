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

"""Tests for DetectionService.

Tests the model-agnostic detection service that uses the detection model registry.
"""

from unittest.mock import patch

import pytest
from rai_perception.services.detection_service import DetectionService

from tests.rai_perception.conftest import (
    setup_mock_clock,
)
from tests.rai_perception.test_helpers import (
    create_detection_request,
    get_detection_weights_path,
    patch_detection_service_dependencies,
    setup_service_parameters,
)
from tests.rai_perception.test_mocks import EmptyBoxer, MockGDBoxer

# Service name default changed from "grounding_dino_classify" to "/detection"
DETECTION_SERVICE_NAME = "/detection"


class TestDetectionService:
    """Tests for DetectionService."""

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test initialization."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_service_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounding_dino", DETECTION_SERVICE_NAME
            )

            instance = DetectionService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            assert instance._boxer is not None
            instance.stop()

    @pytest.mark.timeout(10)
    def test_classify_callback_empty_boxes(self, tmp_path, mock_connector):
        """Test classify callback with no detections."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_service_dependencies(
            mock_connector, EmptyBoxer, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounding_dino", DETECTION_SERVICE_NAME
            )

            instance = DetectionService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = create_detection_request("dinosaur")
            response = RAIGroundingDino.Response()

            setup_mock_clock(instance)
            result = instance._classify_callback(request, response)

            assert len(result.detections.detections) == 0
            assert result.detections.detection_classes == ["dinosaur"]

            instance.stop()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_service_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounding_dino", DETECTION_SERVICE_NAME
            )

            instance = DetectionService(
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
                    call_args[0][0] == DETECTION_SERVICE_NAME
                    or call_args[0][0] == "/detection"
                )
                assert (
                    call_args[1].get("service_type")
                    == "rai_interfaces/srv/RAIGroundingDino"
                    or call_args[0][2] == "rai_interfaces/srv/RAIGroundingDino"
                )

            instance.stop()

    @pytest.mark.timeout(10)
    def test_classify_callback(self, tmp_path, mock_connector):
        """Test classify callback processes request correctly."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_service_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            setup_service_parameters(
                mock_connector, "grounding_dino", DETECTION_SERVICE_NAME
            )

            instance = DetectionService(
                weights_root_path=str(tmp_path),
                ros2_name="test",
                ros2_connector=mock_connector,
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = create_detection_request("dinosaur, dragon")
            response = RAIGroundingDino.Response()

            setup_mock_clock(instance)
            result = instance._classify_callback(request, response)

            # Verify response
            assert len(result.detections.detections) == 2
            assert result.detections.detection_classes == ["dinosaur", "dragon"]
            assert result is response

            instance.stop()
