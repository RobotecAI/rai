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

"""Model-agnostic detection service.

It reads the model name from ROS2 parameters and uses the detection model registry
to dynamically load the appropriate detection algorithm.
"""

from pathlib import Path
from typing import Optional

from rai.communication.ros2 import ROS2Connector

from rai_interfaces.msg import RAIDetectionArray
from rai_perception.models.detection import get_model
from rai_perception.services.base_vision_service import BaseVisionService


class DetectionService(BaseVisionService):
    """Model-agnostic detection service that uses the detection model registry.

    Reads ROS2 parameters:
    - model_name: Detection model to use (default: "grounding_dino")
    - service_name: ROS2 service name to expose (default: "/detection")

    Note: Currently uses hardcoded weights for grounding_dino. Future enhancement:
    move weights URL to registry for full model-agnostic support.
    """

    WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    WEIGHTS_FILENAME = "groundingdino_swint_ogc.pth"

    def __init__(
        self,
        weights_root_path: str | Path = Path.home() / Path(".cache/rai"),
        ros2_name: str = "detection_service",
        ros2_connector: Optional[ROS2Connector] = None,
    ):
        # TODO: After agents are deprecated, make ros2_connector a required parameter
        super().__init__(weights_root_path, ros2_name, ros2_connector=ros2_connector)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize detection model from registry based on ROS2 parameter."""
        self._boxer, _ = self._initialize_model_from_registry(
            get_model, "grounding_dino", "detection"
        )

    def run(self):
        """Start the ROS2 service."""
        service_name = self._get_service_name("/detection")
        self._create_service(
            service_name,
            self._classify_callback,
            "rai_interfaces/srv/RAIGroundingDino",
            "Detection",
        )

    def _classify_callback(self, request, response: RAIDetectionArray):
        """Handle detection service requests."""
        self.logger.info(
            f"Request received: {request.classes}, {request.box_threshold}, {request.text_threshold}"
        )

        class_array = request.classes.split(",")
        class_array = [class_name.strip() for class_name in class_array]
        class_dict = {class_name: i for i, class_name in enumerate(class_array)}

        boxes = self._boxer.get_boxes(
            request.source_img,
            class_array,
            request.box_threshold,
            request.text_threshold,
        )

        ts = self.ros2_connector.node.get_clock().now().to_msg()
        response.detections.detections = [  # type: ignore
            box.to_detection_msg(class_dict, ts)
            for box in boxes  # type: ignore
        ]
        response.detections.header.stamp = ts  # type: ignore
        response.detections.detection_classes = class_array  # type: ignore

        return response
