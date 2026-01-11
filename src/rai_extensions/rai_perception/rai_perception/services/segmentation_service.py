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

"""Model-agnostic segmentation service.

It reads the model name from ROS2 parameters and uses the segmentation model registry
to dynamically load the appropriate segmentation algorithm.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from cv_bridge import CvBridge
from rai.communication.ros2 import ROS2Connector

from rai_interfaces.srv import RAIGroundedSam
from rai_perception.models.segmentation import get_model
from rai_perception.services.base_vision_service import BaseVisionService


class SegmentationService(BaseVisionService):
    """Model-agnostic segmentation service that uses the segmentation model registry.

    Reads ROS2 parameters:
    - model_name: Segmentation model to use (default: "grounded_sam")
    - service_name: ROS2 service name to expose (default: "/segmentation")

    Note: Currently uses hardcoded weights for grounded_sam. Future enhancement:
    move weights URL to registry for full model-agnostic support.
    """

    WEIGHTS_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    )
    WEIGHTS_FILENAME = "sam2_hiera_large.pt"

    def __init__(
        self,
        weights_root_path: str | Path = Path.home() / Path(".cache/rai"),
        ros2_name: str = "segmentation_service",
        ros2_connector: Optional[ROS2Connector] = None,
    ):
        # TODO: After agents are deprecated, make ros2_connector a required parameter
        super().__init__(weights_root_path, ros2_name, ros2_connector=ros2_connector)
        self._bridge = CvBridge()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize segmentation model from registry based on ROS2 parameter."""
        self._segmenter, _ = self._initialize_model_from_registry(
            get_model, "grounded_sam", "segmentation"
        )

    def run(self):
        """Start the ROS2 service."""
        service_name = self._get_service_name("/segmentation")
        self._create_service(
            service_name,
            self._segment_callback,
            "rai_interfaces/srv/RAIGroundedSam",
            "Segmentation",
        )

    def _segment_callback(self, request, response: RAIGroundedSam.Response):
        """Handle segmentation service requests."""
        received_boxes = []
        for detection in request.detections.detections:
            received_boxes.append(detection.bbox)

        image = request.source_img

        assert self._segmenter is not None
        masks = self._segmenter.get_segmentation(image, received_boxes)
        img_arr = []
        for mask in masks:
            if len(mask.shape) > 2:
                mask = np.squeeze(mask)
            arr = (mask * 255).astype(np.uint8)
            img_arr.append(self._bridge.cv2_to_imgmsg(arr, encoding="mono8"))

        response.masks = img_arr
        return response
