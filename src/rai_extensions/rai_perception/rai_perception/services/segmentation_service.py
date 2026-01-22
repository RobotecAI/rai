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
    - enable_legacy_service_names: Register legacy service name "/grounded_sam_segment"
      for backward compatibility (default: True)

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

        # Register legacy service name for backward compatibility
        if self._should_register_legacy_name():
            legacy_service_name = "/grounded_sam_segment"
            self._create_service(
                legacy_service_name,
                self._segment_callback,
                "rai_interfaces/srv/RAIGroundedSam",
                "Segmentation (legacy)",
            )
            self.logger.info(
                f"Legacy service name '{legacy_service_name}' registered for backward compatibility. "
                f"Consider migrating to '{service_name}'."
            )

    def _should_register_legacy_name(self) -> bool:
        """Check if legacy service name should be registered.

        Reads ROS2 parameter: enable_legacy_service_names
        Default: True (for backward compatibility)

        Returns:
            True if legacy service name should be registered, False otherwise
        """
        from rai.communication.ros2 import get_param_value

        return get_param_value(
            self.ros2_connector.node,
            "enable_legacy_service_names",
            default=True,
        )

    def _segment_callback(self, request, response: RAIGroundedSam.Response):
        """Handle segmentation service requests."""
        try:
            # Validate image
            image_data_size = (
                len(request.source_img.data) if request.source_img.data else 0
            )

            if not request.source_img.data or image_data_size == 0:
                self.logger.error("Received empty image data in segmentation request")
                response.masks = []
                return response

            # Validate detections
            num_detections = (
                len(request.detections.detections)
                if request.detections.detections
                else 0
            )

            if num_detections == 0:
                self.logger.warning("Received empty detections in segmentation request")
                response.masks = []
                return response

            received_boxes = []
            for detection in request.detections.detections:
                received_boxes.append(detection.bbox)

            image = request.source_img

            if self._segmenter is None:
                raise RuntimeError("Segmentation model not initialized")

            masks = self._segmenter.get_segmentation(image, received_boxes)

            if masks is None:
                masks = []

            img_arr = []
            total_masks = len(masks)
            successful_masks = 0
            for mask in masks:
                try:
                    if len(mask.shape) > 2:
                        mask = np.squeeze(mask)
                    arr = (mask * 255).astype(np.uint8)
                    img_arr.append(self._bridge.cv2_to_imgmsg(arr, encoding="mono8"))
                    successful_masks += 1
                except Exception as mask_error:
                    self.logger.error(
                        f"Error processing mask: {mask_error}",
                        exc_info=True,
                    )

            if total_masks > 0 and successful_masks < total_masks:
                self.logger.warning(
                    f"Processed {successful_masks}/{total_masks} masks successfully. "
                    f"{total_masks - successful_masks} mask(s) failed to process."
                )

            response.masks = img_arr

        except Exception as e:
            self.logger.error(
                f"Error processing segmentation request: {e}", exc_info=True
            )
            response.masks = []

        return response
