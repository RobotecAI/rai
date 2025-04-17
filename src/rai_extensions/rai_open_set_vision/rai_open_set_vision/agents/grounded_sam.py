# Copyright (C) 2024 Robotec.AI
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
from pathlib import Path
from typing import Optional

import numpy as np
from cv_bridge import CvBridge

from rai_interfaces.srv import RAIGroundedSam
from rai_open_set_vision.agents.base_vision_agent import BaseVisionAgent
from rai_open_set_vision.vision_markup.segmenter import GDSegmenter

GSAM_NODE_NAME = "grounded_sam"
GSAM_SERVICE_NAME = "grounded_sam_segment"


class GroundedSamAgent(BaseVisionAgent):
    WEIGHTS_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    )
    WEIGHTS_FILENAME = "sam2_hiera_large.pt"

    def __init__(
        self,
        weights_path: str | Path = Path.home() / Path(".cache/rai"),
        ros2_name: str = GSAM_NODE_NAME,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(weights_path, ros2_name, logger)
        try:
            self._segmenter = GDSegmenter(self._weights_path)
        except Exception:
            self.get_logger().error(
                "Could not load model. The weights might be corrupted. Redownloading..."
            )
            self._remove_weights(self.weight_path)
            self._init_weight_path()
            self.segmenter = GDSegmenter(self.weight_path)

    def run(self):
        self.connectors["ros2"].create_service(
            service_name=GSAM_SERVICE_NAME,
            on_request=self._segment_callback,
            service_type="rai_interfaces/srv/RAIGroundedSam",
        )

    def _segment_callback(self, request, response: RAIGroundedSam.Response):
        received_boxes = []
        for detection in request.detections.detections:
            received_boxes.append(detection.bbox)

        image = request.source_img

        assert self._segmenter is not None
        masks = self._segmenter.get_segmentation(image, received_boxes)
        bridge = CvBridge()
        img_arr = []
        for mask in masks:
            if len(mask.shape) > 2:  # Check if the mask has multiple channels
                mask = np.squeeze(mask)
            arr = (mask * 255).astype(np.uint8)  # Convert binary 0/1 to 0/255
            img_arr.append(bridge.cv2_to_imgmsg(arr, encoding="mono8"))

        response.masks = img_arr
        return response
