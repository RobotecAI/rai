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

"""Segmentation algorithm: GDSegmenter.

Low-level segmentation algorithm that loads its own config (self-contained).
"""

import logging
from os import PathLike
from typing import List

import hydra
import numpy as np
import torch
from cv_bridge import CvBridge
from rai.communication.ros2.api import convert_ros_img_to_ndarray
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D

from rai_perception.components.exceptions import PerceptionError


class GDSegmenter:
    """Grounded SAM segmentation algorithm.

    Algorithm loads its own config (self-contained).
    """

    def __init__(
        self,
        weight_path: str | PathLike,
        config_path: str | PathLike | None = None,
        use_cuda: bool = True,
    ):
        """Initialize GDSegmenter.

        Args:
            weight_path: Path to model weights file
            config_path: Ignored. SAM2 uses Hydra config module system internally.
            use_cuda: Whether to use CUDA if available
        """
        self.logger = logging.getLogger(__name__)
        self.weight_path = str(weight_path)

        if use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            if use_cuda:
                self.logger.warning("CUDA is not available but requested, using CPU")
            self.device = "cpu"

        # Initialize Hydra with config module and build SAM2 model
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_module("rai_perception.configs")

        try:
            self.sam2_model = build_sam2(
                "seg_config.yml", self.weight_path, device=self.device
            )
        except Exception as e:
            raise PerceptionError(f"Failed to build SAM2 model: {str(e)}") from e

        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.bridge = CvBridge()

    def _get_boxes_xyxy(self, bboxes: List[BoundingBox2D]) -> List[np.ndarray]:
        """Convert ROS2 bounding boxes to xyxy format."""
        data = []
        for bbox in bboxes:
            center_x = bbox.center.position.x
            center_y = bbox.center.position.y
            data.append(
                np.array(
                    [
                        center_x - bbox.size_x / 2,
                        center_y - bbox.size_y / 2,
                        center_x + bbox.size_x / 2,
                        center_y + bbox.size_y / 2,
                    ]
                )
            )
        return data

    def get_segmentation(
        self, image_msg: Image, ros_bboxes: List[BoundingBox2D]
    ) -> List[np.ndarray]:
        """Generate segmentation masks for bounding boxes.

        Args:
            image_msg: ROS2 Image message
            ros_bboxes: List of ROS2 BoundingBox2D messages

        Returns:
            List of segmentation masks as numpy arrays
        """
        img_array = convert_ros_img_to_ndarray(image_msg, image_msg.encoding)
        self.sam2_predictor.set_image(img_array)
        bboxes = self._get_boxes_xyxy(ros_bboxes)

        all_masks: List[np.ndarray] = []
        for box in bboxes:
            mask, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            all_masks.append(mask)

        return all_masks
