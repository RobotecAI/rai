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


class GDSegmenter:
    def __init__(
        self,
        weight_path: str | PathLike,
        use_cuda: bool = True,
    ):
        self.cfg_path = "seg_config.yml"
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_module("rai_perception.configs")

        self.weight_path = str(weight_path)
        if use_cuda:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.sam2_model = build_sam2(
            self.cfg_path, self.weight_path, device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.bridge = CvBridge()

    def _get_boxes_xyxy(self, bboxes: List[BoundingBox2D]) -> List[np.ndarray]:
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
