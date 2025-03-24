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


import os
import subprocess
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node

from rai_interfaces.srv import RAIGroundedSam
from rai_open_set_vision.vision_markup.segmenter import GDSegmenter

GSAM_NODE_NAME = "grounded_sam"
GSAM_SERVICE_NAME = "grounded_sam_segment"


# TODO: Create a base class for vision services
class GSamService(Node):
    WEIGHTS_URL = (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    )
    WEIGHTS_FILENAME = "sam2_hiera_large.pt"

    def __init__(self):
        super().__init__(node_name=GSAM_NODE_NAME, parameter_overrides=[])

        self.declare_parameter("weights_path", "")
        try:
            weight_path = self.get_parameter("weights_path").value
            assert isinstance(weight_path, str)
            if self.get_parameter("weights_path").value == "":
                weight_path = self._init_weight_path()
            try:
                self.segmenter = GDSegmenter(weight_path)
            except Exception:
                self.get_logger().error(
                    "Could not load model. The weights might be corrupted. Redownloading..."
                )
                self._remove_weights(weight_path)
                weight_path = self._init_weight_path()
                self.segmenter = GDSegmenter(weight_path)

        except Exception:
            self.get_logger().error("Could not load model")
            raise Exception("Could not load model")

        self.srv = self.create_service(
            RAIGroundedSam, GSAM_SERVICE_NAME, self.segment_callback
        )

    def _init_weight_path(self):
        try:
            found_path = get_package_share_directory("rai_open_set_vision")
            install_path = (
                Path(found_path.strip()) / "share" / "weights" / self.WEIGHTS_FILENAME
            )
            # make sure the file exists
            if install_path.exists():
                return install_path
            else:
                self._download_weights(install_path)
                return install_path

        except Exception:
            self.get_logger().error("Could not find package path")
            raise Exception("Could not find package path")

    def _download_weights(self, path: Path):
        try:
            os.makedirs(path.parent, exist_ok=True)
            subprocess.run(
                [
                    "wget",
                    self.WEIGHTS_URL,
                    "-O",
                    path,
                    "--progress=dot:giga",
                ]
            )
        except Exception:
            self.get_logger().error("Could not download weights")
            raise Exception("Could not download weights")

    def _remove_weights(self, path: Path):
        if path.exists():
            os.remove(path)

    def segment_callback(self, request, response: RAIGroundedSam.Response):
        received_boxes = []
        for detection in request.detections.detections:
            received_boxes.append(detection.bbox)

        image = request.source_img

        assert self.segmenter is not None
        masks = self.segmenter.get_segmentation(image, received_boxes)
        bridge = CvBridge()
        img_arr = []
        for mask in masks:
            if len(mask.shape) > 2:  # Check if the mask has multiple channels
                mask = np.squeeze(mask)
            arr = (mask * 255).astype(np.uint8)  # Convert binary 0/1 to 0/255
            img_arr.append(bridge.cv2_to_imgmsg(arr, encoding="mono8"))

        response.masks = img_arr
        return response


def main(args=None):
    rclpy.init(args=args)

    gsam_service = GSamService()

    try:
        rclpy.spin(gsam_service)
    except KeyboardInterrupt:
        gsam_service.get_logger().info("Shutting down")
    except Exception as e:
        gsam_service.get_logger().error(f"Error: {e}")
    finally:
        gsam_service.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
