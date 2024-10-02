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
#

import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rai_open_set_vision.vision_markup.segmenter import GDSegmenter
from rclpy.node import Node
from sensor_msgs.msg import Image

from rai_interfaces.srv import RAIGroundedSam

GSAM_NODE_NAME = "grounded_sam"
GSAM_SERVICE_NAME = "grounded_sam_segment"


# TODO: Create a base class for vision services
class GSamService(Node):
    def __init__(self):
        super().__init__(node_name=GSAM_NODE_NAME, parameter_overrides=[])
        self.srv = self.create_service(
            RAIGroundedSam, GSAM_SERVICE_NAME, self.segment_callback
        )

        self.declare_parameter("weights_path", "")
        try:
            weight_path = self.get_parameter("weights_path").value
            assert isinstance(weight_path, str)
            if self.get_parameter("weights_path").value == "":
                weight_path = self._init_weight_path()
            self.segmenter = GDSegmenter(weight_path)
        except Exception:
            self.get_logger().error("Could not load model")
            raise Exception("Could not load model")

    def _init_weight_path(self):
        try:
            found_path = subprocess.check_output(
                ["ros2", "pkg", "prefix", "rai_open_set_vision"]
            ).decode("utf-8")
            install_path = (
                Path(found_path.strip()) / "share" / "weights" / "sam2_hiera_large.pt"
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
                    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                    "-O",
                    path,
                ]
            )
        except Exception:
            self.get_logger().error("Could not download weights")
            raise Exception("Could not download weights")

    def segment_callback(self, request, response: str) -> List[Image]:
        received_boxes = []
        for detection in request.detections.detections:
            received_boxes.append(detection.bbox)

        image = request.source_img

        assert self.segmenter is not None
        masks = self.segmenter.get_segmentation(image, received_boxes)
        self.get_logger().info(masks)
        bridge = CvBridge()
        img_arr = []
        for mask in masks:
            arr = (mask * 255).astype(np.uint8)  # Convert binary 0/1 to 0/255
            img_arr.append(bridge.cv2_to_imgmsg(arr, encoding="mono8"))

        return img_arr


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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
