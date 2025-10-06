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


import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino


class GDClientExample(Node):
    def __init__(self):
        super().__init__(node_name="GDClientExample", parameter_overrides=[])
        self.declare_parameter("image_path", "")
        self.cli = self.create_client(RAIGroundingDino, "grounding_dino_classify")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "service grounding_dino_classify not available, waiting again..."
            )
        self.req = RAIGroundingDino.Request()
        self.bridge = CvBridge()

    def get_image_path(self) -> str:
        image_path = self.get_parameter("image_path").value
        assert isinstance(image_path, str)
        return image_path

    def send_request(self):
        image_path = self.get_parameter("image_path").value
        assert isinstance(image_path, str)
        img = cv2.imread(image_path)
        # convert img to numpy array
        img = np.array(img)
        self.req.source_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.req.classes = "dragon , lizard , dinosaur"
        self.req.box_threshold = 0.4
        self.req.text_threshold = 0.4

        self.future = self.cli.call_async(self.req)


class GSClientExample(Node):
    def __init__(self):
        super().__init__(node_name="GSClientExample", parameter_overrides=[])
        self.cli = self.create_client(RAIGroundedSam, "grounded_sam_segment")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "service grounded_sam_segment not available, waiting again..."
            )
        self.req = RAIGroundedSam.Request()
        self.bridge = CvBridge()

    def send_request(self, image_path: str, data: RAIGroundingDino.Response):
        self.req.detections = data.detections
        img = cv2.imread(image_path)
        # convert img to numpy array
        img = np.array(img)
        self.req.source_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    gdino_client = GDClientExample()
    gdino_client.send_request()

    gsam_client = GSClientExample()

    response = None
    while rclpy.ok():
        rclpy.spin_once(gdino_client)
        if gdino_client.future.done():
            try:
                response: RAIGroundingDino.Response = gdino_client.future.result()  # type: ignore
            except Exception as e:
                gdino_client.get_logger().info("Service call failed %r" % (e,))
            else:
                assert response is not None
                gdino_client.get_logger().info(f"{response.detections}")  # CHANGE
            break
    assert response is not None
    gsam_client.send_request(gdino_client.get_image_path(), response)
    gsam_client.get_logger().info("making segmentation request")
    while rclpy.ok():
        rclpy.spin_once(gsam_client)
        if gsam_client.future.done():
            try:
                gsam_client.get_logger().info("request finished")
                response: RAIGroundedSam.Response = gsam_client.future.result()  # type: ignore
                gsam_client.get_logger().info(f"response: {response}")
            except Exception as e:
                gsam_client.get_logger().info("Service call failed %r" % (e,))
            else:
                assert response is not None
                gsam_client.get_logger().info(f"{response.masks}")  # CHANGE
            break

    gdino_client.destroy_node()
    gsam_client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
