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


from typing import List

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino


class GDClientExample(Node):
    def __init__(self, detection_classes: List[str]):
        super().__init__(node_name="GDClientExample", parameter_overrides=[])
        self.declare_parameter("image_path", "")
        self.cli = self.create_client(RAIGroundingDino, "/detection")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service /detection not available, waiting again...")
        self.req = RAIGroundingDino.Request()
        self.detection_classes = detection_classes
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
        self.req.classes = ", ".join(self.detection_classes)
        self.req.box_threshold = 0.4
        self.req.text_threshold = 0.4

        self.future = self.cli.call_async(self.req)


class GSClientExample(Node):
    def __init__(self):
        super().__init__(node_name="GSClientExample", parameter_overrides=[])
        self.cli = self.create_client(RAIGroundedSam, "/segmentation")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "service /segmentation not available, waiting again..."
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


def draw_bounding_box(img, detection, color=(0, 255, 0)):
    """Draw a single bounding box with label on the image."""
    bbox = detection.bbox
    class_name = detection.results[0].hypothesis.class_id
    confidence = detection.results[0].hypothesis.score

    # Calculate coordinates
    x1 = int(bbox.center.position.x - bbox.size_x / 2)
    y1 = int(bbox.center.position.y - bbox.size_y / 2)
    x2 = int(bbox.center.position.x + bbox.size_x / 2)
    y2 = int(bbox.center.position.y + bbox.size_y / 2)

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Add label
    label = f"{class_name}: {confidence:.2f}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(
        img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1
    )
    cv2.putText(
        img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )


def overlay_mask(img, mask_msg, bridge, mask_index):
    """Overlay a single mask on the image with a unique color."""
    # Convert ROS2 Image message to numpy array
    mask = bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")

    # Use different colors for different masks
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
    color = colors[mask_index % len(colors)]

    # Create colored mask
    color_mask = np.zeros_like(img)
    color_mask[mask > 0] = color

    # Blend with original image
    return cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)


def create_visualization(image, masks, detections, gsam_client):
    """Create final visualization with masks and bounding boxes."""
    img = image.copy()

    # Draw all bounding boxes
    for detection in detections.detections:
        draw_bounding_box(img, detection)

    # Overlay all masks
    for i, mask_msg in enumerate(masks):
        img = overlay_mask(img, mask_msg, gsam_client.bridge, i)

    return img


def wait_for_detection(gdino_client):
    """Wait for GroundingDINO detection to complete and return results."""
    while rclpy.ok():
        rclpy.spin_once(gdino_client)
        if gdino_client.future.done():
            try:
                gdino_response = gdino_client.future.result()
                gdino_client.get_logger().info(
                    f"Number of detections: {len(gdino_response.detections.detections)}"
                )

                # Log detection details
                for i, detection in enumerate(gdino_response.detections.detections):
                    class_name = detection.results[0].hypothesis.class_id
                    confidence = detection.results[0].hypothesis.score
                    bbox = detection.bbox
                    gdino_client.get_logger().info(
                        f"Detection {i}: {class_name} (conf: {confidence:.3f}) "
                        f"at ({bbox.center.position.x:.1f}, {bbox.center.position.y:.1f}) "
                        f"size {bbox.size_x:.1f}x{bbox.size_y:.1f}"
                    )
                return gdino_response
            except Exception as e:
                gdino_client.get_logger().error(f"Detection failed: {e}")
                return None
    return None


def wait_for_segmentation(gsam_client):
    """Wait for GroundedSAM segmentation to complete and return results."""
    while rclpy.ok():
        rclpy.spin_once(gsam_client)
        if gsam_client.future.done():
            try:
                gsam_response = gsam_client.future.result()
                gsam_client.get_logger().info(
                    f"Number of masks: {len(gsam_response.masks)}"
                )
                return gsam_response
            except Exception as e:
                gsam_client.get_logger().error(f"Segmentation failed: {e}")
                return None
    return None


def main(args=None):
    rclpy.init(args=args)

    # Initialize clients
    gdino_client = GDClientExample(detection_classes=["dragon", "lizard", "dinosaur"])
    gsam_client = GSClientExample()

    try:
        # Stage 1: Object Detection
        gdino_client.send_request()
        gdino_response = wait_for_detection(gdino_client)

        if gdino_response is None:
            gdino_client.get_logger().error("Detection failed, exiting")
            return

        # Stage 2: Object Segmentation
        gsam_client.send_request(gdino_client.get_image_path(), gdino_response)
        gsam_response = wait_for_segmentation(gsam_client)

        if gsam_response is None:
            gsam_client.get_logger().error("Segmentation failed, exiting")
            return

        # Stage 3: Create Visualization
        img = cv2.imread(gdino_client.get_image_path())
        result_img = create_visualization(
            img, gsam_response.masks, gdino_response.detections, gsam_client
        )
        cv2.imwrite("masks.png", result_img)
        print("Visualization saved to masks.png")

    finally:
        gdino_client.destroy_node()
        gsam_client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
