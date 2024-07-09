from typing import TypedDict

import rclpy
from rai_interfaces.msg import RAIDetectionArray
from rai_interfaces.srv import RAIGroundingDino
from rclpy.node import Node
from sensor_msgs.msg import Image

from rai_grounding_dino.boxer import GDBoxer


class GDRequest(TypedDict):
    classes: str
    box_threshold: float
    text_threshold: float
    source_img: Image


class GDinoService(Node):
    def __init__(self):
        super().__init__(node_name="grounding_dino", parameter_overrides=[])
        self.srv = self.create_service(
            RAIGroundingDino, "grounding_dino_classify", self.classify_callback
        )  # CHANGE
        self.boxer = GDBoxer()

    def classify_callback(self, request: GDRequest, response: RAIDetectionArray):
        self.get_logger().info(
            f"Request received: {request.classes}, {request.box_threshold}, {request.text_threshold}"
        )  # CHANGE

        class_array = request.classes.split(",")
        class_array = [class_name.strip() for class_name in class_array]
        class_dict = {class_name: i for i, class_name in enumerate(class_array)}

        boxes = self.boxer.get_boxes(
            request.source_img,
            class_array,
            request.box_threshold,
            request.text_threshold,
        )

        response.detections.detections = [
            box.to_detection_msg(class_dict, self.get_clock().now().to_msg())
            for box in boxes
        ]
        response.detections.header.stamp = self.get_clock().now().to_msg()
        response.detections.detection_classes = class_array

        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = GDinoService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
