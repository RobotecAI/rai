from os import PathLike
from typing import Dict

import torch
from cv_bridge import CvBridge
from groundingdino.util.inference import Model
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from torchvision.ops.boxes import box_convert
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)


class Box:
    def __init__(self, center, size_x, size_y, phrase, confidence):
        self.phrase = phrase
        self.center = center
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.confidence = float(confidence)

    def to_detection_msg(
        self, class_dict: Dict[str, int], timestamp: Time
    ) -> Detection2D:
        detection = Detection2D()
        detection.header = Header()
        detection.header.stamp = timestamp
        detection.results = []
        hypothesis_with_pose = ObjectHypothesisWithPose()
        hypothesis_with_pose.hypothesis = ObjectHypothesis()
        hypothesis_with_pose.hypothesis.class_id = self.phrase
        hypothesis_with_pose.hypothesis.score = self.confidence
        detection.results.append(hypothesis_with_pose)
        detection.bbox = BoundingBox2D()
        detection.bbox.center.position.x = self.center[0]
        detection.bbox.center.position.y = self.center[1]
        detection.bbox.size_x = self.size_x
        detection.bbox.size_y = self.size_y
        return detection


class GDBoxer:
    def __init__(
        self,
        cfg_path: str
        | PathLike = "./src/rai_grounding_dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weight_path: str
        | PathLike = "./src/rai_grounding_dino/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        use_cuda: bool = True,
    ):
        self.cfg_path = str(cfg_path)
        self.weight_path = str(weight_path)
        if not use_cuda:
            self.model = Model(self.cfg_path, self.weight_path, device="cpu")
        else:
            self.model = Model(self.cfg_path, self.weight_path)
        self.bridge = CvBridge()

    def get_boxes(
        self,
        image_msg: Image,
        classes: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[Box]:
        image = self.bridge.imgmsg_to_cv2(
            image_msg, desired_encoding=image_msg.encoding
        )

        predictions = self.model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        print("Predictions: ", predictions)

        # boxes, logits, phrases = predict(
        #     model=self.model,
        #     image=image,
        #     caption=text_prompt,
        #     box_threshold=box_threshold,
        #     text_threshold=text_threshold,
        # )
        # # save for future reference
        # self.boxes = boxes
        # self.logits = logits
        # self.phrases = phrases
        # print("Boxes: ", boxes)
        # print("Phrases: ", phrases)
        # print("Logits: ", logits)
        packed_boxes = []
        # h, w, _ = image.shape
        # boxes = boxes * torch.Tensor([w, h, w, h])
        # xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
        xyxy = predictions.xyxy
        class_ids = predictions.class_id
        confidences = predictions.confidence
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            phrase = classes[class_ids[i]]
            confidence = confidences[i]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size_x = x2 - x1
            size_y = y2 - y1
            packed_boxes.append(Box(center, size_x, size_y, phrase, confidence))
        return packed_boxes
