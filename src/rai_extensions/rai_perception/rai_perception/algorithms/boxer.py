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

"""Detection algorithm: GDBoxer.

Low-level detection algorithm that loads its own config from config_path
provided by the model registry.
"""

import logging
from os import PathLike
from typing import Dict

import cv2
import torch
from cv_bridge import CvBridge
from groundingdino.util.inference import Model
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)


class Box:
    """Represents a detected bounding box with metadata."""

    def __init__(self, center, size_x, size_y, phrase, confidence):
        self.phrase = phrase
        self.center = center
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.confidence = float(confidence)

    def to_detection_msg(
        self, class_dict: Dict[str, int], timestamp: Time
    ) -> Detection2D:
        """Convert Box to ROS2 Detection2D message."""
        detection = Detection2D()
        detection.header = Header()
        # TODO(juliaj): Investigate why timestamp is sometimes rclpy.time.Time and sometimes
        # builtin_interfaces.msg.Time. The function signature expects rclpy.time.Time, but
        # grounding_dino.py calls .to_msg() before passing it. Should we fix the caller or
        # change the signature to accept Union[rclpy.time.Time, builtin_interfaces.msg.Time]?
        # Handle both rclpy.time.Time (call to_msg()) and builtin_interfaces.msg.Time (use directly)
        if hasattr(timestamp, "to_msg"):
            detection.header.stamp = timestamp.to_msg()
        else:
            # Already a builtin_interfaces.msg.Time
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
    """GroundingDINO detection algorithm.

    Algorithm loads its own config from config_path provided by registry.
    Example: GDBoxer(weights_path, config_path="configs/gdino_config.py")
    """

    def __init__(
        self,
        weight_path: str | PathLike,
        config_path: str | PathLike | None = None,
        use_cuda: bool = True,
    ):
        """Initialize GDBoxer.

        Args:
            weight_path: Path to model weights file
            config_path: Path to config file. If None, uses default location
            use_cuda: Whether to use CUDA if available
        """
        self.logger = logging.getLogger(__name__)
        if config_path is None:
            # Default config path for backward compatibility
            from pathlib import Path

            config_path = Path(__file__).parent.parent / "configs" / "gdino_config.py"
        self.cfg_path = str(config_path)
        self.weight_path = str(weight_path)
        if use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            if use_cuda:
                self.logger.warning("CUDA is not available but requested, using CPU")
            self.device = "cpu"
        self.model = Model(self.cfg_path, self.weight_path, device=self.device)
        self.bridge = CvBridge()

    def get_boxes(
        self,
        image_msg: Image,
        classes: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[Box]:
        """Detect objects in image and return bounding boxes.

        Args:
            image_msg: ROS2 Image message
            classes: List of class names to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold

        Returns:
            List of Box objects with detections
        """
        # TODO: move this to method, or use RAI canonical one
        image = self.bridge.imgmsg_to_cv2(
            image_msg, desired_encoding=image_msg.encoding
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictions = self.model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        packed_boxes = []
        xyxy = predictions.xyxy
        class_ids = predictions.class_id
        confidences = predictions.confidence
        if len(xyxy) == 0:
            return []
        else:
            assert class_ids is not None
            assert confidences is not None

        for i in range(len(xyxy)):
            if class_ids[i] is None:
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            phrase = classes[class_ids[i]]
            confidence = confidences[i]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size_x = x2 - x1
            size_y = y2 - y1
            packed_boxes.append(Box(center, size_x, size_y, phrase, confidence))
        return packed_boxes
