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

Grounding DINO open-vocabulary detector, backed by the Hugging Face transformers
implementation (``AutoModelForZeroShotObjectDetection``). The default checkpoint,
``IDEA-Research/grounding-dino-tiny``, is the transformers conversion of the Swin-T OGC
checkpoint (``groundingdino_swint_ogc.pth``) that RAI used previously.
"""

import bisect
import logging
import re
import warnings
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from rai.communication.ros2.image_bridge import CvBridge
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

GDINO_HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# Pinned commit of GDINO_HF_MODEL_ID so a changed upstream repo cannot silently change the model.
GDINO_HF_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"

_HUB_ID_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


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
    """Grounding DINO detector.

    Args:
        weight_path: One of:
            - a local directory with a transformers Grounding DINO checkpoint
              (``config.json``, ``model.safetensors``, tokenizer and preprocessor files),
              as downloaded by ``DetectionService`` / ``download-perception-models``;
            - a Hugging Face Hub model id (e.g. ``"IDEA-Research/grounding-dino-tiny"``);
            - a legacy original-format checkpoint file (``groundingdino_swint_ogc.pth``).
              Those can no longer be loaded, so the equivalent transformers checkpoint
              ``GDINO_HF_MODEL_ID`` is loaded from the Hub instead, with a warning.
        config_path: Ignored. The transformers checkpoint carries its own config.
            Kept for backward compatibility with the model registry interface.
        use_cuda: Whether to use CUDA if available.
    """

    def __init__(
        self,
        weight_path: str | PathLike,
        config_path: str | PathLike | None = None,
        use_cuda: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        if config_path is not None:
            warnings.warn(
                "GDBoxer config_path is ignored: the transformers Grounding DINO "
                "checkpoint carries its own config.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.weight_path = str(weight_path)
        if use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            if use_cuda:
                self.logger.warning("CUDA is not available but requested, using CPU")
            self.device = "cpu"

        source, revision = self._resolve_model_source(self.weight_path)
        self.processor = AutoProcessor.from_pretrained(source, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            source, revision=revision, use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.bridge = CvBridge()

    def _resolve_model_source(self, weight_path: str) -> Tuple[str, Optional[str]]:
        """Map ``weight_path`` to a ``from_pretrained`` source and revision."""
        path = Path(weight_path)
        if path.is_dir():
            return str(path), None
        if path.is_file():
            self.logger.warning(
                f"'{weight_path}' is an original-format Grounding DINO checkpoint, which "
                f"is no longer supported. Loading the equivalent transformers checkpoint "
                f"'{GDINO_HF_MODEL_ID}' from the Hugging Face Hub instead."
            )
            return GDINO_HF_MODEL_ID, GDINO_HF_MODEL_REVISION
        if _HUB_ID_PATTERN.match(weight_path):
            if weight_path == GDINO_HF_MODEL_ID:
                return weight_path, GDINO_HF_MODEL_REVISION
            return weight_path, None
        raise FileNotFoundError(f"Grounding DINO weights not found: {weight_path}")

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
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        xyxy, confidences, phrases = self._predict(
            image, ". ".join(classes), box_threshold, text_threshold
        )
        class_ids = self._phrases_to_class_ids(phrases, classes)

        packed_boxes = []
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

    def _predict(
        self,
        image_rgb: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run the model and return pixel xyxy boxes, confidences and matched phrases.

        Post-processing mirrors the original GroundingDINO ``predict(..., remove_combined=True)``:
        each box keeps the tokens above ``text_threshold`` within the caption segment
        (between separators) that holds its highest-scoring token.
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        inputs = self.processor(
            images=PILImage.fromarray(image_rgb), text=caption, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.sigmoid()[0].cpu()  # (num_queries, max_text_len)
        boxes = outputs.pred_boxes[0].cpu()  # (num_queries, 4), normalized cxcywh
        keep = logits.max(dim=1)[0] > box_threshold
        logits = logits[keep]
        boxes = boxes[keep]

        input_ids: List[int] = inputs["input_ids"][0].tolist()
        tokenizer = self.processor.tokenizer
        separator_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.convert_tokens_to_ids("."),
        }
        separator_idx = [i for i, t in enumerate(input_ids) if t in separator_ids]

        phrases = []
        for logit in logits:
            max_idx = int(logit.argmax())
            insert_idx = bisect.bisect_left(separator_idx, max_idx)
            right_idx = separator_idx[insert_idx]
            left_idx = separator_idx[insert_idx - 1]
            token_mask = logit > text_threshold
            token_mask[: left_idx + 1] = False
            token_mask[right_idx:] = False
            token_ids = [
                input_ids[i] for i in token_mask.nonzero(as_tuple=True)[0].tolist()
            ]
            phrases.append(tokenizer.decode(token_ids).replace(".", ""))

        height, width = image_rgb.shape[:2]
        cx, cy, w, h = (boxes * torch.tensor([width, height, width, height])).unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        return xyxy.numpy(), logits.max(dim=1)[0].numpy(), phrases

    @staticmethod
    def _phrases_to_class_ids(
        phrases: Sequence[str], classes: Sequence[str]
    ) -> List[Optional[int]]:
        """Map each phrase to the index of the first class contained in it, else None."""
        class_ids: List[Optional[int]] = []
        for phrase in phrases:
            for class_id, class_name in enumerate(classes):
                if class_name in phrase:
                    class_ids.append(class_id)
                    break
            else:
                class_ids.append(None)
        return class_ids
