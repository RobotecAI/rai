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

"""Base test class for GDBoxer - shared test logic for algorithms and vision_markup."""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rclpy
import torch
from sensor_msgs.msg import Image

from tests.rai_perception.algorithms.test_utils import (
    create_mock_image_array,
    create_test_model_dir,
)

# Fake BERT-style vocabulary: [CLS]=101, [SEP]=102, "."=1012, class words below.
CLS_ID, SEP_ID, DOT_ID = 101, 102, 1012
FAKE_VOCAB = {
    1001: "dinosaur",
    1002: "dragon",
    CLS_ID: "[CLS]",
    SEP_ID: "[SEP]",
    DOT_ID: ".",
}
FAKE_INPUT_IDS = {
    "dinosaur.": [CLS_ID, 1001, DOT_ID, SEP_ID],
    "dinosaur. dragon.": [CLS_ID, 1001, DOT_ID, 1002, DOT_ID, SEP_ID],
}
NUM_QUERIES, MAX_TEXT_LEN = 4, 256


class FakeTokenizer:
    cls_token_id = CLS_ID
    sep_token_id = SEP_ID

    def convert_tokens_to_ids(self, token):
        return {".": DOT_ID}[token]

    def decode(self, token_ids):
        return " ".join(FAKE_VOCAB[t] for t in token_ids)


class FakeInputs(dict):
    def to(self, _device):
        return self


def make_fake_processor():
    """Processor mock that tokenizes captions with FAKE_INPUT_IDS."""
    processor = MagicMock()
    processor.tokenizer = FakeTokenizer()

    def _call(images, text, return_tensors):
        assert return_tensors == "pt"
        return FakeInputs(input_ids=torch.tensor([FAKE_INPUT_IDS[text]]))

    processor.side_effect = _call
    return processor


def make_fake_outputs(detections):
    """Build model outputs. detections: list of (token_index, probability, cxcywh_normalized)."""
    logits = torch.full((1, NUM_QUERIES, MAX_TEXT_LEN), -10.0)
    boxes = torch.zeros((1, NUM_QUERIES, 4))
    for q, (token_idx, prob, cxcywh) in enumerate(detections):
        logits[0, q, token_idx] = torch.logit(torch.tensor(prob))
        boxes[0, q] = torch.tensor(cxcywh)
    return SimpleNamespace(logits=logits, pred_boxes=boxes)


class TestGDBoxerBase(ABC):
    """Base test class for GDBoxer - shared test logic."""

    @abstractmethod
    def get_boxer_class(self):
        """Return the GDBoxer class to test (from algorithms or vision_markup)."""
        pass

    @abstractmethod
    def get_patch_path(self, target):
        """Return the patch path for the given target (algorithms or vision_markup)."""
        pass

    @contextmanager
    def patch_hf(self, outputs=None):
        """Patch the transformers processor/model classes used by GDBoxer."""
        with (
            patch(self.get_patch_path("AutoProcessor")) as mock_processor_cls,
            patch(self.get_patch_path("AutoModel")) as mock_model_cls,
        ):
            mock_processor_cls.from_pretrained.return_value = make_fake_processor()
            mock_model = MagicMock()
            mock_model.return_value = outputs or make_fake_outputs([])
            mock_model_cls.from_pretrained.return_value = mock_model
            yield mock_processor_cls, mock_model_cls, mock_model

    def setup_method(self):
        """Initialize ROS2 before tests that use Time() or ROS2 messages."""
        if not rclpy.ok():
            rclpy.init()

    def teardown_method(self):
        """Clean up ROS2 context after each test to prevent thread exceptions."""
        try:
            if rclpy.ok():
                time.sleep(0.1)
                rclpy.shutdown()
        except Exception:
            pass

    def test_gdboxer_initialization(self, tmp_path):
        """Test GDBoxer initialization from a local model directory."""
        model_dir = create_test_model_dir(tmp_path)
        GDBoxer = self.get_boxer_class()

        with self.patch_hf() as (mock_processor_cls, mock_model_cls, mock_model):
            boxer = GDBoxer(str(model_dir), use_cuda=True)

            assert boxer.weight_path == str(model_dir)
            assert boxer.model is mock_model
            mock_processor_cls.from_pretrained.assert_called_once_with(
                str(model_dir), revision=None
            )
            mock_model_cls.from_pretrained.assert_called_once_with(
                str(model_dir), revision=None, use_safetensors=True
            )
            mock_model.eval.assert_called_once()

    def test_gdboxer_initialization_use_cuda_false(self, tmp_path):
        """Test GDBoxer initialization with use_cuda=False."""
        model_dir = create_test_model_dir(tmp_path)
        GDBoxer = self.get_boxer_class()
        logger_path = self.get_patch_path("logging.getLogger")

        with (
            self.patch_hf() as (_, _, mock_model),
            patch(logger_path) as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            boxer = GDBoxer(str(model_dir), use_cuda=False)

            assert boxer.device == "cpu"
            assert boxer.weight_path == str(model_dir)
            mock_model.to.assert_called_once_with("cpu")
            mock_logger.warning.assert_not_called()

    def test_gdboxer_get_boxes(self, tmp_path):
        """Test get_boxes maps detections to classes, pixel boxes and confidences."""
        model_dir = create_test_model_dir(tmp_path)
        GDBoxer = self.get_boxer_class()
        to_numpy_patch = self.get_patch_path("imgmsg_to_numpy")
        # Token 1 is "dinosaur", token 3 is "dragon" in "dinosaur. dragon."
        outputs = make_fake_outputs(
            [
                (1, 0.9, [0.3, 0.3, 0.4, 0.4]),
                (3, 0.8, [0.75, 0.75, 0.3, 0.3]),
                (3, 0.2, [0.5, 0.5, 0.1, 0.1]),  # below box_threshold
            ]
        )

        with (
            self.patch_hf(outputs),
            patch(to_numpy_patch, return_value=create_mock_image_array()) as to_numpy,
        ):
            boxer = GDBoxer(str(model_dir), use_cuda=True)

            image_msg = Image()
            classes = ["dinosaur", "dragon"]
            boxes = boxer.get_boxes(image_msg, classes, 0.4, 0.4)

            to_numpy.assert_called_once_with(image_msg, "rgb8")
            assert len(boxes) == 2
            assert boxes[0].phrase == "dinosaur"
            assert abs(boxes[0].confidence - 0.9) < 1e-5
            assert boxes[0].center == (30.0, 30.0)
            assert (boxes[0].size_x, boxes[0].size_y) == (40.0, 40.0)
            assert boxes[1].phrase == "dragon"
            assert abs(boxes[1].confidence - 0.8) < 1e-5
            assert boxes[1].center == (75.0, 75.0)

    def test_gdboxer_get_boxes_text_threshold_drops_unmatched(self, tmp_path):
        """A box whose tokens all fall below text_threshold matches no class and is dropped."""
        model_dir = create_test_model_dir(tmp_path)
        GDBoxer = self.get_boxer_class()
        to_numpy_patch = self.get_patch_path("imgmsg_to_numpy")
        outputs = make_fake_outputs([(1, 0.5, [0.5, 0.5, 0.2, 0.2])])

        with (
            self.patch_hf(outputs),
            patch(to_numpy_patch, return_value=create_mock_image_array()),
        ):
            boxer = GDBoxer(str(model_dir), use_cuda=False)

            assert boxer.get_boxes(Image(), ["dinosaur"], 0.4, 0.4) != []
            assert boxer.get_boxes(Image(), ["dinosaur"], 0.4, 0.6) == []

    def test_gdboxer_get_boxes_empty(self, tmp_path):
        """Test GDBoxer get_boxes with no detections."""
        model_dir = create_test_model_dir(tmp_path)
        GDBoxer = self.get_boxer_class()
        to_numpy_patch = self.get_patch_path("imgmsg_to_numpy")

        with (
            self.patch_hf(),
            patch(to_numpy_patch, return_value=create_mock_image_array()),
        ):
            boxer = GDBoxer(str(model_dir), use_cuda=False)

            image_msg = Image()
            classes = ["dinosaur"]
            boxes = boxer.get_boxes(image_msg, classes, 0.4, 0.4)

            assert len(boxes) == 0
