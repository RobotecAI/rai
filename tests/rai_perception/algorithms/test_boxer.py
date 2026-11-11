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

from unittest.mock import MagicMock, patch

import pytest
from rai_perception.algorithms.boxer import (
    GDINO_HF_MODEL_ID,
    GDINO_HF_MODEL_REVISION,
    Box,
    GDBoxer,
)
from rclpy.time import Time
from vision_msgs.msg import Detection2D

from tests.rai_perception.algorithms.test_base_boxer import TestGDBoxerBase
from tests.rai_perception.algorithms.test_utils import (
    create_test_model_dir,
    create_test_weights_file,
)


class TestBox:
    """Test cases for Box class."""

    def test_box_initialization(self):
        """Test Box initialization."""
        box = Box((50.0, 50.0), 40.0, 40.0, "dinosaur", 0.9)

        assert box.center == (50.0, 50.0)
        assert box.size_x == 40.0
        assert box.size_y == 40.0
        assert box.phrase == "dinosaur"
        assert box.confidence == 0.9

    def test_box_to_detection_msg(self):
        """Test Box conversion to Detection2D message."""
        box = Box((50.0, 50.0), 40.0, 40.0, "dinosaur", 0.9)

        class_dict = {"dinosaur": 0, "dragon": 1}
        timestamp = Time()

        detection = box.to_detection_msg(class_dict, timestamp)

        assert isinstance(detection, Detection2D)
        assert detection.bbox.center.position.x == 50.0
        assert detection.bbox.center.position.y == 50.0
        assert detection.bbox.size_x == 40.0
        assert detection.bbox.size_y == 40.0
        assert detection.results[0].hypothesis.class_id == "dinosaur"
        assert detection.results[0].hypothesis.score == 0.9
        assert detection.header.stamp == timestamp.to_msg()


class TestGDBoxer(TestGDBoxerBase):
    """Test cases for algorithms.boxer.GDBoxer class."""

    def get_boxer_class(self):
        """Return the GDBoxer class from algorithms."""
        return GDBoxer

    def get_patch_path(self, target):
        """Return patch path for algorithms module."""
        patch_map = {
            "AutoProcessor": "rai_perception.algorithms.boxer.AutoProcessor",
            "AutoModel": "rai_perception.algorithms.boxer.AutoModelForZeroShotObjectDetection",
            "imgmsg_to_numpy": "rai_perception.algorithms.boxer.imgmsg_to_numpy",
            "logging.getLogger": "rai_perception.algorithms.boxer.logging.getLogger",
        }
        return patch_map[target]


class TestGDBoxerModelSource:
    """Test how GDBoxer resolves weight_path into a transformers source."""

    def _boxer(self):
        boxer = GDBoxer.__new__(GDBoxer)
        boxer.logger = MagicMock()
        return boxer

    def test_local_directory(self, tmp_path):
        model_dir = create_test_model_dir(tmp_path)
        assert self._boxer()._resolve_model_source(str(model_dir)) == (
            str(model_dir),
            None,
        )

    def test_legacy_checkpoint_file_falls_back_to_pinned_hub_model(self, tmp_path):
        legacy = create_test_weights_file(tmp_path, "groundingdino_swint_ogc.pth")
        boxer = self._boxer()
        assert boxer._resolve_model_source(str(legacy)) == (
            GDINO_HF_MODEL_ID,
            GDINO_HF_MODEL_REVISION,
        )
        boxer.logger.warning.assert_called_once()

    def test_default_hub_id_uses_pinned_revision(self):
        assert self._boxer()._resolve_model_source(GDINO_HF_MODEL_ID) == (
            GDINO_HF_MODEL_ID,
            GDINO_HF_MODEL_REVISION,
        )

    def test_other_hub_id_uses_default_revision(self):
        assert self._boxer()._resolve_model_source(
            "IDEA-Research/grounding-dino-base"
        ) == ("IDEA-Research/grounding-dino-base", None)

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._boxer()._resolve_model_source(str(tmp_path / "missing"))

    def test_config_path_is_ignored_with_deprecation_warning(self, tmp_path):
        model_dir = create_test_model_dir(tmp_path)
        with (
            patch("rai_perception.algorithms.boxer.AutoProcessor"),
            patch(
                "rai_perception.algorithms.boxer.AutoModelForZeroShotObjectDetection"
            ),
            pytest.warns(DeprecationWarning, match="config_path is ignored"),
        ):
            GDBoxer(str(model_dir), config_path="gdino_config.py", use_cuda=False)
