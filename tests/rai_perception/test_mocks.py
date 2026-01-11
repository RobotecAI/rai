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

"""Shared mock classes for testing."""

import numpy as np
from rai_perception.algorithms.boxer import Box


class MockGDBoxer:
    """Mock GDBoxer for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
        """Mock box detection."""
        box1 = Box((50.0, 50.0), 40.0, 40.0, classes[0], 0.9)
        box2 = Box((100.0, 100.0), 30.0, 30.0, classes[1], 0.8)
        return [box1, box2]


class MockGDSegmenter:
    """Mock GDSegmenter for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_segmentation(self, image, boxes):
        """Mock segmentation that returns simple masks."""
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[10:50, 10:50] = 1.0
        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[60:90, 60:90] = 1.0
        return [mask1, mask2]


class EmptyBoxer:
    """Mock GDBoxer that returns empty results for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_boxes(self, image_msg, classes, box_threshold, text_threshold):
        """Return empty list of boxes."""
        return []


class EmptySegmenter:
    """Mock GDSegmenter that returns empty results for testing."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def get_segmentation(self, image, boxes):
        """Return empty list of masks."""
        return []
