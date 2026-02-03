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

"""Shared test utilities for algorithm tests."""

import numpy as np
from vision_msgs.msg import BoundingBox2D


def create_test_weights_file(tmp_path, filename="weights.pth"):
    """Helper to create a test weights file.

    Args:
        tmp_path: Temporary directory path
        filename: Name of the weights file

    Returns:
        Path to created weights file
    """
    weights_path = tmp_path / filename
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"test")
    return weights_path


def create_mock_image_array(shape=(100, 100, 3), dtype=np.uint8):
    """Helper to create a mock image array for testing.

    Args:
        shape: Shape of the image array
        dtype: Data type of the array

    Returns:
        Numpy array with zeros
    """
    return np.zeros(shape, dtype=dtype)


def create_test_bbox(center_x, center_y, size_x, size_y):
    """Helper to create a test bounding box.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        size_x: Width of bounding box
        size_y: Height of bounding box

    Returns:
        BoundingBox2D message
    """
    bbox = BoundingBox2D()
    bbox.center.position.x = center_x
    bbox.center.position.y = center_y
    bbox.size_x = size_x
    bbox.size_y = size_y
    return bbox
