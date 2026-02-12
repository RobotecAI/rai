# Copyright (C) 2025 Julia Jia
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

"""Unit tests for depth_to_point_cloud algorithm."""

import numpy as np
from rai_perception.algorithms.point_cloud import depth_to_point_cloud


class TestDepthToPointCloud:
    """Test cases for depth_to_point_cloud function."""

    @staticmethod
    def _default_intrinsics():
        """Return default camera intrinsics for testing."""
        return 100.0, 100.0, 1.0, 1.0  # fx, fy, cx, cy

    def test_depth_to_point_cloud_basic(self):
        """Test basic depth to point cloud conversion."""
        depth_image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        fx, fy, cx, cy = self._default_intrinsics()

        points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

        assert points.shape == (4, 3)
        assert points.dtype == np.float32

        expected_z_values = [1.0, 2.0, 3.0, 4.0]
        actual_z_values = sorted(points[:, 2].tolist())
        np.testing.assert_array_almost_equal(actual_z_values, expected_z_values)

    def test_depth_to_point_cloud_filters_zero_depth(self):
        """Test that zero depth points are filtered out."""
        depth_image = np.zeros((2, 2), dtype=np.float32)
        fx, fy, cx, cy = self._default_intrinsics()

        points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

        assert points.shape[0] == 0

    def test_depth_to_point_cloud_partial_zero_depth(self):
        """Test filtering of partial zero depth regions."""
        depth_image = np.zeros((100, 100), dtype=np.float32)
        depth_image[20:80, 20:80] = 1.0

        fx, fy, cx, cy = 500.0, 500.0, 50.0, 50.0

        points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

        assert points.shape[0] > 0
        assert all(points[:, 2] > 0)
        assert points.shape[0] == (60 * 60)

    def test_depth_to_point_cloud_coordinate_calculation(self):
        """Test that 3D coordinates are calculated correctly."""
        depth_value = 2.0
        depth_image = np.ones((3, 3), dtype=np.float32) * depth_value
        fx, fy, cx, cy = self._default_intrinsics()

        points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

        assert points.shape == (9, 3)

        # Verify coordinates for each pixel
        for i, point in enumerate(points):
            u, v = i % 3, i // 3
            expected_x = (u - cx) * depth_value / fx
            expected_y = (v - cy) * depth_value / fy

            np.testing.assert_almost_equal(point[0], expected_x, decimal=5)
            np.testing.assert_almost_equal(point[1], expected_y, decimal=5)
            np.testing.assert_almost_equal(point[2], depth_value, decimal=5)

    def test_depth_to_point_cloud_different_focal_lengths(self):
        """Test with different focal lengths in x and y."""
        depth_value = 1.0
        depth_image = np.ones((2, 2), dtype=np.float32) * depth_value
        fx, fy = 200.0, 100.0
        cx, cy = 1.0, 1.0

        points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

        assert points.shape == (4, 3)

        # Test first point (u=0, v=0)
        point_00 = points[0]
        expected_x = (0 - cx) * depth_value / fx
        expected_y = (0 - cy) * depth_value / fy

        np.testing.assert_almost_equal(point_00[0], expected_x)
        np.testing.assert_almost_equal(point_00[1], expected_y)
