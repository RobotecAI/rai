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

import pytest

try:
    import rclpy  # noqa: F401

    _ = rclpy  # noqa: F841
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from unittest.mock import Mock

import numpy as np
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.detection import GetGrippingPointTool
from rai.tools.ros2.detection.pcl import (
    GrippingPointEstimator,
    PointCloudFilter,
    PointCloudFromSegmentation,
    depth_to_point_cloud,
)


def test_depth_to_point_cloud():
    """Test depth image to point cloud conversion algorithm."""
    # Create a simple 2x2 depth image with known values
    depth_image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Camera intrinsics
    fx, fy, cx, cy = 100.0, 100.0, 1.0, 1.0

    # Convert to point cloud
    points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

    # Should have 4 points (2x2 image)
    assert points.shape[0] == 4
    assert points.shape[1] == 3  # X, Y, Z coordinates

    # Check that all Z values match the depth image
    expected_z_values = [1.0, 2.0, 3.0, 4.0]
    actual_z_values = sorted(points[:, 2].tolist())
    np.testing.assert_array_almost_equal(actual_z_values, expected_z_values)

    # Verify no points with zero depth are included
    zero_depth = np.zeros((2, 2), dtype=np.float32)
    points_zero = depth_to_point_cloud(zero_depth, fx, fy, cx, cy)
    assert points_zero.shape[0] == 0


def test_gripping_point_estimator():
    """Test gripping point estimation strategies."""
    # Create test point cloud data - a simple box shape
    points1 = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 2.0],
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    points2 = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 6.0],
            [6.0, 5.0, 5.0],
            [6.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    segmented_clouds = [points1, points2]

    # Test centroid strategy
    estimator = GrippingPointEstimator(strategy="centroid")
    grip_points = estimator.run(segmented_clouds)

    assert len(grip_points) == 2
    # Check centroid of first cloud
    expected_centroid1 = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    np.testing.assert_array_almost_equal(grip_points[0], expected_centroid1)

    # Test top_plane strategy
    estimator_top = GrippingPointEstimator(strategy="top_plane", top_percentile=0.5)
    grip_points_top = estimator_top.run(segmented_clouds)

    assert len(grip_points_top) == 2
    # Top plane should have higher Z values
    assert grip_points_top[0][2] >= grip_points[0][2]

    # Test with empty point cloud
    empty_clouds = [np.array([]).reshape(0, 3).astype(np.float32)]
    grip_points_empty = estimator.run(empty_clouds)
    assert len(grip_points_empty) == 0


def test_point_cloud_filter():
    """Test point cloud filtering strategies."""
    # Create test data with noise points
    main_cluster = np.random.normal([0, 0, 0], 0.1, (50, 3)).astype(np.float32)
    noise_points = np.random.normal([5, 5, 5], 0.1, (5, 3)).astype(np.float32)
    noisy_cloud = np.vstack([main_cluster, noise_points])

    clouds = [noisy_cloud]

    # Test DBSCAN filtering
    filter_dbscan = PointCloudFilter(
        strategy="dbscan", dbscan_eps=0.5, dbscan_min_samples=5
    )
    filtered_dbscan = filter_dbscan.run(clouds)

    assert len(filtered_dbscan) == 1
    # Should remove most noise points
    assert filtered_dbscan[0].shape[0] < noisy_cloud.shape[0]
    assert filtered_dbscan[0].shape[0] >= 40  # Should keep most of main cluster

    # Test with too few points (should return original)
    small_cloud = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    filter_small = PointCloudFilter(strategy="dbscan", min_points=20)
    filtered_small = filter_small.run([small_cloud])

    assert len(filtered_small) == 1
    np.testing.assert_array_equal(filtered_small[0], small_cloud)

    # Test kmeans_largest_cluster strategy
    filter_kmeans = PointCloudFilter(strategy="kmeans_largest_cluster", kmeans_k=2)
    filtered_kmeans = filter_kmeans.run(clouds)

    assert len(filtered_kmeans) == 1
    assert filtered_kmeans[0].shape[0] > 0


def test_get_gripping_point_tool_timeout():
    """Test GetGrippingPointTool timeout behavior."""
    # Mock the connector and components
    mock_connector = Mock(spec=ROS2Connector)

    # Create mock components that will simulate timeout
    mock_pcl_gen = Mock(spec=PointCloudFromSegmentation)
    mock_pcl_gen.run.side_effect = lambda x: []  # Return empty to simulate no detection

    mock_filter = Mock(spec=PointCloudFilter)
    mock_filter.run.return_value = []

    mock_estimator = Mock(spec=GrippingPointEstimator)
    mock_estimator.run.return_value = []

    # Create tool with short timeout
    tool = GetGrippingPointTool(
        connector=mock_connector,
        point_cloud_from_segmentation=mock_pcl_gen,
        point_cloud_filter=mock_filter,
        gripping_point_estimator=mock_estimator,
        timeout_sec=0.1,
    )

    # Test successful run with no gripping points found
    result = tool._run("test_object")
    assert "No gripping point found" in result
    assert "test_object" in result

    # Test with mock that simulates found gripping points
    mock_estimator.run.return_value = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    result = tool._run("test_object")
    assert "gripping point of the object test_object is" in result
    assert "[1. 2. 3.]" in result

    # Test with multiple gripping points
    mock_estimator.run.return_value = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
    ]
    result = tool._run("test_object")
    assert "Multiple gripping points found" in result


def test_get_gripping_point_tool_validation():
    """Test GetGrippingPointTool input validation."""
    mock_connector = Mock(spec=ROS2Connector)
    mock_pcl_gen = Mock(spec=PointCloudFromSegmentation)
    mock_filter = Mock(spec=PointCloudFilter)
    mock_estimator = Mock(spec=GrippingPointEstimator)

    # Test tool creation
    tool = GetGrippingPointTool(
        connector=mock_connector,
        point_cloud_from_segmentation=mock_pcl_gen,
        point_cloud_filter=mock_filter,
        gripping_point_estimator=mock_estimator,
    )

    # Verify tool properties
    assert tool.name == "get_gripping_point"
    assert "gripping points" in tool.description
    assert tool.timeout_sec == 10.0  # default value

    # Test args schema
    from rai.tools.ros2.detection.tools import GetGrippingPointToolInput

    assert tool.args_schema == GetGrippingPointToolInput
