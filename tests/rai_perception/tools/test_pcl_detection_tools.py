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

import time

import pytest

try:
    import rclpy  # noqa: F401

    _ = rclpy  # noqa: F841
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from unittest.mock import Mock, patch

import numpy as np
from rai.communication.ros2.connectors import ROS2Connector
from rai_perception import (
    GetObjectGrippingPointsTool,
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
)
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX


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
    estimator = GrippingPointEstimator(
        config=GrippingPointEstimatorConfig(strategy="centroid")
    )
    grip_points = estimator.run(segmented_clouds)

    assert len(grip_points) == 2
    # Check centroid of first cloud
    expected_centroid1 = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    np.testing.assert_array_almost_equal(grip_points[0], expected_centroid1)

    # Test top_plane strategy
    estimator_top = GrippingPointEstimator(
        config=GrippingPointEstimatorConfig(strategy="top_plane", top_percentile=0.5)
    )
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

    # Test density-based filtering (maps to DBSCAN)
    filter_dbscan = PointCloudFilter(
        config=PointCloudFilterConfig(
            strategy="density_based", cluster_radius_m=0.5, min_cluster_size=5
        )
    )
    filtered_dbscan = filter_dbscan.run(clouds)

    assert len(filtered_dbscan) == 1
    # Should remove most noise points
    assert filtered_dbscan[0].shape[0] < noisy_cloud.shape[0]
    assert filtered_dbscan[0].shape[0] >= 40  # Should keep most of main cluster

    # Test with too few points (should return original)
    small_cloud = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    filter_small = PointCloudFilter(
        config=PointCloudFilterConfig(strategy="density_based", min_points=20)
    )
    filtered_small = filter_small.run([small_cloud])

    assert len(filtered_small) == 1
    np.testing.assert_array_equal(filtered_small[0], small_cloud)

    # Test cluster-based strategy (maps to KMeans)
    filter_kmeans = PointCloudFilter(
        config=PointCloudFilterConfig(strategy="cluster_based", num_clusters=2)
    )
    filtered_kmeans = filter_kmeans.run(clouds)

    assert len(filtered_kmeans) == 1
    assert filtered_kmeans[0].shape[0] > 0


def test_get_gripping_point_tool_timeout():
    # Complete mock setup
    mock_connector = Mock(spec=ROS2Connector)
    mock_pcl_gen = Mock(spec=PointCloudFromSegmentation)
    mock_filter = Mock(spec=PointCloudFilter)
    mock_estimator = Mock(spec=GrippingPointEstimator)

    # Test 1: No timeout (fast execution)
    mock_pcl_gen.run.return_value = []
    mock_filter.run.return_value = []
    mock_estimator.run.return_value = []

    tool = GetObjectGrippingPointsTool(
        connector=mock_connector,
        segmentation_config=PointCloudFromSegmentationConfig(),
        estimator_config=GrippingPointEstimatorConfig(),
        filter_config=PointCloudFilterConfig(),
    )
    # Mock the initialized components
    tool.gripping_point_estimator = mock_estimator
    tool.point_cloud_filter = mock_filter
    tool.timeout_sec = 5.0
    tool.point_cloud_from_segmentation = mock_pcl_gen

    # Test fast execution - should complete without timeout
    result = tool._run("test_object")
    assert "No test_objects detected" in result
    assert "timed out" not in result.lower()

    # Test 2: Actual timeout behavior - should raise TimeoutError
    def slow_operation(obj_name):
        time.sleep(2.0)  # Longer than timeout
        return []

    mock_pcl_gen.run.side_effect = slow_operation
    tool.timeout_sec = 1.0  # Short timeout

    # Expect TimeoutError
    assert (
        tool._run("test")
        == "Timeout: Gripping point detection for object 'test' exceeded 1.0 seconds"
    )


def test_get_object_gripping_points_tool_auto_declaration():
    """Test that GetObjectGrippingPointsTool auto-declares parameters with defaults."""
    rclpy.init()
    try:
        connector = ROS2Connector(executor_type="single_threaded")
        node = connector.node

        # Test 1: Auto-declaration with defaults when no parameters are set
        # Clear any existing parameters
        param_prefix = GRIPPING_POINTS_TOOL_PARAM_PREFIX
        for param_key in [
            "target_frame",
            "source_frame",
            "camera_topic",
            "depth_topic",
            "camera_info_topic",
            "timeout_sec",
            "conversion_ratio",
        ]:
            param_name = f"{param_prefix}.{param_key}"
            if node.has_parameter(param_name):
                node.undeclare_parameter(param_name)

        # Initialize tool - should auto-declare with defaults
        with patch("rai_perception.tools.gripping_points_tools.logger") as mock_logger:
            tool = GetObjectGrippingPointsTool(connector=connector)

            # Verify defaults are used
            assert tool.target_frame == "base_link"
            assert tool.source_frame == "camera_link"
            assert tool.camera_topic == "/camera/rgb/image_raw"
            assert tool.depth_topic == "/camera/depth/image_raw"
            assert tool.camera_info_topic == "/camera/rgb/camera_info"
            assert tool.timeout_sec == 10.0
            assert tool.conversion_ratio == 0.001

            # Verify logging occurred
            assert mock_logger.info.called
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Auto-declared parameter" in str(call) for call in log_calls)
            assert any(
                "GetObjectGrippingPointsTool initialized" in str(call)
                for call in log_calls
            )

        # Test 2: Override parameters before initialization
        # Undeclare parameters from Test 1 first
        for param_key in ["target_frame", "camera_topic"]:
            param_name = f"{param_prefix}.{param_key}"
            if node.has_parameter(param_name):
                node.undeclare_parameter(param_name)

        # Now declare with custom values
        node.declare_parameter(f"{param_prefix}.target_frame", "custom_frame")
        node.declare_parameter(f"{param_prefix}.camera_topic", "/custom/camera")

        with patch("rai_perception.tools.gripping_points_tools.logger") as mock_logger:
            tool2 = GetObjectGrippingPointsTool(connector=connector)

            # Verify overrides are used
            assert tool2.target_frame == "custom_frame"
            assert tool2.camera_topic == "/custom/camera"
            # Other params should still use defaults
            assert tool2.source_frame == "camera_link"

            # Verify override logging
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any(
                "overridden parameter" in str(call).lower() for call in log_calls
            )

        # Test 3: get_config() method
        config = tool2.get_config()
        assert isinstance(config, dict)
        assert config["target_frame"] == "custom_frame"
        assert config["camera_topic"] == "/custom/camera"
        assert config["timeout_sec"] == 10.0
        assert "detection_service_name" in config
        assert "segmentation_service_name" in config

    finally:
        rclpy.shutdown()
