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

import asyncio

import numpy as np
import pytest
from rai.communication.ros2.images import numpy_to_imgmsg
from rai_perception.components.gripping_points import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
)


@pytest.fixture
def sample_point_cloud():
    """Create a sample point cloud."""
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


@pytest.fixture
def multi_object_point_clouds():
    """Create multiple point clouds for testing."""
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]], dtype=np.float32),
    ]


class TestGrippingPointEstimator:
    """Test cases for GrippingPointEstimator."""

    def test_centroid_strategy(self, sample_point_cloud):
        """Test centroid strategy."""
        config = GrippingPointEstimatorConfig(strategy="centroid")
        estimator = GrippingPointEstimator(config)
        result = estimator.run([sample_point_cloud])

        assert len(result) == 1
        assert result[0].shape == (3,)
        # Centroid should be mean of all points
        expected = sample_point_cloud.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_centroid_empty_points(self):
        """Test centroid strategy with empty point cloud."""
        config = GrippingPointEstimatorConfig(strategy="centroid")
        estimator = GrippingPointEstimator(config)
        empty_cloud = np.zeros((0, 3), dtype=np.float32)
        result = estimator.run([empty_cloud])

        assert len(result) == 0

    def test_top_plane_strategy(self):
        """Test top_plane strategy."""
        # Create points with varying Z values
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 2.0],  # Top point
            ],
            dtype=np.float32,
        )
        config = GrippingPointEstimatorConfig(
            strategy="top_plane", top_percentile=0.2, min_points=3
        )
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Should be close to top point
        assert result[0][2] > 1.0

    def test_top_plane_fallback_to_centroid(self):
        """Test top_plane falls back to centroid when insufficient points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        config = GrippingPointEstimatorConfig(strategy="top_plane", min_points=5)
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Should use centroid fallback
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_biggest_plane_strategy(self):
        """Test biggest_plane strategy with planar points."""
        # Create points on a plane
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.2, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        config = GrippingPointEstimatorConfig(
            strategy="biggest_plane",
            min_points=3,
            ransac_iterations=50,
            distance_threshold_m=0.01,
        )
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Z should be close to 1.0 (plane height)
        assert abs(result[0][2] - 1.0) < 0.1

    def test_biggest_plane_fallback_to_centroid(self):
        """Test biggest_plane falls back to centroid when insufficient points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        config = GrippingPointEstimatorConfig(strategy="biggest_plane", min_points=5)
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_multiple_clouds(self, multi_object_point_clouds):
        """Test processing multiple point clouds."""
        config = GrippingPointEstimatorConfig(strategy="centroid")
        estimator = GrippingPointEstimator(config)
        result = estimator.run(multi_object_point_clouds)

        assert len(result) == 2
        assert all(gp.shape == (3,) for gp in result)

    def test_unknown_strategy_fallback(self, sample_point_cloud):
        """Test that unknown strategy falls back to centroid."""
        config = GrippingPointEstimatorConfig()
        config.strategy = "unknown_strategy"  # type: ignore
        estimator = GrippingPointEstimator(config)
        result = estimator.run([sample_point_cloud])

        assert len(result) == 1
        # Should use centroid as fallback
        expected = sample_point_cloud.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_estimator_check_min_points_helper(self):
        """Test _check_min_points helper method."""
        config = GrippingPointEstimatorConfig(min_points=10)
        estimator = GrippingPointEstimator(config)

        # Test with sufficient points
        points = np.random.rand(20, 3).astype(np.float32)
        result = estimator._check_min_points(points)
        assert result is None  # Should return None when points are sufficient

        # Test with insufficient points (should return centroid)
        points = np.random.rand(5, 3).astype(np.float32)
        result = estimator._check_min_points(points)
        assert result is not None
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, points.mean(axis=0), decimal=5)

    def test_estimator_ransac_plane_detection(self):
        """Test _ransac_plane_detection helper method."""
        config = GrippingPointEstimatorConfig(
            min_points=5,
            ransac_iterations=50,
            distance_threshold_m=0.01,
        )
        estimator = GrippingPointEstimator(config)

        # Test with planar points
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        mask = estimator._ransac_plane_detection(points)
        assert mask is not None
        assert mask.dtype == bool
        assert mask.shape[0] == points.shape[0]
        assert np.sum(mask) >= config.min_points

        # Test with insufficient points
        points = np.random.rand(3, 3).astype(np.float32)
        mask = estimator._ransac_plane_detection(points)
        assert mask is None

    def test_estimator_get_strategy_method(self):
        """Test _get_strategy_method helper."""
        strategies = ["centroid", "top_plane", "biggest_plane"]
        for strategy in strategies:
            config = GrippingPointEstimatorConfig(strategy=strategy)
            estimator = GrippingPointEstimator(config)
            method = estimator._get_strategy_method()
            assert callable(method)

        # Test unknown strategy returns centroid
        config = GrippingPointEstimatorConfig()
        config.strategy = "unknown"  # type: ignore
        estimator = GrippingPointEstimator(config)
        method = estimator._get_strategy_method()
        points = np.random.rand(10, 3).astype(np.float32)
        result = method(points)
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_top_plane_empty_result_fallback(self):
        """Test top_plane falls back to centroid when no top points found."""
        # Create points where top percentile filter results in empty set
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        config = GrippingPointEstimatorConfig(
            strategy="top_plane", top_percentile=0.01, min_points=2
        )
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Should fall back to centroid
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_biggest_plane_empty_inliers_fallback(self):
        """Test biggest_plane falls back to centroid when no inliers found."""
        # Create non-planar points that won't form a good plane
        points = np.random.rand(20, 3).astype(np.float32) * 10.0
        config = GrippingPointEstimatorConfig(
            strategy="biggest_plane",
            min_points=5,
            ransac_iterations=10,
            distance_threshold_m=0.001,  # Very strict threshold
        )
        estimator = GrippingPointEstimator(config)
        result = estimator.run([points])

        assert len(result) == 1
        # Should fall back to centroid when plane detection fails
        expected = points.mean(axis=0)
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)


class TestPointCloudFilter:
    """Test cases for PointCloudFilter."""

    def test_aggressive_outlier_removal_strategy(self):
        """Test aggressive_outlier_removal filtering strategy (maps to Isolation Forest)."""
        # Create points with outliers
        inliers = np.random.rand(50, 3).astype(np.float32) * 0.1
        outliers = np.array(
            [[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]], dtype=np.float32
        )
        points = np.vstack([inliers, outliers])

        config = PointCloudFilterConfig(
            strategy="aggressive_outlier_removal",
            min_points=10,
            outlier_fraction=0.1,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        # Should have fewer points (outliers removed)
        assert result[0].shape[0] < points.shape[0]
        assert result[0].shape[0] >= inliers.shape[0] * 0.8  # Most inliers kept

    def test_density_based_strategy(self):
        """Test density_based filtering strategy (maps to DBSCAN)."""
        # Create two clusters
        cluster1 = np.random.rand(30, 3).astype(np.float32) * 0.1
        cluster2 = (np.random.rand(20, 3).astype(np.float32) * 0.1) + np.array(
            [5.0, 5.0, 5.0]
        )
        points = np.vstack([cluster1, cluster2])

        config = PointCloudFilterConfig(
            strategy="density_based",
            min_points=10,
            cluster_radius_m=0.5,
            min_cluster_size=5,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        # Should keep largest cluster
        assert result[0].shape[0] > 0

    def test_cluster_based_strategy(self):
        """Test cluster_based filtering strategy (maps to KMeans)."""
        # Create two clusters
        cluster1 = np.random.rand(40, 3).astype(np.float32) * 0.1
        cluster2 = (np.random.rand(20, 3).astype(np.float32) * 0.1) + np.array(
            [5.0, 5.0, 5.0]
        )
        points = np.vstack([cluster1, cluster2])

        config = PointCloudFilterConfig(
            strategy="cluster_based",
            min_points=10,
            num_clusters=2,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        assert result[0].shape[0] > 0

    def test_filter_insufficient_points(self):
        """Test filtering with insufficient points returns original."""
        points = np.random.rand(5, 3).astype(np.float32)
        config = PointCloudFilterConfig(
            strategy="aggressive_outlier_removal", min_points=10
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        assert result[0].shape == points.shape

    def test_filter_empty_points(self):
        """Test filtering empty point cloud."""
        empty_cloud = np.zeros((0, 3), dtype=np.float32)
        config = PointCloudFilterConfig(strategy="aggressive_outlier_removal")
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([empty_cloud])

        assert len(result) == 0

    def test_multiple_clouds(self, multi_object_point_clouds):
        """Test filtering multiple point clouds."""
        config = PointCloudFilterConfig(
            strategy="aggressive_outlier_removal", min_points=2
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run(multi_object_point_clouds)

        assert len(result) == 2
        assert all(filt.shape[1] == 3 for filt in result)

    def test_unknown_strategy_returns_original(self, sample_point_cloud):
        """Test that unknown strategy returns original points."""
        config = PointCloudFilterConfig()
        config.strategy = "unknown_strategy"  # type: ignore
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([sample_point_cloud])

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], sample_point_cloud)

    def test_conservative_outlier_removal_strategy(self):
        """Test conservative_outlier_removal filtering strategy (maps to Local Outlier Factor)."""
        # Create points with outliers
        inliers = np.random.rand(50, 3).astype(np.float32) * 0.1
        outliers = np.array(
            [[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]], dtype=np.float32
        )
        points = np.vstack([inliers, outliers])

        config = PointCloudFilterConfig(
            strategy="conservative_outlier_removal",
            min_points=10,
            neighborhood_size=20,
            outlier_fraction=0.1,
        )
        filter_obj = PointCloudFilter(config)
        result = filter_obj.run([points])

        assert len(result) == 1
        # Should have fewer points (outliers removed)
        assert result[0].shape[0] < points.shape[0]

    def test_filter_check_min_points_helper(self):
        """Test _check_min_points helper method."""
        config = PointCloudFilterConfig(min_points=10)
        filter_obj = PointCloudFilter(config)

        # Test with sufficient points
        points = np.random.rand(20, 3).astype(np.float32)
        assert filter_obj._check_min_points(points) is True

        # Test with insufficient points
        points = np.random.rand(5, 3).astype(np.float32)
        assert filter_obj._check_min_points(points) is False

        # Test with custom minimum requirement
        assert filter_obj._check_min_points(points, min_required=15) is False
        assert filter_obj._check_min_points(points, min_required=3) is True

    def test_filter_apply_mask_or_fallback_helper(self):
        """Test _apply_mask_or_fallback helper method."""
        config = PointCloudFilterConfig()
        filter_obj = PointCloudFilter(config)

        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
        )

        # Test with valid mask
        mask = np.array([True, False, True], dtype=bool)
        result = filter_obj._apply_mask_or_fallback(points, mask)
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result, points[mask])

        # Test with empty mask (should return original)
        mask = np.array([False, False, False], dtype=bool)
        result = filter_obj._apply_mask_or_fallback(points, mask)
        np.testing.assert_array_equal(result, points)

    def test_filter_get_largest_cluster_helper(self):
        """Test _get_largest_cluster helper method."""
        config = PointCloudFilterConfig()
        filter_obj = PointCloudFilter(config)

        # Test with valid labels
        labels = np.array([0, 0, 0, 1, 1, -1], dtype=np.int64)
        mask = filter_obj._get_largest_cluster(labels)
        assert np.sum(mask) == 3  # Largest cluster has 3 points

        # Test with empty labels
        labels = np.array([], dtype=np.int64)
        mask = filter_obj._get_largest_cluster(labels)
        assert mask.shape[0] == 0

        # Test with all outliers (label -1)
        labels = np.array([-1, -1, -1], dtype=np.int64)
        mask = filter_obj._get_largest_cluster(labels)
        assert np.sum(mask) == 0

    def test_filter_get_strategy_method(self):
        """Test _get_strategy_method helper."""
        strategies = [
            "density_based",
            "cluster_based",
            "aggressive_outlier_removal",
            "conservative_outlier_removal",
        ]
        for strategy in strategies:
            config = PointCloudFilterConfig(strategy=strategy)
            filter_obj = PointCloudFilter(config)
            method = filter_obj._get_strategy_method()
            assert callable(method)

        # Test unknown strategy returns identity function
        config = PointCloudFilterConfig()
        config.strategy = "unknown"  # type: ignore
        filter_obj = PointCloudFilter(config)
        method = filter_obj._get_strategy_method()
        points = np.random.rand(10, 3).astype(np.float32)
        result = method(points)
        np.testing.assert_array_equal(result, points)


def make_pipeline_connector(connector, depth, mask, translation=(0.0, 0.0, 0.0)):
    """Wire mock_connector to serve a camera, depth and camera info, one
    detection with the given mask, and a TF translation."""
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo

    from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino

    info = CameraInfo(k=[100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0])
    messages = {
        "/test/camera": numpy_to_imgmsg(np.zeros((2, 2, 3), np.uint8), "rgb8"),
        "/test/depth": depth,
        "/test/camera_info": info,
    }
    connector.receive.side_effect = lambda topic, **_: messages[topic]

    def call(service, request, **_):
        if isinstance(request, RAIGroundingDino.Request):
            return RAIGroundingDino.Response()
        masks = [] if mask is None else [numpy_to_imgmsg(mask, "mono8")]
        return RAIGroundedSam.Response(masks=masks)

    connector.call.side_effect = call
    transform = TransformStamped()
    t = transform.transform.translation
    t.x, t.y, t.z = translation
    transform.transform.rotation.w = 1.0
    connector.transform.return_value = transform
    return connector


def make_pcl(connector, **kwargs):
    return PointCloudFromSegmentation(
        connector=connector,
        camera_topic="/test/camera",
        depth_topic="/test/depth",
        camera_info_topic="/test/camera_info",
        source_frame="camera_frame",
        target_frame="base_frame",
        **{"config": PointCloudFromSegmentationConfig(), **kwargs},
    )


class TestPointCloudFromSegmentation:
    MASK = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    def test_points_in_target_frame(self, mock_connector):
        depth = numpy_to_imgmsg(np.full((2, 2), 2000, np.uint16), "16UC1")  # mm
        make_pipeline_connector(mock_connector, depth, self.MASK, (1.0, 0.0, 0.0))
        clouds = asyncio.run(make_pcl(mock_connector).run("cup"))
        assert len(clouds) == 1
        np.testing.assert_allclose(clouds[0][:, 2], [2.0, 2.0])  # meters
        np.testing.assert_allclose(clouds[0][:, 0], [0.98, 1.0], atol=1e-6)
        mock_connector.transform.assert_awaited_with("base_frame", "camera_frame")

    def test_conversion_ratio_scales_depth(self, mock_connector):
        depth = numpy_to_imgmsg(np.full((2, 2), 2.0, np.float32), "32FC1")
        make_pipeline_connector(mock_connector, depth, self.MASK)
        pcl = make_pcl(mock_connector, conversion_ratio=0.5)
        assert make_pcl(mock_connector).conversion_ratio == 1.0
        clouds = asyncio.run(pcl.run("cup"))
        np.testing.assert_allclose(clouds[0][:, 2], [1.0, 1.0])

    def test_no_masks_means_no_clouds(self, mock_connector):
        depth = numpy_to_imgmsg(np.full((2, 2), 2.0, np.float32), "32FC1")
        make_pipeline_connector(mock_connector, depth, None)
        assert asyncio.run(make_pcl(mock_connector).run("cup")) == []

    def test_calls_use_configured_services_and_timeout(self, mock_connector):
        depth = numpy_to_imgmsg(np.full((2, 2), 2.0, np.float32), "32FC1")
        make_pipeline_connector(mock_connector, depth, self.MASK)
        config = PointCloudFromSegmentationConfig(service_timeout=90.0)
        assert PointCloudFromSegmentationConfig().service_timeout == 60.0
        pcl = make_pcl(
            mock_connector,
            config=config,
            detection_service="/d",
            segmentation_service="/s",
        )
        asyncio.run(pcl.run("cup"))
        calls = mock_connector.call.call_args_list
        assert [c.args[0] for c in calls] == ["/d", "/s"]
        assert all(c.kwargs["timeout"] == 90.0 for c in calls)

    def test_encoding_aware_conversion(self):
        from rai_perception.components.perception_utils import convert_depth_to_meters

        depth_mm = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        result_mm = convert_depth_to_meters(
            depth_mm, numpy_to_imgmsg(depth_mm, "16UC1")
        )
        np.testing.assert_array_almost_equal(result_mm, depth_mm / 1000.0)
        depth_m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result_m = convert_depth_to_meters(depth_m, numpy_to_imgmsg(depth_m, "32FC1"))
        np.testing.assert_array_almost_equal(result_m, depth_m)
