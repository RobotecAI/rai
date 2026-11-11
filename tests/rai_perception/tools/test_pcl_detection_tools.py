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
import time

import numpy as np
import pytest
from rai.communication.ros2 import QoS
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.tools import Effect
from rai_perception import (
    GetObjectGrippingPointsTool,
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
)
from rai_perception.components.perception_presets import apply_preset
from rai_perception.tools import GetObjectPositionsTool

from tests.rai_perception.components.test_gripping_points import (
    make_pipeline_connector,
)

TOPICS = {
    "camera_topic": "/test/camera",
    "depth_topic": "/test/depth",
    "camera_info_topic": "/test/camera_info",
}


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


@pytest.fixture
def pipeline(mock_connector):
    """One 2x2 cube-ish object 2 m in front of the camera."""
    depth = numpy_to_imgmsg(np.full((2, 2), 2.0, np.float32), "32FC1")
    mask = np.full((2, 2), 255, np.uint8)
    return make_pipeline_connector(mock_connector, depth, mask, (1.0, 0.0, 0.0))


def test_defaults_effect_and_targets(mock_connector):
    tool = GetObjectGrippingPointsTool(connector=mock_connector)
    assert tool.effect is Effect.READ
    assert tool.timeout_s == 60.0
    assert tool.targets == (
        "/camera/rgb/image_raw",
        "/camera/depth/image_raw",
        "/camera/rgb/camera_info",
        "/detection",
        "/segmentation",
    )
    assert tool.get_config() == {
        "target_frame": "base_link",
        "source_frame": "camera_link",
        "camera_topic": "/camera/rgb/image_raw",
        "depth_topic": "/camera/depth/image_raw",
        "camera_info_topic": "/camera/rgb/camera_info",
        "timeout_s": 60.0,
        "conversion_ratio": 1.0,
        "detection_service_name": "/detection",
        "segmentation_service_name": "/segmentation",
    }
    assert tool.estimator_config == apply_preset("default_grasp")[1]


def test_configuration_reaches_the_pipeline(mock_connector):
    tool = GetObjectGrippingPointsTool(
        connector=mock_connector,
        target_frame="panda_link0",
        detection_service_name="/det",
        conversion_ratio=0.5,
        **TOPICS,
    )
    extraction = tool.point_cloud_from_segmentation
    assert extraction.target_frame == "panda_link0"
    assert extraction.camera_topic == "/test/camera"
    assert extraction.detection_service == "/det"
    assert extraction.conversion_ratio == 0.5


def test_pipeline_result(pipeline):
    tool = GetObjectGrippingPointsTool(connector=pipeline, **TOPICS)
    result = tool.invoke({"object_name": "cube"})
    assert result.startswith("Centroids of detected cube in base_link frame: Centroid(")
    pipeline.publish.assert_not_called()


def test_debug_publishes_latched_pipeline_data(pipeline):
    tool = GetObjectGrippingPointsTool(
        connector=pipeline,
        filter_config=PointCloudFilterConfig(min_points=100),  # keeps 4 points
        **TOPICS,
    )
    asyncio.run(tool.ainvoke({"object_name": "cube", "debug": True}))
    published = {c.args[0]: c.kwargs["qos"] for c in pipeline.publish.call_args_list}
    assert set(published) == {
        "/debug/gripping_points/raw_point_clouds",
        "/debug/gripping_points/filtered_point_clouds",
        "/debug_gripping_points_pointcloud",
        "/debug_gripping_points_markerarray",
    }
    assert all(qos is QoS.LATCHED for qos in published.values())


def test_no_objects(pipeline):
    pipeline.call.side_effect = None
    pipeline.call.return_value.masks = []
    tool = GetObjectGrippingPointsTool(connector=pipeline, **TOPICS)
    assert tool.invoke({"object_name": "cube"}) == "No cubes detected."


@pytest.mark.parametrize("path", ["sync", "async"])
@pytest.mark.parametrize("stage", ["ros", "cpu"])
def test_timeout_is_enforced_on_every_path(mock_connector, monkeypatch, path, stage):
    """The A3 regression: timeout_s bounds the call on the sync path too, and
    also while a CPU stage runs in a thread."""
    tool = GetObjectGrippingPointsTool(connector=mock_connector, timeout_s=0.2)

    async def slow_ros(object_name):
        await asyncio.sleep(5.0)

    def slow_cpu(clouds):
        time.sleep(1.0)
        return clouds

    if stage == "ros":
        monkeypatch.setattr(tool.point_cloud_from_segmentation, "run", slow_ros)
    else:

        async def one_cloud(object_name):
            return [np.zeros((3, 3), np.float32)]

        monkeypatch.setattr(tool.point_cloud_from_segmentation, "run", one_cloud)
        monkeypatch.setattr(tool.point_cloud_filter, "run", slow_cpu)

    start = time.monotonic()
    if path == "sync":
        result = tool.invoke({"object_name": "cube"})
    else:
        result = asyncio.run(tool.ainvoke({"object_name": "cube"}))
    assert result == "Timed out after 0.2s"
    assert time.monotonic() - start < 0.8


def test_custom_output_labels(mock_connector):
    tool = GetObjectGrippingPointsTool(
        connector=mock_connector,
        output_point_label="GraspPoint",
        output_description_suffix=" (top)",
        estimator_config=GrippingPointEstimatorConfig(strategy="top_plane"),
    )
    points = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    result = tool._format_result_message("cup", points)
    assert result.startswith("GraspPoints of detected cups (top) in base_link frame:")
    assert "GraspPoint(x=1.000000, y=2.000000, z=3.000000)" in result


def test_get_object_positions_tool(pipeline):
    tool = GetObjectPositionsTool(
        connector=pipeline,
        estimator_config=GrippingPointEstimatorConfig(strategy="top_plane"),
        target_frame="panda_link0",
        **TOPICS,
    )
    assert tool.name == "get_object_positions"
    assert set(tool.args) == {"object_name"}
    assert tool.estimator_config.strategy == "centroid"  # default_grasp preset
    assert tool.get_config()["target_frame"] == "panda_link0"
    assert tool.invoke({"object_name": "cube"}).startswith(
        "Centroids of detected cube in panda_link0 frame"
    )


def test_estimator_and_filter_are_used(mock_connector):
    tool = GetObjectGrippingPointsTool(connector=mock_connector)
    assert isinstance(tool.point_cloud_filter, PointCloudFilter)
    assert isinstance(tool.gripping_point_estimator, GrippingPointEstimator)
