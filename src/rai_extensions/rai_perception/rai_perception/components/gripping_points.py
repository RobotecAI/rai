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
"""Gripping points extraction components and pipeline.

This module provides components for extracting gripping points from segmented objects:

**Component Pipeline:**
    PointCloudFromSegmentation → PointCloudFilter → GrippingPointEstimator

**Components:**
    - PointCloudFromSegmentation: Extracts segmented point clouds from camera/depth data
    - PointCloudFilter: Filters outliers from point clouds using clustering/outlier detection
    - GrippingPointEstimator: Estimates gripping points from filtered point clouds

**Pipeline Composition:**
    Use GrippingPointsPipeline for a composed API that makes the data flow explicit.
    Individual components can also be used independently for progressive evaluation.

**Configuration:**
    Each component has a corresponding Config class:
    - PointCloudFromSegmentationConfig
    - PointCloudFilterConfig
    - GrippingPointEstimatorConfig
"""

import time
from typing import List, Literal, Optional, cast

import numpy as np
import sensor_msgs.msg
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from rai.communication.ros2.api import (
    convert_ros_img_to_ndarray,  # type: ignore[reportUnknownVariableType]
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2.ros_async import get_future_result
from rclpy import Future

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino
from rai_perception.algorithms.point_cloud import depth_to_point_cloud


class PointCloudFromSegmentationConfig(BaseModel):
    box_threshold: float = Field(
        default=0.35, description="Box threshold for GDINO object detection"
    )
    text_threshold: float = Field(
        default=0.45, description="Text threshold for GDINO object detection"
    )


class GrippingPointEstimatorConfig(BaseModel):
    strategy: Literal["centroid", "top_plane", "biggest_plane"] = Field(
        default="centroid",
        description="Strategy for estimating gripping points from point clouds",
    )
    top_percentile: float = Field(
        default=0.05,
        description="Fraction of highest Z points to consider (0.05 = top 5%)",
    )
    plane_bin_size_m: float = Field(
        default=0.01, description="Bin size in meters for plane detection"
    )
    ransac_iterations: int = Field(
        default=200, description="Number of RANSAC iterations for plane fitting"
    )
    distance_threshold_m: float = Field(
        default=0.01,
        description="Distance threshold in meters for RANSAC plane fitting",
    )
    min_points: int = Field(
        default=10, description="Minimum number of points required for processing"
    )


class PointCloudFilterConfig(BaseModel):
    """Configuration for point cloud filtering with domain-oriented parameters.

    This config uses semantic parameter names that map to robotics domain concepts
    rather than exposing algorithm-specific details. The strategy parameter selects
    the filtering approach, and other parameters control behavior in domain terms.

    Strategy mapping to algorithms:
    - "density_based": Uses DBSCAN algorithm for density-based clustering
    - "cluster_based": Uses KMeans algorithm to find largest cluster
    - "aggressive_outlier_removal": Uses Isolation Forest for aggressive noise removal
    - "conservative_outlier_removal": Uses Local Outlier Factor for conservative noise removal
    """

    strategy: Literal[
        "density_based",
        "cluster_based",
        "aggressive_outlier_removal",
        "conservative_outlier_removal",
    ] = Field(
        default="aggressive_outlier_removal",
        description=(
            "Filtering strategy for removing outliers from point clouds. "
            "Options: 'density_based' (DBSCAN), 'cluster_based' (KMeans), "
            "'aggressive_outlier_removal' (Isolation Forest), "
            "'conservative_outlier_removal' (Local Outlier Factor)"
        ),
    )
    min_points: int = Field(
        default=20, description="Minimum number of points required for filtering"
    )
    # Semantic parameters that map to algorithm-specific settings
    cluster_radius_m: float = Field(
        default=0.02,
        description=(
            "Neighborhood radius in meters for density-based clustering. "
            "Maps to DBSCAN eps parameter when strategy='density_based'"
        ),
    )
    min_cluster_size: int = Field(
        default=10,
        description=(
            "Minimum number of points required to form a cluster. "
            "Maps to DBSCAN min_samples when strategy='density_based'"
        ),
    )
    num_clusters: int = Field(
        default=2,
        description=(
            "Number of clusters to identify. "
            "Maps to KMeans n_clusters when strategy='cluster_based'"
        ),
    )
    max_samples: int | float | Literal["auto"] = Field(
        default="auto",
        description=(
            "Maximum number of samples to use for outlier detection. "
            "Maps to Isolation Forest max_samples when strategy='aggressive_outlier_removal'. "
            "Use 'auto' for automatic selection."
        ),
    )
    outlier_fraction: float = Field(
        default=0.05,
        description=(
            "Expected fraction of outliers in the point cloud (0.0 to 1.0). "
            "Maps to Isolation Forest contamination when strategy='aggressive_outlier_removal', "
            "or Local Outlier Factor contamination when strategy='conservative_outlier_removal'"
        ),
    )
    neighborhood_size: int = Field(
        default=20,
        description=(
            "Number of neighbors to consider for local density estimation. "
            "Maps to Local Outlier Factor n_neighbors when strategy='conservative_outlier_removal'"
        ),
    )


def _publish_gripping_point_debug_data(
    connector: ROS2Connector,
    obj_points_xyz: NDArray[np.float32],
    gripping_points_xyz: list[NDArray[np.float32]],
    base_frame_id: str = "egoarm_base_link",
    publish_duration: float = 5.0,
) -> None:
    """Publish the gripping point debug data to ROS2 topics which can be visualized in RVIZ.

    Args:
        connector: The ROS2 connector.
        obj_points_xyz: The list of objects found in the image.
        gripping_points_xyz: The list of gripping points in the base frame.
        base_frame_id: The base frame id.
        publish_duration: Duration in seconds to publish the data (default: 10.0).
    """

    from geometry_msgs.msg import Point, Point32, Pose, Vector3
    from sensor_msgs.msg import PointCloud
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker, MarkerArray

    debug_gripping_points_pointcloud_topic = "/debug_gripping_points_pointcloud"
    debug_gripping_points_markerarray_topic = "/debug_gripping_points_markerarray"

    connector.node.get_logger().warning(
        "Debug data publishing adds computational overhead and network traffic and impact the performance - not suitable for production. "
        f"Data will be published to {debug_gripping_points_pointcloud_topic} and {debug_gripping_points_markerarray_topic} for {publish_duration} seconds."
    )

    points = (
        np.concatenate(obj_points_xyz, axis=0)
        if obj_points_xyz
        else np.zeros((0, 3), dtype=np.float32)
    )

    msg = PointCloud()  # type: ignore[reportUnknownArgumentType]
    msg.header.frame_id = base_frame_id  # type: ignore[reportUnknownMemberType]
    msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]  # type: ignore[reportUnknownArgumentType]
    pub = connector.node.create_publisher(  # type: ignore[reportUnknownMemberType]
        PointCloud, debug_gripping_points_pointcloud_topic, 10
    )

    marker_pub = connector.node.create_publisher(  # type: ignore[reportUnknownMemberType]
        MarkerArray, debug_gripping_points_markerarray_topic, 10
    )
    marker_array = MarkerArray()
    header = Header()
    header.frame_id = base_frame_id
    header.stamp = connector.node.get_clock().now().to_msg()
    markers = []
    for i, p in enumerate(gripping_points_xyz):
        m = Marker()
        m.header = header
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = Pose(position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        m.scale = Vector3(x=0.04, y=0.04, z=0.04)
        m.id = i
        m.color.r = 1.0  # type: ignore[reportUnknownMemberType]
        m.color.g = 0.0  # type: ignore[reportUnknownMemberType]
        m.color.b = 0.0  # type: ignore[reportUnknownMemberType]
        m.color.a = 1.0  # type: ignore[reportUnknownMemberType]

        markers.append(m)  # type: ignore[reportUnknownArgumentType]
    marker_array.markers = markers

    start_time = time.time()
    publish_rate = 10.0  # Hz
    sleep_duration = 1.0 / publish_rate

    while time.time() - start_time < publish_duration:
        marker_pub.publish(marker_array)
        pub.publish(msg)
        time.sleep(sleep_duration)


class PointCloudFromSegmentation:
    """Generate a masked point cloud for an object and transform it to a target frame.

    Configure with source/target TF frames and ROS2 topics. Call run(object_name) to
    get an Nx3 numpy array of points [X, Y, Z] expressed in the target frame.
    """

    def __init__(
        self,
        *,
        connector: ROS2Connector,
        camera_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        source_frame: str,
        target_frame: str,
        conversion_ratio: float = 0.001,
        config: PointCloudFromSegmentationConfig,
    ) -> None:
        self.connector = connector
        self.camera_topic = camera_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.config = config
        self.conversion_ratio = conversion_ratio

    # --------------------- ROS helpers ---------------------
    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.connector.receive_message(topic).payload
        if isinstance(msg, sensor_msgs.msg.Image):
            return msg
        raise TypeError("Received wrong message type for Image")

    def _get_camera_info_message(self, topic: str) -> sensor_msgs.msg.CameraInfo:
        for _ in range(3):
            msg = self.connector.receive_message(topic, timeout_sec=3.0).payload
            if isinstance(msg, sensor_msgs.msg.CameraInfo):
                return msg
            self.connector.node.get_logger().warn(  # type: ignore[reportUnknownMemberType]
                "Received wrong CameraInfo message type. Retrying..."
            )
        raise RuntimeError("Failed to receive correct CameraInfo after 3 attempts")

    def _get_intrinsic_from_camera_info(
        self, camera_info: sensor_msgs.msg.CameraInfo
    ) -> tuple[float, float, float, float]:
        k = camera_info.k  # type: ignore[reportUnknownMemberType]
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        return fx, fy, cx, cy

    def _get_detection_service_name(self) -> str:
        """Get detection service name from ROS2 parameter or use default."""
        from rai_perception.components.service_utils import get_detection_service_name

        return get_detection_service_name(self.connector)

    def _get_segmentation_service_name(self) -> str:
        """Get segmentation service name from ROS2 parameter or use default."""
        from rai_perception.components.service_utils import (
            get_segmentation_service_name,
        )

        return get_segmentation_service_name(self.connector)

    def _create_service_client(self, service_type, service_name: str):
        """Create a service client for the given service type and name."""
        from rai_perception.components.service_utils import create_service_client

        return create_service_client(self.connector, service_type, service_name)

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        service_name = self._get_detection_service_name()
        cli = self._create_service_client(RAIGroundingDino, service_name)  # type: ignore[reportUnknownMemberType]
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.config.box_threshold
        req.text_threshold = self.config.text_threshold
        return cli.call_async(req)

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ) -> Future:
        service_name = self._get_segmentation_service_name()
        cli = self._create_service_client(RAIGroundedSam, service_name)  # type: ignore[reportUnknownMemberType]
        req = RAIGroundedSam.Request()
        req.detections = data.detections  # type: ignore[reportUnknownMemberType]
        req.source_img = camera_img_message
        return cli.call_async(req)

    # --------------------- Geometry helpers ---------------------
    @staticmethod
    def _quaternion_to_rotation_matrix(
        qx: float, qy: float, qz: float, qw: float
    ) -> NDArray[np.float64]:
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def _transform_points_source_to_target(
        self, points_xyz: NDArray[np.float32]
    ) -> NDArray[np.float64]:
        if points_xyz.size == 0:
            return points_xyz.reshape(0, 3).astype(np.float64)

        transform = self.connector.get_transform(self.target_frame, self.source_frame)
        t = transform.transform.translation  # type: ignore[reportUnknownMemberType]
        r = transform.transform.rotation  # type: ignore[reportUnknownMemberType]
        qw = float(r.w)  # type: ignore[reportUnknownMemberType]
        qx = float(r.x)  # type: ignore[reportUnknownMemberType]
        qy = float(r.y)  # type: ignore[reportUnknownMemberType]
        qz = float(r.z)  # type: ignore[reportUnknownMemberType]
        rotation_matrix = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
        translation = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float64)  # type: ignore[reportUnknownMemberType]

        return (points_xyz.astype(np.float64) @ rotation_matrix.T) + translation

    # --------------------- Public API ---------------------
    def run(self, object_name: str) -> list[NDArray[np.float32]]:
        """Return Nx3 numpy array [X, Y, Z] of the object's masked point cloud in target frame."""

        camera_img_msg = self._get_image_message(self.camera_topic)
        depth_msg = self.connector.receive_message(self.depth_topic).payload
        camera_info = self._get_camera_info_message(self.camera_info_topic)

        fx, fy, cx, cy = self._get_intrinsic_from_camera_info(camera_info)

        gdino_future = self._call_gdino_node(camera_img_msg, object_name)

        gdino_resolved = get_future_result(gdino_future)
        if gdino_resolved is None:
            return []

        gsam_future = self._call_gsam_node(camera_img_msg, gdino_resolved)
        gsam_resolved = get_future_result(gsam_future)
        if gsam_resolved is None or len(gsam_resolved.masks) == 0:
            return []

        depth = convert_ros_img_to_ndarray(depth_msg).astype(np.float32)
        all_points: List[NDArray[np.float32]] = []
        for mask_msg in gsam_resolved.masks:
            mask = cast(NDArray[np.uint8], convert_ros_img_to_ndarray(mask_msg))
            binary_mask = mask == 255
            masked_depth_image: NDArray[np.float32] = np.zeros_like(
                depth, dtype=np.float32
            )
            masked_depth_image[binary_mask] = depth[binary_mask]
            masked_depth_image = masked_depth_image * float(self.conversion_ratio)

            points_camera: NDArray[np.float32] = depth_to_point_cloud(
                masked_depth_image, fx, fy, cx, cy
            )
            if points_camera.size:
                all_points.append(points_camera)

        if not all_points:
            return []

        points_target = [
            self._transform_points_source_to_target(points_source).astype(np.float32)
            for points_source in all_points
        ]
        return points_target


class GrippingPointEstimator:
    """Estimate gripping points from segmented point clouds using different strategies.

    This class operates on the output of `PointCloudFromSegmentation.run`, which is
    a list of numpy arrays, one per segmented instance, each of shape (N, 3).

    Supported strategies:
      - "centroid": centroid of all points
      - "top_plane": centroid of points in the top-Z percentile (proxy for top plane)
      - "biggest_plane": centroid of the most populated horizontal plane bin (RANSAC-free)
    """

    def __init__(self, config: GrippingPointEstimatorConfig) -> None:
        self.config = config

    def _centroid(self, points: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Compute centroid of points."""
        if points.size == 0:
            return None
        return points.mean(axis=0).astype(np.float32)

    def _check_min_points(
        self, points: NDArray[np.float32]
    ) -> Optional[NDArray[np.float32]]:
        """Check if points meet minimum requirement, return centroid fallback if not."""
        if points.shape[0] < self.config.min_points:
            return self._centroid(points)
        return None

    def _top_plane_centroid(
        self, points: NDArray[np.float32]
    ) -> Optional[NDArray[np.float32]]:
        fallback = self._check_min_points(points)
        if fallback is not None:
            return fallback

        z_vals = points[:, 2]
        threshold = np.quantile(z_vals, 1.0 - self.config.top_percentile)
        top_points = points[z_vals >= threshold]

        if top_points.shape[0] == 0:
            return self._centroid(points)
        return top_points.mean(axis=0).astype(np.float32)

    def _biggest_plane_centroid(
        self, points: NDArray[np.float32]
    ) -> Optional[NDArray[np.float32]]:
        """RANSAC plane detection: find largest plane and return its centroid."""
        fallback = self._check_min_points(points)
        if fallback is not None:
            return fallback

        best_inlier_mask = self._ransac_plane_detection(points)
        if best_inlier_mask is None:
            return self._centroid(points)

        inlier_points = points[best_inlier_mask]
        if inlier_points.shape[0] == 0:
            return self._centroid(points)
        return inlier_points.mean(axis=0).astype(np.float32)

    def _ransac_plane_detection(
        self, points: NDArray[np.float32]
    ) -> Optional[NDArray[np.bool_]]:
        """Perform RANSAC plane detection and return inlier mask."""
        num_points = points.shape[0]
        pts64 = points.astype(np.float64, copy=False)
        threshold = float(self.config.distance_threshold_m)
        rng = np.random.default_rng()

        best_inlier_count = 0
        best_inlier_mask: Optional[NDArray[np.bool_]] = None

        for _ in range(self.config.ransac_iterations):
            # Sample 3 unique points
            idxs = rng.choice(num_points, size=3, replace=False)
            p0, p1, p2 = pts64[idxs[0]], pts64[idxs[1]], pts64[idxs[2]]

            # Compute plane normal
            v1, v2 = p1 - p0, p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue  # Degenerate triplet
            normal /= norm_len

            # Compute distances to plane: |normal · (x - p0)|
            dists = np.abs((pts64 - p0) @ normal)
            inliers = dists <= threshold
            count = int(inliers.sum())

            if count > best_inlier_count:
                best_inlier_count = count
                best_inlier_mask = inliers

        if best_inlier_count < self.config.min_points:
            return None
        return best_inlier_mask

    def _get_strategy_method(self):
        """Get the strategy method based on config."""
        strategy_map = {
            "centroid": self._centroid,
            "top_plane": self._top_plane_centroid,
            "biggest_plane": self._biggest_plane_centroid,
        }
        return strategy_map.get(self.config.strategy, self._centroid)

    def run(
        self, segmented_point_clouds: list[NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        """Compute gripping points for each segmented point cloud.

        Parameters
        ----------
        segmented_point_clouds: list of (N, 3) arrays in target frame.

        Returns
        -------
        list of np.array points [[x, y, z], ...], one per input cloud.
        """
        strategy_method = self._get_strategy_method()
        gripping_points: list[NDArray[np.float32]] = []

        for pts in segmented_point_clouds:
            if pts.size == 0:
                continue

            gp = strategy_method(pts)
            if gp is not None:
                gripping_points.append(gp.astype(np.float32))

        return gripping_points


class PointCloudFilter:
    """Filter segmented point clouds using domain-oriented filtering strategies.

    This class provides semantic filtering strategies that map to underlying algorithms:

    - "density_based": Uses DBSCAN for density-based clustering, keeps largest cluster
    - "cluster_based": Uses KMeans clustering, keeps largest cluster
    - "aggressive_outlier_removal": Uses Isolation Forest for aggressive noise removal
    - "conservative_outlier_removal": Uses Local Outlier Factor for conservative noise removal

    All strategies use domain-oriented parameters (e.g., cluster_radius_m, outlier_fraction)
    rather than exposing algorithm-specific details.
    """

    def __init__(self, config: PointCloudFilterConfig) -> None:
        self.config = config

    def _check_min_points(
        self, pts: NDArray[np.float32], min_required: int = 0
    ) -> bool:
        """Check if points meet minimum requirement.

        Args:
            pts: Point cloud array
            min_required: Custom minimum requirement. If > 0, overrides config.min_points.
        """
        min_threshold = min_required if min_required > 0 else self.config.min_points
        return pts.shape[0] >= min_threshold

    def _apply_mask_or_fallback(
        self, pts: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """Apply mask to points, return original if mask is empty."""
        if not np.any(mask):
            return pts
        return pts[mask]

    def _get_largest_cluster(self, labels: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Get mask for the largest cluster from labels."""
        if labels.size == 0:
            return np.zeros(0, dtype=bool)

        # For DBSCAN, exclude outliers (label -1)
        valid = labels >= 0
        if not np.any(valid):
            return np.zeros(labels.shape[0], dtype=bool)

        labels_valid = labels[valid]
        unique_labels, counts = np.unique(labels_valid, return_counts=True)
        dominant = unique_labels[np.argmax(counts)]
        return labels == dominant

    def _filter_dbscan(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter using DBSCAN density-based clustering.

        Maps semantic parameters:
        - cluster_radius_m -> DBSCAN eps
        - min_cluster_size -> DBSCAN min_samples
        """
        from sklearn.cluster import DBSCAN  # type: ignore[reportMissingImports]

        if not self._check_min_points(pts):
            return pts

        db = DBSCAN(
            eps=self.config.cluster_radius_m, min_samples=self.config.min_cluster_size
        )
        labels = cast(NDArray[np.int64], db.fit_predict(pts))  # type: ignore[no-any-return]
        mask = self._get_largest_cluster(labels)
        return self._apply_mask_or_fallback(pts, mask)

    def _filter_kmeans_largest(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter using KMeans clustering, keeping largest cluster.

        Maps semantic parameters:
        - num_clusters -> KMeans n_clusters
        """
        from sklearn.cluster import KMeans  # type: ignore[reportMissingImports]

        if not self._check_min_points(pts, self.config.num_clusters):
            return pts

        kmeans = KMeans(n_clusters=self.config.num_clusters, n_init="auto")
        labels = cast(NDArray[np.int64], kmeans.fit_predict(pts))  # type: ignore[no-any-return]
        mask = self._get_largest_cluster(labels)
        return self._apply_mask_or_fallback(pts, mask)

    def _filter_isolation_forest(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter using Isolation Forest for aggressive outlier removal.

        Maps semantic parameters:
        - max_samples -> Isolation Forest max_samples
        - outlier_fraction -> Isolation Forest contamination
        """
        from sklearn.ensemble import (
            IsolationForest,  # type: ignore[reportMissingImports]
        )

        if not self._check_min_points(pts):
            return pts

        iso = IsolationForest(
            max_samples=self.config.max_samples,
            contamination=self.config.outlier_fraction,
            random_state=42,
        )
        pred = cast(NDArray[np.int64], iso.fit_predict(pts))  # type: ignore[no-any-return]
        mask = pred == 1  # 1 = inlier, -1 = outlier
        return self._apply_mask_or_fallback(pts, mask)

    def _filter_lof(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter using Local Outlier Factor for conservative outlier removal.

        Maps semantic parameters:
        - neighborhood_size -> Local Outlier Factor n_neighbors
        - outlier_fraction -> Local Outlier Factor contamination
        """
        from sklearn.neighbors import (
            LocalOutlierFactor,  # type: ignore[reportMissingImports]
        )

        if not self._check_min_points(pts, self.config.neighborhood_size + 1):
            return pts

        lof = LocalOutlierFactor(
            n_neighbors=self.config.neighborhood_size,
            contamination=self.config.outlier_fraction,
        )
        pred = cast(NDArray[np.int64], lof.fit_predict(pts))  # type: ignore[no-any-return]
        mask = pred == 1  # 1 = inlier, -1 = outlier
        return self._apply_mask_or_fallback(pts, mask)

    def _get_strategy_method(self):
        """Get the filter strategy method based on config.

        Maps semantic strategy names to algorithm implementations:
        - "density_based" -> DBSCAN algorithm
        - "cluster_based" -> KMeans algorithm
        - "aggressive_outlier_removal" -> Isolation Forest algorithm
        - "conservative_outlier_removal" -> Local Outlier Factor algorithm
        """
        strategy_map = {
            "density_based": self._filter_dbscan,
            "cluster_based": self._filter_kmeans_largest,
            "aggressive_outlier_removal": self._filter_isolation_forest,
            "conservative_outlier_removal": self._filter_lof,
        }
        return strategy_map.get(self.config.strategy, lambda pts: pts)

    def run(
        self, segmented_point_clouds: list[NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        """Filter each point cloud using the configured strategy.

        Parameters
        ----------
        segmented_point_clouds: list of (N, 3) arrays to filter.

        Returns
        -------
        list of filtered (N, 3) arrays.
        """
        filter_method = self._get_strategy_method()
        filtered: list[NDArray[np.float32]] = []

        for pts in segmented_point_clouds:
            if pts.size == 0:
                continue
            filtered.append(filter_method(pts).astype(np.float32, copy=False))

        return filtered
