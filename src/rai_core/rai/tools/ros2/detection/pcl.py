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
import time
from typing import List, Literal, Optional, cast

import numpy as np
import sensor_msgs.msg
from numpy.typing import NDArray
from rai_open_set_vision import GDINO_SERVICE_NAME
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai.communication.ros2.api import (
    convert_ros_img_to_ndarray,  # type: ignore[reportUnknownVariableType]
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2.ros_async import get_future_result
from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino


def depth_to_point_cloud(
    depth_image: NDArray[np.float32], fx: float, fy: float, cx: float, cy: float
) -> NDArray[np.float32]:
    height, width = depth_image.shape
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    z = depth_image
    x = (x_grid - float(cx)) * z / float(fx)
    y = (y_grid - float(cy)) * z / float(fy)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] > 0]
    return points.astype(np.float32, copy=False)


def _publish_gripping_point_debug_data(
    connector: ROS2Connector,
    obj_points_xyz: NDArray[np.float32],
    gripping_points_xyz: list[NDArray[np.float32]],
    base_frame_id: str = "egoarm_base_link",
    publish_duration: float = 10.0,
) -> None:
    """Publish the gripping point debug data for visualization in RVIZ via point cloud and marker array.

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

    points = (
        np.concatenate(obj_points_xyz, axis=0)
        if obj_points_xyz
        else np.zeros((0, 3), dtype=np.float32)
    )

    msg = PointCloud()  # type: ignore[reportUnknownArgumentType]
    msg.header.frame_id = base_frame_id  # type: ignore[reportUnknownMemberType]
    msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]  # type: ignore[reportUnknownArgumentType]
    pub = connector.node.create_publisher(  # type: ignore[reportUnknownMemberType]
        PointCloud, "/debug_gripping_points_pointcloud", 10
    )

    marker_pub = connector.node.create_publisher(  # type: ignore[reportUnknownMemberType]
        MarkerArray, "/debug_gripping_points_markerarray", 10
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

        # m.ns = str(i)

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

    connector: ROS2Connector
    camera_topic: str
    depth_topic: str
    camera_info_topic: str
    source_frame: str
    target_frame: str

    box_threshold: float = 0.35
    text_threshold: float = 0.45

    def __init__(
        self,
        *,
        connector: ROS2Connector,
        camera_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        source_frame: str,
        target_frame: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.45,
    ) -> None:
        self.connector = connector
        self.camera_topic = camera_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

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

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)  # type: ignore[reportUnknownMemberType]
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(  # type: ignore[reportUnknownMemberType]
                f"service {GDINO_SERVICE_NAME} not available, waiting again..."
            )
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold
        return cli.call_async(req)

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundedSam, "grounded_sam_segment")  # type: ignore[reportUnknownMemberType]
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(  # type: ignore[reportUnknownMemberType]
                "service grounded_sam_segment not available, waiting again..."
            )
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

        logger = self.connector.node.get_logger()
        try:
            conversion_ratio_value = self.connector.node.get_parameter(
                "conversion_ratio"
            ).value  # type: ignore[reportUnknownMemberType]
            conversion_ratio: float
            if isinstance(conversion_ratio_value, float):
                conversion_ratio = conversion_ratio_value
            else:
                logger.error(  # type: ignore[reportUnknownMemberType]
                    "Parameter conversion_ratio has wrong type. Using default 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning("Parameter conversion_ratio not found. Using default 0.001")  # type: ignore[reportUnknownMemberType]
            conversion_ratio = 0.001

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
            masked_depth_image = masked_depth_image * float(conversion_ratio)

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

    strategy: Literal["centroid", "top_plane", "biggest_plane"]
    top_percentile: float
    plane_bin_size_m: float
    ransac_iterations: int
    distance_threshold_m: float
    min_points: int

    def __init__(
        self,
        *,
        strategy: Literal["centroid", "top_plane", "biggest_plane"] = "centroid",
        top_percentile: float = 0.05,
        plane_bin_size_m: float = 0.01,
        ransac_iterations: int = 200,
        distance_threshold_m: float = 0.01,
        min_points: int = 10,
    ) -> None:
        self.strategy = strategy
        self.top_percentile = top_percentile
        self.plane_bin_size_m = plane_bin_size_m
        self.ransac_iterations = int(max(1, ransac_iterations))
        self.distance_threshold_m = float(max(1e-6, distance_threshold_m))
        self.min_points = min_points

    def _centroid(self, points: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        if points.size == 0:
            return None
        return points.mean(axis=0).astype(np.float32)

    def _top_plane_centroid(
        self, points: NDArray[np.float32]
    ) -> Optional[NDArray[np.float32]]:
        if points.shape[0] < self.min_points:
            return self._centroid(points)
        z_vals = points[:, 2]
        threshold = np.quantile(z_vals, 1.0 - self.top_percentile)
        mask = z_vals >= threshold
        top_points = points[mask]
        if top_points.shape[0] == 0:
            return self._centroid(points)
        return top_points.mean(axis=0).astype(np.float32)

    def _biggest_plane_centroid(
        self, points: NDArray[np.float32]
    ) -> Optional[NDArray[np.float32]]:
        # RANSAC plane detection: not restricted to horizontal planes
        num_points = points.shape[0]
        if num_points < self.min_points:
            return self._centroid(points)

        best_inlier_count = 0
        best_inlier_mask: Optional[NDArray[np.bool_]] = None

        # Precompute for speed
        pts64 = points.astype(np.float64, copy=False)
        threshold = float(self.distance_threshold_m)

        rng = np.random.default_rng()

        for _ in range(self.ransac_iterations):
            # Sample 3 unique points
            idxs = rng.choice(num_points, size=3, replace=False)
            p0, p1, p2 = pts64[idxs[0]], pts64[idxs[1]], pts64[idxs[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue  # degenerate triplet
            normal /= norm_len
            # Distance from points to plane
            # Plane eq: normal · (x - p0) = 0 -> distance = |normal · (x - p0)|
            diffs = pts64 - p0
            dists = np.abs(diffs @ normal)
            inliers = dists <= threshold
            count = int(inliers.sum())
            if count > best_inlier_count:
                best_inlier_count = count
                best_inlier_mask = inliers

        if best_inlier_mask is None or best_inlier_count < self.min_points:
            return self._centroid(points)

        inlier_points = points[best_inlier_mask]
        if inlier_points.shape[0] == 0:
            return self._centroid(points)
        return inlier_points.mean(axis=0).astype(np.float32)

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
        gripping_points: list[NDArray[np.float32]] = []

        for pts in segmented_point_clouds:
            if pts.size == 0:
                continue
            if self.strategy == "centroid":
                gp = self._centroid(pts)
            elif self.strategy == "top_plane":
                gp = self._top_plane_centroid(pts)
            elif self.strategy == "biggest_plane":
                gp = self._biggest_plane_centroid(pts)
            else:
                gp = self._centroid(pts)

            if gp is not None:
                gripping_points.append(gp.astype(np.float32))

        return gripping_points


class PointCloudFilter:
    """Filter segmented point clouds using various sklearn strategies.

    Strategies:
      - "dbscan": keep the largest DBSCAN cluster (exclude label -1)
      - "kmeans_largest_cluster": keep the largest KMeans cluster
      - "isolation_forest": keep inliers (pred == 1)
      - "lof": keep inliers (pred == 1)
    """

    strategy: Literal["dbscan", "kmeans_largest_cluster", "isolation_forest", "lof"]
    min_points: int
    # DBSCAN
    dbscan_eps: float
    dbscan_min_samples: int
    # KMeans
    kmeans_k: int
    # Isolation Forest
    if_max_samples: int | float | Literal["auto"]
    if_contamination: float
    # LOF
    lof_n_neighbors: int
    lof_contamination: float

    def __init__(
        self,
        *,
        strategy: Literal[
            "dbscan", "kmeans_largest_cluster", "isolation_forest", "lof"
        ] = "dbscan",
        min_points: int = 20,
        dbscan_eps: float = 0.02,
        dbscan_min_samples: int = 10,
        kmeans_k: int = 2,
        if_max_samples: int | float | Literal["auto"] = "auto",
        if_contamination: float = 0.05,
        lof_n_neighbors: int = 20,
        lof_contamination: float = 0.05,
    ) -> None:
        self.strategy = strategy
        self.min_points = min_points
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kmeans_k = kmeans_k
        self.if_max_samples = if_max_samples
        self.if_contamination = if_contamination
        self.lof_n_neighbors = lof_n_neighbors
        self.lof_contamination = lof_contamination

    def _filter_dbscan(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        from sklearn.cluster import DBSCAN  # type: ignore[reportMissingImports]

        if pts.shape[0] < self.min_points:
            return pts
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = cast(NDArray[np.int64], db.fit_predict(pts))  # type: ignore[no-any-return]
        if labels.size == 0:
            return pts
        valid = labels >= 0
        if not np.any(valid):
            return pts
        labels_valid = labels[valid]
        unique_labels, counts = np.unique(labels_valid, return_counts=True)
        dominant = unique_labels[np.argmax(counts)]
        mask = labels == dominant
        return pts[mask]

    def _filter_kmeans_largest(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        from sklearn.cluster import KMeans  # type: ignore[reportMissingImports]

        if pts.shape[0] < max(self.min_points, self.kmeans_k):
            return pts
        kmeans = KMeans(n_clusters=self.kmeans_k, n_init="auto")
        labels = cast(NDArray[np.int64], kmeans.fit_predict(pts))  # type: ignore[no-any-return]
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant = unique_labels[np.argmax(counts)]
        mask = labels == dominant
        return pts[mask]

    def _filter_isolation_forest(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        from sklearn.ensemble import (
            IsolationForest,  # type: ignore[reportMissingImports]
        )

        if pts.shape[0] < self.min_points:
            return pts
        iso = IsolationForest(
            max_samples=self.if_max_samples,
            contamination=self.if_contamination,
            random_state=42,
        )
        pred = cast(NDArray[np.int64], iso.fit_predict(pts))  # type: ignore[no-any-return]  # 1 inlier, -1 outlier
        mask = pred == 1
        if not np.any(mask):
            return pts
        return pts[mask]

    def _filter_lof(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        from sklearn.neighbors import (
            LocalOutlierFactor,  # type: ignore[reportMissingImports]
        )

        if pts.shape[0] < max(self.min_points, self.lof_n_neighbors + 1):
            return pts
        lof = LocalOutlierFactor(
            n_neighbors=self.lof_n_neighbors, contamination=self.lof_contamination
        )
        pred = cast(NDArray[np.int64], lof.fit_predict(pts))  # type: ignore[no-any-return]  # 1 inlier, -1 outlier
        mask = pred == 1
        if not np.any(mask):
            return pts
        return pts[mask]

    def run(
        self, segmented_point_clouds: list[NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        filtered: list[NDArray[np.float32]] = []
        for pts in segmented_point_clouds:
            if pts.size == 0:
                continue
            if self.strategy == "dbscan":
                f = self._filter_dbscan(pts)
            elif self.strategy == "kmeans_largest_cluster":
                f = self._filter_kmeans_largest(pts)
            elif self.strategy == "isolation_forest":
                f = self._filter_isolation_forest(pts)
            elif self.strategy == "lof":
                f = self._filter_lof(pts)
            else:
                f = pts
            filtered.append(f.astype(np.float32, copy=False))
        return filtered
