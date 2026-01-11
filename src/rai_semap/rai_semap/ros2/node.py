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

import logging
from pathlib import Path
from typing import Optional, Tuple

# ROS2 core
import rclpy
import yaml
from cv_bridge import CvBridge

# ROS2 geometry and transforms
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rai.communication.ros2 import ROS2Connector
from rai_perception.components.perception_utils import extract_pointcloud_from_bbox
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_pose_stamped
from vision_msgs.msg import Detection2D

# RAI interfaces
from rai_interfaces.msg import RAIDetectionArray

# Local imports
from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticAnnotation, SemanticMapMemory
from rai_semap.utils.ros2_log import ROS2LogHandler

# Constants
DEFAULT_QUEUE_SIZE = 10
TF_LOOKUP_TIMEOUT_SEC = 1.0


class SemanticMapNode:
    """ROS2 node for semantic map processing."""

    def __init__(
        self,
        connector: ROS2Connector,
        database_path: Optional[str] = None,
    ):
        """Initialize SemanticMapNode.

        Args:
            connector: ROS2Connector instance for ROS2 communication.
            database_path: Optional path to SQLite database file.
        """
        self.connector = connector

        # Configure Python logging to forward to ROS2 logger
        # Configure all rai_semap loggers (including submodules)
        handler = ROS2LogHandler(self.connector.node)
        handler.setLevel(logging.DEBUG)

        # Configure root rai_semap logger
        python_logger = logging.getLogger("rai_semap")
        python_logger.setLevel(logging.DEBUG)
        python_logger.handlers.clear()  # Remove any existing handlers
        python_logger.addHandler(handler)
        python_logger.propagate = False  # Prevent propagation to root logger

        # Also explicitly configure SQLite backend logger
        sqlite_logger = logging.getLogger("rai_semap.core.backend.sqlite_backend")
        sqlite_logger.setLevel(logging.DEBUG)
        sqlite_logger.handlers.clear()
        sqlite_logger.addHandler(handler)
        sqlite_logger.propagate = False

        self._initialize_parameters()
        if database_path is not None:
            self.connector.node.set_parameters(
                [
                    rclpy.parameter.Parameter(
                        "database_path",
                        rclpy.parameter.Parameter.Type.STRING,
                        database_path,
                    )
                ]
            )
        self._parse_class_thresholds()
        self._parse_class_merge_thresholds()
        self.bridge = CvBridge()
        self.last_depth_image: Optional[Image] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self._initialize_memory()
        self._initialize_subscriptions()

    def _initialize_parameters(self):
        """Initialize ROS2 parameters from YAML config file."""
        current_dir = Path(__file__).parent
        config_dir = current_dir / "config"

        # Declare config file path parameter
        self.connector.node.declare_parameter(
            "node_config",
            "",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Path to node YAML config file (empty = use default in config/)",
            ),
        )

        # Get config file path
        node_config_path = (
            self.connector.node.get_parameter("node_config")
            .get_parameter_value()
            .string_value
        )
        if node_config_path:
            node_yaml = Path(node_config_path)
        else:
            node_yaml = config_dir / "node.yaml"

        # Load YAML config
        with open(node_yaml, "r") as f:
            node_config = yaml.safe_load(f)
        node_params = node_config.get("rai_semap_node", {}).get("ros__parameters", {})

        # Extract grouped parameters with defaults
        storage = node_params.get("storage", {})
        detection_filtering = node_params.get("detection_filtering", {})
        deduplication = node_params.get("deduplication", {})
        topics = node_params.get("topics", {})
        map_config = node_params.get("map", {})

        # Declare all parameters with descriptions
        parameters = [
            # Storage
            (
                "backend_type",
                storage.get("backend_type", "sqlite"),
                ParameterType.PARAMETER_STRING,
                "Database backend type: 'sqlite' (default) or 'postgres' (future)",
            ),
            (
                "database_path",
                storage.get("database_path", "semantic_map.db"),
                ParameterType.PARAMETER_STRING,
                "Path to database file (SQLite) or connection string (PostgreSQL)",
            ),
            (
                "location_id",
                storage.get("location_id", "default_location"),
                ParameterType.PARAMETER_STRING,
                "Identifier for the physical location",
            ),
            # Detection filtering
            (
                "confidence_threshold",
                detection_filtering.get("confidence_threshold", 0.5),
                ParameterType.PARAMETER_DOUBLE,
                "Minimum confidence score (0.0-1.0) for storing detections",
            ),
            (
                "class_confidence_thresholds",
                detection_filtering.get("class_confidence_thresholds", ""),
                ParameterType.PARAMETER_STRING,
                "Class-specific thresholds as 'class1:threshold1,class2:threshold2' (e.g., 'person:0.7,window:0.6')",
            ),
            (
                "min_bbox_area",
                detection_filtering.get("min_bbox_area", 100.0),
                ParameterType.PARAMETER_DOUBLE,
                "Minimum bounding box area (pixels^2) to filter small false positives",
            ),
            # Deduplication
            (
                "class_merge_thresholds",
                deduplication.get("class_merge_thresholds", ""),
                ParameterType.PARAMETER_STRING,
                "Class-specific merge radii (meters) for deduplication as 'class1:radius1,class2:radius2' (e.g., 'couch:2.5,table:1.5')",
            ),
            (
                "use_pointcloud_dedup",
                deduplication.get("use_pointcloud_dedup", True),
                ParameterType.PARAMETER_BOOL,
                "Use point cloud features for improved deduplication matching",
            ),
            # Topics
            (
                "detection_topic",
                topics.get("detection_topic", "/detection_array"),
                ParameterType.PARAMETER_STRING,
                "Topic for RAIDetectionArray messages",
            ),
            (
                "map_topic",
                topics.get("map_topic", "/map"),
                ParameterType.PARAMETER_STRING,
                "Topic for OccupancyGrid map messages",
            ),
            (
                "depth_topic",
                topics.get("depth_topic", ""),
                ParameterType.PARAMETER_STRING,
                "Depth image topic (optional, for point cloud extraction)",
            ),
            (
                "camera_info_topic",
                topics.get("camera_info_topic", ""),
                ParameterType.PARAMETER_STRING,
                "Camera info topic (optional, for point cloud extraction)",
            ),
            # Map/SLAM
            (
                "map_frame_id",
                map_config.get("map_frame_id", "map"),
                ParameterType.PARAMETER_STRING,
                "Frame ID of the SLAM map",
            ),
            (
                "map_resolution",
                map_config.get("map_resolution", 0.05),
                ParameterType.PARAMETER_DOUBLE,
                "OccupancyGrid resolution (meters/pixel)",
            ),
        ]

        for name, default, param_type, description in parameters:
            self.connector.node.declare_parameter(
                name,
                default,
                descriptor=ParameterDescriptor(
                    type=param_type,
                    description=description,
                ),
            )

    def _get_string_parameter(self, name: str) -> str:
        """Get string parameter value."""
        return (
            self.connector.node.get_parameter(name).get_parameter_value().string_value
        )

    def _get_double_parameter(self, name: str) -> float:
        """Get double parameter value."""
        return (
            self.connector.node.get_parameter(name).get_parameter_value().double_value
        )

    def _parse_class_thresholds(self):
        """Parse class-specific confidence thresholds from parameter."""
        thresholds_str = self._get_string_parameter("class_confidence_thresholds")
        self.class_thresholds = {}
        if thresholds_str:
            for item in thresholds_str.split(","):
                item = item.strip()
                if ":" in item:
                    class_name, threshold_str = item.split(":", 1)
                    try:
                        threshold = float(threshold_str.strip())
                        self.class_thresholds[class_name.strip()] = threshold
                        self.connector.node.get_logger().info(
                            f"Class-specific threshold: {class_name.strip()}={threshold:.3f}"
                        )
                    except ValueError:
                        self.connector.node.get_logger().warning(
                            f"Invalid threshold value in '{item}', skipping"
                        )

    def _parse_class_merge_thresholds(self):
        """Parse class-specific merge thresholds from parameter."""
        merge_thresholds_str = self._get_string_parameter("class_merge_thresholds")
        self.class_merge_thresholds = {}
        if merge_thresholds_str:
            for item in merge_thresholds_str.split(","):
                item = item.strip()
                if ":" in item:
                    class_name, radius_str = item.split(":", 1)
                    try:
                        radius = float(radius_str.strip())
                        self.class_merge_thresholds[class_name.strip()] = radius
                        self.connector.node.get_logger().info(
                            f"Class-specific merge radius: {class_name.strip()}={radius:.2f}m"
                        )
                    except ValueError:
                        self.connector.node.get_logger().warning(
                            f"Invalid merge radius value in '{item}', skipping"
                        )

    def _initialize_memory(self):
        """Initialize semantic map memory backend."""
        backend_type = self._get_string_parameter("backend_type")
        database_path = self._get_string_parameter("database_path")
        location_id = self._get_string_parameter("location_id")
        map_frame_id = self._get_string_parameter("map_frame_id")
        map_resolution = self._get_double_parameter("map_resolution")

        if backend_type == "sqlite":
            backend = SQLiteBackend(database_path)
        elif backend_type == "postgres":
            raise NotImplementedError(
                "PostgreSQL backend not yet implemented. Use 'sqlite' for now."
            )
        else:
            raise ValueError(
                f"Unknown backend_type: {backend_type}. Supported: 'sqlite'"
            )

        backend.init_schema()
        self.memory = SemanticMapMemory(
            backend=backend,
            location_id=location_id,
            map_frame_id=map_frame_id,
            resolution=map_resolution,
        )
        self.connector.node.get_logger().info(
            f"Initialized semantic map memory: backend={backend_type}, "
            f"location_id={location_id}, map_frame_id={map_frame_id}, "
            f"database_path={database_path}"
        )

    def _initialize_subscriptions(self):
        """Initialize ROS2 subscriptions."""
        detection_topic = self._get_string_parameter("detection_topic")
        map_topic = self._get_string_parameter("map_topic")

        self.detection_subscription = self.connector.node.create_subscription(
            RAIDetectionArray,
            detection_topic,
            self.detection_callback,
            qos_profile_sensor_data,
        )
        self.connector.node.get_logger().info(
            f"Subscribed to detection topic: {detection_topic} "
            f"(QoS: reliability={qos_profile_sensor_data.reliability.name})"
        )
        self.map_subscription = self.connector.node.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, DEFAULT_QUEUE_SIZE
        )

        # Optional depth and camera info for point cloud extraction
        depth_topic = self._get_string_parameter("depth_topic")
        camera_info_topic = self._get_string_parameter("camera_info_topic")
        use_pointcloud = (
            self.connector.node.get_parameter("use_pointcloud_dedup")
            .get_parameter_value()
            .bool_value
        )

        if use_pointcloud and depth_topic:
            self.depth_subscription = self.connector.node.create_subscription(
                Image, depth_topic, self.depth_callback, qos_profile_sensor_data
            )
            self.connector.node.get_logger().info(
                f"Subscribed to depth topic: {depth_topic}"
            )
        else:
            self.depth_subscription = None

        if use_pointcloud and camera_info_topic:
            self.camera_info_subscription = self.connector.node.create_subscription(
                CameraInfo,
                camera_info_topic,
                self.camera_info_callback,
                qos_profile_sensor_data,
            )
            self.connector.node.get_logger().info(
                f"Subscribed to camera info topic: {camera_info_topic}"
            )
        else:
            self.camera_info_subscription = None

        self.connector.node.get_logger().info(
            f"Subscribed to detection_topic={detection_topic}, map_topic={map_topic}"
        )

    def depth_callback(self, msg: Image):
        """Store latest depth image."""
        self.last_depth_image = msg

    def camera_info_callback(self, msg: CameraInfo):
        """Store latest camera info."""
        self.last_camera_info = msg

    def _extract_pointcloud_from_bbox(
        self, detection, source_frame: str
    ) -> Optional[Tuple[Point, float, int]]:
        """Extract point cloud from bounding box region and compute features.

        Args:
            detection: Detection2D message with bounding box
            source_frame: Frame ID of the detection (unused, kept for compatibility)

        Returns:
            Tuple of (centroid_3d, pointcloud_size, point_count) or None if extraction fails.
            centroid_3d: 3D centroid of point cloud in source frame
            pointcloud_size: Approximate 3D size (diagonal of bounding box in meters)
            point_count: Number of valid 3D points
        """
        if self.last_depth_image is None or self.last_camera_info is None:
            return None

        result = extract_pointcloud_from_bbox(
            detection,
            self.last_depth_image,
            self.last_camera_info,
            self.bridge,
        )

        if result is None:
            self.connector.node.get_logger().warning(
                "Failed to extract point cloud from bbox"
            )

        return result

    def detection_callback(self, msg: RAIDetectionArray):
        """Process detection array and store annotations."""
        self.connector.node.get_logger().debug("Entering detection_callback")
        confidence_threshold = self._get_double_parameter("confidence_threshold")
        map_frame_id = self._get_string_parameter("map_frame_id")

        self.connector.node.get_logger().info(
            f"Received detection array with {len(msg.detections)} detections: {msg.detection_classes}, "
            f"header.frame_id={msg.header.frame_id}, confidence_threshold={confidence_threshold}"
        )

        # Log details of each detection
        for i, det in enumerate(msg.detections):
            results_count = len(det.results) if det.results else 0
            if results_count > 0:
                result = det.results[0]
                self.connector.node.get_logger().debug(
                    f"  Detection {i}: class={result.hypothesis.class_id}, "
                    f"score={result.hypothesis.score:.3f}, "
                    f"frame_id={det.header.frame_id}"
                )
            else:
                self.connector.node.get_logger().warning(
                    f"  Detection {i} has no results!"
                )

        timestamp_ros = rclpy.time.Time.from_msg(msg.header.stamp)
        timestamp = (
            timestamp_ros.nanoseconds / 1e9
        )  # Convert to Unix timestamp (seconds)
        detection_source = msg.header.frame_id or "unknown"

        stored_count = 0
        skipped_count = 0
        default_frame_id = msg.header.frame_id
        self.connector.node.get_logger().debug(
            f"Processing {len(msg.detections)} detections from source={detection_source}, "
            f"confidence_threshold={confidence_threshold}"
        )
        for detection in msg.detections:
            if self._process_detection(
                detection,
                confidence_threshold,
                map_frame_id,
                timestamp,
                detection_source,
                default_frame_id,
            ):
                stored_count += 1
            else:
                skipped_count += 1

        if stored_count > 0:
            self.connector.node.get_logger().info(
                f"Stored {stored_count} annotations, skipped {skipped_count} (low confidence or transform failed)"
            )
        elif len(msg.detections) > 0:
            self.connector.node.get_logger().warning(
                f"Received {len(msg.detections)} detections but none were stored "
                f"(confidence threshold: {confidence_threshold})"
            )

    def _process_detection(
        self,
        detection: Detection2D,
        confidence_threshold: float,
        map_frame_id: str,
        timestamp: float,
        detection_source: str,
        default_frame_id: str,
    ) -> bool:
        """Process a single detection and store annotation if valid.

        Returns:
            True if annotation was stored, False otherwise.
        """
        self.connector.node.get_logger().debug(
            f"Entering _process_detection: source={detection_source}"
        )

        # Validate detection and extract basic data
        validation_result = self._validate_and_extract_detection_data(
            detection, confidence_threshold, default_frame_id
        )
        if not validation_result:
            return False

        object_class, confidence, source_frame, pose_in_source_frame = validation_result

        # Validate and transform pose to map frame
        pose_in_map_frame = self._validate_and_transform_pose(
            pose_in_source_frame, source_frame, map_frame_id, object_class, detection
        )
        if pose_in_map_frame is None:
            return False

        # Extract point cloud features if enabled
        pointcloud_features, pointcloud_centroid_map, pc_size = (
            self._extract_pointcloud_features(detection, source_frame, map_frame_id)
        )

        # Store or update annotation
        vision_detection_id = detection.id if hasattr(detection, "id") else None
        return self._store_or_update_annotation(
            object_class=object_class,
            confidence=confidence,
            pose_in_map_frame=pose_in_map_frame,
            pointcloud_centroid_map=pointcloud_centroid_map,
            pointcloud_features=pointcloud_features,
            pc_size=pc_size,
            timestamp=timestamp,
            detection_source=detection_source,
            source_frame=source_frame,
            vision_detection_id=vision_detection_id,
        )

    def _validate_and_extract_detection_data(
        self, detection: Detection2D, confidence_threshold: float, default_frame_id: str
    ) -> Optional[Tuple[str, float, str, Pose]]:
        """Validate detection and extract basic data.

        Returns:
            Tuple of (object_class, confidence, source_frame, pose_in_source_frame) if valid,
            None otherwise.
        """
        if not detection.results:
            self.connector.node.get_logger().debug("Detection has no results, skipping")
            return None

        result = detection.results[0]
        confidence = result.hypothesis.score
        object_class = result.hypothesis.class_id

        # Check bounding box size
        min_bbox_area = self._get_double_parameter("min_bbox_area")
        bbox_area = detection.bbox.size_x * detection.bbox.size_y

        if bbox_area < min_bbox_area:
            self.connector.node.get_logger().debug(
                f"Bounding box too small: area={bbox_area:.1f} < {min_bbox_area:.1f} pixels^2, "
                f"skipping {object_class}"
            )
            return None

        # Use class-specific threshold if available
        effective_threshold = self.class_thresholds.get(
            object_class, confidence_threshold
        )

        self.connector.node.get_logger().info(
            f"Processing detection: class={object_class}, confidence={confidence:.3f}, "
            f"threshold={effective_threshold:.3f}, bbox_area={bbox_area:.1f}"
        )

        if confidence < effective_threshold:
            self.connector.node.get_logger().debug(
                f"Confidence {confidence:.3f} below threshold {effective_threshold:.3f}, skipping"
            )
            return None

        # Use detection frame_id, fallback to message header frame_id if empty
        source_frame = detection.header.frame_id or default_frame_id
        if not source_frame:
            self.connector.node.get_logger().warning(
                f"Detection has no frame_id (detection.frame_id='{detection.header.frame_id}', "
                f"default_frame_id='{default_frame_id}'), skipping"
            )
            return None

        pose_in_source_frame = result.pose.pose
        return (object_class, confidence, source_frame, pose_in_source_frame)

    def _validate_and_transform_pose(
        self,
        pose_in_source_frame: Pose,
        source_frame: str,
        map_frame_id: str,
        object_class: str,
        detection: Detection2D,
    ) -> Optional[Pose]:
        """Validate pose and transform to map frame.

        Returns:
            Transformed pose in map frame if successful, None otherwise.
        """
        # Check if pose is empty (all zeros)
        pose_is_empty = (
            pose_in_source_frame.position.x == 0.0
            and pose_in_source_frame.position.y == 0.0
            and pose_in_source_frame.position.z == 0.0
        )

        if pose_is_empty:
            self.connector.node.get_logger().warning(
                f"Detection for {object_class} has empty pose (0,0,0). "
                f"GroundingDINO provides 2D bounding boxes but no 3D pose. "
                f"Cannot store annotation without 3D position. "
                f"Bounding box center: ({detection.bbox.center.position.x:.1f}, "
                f"{detection.bbox.center.position.y:.1f})"
            )
            return None

        self.connector.node.get_logger().debug(
            f"Pose in source frame ({source_frame}): "
            f"x={pose_in_source_frame.position.x:.3f}, "
            f"y={pose_in_source_frame.position.y:.3f}, "
            f"z={pose_in_source_frame.position.z:.3f}"
        )

        try:
            pose_in_map_frame = self._transform_pose_to_map(
                pose_in_source_frame, source_frame, map_frame_id
            )
            self.connector.node.get_logger().info(
                f"Transformed pose to map frame ({map_frame_id}): "
                f"x={pose_in_map_frame.position.x:.3f}, "
                f"y={pose_in_map_frame.position.y:.3f}, "
                f"z={pose_in_map_frame.position.z:.3f}"
            )
            return pose_in_map_frame
        except Exception as e:
            self.connector.node.get_logger().warning(
                f"Failed to transform pose from {source_frame} to {map_frame_id}: {e}"
            )
            return None

    def _extract_pointcloud_features(
        self, detection, source_frame: str, map_frame_id: str
    ) -> Tuple[Optional[dict], Optional[Point], Optional[float]]:
        """Extract point cloud features if enabled.

        Returns:
            Tuple of (pointcloud_features_dict, pointcloud_centroid_map, pc_size)
        """
        use_pointcloud = (
            self.connector.node.get_parameter("use_pointcloud_dedup")
            .get_parameter_value()
            .bool_value
        )

        if not use_pointcloud:
            return (None, None, None)

        pc_result = self._extract_pointcloud_from_bbox(detection, source_frame)
        if not pc_result:
            return (None, None, None)

        pc_centroid_source, pc_size, pc_count = pc_result

        try:
            pc_centroid_map = self._transform_pose_to_map(
                Pose(position=pc_centroid_source), source_frame, map_frame_id
            )
            pointcloud_features = {
                "centroid": {
                    "x": pc_centroid_map.position.x,
                    "y": pc_centroid_map.position.y,
                    "z": pc_centroid_map.position.z,
                },
                "size_3d": pc_size,
                "point_count": pc_count,
            }
            pointcloud_centroid_map = Point(
                x=pc_centroid_map.position.x,
                y=pc_centroid_map.position.y,
                z=pc_centroid_map.position.z,
            )
            self.connector.node.get_logger().debug(
                f"Point cloud features: size={pc_size:.2f}m, points={pc_count}, "
                f"centroid=({pc_centroid_map.position.x:.2f}, {pc_centroid_map.position.y:.2f}, "
                f"{pc_centroid_map.position.z:.2f})"
            )
            return (pointcloud_features, pointcloud_centroid_map, pc_size)
        except Exception as e:
            self.connector.node.get_logger().warning(
                f"Failed to transform point cloud centroid: {e}"
            )
            return (None, None, None)

    def _determine_merge_decision(
        self,
        nearby: list,
        pointcloud_features: Optional[dict],
        pc_size: Optional[float],
        use_pointcloud: bool,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if detection should merge with existing annotation.

        Returns:
            Tuple of (should_merge, existing_id)
        """
        if not nearby:
            return (False, None)

        existing = nearby[0]

        # If both have point cloud data, check size similarity
        if pointcloud_features and use_pointcloud and pc_size is not None:
            if existing.metadata and "pointcloud" in existing.metadata:
                existing_pc = existing.metadata["pointcloud"]
                existing_size = existing_pc.get("size_3d", 0)

                if existing_size > 0:
                    size_ratio = min(pc_size, existing_size) / max(
                        pc_size, existing_size
                    )
                    size_diff = abs(existing_size - pc_size)

                    # If sizes are very different, likely different objects
                    if size_ratio < 0.5 and size_diff > 0.5:
                        self.connector.node.get_logger().info(
                            f"Point cloud size mismatch: existing={existing_size:.2f}m, "
                            f"new={pc_size:.2f}m, ratio={size_ratio:.2f}. Treating as different object."
                        )
                        return (False, None)
                    else:
                        self.connector.node.get_logger().debug(
                            f"Point cloud size match: existing={existing_size:.2f}m, "
                            f"new={pc_size:.2f}m, ratio={size_ratio:.2f}"
                        )
                        return (True, existing.id)

        # Use spatial matching (either no point cloud or sizes match)
        return (True, existing.id)

    def _store_or_update_annotation(
        self,
        object_class: str,
        confidence: float,
        pose_in_map_frame: Pose,
        pointcloud_centroid_map: Optional[Point],
        pointcloud_features: Optional[dict],
        pc_size: Optional[float],
        timestamp: float,
        detection_source: str,
        source_frame: str,
        vision_detection_id: Optional[str],
    ) -> bool:
        """Store or update annotation based on matching logic.

        Returns:
            True if annotation was stored/updated, False otherwise.
        """
        # Use class-specific merge threshold if available
        merge_threshold = self.class_merge_thresholds.get(object_class, 0.5)

        # Use point cloud centroid for matching if available, otherwise use pose
        match_center = (
            pointcloud_centroid_map
            if pointcloud_centroid_map
            else Point(
                x=pose_in_map_frame.position.x,
                y=pose_in_map_frame.position.y,
                z=pose_in_map_frame.position.z,
            )
        )

        try:
            self.connector.node.get_logger().info(
                f"Storing annotation: class={object_class}, confidence={confidence:.3f}, "
                f"merge_radius={merge_threshold:.2f}m, location_id={self.memory.location_id}"
            )

            # Query nearby annotations
            nearby = self.memory.query_by_location(
                match_center,
                radius=merge_threshold,
                object_class=object_class,
                location_id=self.memory.location_id,
            )

            use_pointcloud = (
                self.connector.node.get_parameter("use_pointcloud_dedup")
                .get_parameter_value()
                .bool_value
            )
            should_merge, existing_id = self._determine_merge_decision(
                nearby, pointcloud_features, pc_size, use_pointcloud
            )

            # Prepare metadata
            metadata = {}
            if pointcloud_features:
                metadata["pointcloud"] = pointcloud_features

            if should_merge and existing_id:
                # Update existing annotation
                existing_ann = nearby[0]
                updated = SemanticAnnotation(
                    id=existing_id,
                    object_class=object_class,
                    pose=pose_in_map_frame,
                    confidence=max(existing_ann.confidence, confidence),
                    timestamp=timestamp,
                    detection_source=detection_source,
                    source_frame=source_frame,
                    location_id=self.memory.location_id,
                    vision_detection_id=vision_detection_id,
                    metadata=metadata if metadata else existing_ann.metadata,
                )
                self.memory.update_annotation(updated)
                self.connector.node.get_logger().info(
                    f"Updated existing annotation for {object_class}"
                )
            else:
                # Insert new annotation
                self.memory.store_annotation(
                    object_class=object_class,
                    pose=pose_in_map_frame,
                    confidence=confidence,
                    timestamp=timestamp,
                    detection_source=detection_source,
                    source_frame=source_frame,
                    location_id=self.memory.location_id,
                    vision_detection_id=vision_detection_id,
                    metadata=metadata if metadata else None,
                )
                self.connector.node.get_logger().info(
                    f"Created new annotation for {object_class}"
                )

            return True

        except Exception as e:
            self.connector.node.get_logger().error(f"Failed to store annotation: {e}")
            return False

    def map_callback(self, msg: OccupancyGrid):
        """Process map update and store metadata."""
        map_frame_id = self._get_string_parameter("map_frame_id")

        if msg.header.frame_id != map_frame_id:
            self.connector.node.get_logger().warning(
                f"Map frame_id mismatch: expected {map_frame_id}, got {msg.header.frame_id}"
            )

        self.memory.map_frame_id = msg.header.frame_id
        self.memory.resolution = msg.info.resolution

        self.connector.node.get_logger().debug(
            f"Updated map metadata: frame_id={msg.header.frame_id}, "
            f"resolution={msg.info.resolution}"
        )

    def _transform_pose_to_map(
        self, pose: Pose, source_frame: str, target_frame: str
    ) -> Pose:
        """Transform pose from source frame to map frame.

        If source and target frames are the same, returns the pose unchanged.

        Raises:
            Exception: If transform lookup fails.
        """
        # No transform needed if frames are identical
        if source_frame == target_frame:
            return pose

        try:
            # Use ROS2Connector's get_transform method which handles waiting and errors
            transform = self.connector.get_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                timeout_sec=TF_LOOKUP_TIMEOUT_SEC,
            )
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = source_frame
            pose_stamped.header.stamp = transform.header.stamp

            transformed_pose_stamped = do_transform_pose_stamped(
                pose_stamped, transform
            )
            return transformed_pose_stamped.pose
        except Exception as e:
            raise Exception(f"Transform lookup failed: {e}") from e


def main(args=None):
    """Main entry point for the semantic map node."""
    rclpy.init(args=args)
    connector = ROS2Connector(
        node_name="rai_semap_node", executor_type="multi_threaded"
    )
    semantic_map_node = SemanticMapNode(connector=connector)
    try:
        rclpy.spin(semantic_map_node.connector.node)
    except KeyboardInterrupt:
        pass
    finally:
        semantic_map_node.connector.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
