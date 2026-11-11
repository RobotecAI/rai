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


"""The semantic map node: stores RAIDetectionArray detections as map annotations."""

import logging
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rai.communication.ros2 import QoS, ROS2Connector
from rai.communication.ros2.runtime import ROS2Runtime
from rai_perception.components.perception_utils import extract_pointcloud_from_bbox
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_pose_stamped
from vision_msgs.msg import Detection2D

from rai_interfaces.msg import RAIDetectionArray
from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticAnnotation, SemanticMapMemory

logger = logging.getLogger(__name__)

TF_LOOKUP_TIMEOUT_SEC = 1.0
DEFAULT_CONFIG = Path(__file__).parent / "config" / "node.yaml"


@dataclass
class SemanticMapConfig:
    """The node's ROS parameters; defaults come from the node_config YAML file."""

    backend_type: str = "sqlite"  # 'sqlite' ('postgres' in the future)
    database_path: str = "semantic_map.db"
    location_id: str = "default_location"
    confidence_threshold: float = 0.5
    class_confidence_thresholds: str = ""  # "person:0.7,window:0.6"
    min_bbox_area: float = 100.0  # pixels^2
    class_merge_thresholds: str = ""  # merge radii in meters: "couch:2.5,table:1.5"
    use_pointcloud_dedup: bool = True
    detection_topic: str = "/detection_array"
    map_topic: str = "/map"
    depth_topic: str = ""
    camera_info_topic: str = ""
    map_frame_id: str = "map"
    map_resolution: float = 0.05  # meters/pixel


def _parse_class_values(text: str, what: str) -> dict[str, float]:
    """'class1:value1,class2:value2' as a dict; malformed items are skipped."""
    values: dict[str, float] = {}
    for item in filter(None, (i.strip() for i in text.split(","))):
        name, _, value = item.partition(":")
        try:
            values[name.strip()] = float(value)
            logger.info(f"Class-specific {what}: {name.strip()}={float(value):.3f}")
        except ValueError:
            logger.warning(f"Invalid {what} in '{item}', skipping")
    return values


class SemanticMapNode:
    """Subscribes to detections, the map and optionally depth and camera info, and
    stores detections transformed to the map frame in a SemanticMapMemory."""

    def __init__(self, connector: ROS2Connector, database_path: Optional[str] = None):
        self.connector = connector
        config_path = connector.declare_parameter("node_config", "") or DEFAULT_CONFIG
        with open(config_path) as f:
            groups = (yaml.safe_load(f) or {}).get("rai_semap_node", {})
        from_yaml = {
            name: value
            for group in groups.get("ros__parameters", {}).values()
            if isinstance(group, dict)
            for name, value in group.items()
        }
        self.config = SemanticMapConfig(
            **{
                name: connector.declare_parameter(name, from_yaml.get(name, default))
                for name, default in asdict(SemanticMapConfig()).items()
            }
        )
        if database_path is not None:
            self.config.database_path = database_path
        self.class_thresholds = _parse_class_values(
            self.config.class_confidence_thresholds, "threshold"
        )
        self.class_merge_thresholds = _parse_class_values(
            self.config.class_merge_thresholds, "merge radius"
        )
        self.last_depth_image: Optional[Image] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self._initialize_memory()
        self._initialize_subscriptions()

    def _initialize_memory(self):
        c = self.config
        if c.backend_type == "postgres":
            raise NotImplementedError(
                "PostgreSQL backend not yet implemented. Use 'sqlite' for now."
            )
        if c.backend_type != "sqlite":
            raise ValueError(
                f"Unknown backend_type: {c.backend_type}. Supported: 'sqlite'"
            )
        backend = SQLiteBackend(c.database_path)
        backend.init_schema()
        self.memory = SemanticMapMemory(
            backend=backend,
            location_id=c.location_id,
            map_frame_id=c.map_frame_id,
            resolution=c.map_resolution,
        )
        logger.info(
            f"Initialized semantic map memory: backend={c.backend_type}, "
            f"location_id={c.location_id}, map_frame_id={c.map_frame_id}, "
            f"database_path={c.database_path}"
        )

    def _initialize_subscriptions(self):
        c, subscribe = self.config, self.connector.subscribe
        self.detection_subscription = subscribe(
            c.detection_topic,
            self.detection_callback,
            msg_type=RAIDetectionArray,
            qos=QoS.SENSOR,
        )
        self.map_subscription = subscribe(
            c.map_topic, self.map_callback, msg_type=OccupancyGrid
        )
        self.depth_subscription = self.camera_info_subscription = None
        if c.use_pointcloud_dedup and c.depth_topic:
            self.depth_subscription = subscribe(
                c.depth_topic, self.depth_callback, msg_type=Image, qos=QoS.SENSOR
            )
        if c.use_pointcloud_dedup and c.camera_info_topic:
            self.camera_info_subscription = subscribe(
                c.camera_info_topic,
                self.camera_info_callback,
                msg_type=CameraInfo,
                qos=QoS.SENSOR,
            )
        logger.info(
            f"Subscribed to detection_topic={c.detection_topic}, map_topic={c.map_topic},"
            f" depth_topic={c.depth_topic!r}, camera_info_topic={c.camera_info_topic!r}"
        )

    def depth_callback(self, msg: Image):
        self.last_depth_image = msg

    def camera_info_callback(self, msg: CameraInfo):
        self.last_camera_info = msg

    def _extract_pointcloud_from_bbox(
        self, detection
    ) -> Optional[Tuple[Point, float, int]]:
        """(centroid in the source frame, 3D size in meters, point count), or None."""
        if self.last_depth_image is None or self.last_camera_info is None:
            return None
        result = extract_pointcloud_from_bbox(
            detection, self.last_depth_image, self.last_camera_info
        )
        if result is None:
            logger.warning("Failed to extract point cloud from bbox")
        return result

    def detection_callback(self, msg: RAIDetectionArray):
        """Store the detections of msg that pass the filters."""
        threshold = self.config.confidence_threshold
        logger.info(
            f"Received detection array with {len(msg.detections)} detections: "
            f"{msg.detection_classes}, header.frame_id={msg.header.frame_id}, "
            f"confidence_threshold={threshold}"
        )
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        source = msg.header.frame_id or "unknown"
        stored = sum(
            self._process_detection(
                detection,
                threshold,
                self.config.map_frame_id,
                timestamp,
                source,
                msg.header.frame_id,
            )
            for detection in msg.detections
        )
        if stored:
            logger.info(
                f"Stored {stored} annotations, skipped {len(msg.detections) - stored}"
                " (low confidence or transform failed)"
            )
        elif msg.detections:
            logger.warning(
                f"Received {len(msg.detections)} detections but none were stored "
                f"(confidence threshold: {threshold})"
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
        """Store or merge one detection; True if an annotation was written."""
        data = self._validate_and_extract_detection_data(
            detection, confidence_threshold, default_frame_id
        )
        if not data:
            return False
        object_class, confidence, source_frame, pose_in_source_frame = data
        pose_in_map_frame = self._validate_and_transform_pose(
            pose_in_source_frame, source_frame, map_frame_id, object_class, detection
        )
        if pose_in_map_frame is None:
            return False
        features, centroid_map, pc_size = self._extract_pointcloud_features(
            detection, source_frame, map_frame_id
        )
        return self._store_or_update_annotation(
            object_class=object_class,
            confidence=confidence,
            pose_in_map_frame=pose_in_map_frame,
            pointcloud_centroid_map=centroid_map,
            pointcloud_features=features,
            pc_size=pc_size,
            timestamp=timestamp,
            detection_source=detection_source,
            source_frame=source_frame,
            vision_detection_id=getattr(detection, "id", None),
        )

    def _validate_and_extract_detection_data(
        self, detection: Detection2D, confidence_threshold: float, default_frame_id: str
    ) -> Optional[Tuple[str, float, str, Pose]]:
        """(class, confidence, source frame, pose in the source frame), or None if
        the detection is filtered out."""
        if not detection.results:
            logger.debug("Detection has no results, skipping")
            return None
        result = detection.results[0]
        confidence = result.hypothesis.score
        object_class = result.hypothesis.class_id
        bbox_area = detection.bbox.size_x * detection.bbox.size_y
        if bbox_area < self.config.min_bbox_area:
            logger.debug(
                f"Bounding box too small: area={bbox_area:.1f} < "
                f"{self.config.min_bbox_area:.1f} pixels^2, skipping {object_class}"
            )
            return None
        threshold = self.class_thresholds.get(object_class, confidence_threshold)
        if confidence < threshold:
            logger.debug(
                f"{object_class}: confidence {confidence:.3f} below threshold"
                f" {threshold:.3f}, skipping"
            )
            return None
        source_frame = detection.header.frame_id or default_frame_id
        if not source_frame:
            logger.warning(f"Detection of {object_class} has no frame_id, skipping")
            return None
        return (object_class, confidence, source_frame, result.pose.pose)

    def _validate_and_transform_pose(
        self,
        pose_in_source_frame: Pose,
        source_frame: str,
        map_frame_id: str,
        object_class: str,
        detection: Detection2D,
    ) -> Optional[Pose]:
        """The pose in the map frame, or None if it is empty or TF fails."""
        p = pose_in_source_frame.position
        if p.x == 0.0 and p.y == 0.0 and p.z == 0.0:
            logger.warning(
                f"Detection for {object_class} has empty pose (0,0,0): a 2D detection"
                " without a 3D pose cannot be stored. Bounding box center: "
                f"({detection.bbox.center.position.x:.1f}, "
                f"{detection.bbox.center.position.y:.1f})"
            )
            return None
        try:
            return self._transform_pose_to_map(
                pose_in_source_frame, source_frame, map_frame_id
            )
        except Exception as e:
            logger.warning(
                f"Failed to transform pose from {source_frame} to {map_frame_id}: {e}"
            )
            return None

    def _extract_pointcloud_features(
        self, detection, source_frame: str, map_frame_id: str
    ) -> Tuple[Optional[dict], Optional[Point], Optional[float]]:
        """(features, centroid in the map frame, size), all None if disabled or
        unavailable."""
        if not self.config.use_pointcloud_dedup:
            return (None, None, None)
        pc_result = self._extract_pointcloud_from_bbox(detection)
        if not pc_result:
            return (None, None, None)
        centroid_source, pc_size, pc_count = pc_result
        try:
            centroid = self._transform_pose_to_map(
                Pose(position=centroid_source), source_frame, map_frame_id
            ).position
        except Exception as e:
            logger.warning(f"Failed to transform point cloud centroid: {e}")
            return (None, None, None)
        features = {
            "centroid": {"x": centroid.x, "y": centroid.y, "z": centroid.z},
            "size_3d": pc_size,
            "point_count": pc_count,
        }
        return (features, Point(x=centroid.x, y=centroid.y, z=centroid.z), pc_size)

    def _determine_merge_decision(
        self,
        nearby: list,
        pointcloud_features: Optional[dict],
        pc_size: Optional[float],
        use_pointcloud: bool,
    ) -> Tuple[bool, Optional[str]]:
        """(should merge, id of the annotation to merge into)."""
        if not nearby:
            return (False, None)
        existing = nearby[0]
        if pointcloud_features and use_pointcloud and pc_size is not None:
            existing_size = (existing.metadata or {}).get("pointcloud", {})
            existing_size = existing_size.get("size_3d", 0)
            if existing_size > 0:
                ratio = min(pc_size, existing_size) / max(pc_size, existing_size)
                if ratio < 0.5 and abs(existing_size - pc_size) > 0.5:
                    logger.info(
                        f"Point cloud size mismatch: existing={existing_size:.2f}m, "
                        f"new={pc_size:.2f}m, ratio={ratio:.2f}. Treating as a"
                        " different object."
                    )
                    return (False, None)
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
        """Merge into a nearby annotation of the same class, or insert a new one."""
        radius = self.class_merge_thresholds.get(object_class, 0.5)
        position = pose_in_map_frame.position
        center = pointcloud_centroid_map or Point(
            x=position.x, y=position.y, z=position.z
        )
        location_id = self.memory.location_id
        metadata = {"pointcloud": pointcloud_features} if pointcloud_features else {}
        try:
            nearby = self.memory.query_by_location(
                center,
                radius=radius,
                object_class=object_class,
                location_id=location_id,
            )
            merge, existing_id = self._determine_merge_decision(
                nearby, pointcloud_features, pc_size, self.config.use_pointcloud_dedup
            )
            if merge and existing_id:
                existing = nearby[0]
                self.memory.update_annotation(
                    SemanticAnnotation(
                        id=existing_id,
                        object_class=object_class,
                        pose=pose_in_map_frame,
                        confidence=max(existing.confidence, confidence),
                        timestamp=timestamp,
                        detection_source=detection_source,
                        source_frame=source_frame,
                        location_id=location_id,
                        vision_detection_id=vision_detection_id,
                        metadata=metadata or existing.metadata,
                    )
                )
                logger.info(f"Updated existing annotation for {object_class}")
            else:
                self.memory.store_annotation(
                    object_class=object_class,
                    pose=pose_in_map_frame,
                    confidence=confidence,
                    timestamp=timestamp,
                    detection_source=detection_source,
                    source_frame=source_frame,
                    location_id=location_id,
                    vision_detection_id=vision_detection_id,
                    metadata=metadata or None,
                )
                logger.info(
                    f"Created new annotation for {object_class} "
                    f"(confidence={confidence:.3f}, merge_radius={radius:.2f}m)"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store annotation: {e}")
            return False

    def map_callback(self, msg: OccupancyGrid):
        """Track the map's frame and resolution."""
        if msg.header.frame_id != self.config.map_frame_id:
            logger.warning(
                f"Map frame_id mismatch: expected {self.config.map_frame_id},"
                f" got {msg.header.frame_id}"
            )
        self.memory.map_frame_id = msg.header.frame_id
        self.memory.resolution = msg.info.resolution

    def _transform_pose_to_map(
        self, pose: Pose, source_frame: str, target_frame: str
    ) -> Pose:
        """pose from source_frame to target_frame; raises if TF lookup fails."""
        if source_frame == target_frame:
            return pose
        transform = self.connector.sync.transform(
            target_frame, source_frame, timeout=TF_LOOKUP_TIMEOUT_SEC
        )
        stamped = PoseStamped(pose=pose)
        stamped.header.frame_id = source_frame
        stamped.header.stamp = transform.header.stamp
        return do_transform_pose_stamped(stamped, transform).pose


def main():
    """Run the semantic map node; --ros-args -p name:=value overrides parameters."""
    logging.basicConfig(level=logging.INFO)
    runtime = ROS2Runtime("rai_semap_node", cli_args=sys.argv)
    with ROS2Connector(runtime=runtime) as connector:
        SemanticMapNode(connector=connector)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
