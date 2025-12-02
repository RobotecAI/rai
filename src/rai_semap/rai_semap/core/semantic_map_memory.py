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

import uuid
from typing import Any, Dict, List, Optional, Tuple

from geometry_msgs.msg import Point, Pose

from rai_semap.core.backend.spatial_db_backend import SpatialDBBackend
from rai_semap.core.base_memory import BaseMemory


def _pose_to_dict(pose: Pose) -> Dict[str, Any]:
    """Convert Pose to JSON-serializable dictionary."""
    return {
        "position": {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z},
        "orientation": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


class SemanticAnnotation:
    """Spatial-semantic annotation data model."""

    def __init__(
        self,
        id: str,
        object_class: str,
        pose: Pose,
        confidence: float,
        timestamp: Any,
        detection_source: str,
        source_frame: str,
        location_id: str,
        vision_detection_id: Optional[
            str
        ] = None,  # id from vision pipeline, mostly for debugging
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.object_class = object_class
        self.pose = pose
        self.confidence = confidence
        self.timestamp = timestamp
        self.detection_source = detection_source
        self.source_frame = source_frame
        self.location_id = location_id
        self.vision_detection_id = vision_detection_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "object_class": self.object_class,
            "pose": _pose_to_dict(self.pose),
            "confidence": self.confidence,
            "timestamp": str(self.timestamp) if self.timestamp is not None else None,
            "detection_source": self.detection_source,
            "source_frame": self.source_frame,
            "location_id": self.location_id,
            "vision_detection_id": self.vision_detection_id,
            "metadata": self.metadata,
        }


class MapMetadata:
    """Metadata structure for a SLAM map (one per location, not per annotation).

    Tracks properties of the underlying SLAM map and semantic annotation activity.
    """

    def __init__(
        self,
        location_id: str,
        map_frame_id: str,
        resolution: float,
        origin: Optional[Pose] = None,
        last_updated: Optional[Any] = None,
    ):
        """
        Args:
            location_id: Identifier for the physical location (e.g., "warehouse_a", "warehouse_b")
            map_frame_id: Frame ID of the SLAM map
            resolution: OccupancyGrid resolution (meters/pixel) from SLAM map configuration
            origin: Optional map origin pose. Only needed for coordinate transformations
                between map frame and world frame. Not required for semantic annotations
                that are already stored in map frame.
            last_updated: Optional timestamp of last semantic annotation update to this map
        """
        self.location_id = location_id
        self.map_frame_id = map_frame_id
        self.resolution = resolution
        self.origin = origin
        self.last_updated = last_updated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "location_id": self.location_id,
            "map_frame_id": self.map_frame_id,
            "resolution": self.resolution,
            "last_updated": str(self.last_updated)
            if self.last_updated is not None
            else None,
        }
        if self.origin is not None:
            result["origin"] = _pose_to_dict(self.origin)
        return result


class SemanticMapMemory(BaseMemory):
    """Spatial-semantic memory for storing and querying object annotations."""

    def __init__(
        self,
        backend: SpatialDBBackend,
        location_id: str,
        map_frame_id: str = "map",
        resolution: float = 0.05,
    ):
        self.backend = backend
        self.location_id = location_id
        self.map_frame_id = map_frame_id
        self.resolution = resolution

    def store(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a value with optional metadata. Returns storage ID.

        Uses temporal consistency: if nearby annotation of same class exists,
        updates it; otherwise inserts new. The key parameter is ignored
        (annotation ID is determined by temporal consistency).

        Args:
            key: Ignored (kept for BaseMemory interface compatibility)
            value: Dict containing required fields including 'object_class'
            metadata: Optional additional metadata
        """
        if not isinstance(value, dict):
            raise TypeError(f"value must be a dict, got {type(value).__name__}")

        required_fields = [
            "object_class",
            "pose",
            "confidence",
            "timestamp",
            "detection_source",
            "source_frame",
            "location_id",
        ]
        missing = [field for field in required_fields if field not in value]
        if missing:
            raise ValueError(f"Missing required fields in value: {missing}")

        return self.store_or_update_annotation(
            object_class=value["object_class"],
            pose=value["pose"],
            confidence=value["confidence"],
            timestamp=value["timestamp"],
            detection_source=value["detection_source"],
            source_frame=value["source_frame"],
            location_id=value["location_id"],
            vision_detection_id=value.get("vision_detection_id"),
            metadata=metadata,
        )

    def retrieve(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Retrieve values matching query and filters.

        Not implemented. Use concrete query methods instead:
        - query_by_class: Query by object class
        - query_by_location: Query by center point and radius
        - query_by_region: Query by bounding box
        """
        raise NotImplementedError(
            "Use concrete query methods: query_by_class, query_by_location, or query_by_region"
        )

    def delete(self, key: str) -> bool:
        """Delete a stored value by annotation ID. Returns success status."""
        return self.backend.delete_annotation(key)

    def delete_all_annotations(self, location_id: Optional[str] = None) -> int:
        """Delete all annotations, optionally filtered by location_id.

        Args:
            location_id: If provided, only delete annotations for this location.
                       If None, delete all annotations. If not provided and
                       self.location_id is set, defaults to self.location_id.

        Returns:
            Number of annotations deleted.
        """
        if location_id is None:
            location_id = self.location_id
        return self.backend.delete_all_annotations(location_id)

    def store_annotation(
        self,
        object_class: str,
        pose: Pose,
        confidence: float,
        timestamp: Any,
        detection_source: str,
        source_frame: str,
        location_id: str,
        vision_detection_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        annotation_id: Optional[str] = None,
    ) -> str:
        """Store a semantic annotation. Returns annotation ID."""
        if annotation_id is None:
            annotation_id = str(uuid.uuid4())
        annotation = SemanticAnnotation(
            id=annotation_id,
            object_class=object_class,
            pose=pose,
            confidence=confidence,
            timestamp=timestamp,
            detection_source=detection_source,
            source_frame=source_frame,
            location_id=location_id,
            vision_detection_id=vision_detection_id,
            metadata=metadata,
        )
        return self.backend.insert_annotation(annotation)

    def store_or_update_annotation(
        self,
        object_class: str,
        pose: Pose,
        confidence: float,
        timestamp: Any,
        detection_source: str,
        source_frame: str,
        location_id: str,
        vision_detection_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        merge_threshold: float = 0.5,
    ) -> str:
        """Store or update annotation with temporal consistency.

        Queries nearby annotations of the same class and location. If found within merge_threshold,
        updates existing annotation. Otherwise, inserts new annotation.

        Args:
            location_id: Identifier for the physical location
            merge_threshold: Distance threshold (meters) for merging duplicate detections
            Other args: Same as store_annotation

        Returns:
            Annotation ID (existing if updated, new if inserted)
        """
        center = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
        nearby = self.query_by_location(
            center,
            radius=merge_threshold,
            object_class=object_class,
            location_id=location_id,
        )

        if nearby:
            # Update first match (closest)
            existing = nearby[0]
            updated = SemanticAnnotation(
                id=existing.id,
                object_class=object_class,
                pose=pose,
                confidence=max(
                    existing.confidence, confidence
                ),  # Keep higher confidence
                timestamp=timestamp,  # Update to latest timestamp
                detection_source=detection_source,
                source_frame=source_frame,
                location_id=location_id,
                vision_detection_id=vision_detection_id,
                metadata=metadata or existing.metadata,
            )
            self.backend.update_annotation(updated)
            return existing.id
        else:
            # Insert new
            return self.store_annotation(
                object_class=object_class,
                pose=pose,
                confidence=confidence,
                timestamp=timestamp,
                detection_source=detection_source,
                source_frame=source_frame,
                location_id=location_id,
                vision_detection_id=vision_detection_id,
                metadata=metadata,
            )

    def query_by_class(
        self,
        object_class: str,
        confidence_threshold: float = 0.5,
        limit: Optional[int] = None,
        location_id: Optional[str] = None,
    ) -> List[SemanticAnnotation]:
        """Query annotations by object class."""
        filters = {
            "object_class": object_class,
            "confidence_threshold": confidence_threshold,
            "location_id": location_id or self.location_id,
        }
        center = Point(x=0.0, y=0.0, z=0.0)
        results = self.backend.spatial_query(center, radius=1e10, filters=filters)
        if limit is not None:
            results = results[:limit]
        return results

    def query_by_location(
        self,
        center: Point,
        radius: float,
        object_class: Optional[str] = None,
        location_id: Optional[str] = None,
    ) -> List[SemanticAnnotation]:
        """Query annotations within radius of center point."""
        filters = {"location_id": location_id or self.location_id}
        if object_class:
            filters["object_class"] = object_class
        return self.backend.spatial_query(center, radius, filters=filters)

    def query_by_region(
        self,
        bbox: Tuple[float, float, float, float],  # (min_x, min_y, max_x, max_y)
        object_class: Optional[str] = None,
        location_id: Optional[str] = None,
    ) -> List[SemanticAnnotation]:
        """Query annotations within bounding box region."""
        min_x, min_y, max_x, max_y = bbox
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        radius = max(max_x - min_x, max_y - min_y) / 2.0

        center = Point(x=center_x, y=center_y, z=0.0)
        filters = {"location_id": location_id or self.location_id}
        if object_class:
            filters["object_class"] = object_class

        results = self.backend.spatial_query(center, radius, filters=filters)

        filtered_results = []
        for annotation in results:
            x = annotation.pose.position.x
            y = annotation.pose.position.y
            if min_x <= x <= max_x and min_y <= y <= max_y:
                filtered_results.append(annotation)

        return filtered_results

    def get_map_metadata(self) -> MapMetadata:
        """Get metadata for the current SLAM map.

        Returns one MapMetadata instance per location (not per annotation).
        Computes last_updated from the most recent annotation timestamp for this location.
        map_frame_id and resolution come from instance configuration.
        """
        # Get most recent annotation timestamp for this location
        center = Point(x=0.0, y=0.0, z=0.0)
        filters = {"location_id": self.location_id}
        all_annotations = self.backend.spatial_query(
            center, radius=1e10, filters=filters
        )

        last_updated = None
        if all_annotations:
            timestamps = [
                ann.timestamp for ann in all_annotations if ann.timestamp is not None
            ]
            if timestamps:
                last_updated = max(timestamps)

        return MapMetadata(
            location_id=self.location_id,
            map_frame_id=self.map_frame_id,
            resolution=self.resolution,
            origin=None,
            last_updated=last_updated,
        )
