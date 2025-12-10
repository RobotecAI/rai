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

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rai.types import Point

from rai_semap.core.backend.spatial_db_backend import SpatialDBBackend
from rai_semap.core.base_memory import BaseMemory

# Type alias for Pose - accepts both rai.types.Pose (Pydantic model) and geometry_msgs.msg.Pose (ROS2 message)
# With arbitrary_types_allowed=True, Pydantic accepts ROS2 messages even though type annotation is rai.types.Pose
# ROS2 messages are required because ROS2 transform functions (tf2_geometry_msgs) require and return ROS2 message types
PoseType = Any


class SemanticAnnotation(BaseModel):
    """Spatial-semantic annotation data model."""

    # Allow ROS2 message types (e.g., Pose) that Pydantic doesn't validate natively.
    # Other fields are still validated; ROS2 types are validated by ROS2.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    object_class: str
    pose: PoseType
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    timestamp: float = Field(description="Unix timestamp in seconds (timezone-naive)")
    detection_source: str
    source_frame: str
    location_id: str
    vision_detection_id: Optional[str] = Field(
        default=None, description="ID from vision pipeline, mostly for debugging"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {v}")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, v: Any) -> Dict[str, Any]:
        """Convert None metadata to empty dict."""
        if v is None:
            return {}
        return v


class MapMetadata(BaseModel):
    """Metadata structure for a SLAM map (one per location, not per annotation).

    Tracks properties of the underlying SLAM map and semantic annotation activity.
    """

    # Allow ROS2 message types (e.g., Pose) that Pydantic doesn't validate natively.
    # Other fields are still validated; ROS2 types are validated by ROS2.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    location_id: str = Field(
        description="Identifier for the physical location (e.g., 'warehouse_a', 'warehouse_b')"
    )
    map_frame_id: str = Field(description="Frame ID of the SLAM map")
    resolution: float = Field(
        default=0.05,
        gt=0.0,
        description="OccupancyGrid resolution (meters/pixel) from SLAM map configuration",
    )
    origin: Optional[PoseType] = Field(
        default=None,
        description="Optional map origin pose (rai.types.Pose or geometry_msgs.msg.Pose). Only needed for coordinate transformations between map frame and world frame. Not required for semantic annotations that are already stored in map frame.",
    )
    last_updated: Optional[float] = Field(
        default=None,
        description="Optional Unix timestamp (seconds) of last semantic annotation update to this map",
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: float) -> float:
        """Validate resolution is positive."""
        if v <= 0.0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


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
        pose: PoseType,
        confidence: float,
        timestamp: float,
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

    def update_annotation(self, annotation: SemanticAnnotation) -> bool:
        """Update an existing annotation by ID. Returns success status."""
        return self.backend.update_annotation(annotation)

    def store_or_update_annotation(
        self,
        object_class: str,
        pose: PoseType,
        confidence: float,
        timestamp: float,
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

    def get_seen_object_classes(self, location_id: Optional[str] = None) -> List[str]:
        """Get list of distinct object classes seen in a location.

        Args:
            location_id: If provided, only return classes for this location.
                         If None, use the instance's location_id.

        Returns:
            List of unique object class names, sorted alphabetically.
        """
        pass
