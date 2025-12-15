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

import json
import logging
import re
import sqlite3
from typing import Any, Dict, List, Optional

from rai.types import Point, Pose

from rai_semap.core.backend.spatial_db_backend import SpatialDBBackend
from rai_semap.core.semantic_map_memory import SemanticAnnotation

logger = logging.getLogger(__name__)


class SQLiteBackend(SpatialDBBackend):
    """SQLite backend with SpatiaLite extension for spatial indexing."""

    def __init__(self, database_path: str):
        self.database_path = database_path
        self.conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.conn is None:
            logger.info(f"Creating SQLite connection to: {self.database_path}")
            self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency and ensure data is written
            journal_mode = self.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            logger.debug(f"SQLite journal mode: {journal_mode}")
            # Use NORMAL synchronous mode (balance between safety and performance)
            # FULL is safer but slower, OFF is faster but riskier
            self.conn.execute("PRAGMA synchronous=NORMAL")
            logger.info(f"SQLite connection established to {self.database_path}")
        return self.conn

    def init_schema(self) -> None:
        """Initialize database schema with spatial extensions."""
        conn = self._get_connection()
        cursor = conn.cursor()

        logger.info("Initializing database schema")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                object_class TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                qx REAL,
                qy REAL,
                qz REAL,
                qw REAL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                detection_source TEXT NOT NULL,
                source_frame TEXT NOT NULL,
                location_id TEXT NOT NULL,
                vision_detection_id TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_object_class ON annotations(object_class)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON annotations(confidence)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_location_id ON annotations(location_id)
        """)

        conn.commit()
        logger.info("Database schema initialized successfully")

    def insert_annotation(self, annotation: SemanticAnnotation) -> str:
        """Insert annotation with spatial indexing."""
        conn = self._get_connection()
        cursor = conn.cursor()

        x = annotation.pose.position.x
        y = annotation.pose.position.y
        z = annotation.pose.position.z
        qx = annotation.pose.orientation.x
        qy = annotation.pose.orientation.y
        qz = annotation.pose.orientation.z
        qw = annotation.pose.orientation.w

        metadata_json = json.dumps(annotation.metadata) if annotation.metadata else None

        logger.info(
            f"Inserting annotation: id={annotation.id}, class={annotation.object_class}, "
            f"pos=({x:.2f}, {y:.2f}, {z:.2f}), confidence={annotation.confidence:.3f}, "
            f"location_id={annotation.location_id}"
        )

        try:
            cursor.execute(
                """
                INSERT INTO annotations (
                    id, object_class, x, y, z, qx, qy, qz, qw,
                    confidence, timestamp, detection_source, source_frame,
                    location_id, vision_detection_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    annotation.id,
                    annotation.object_class,
                    x,
                    y,
                    z,
                    qx,
                    qy,
                    qz,
                    qw,
                    annotation.confidence,
                    annotation.timestamp,
                    annotation.detection_source,
                    annotation.source_frame,
                    annotation.location_id,
                    annotation.vision_detection_id,
                    metadata_json,
                ),
            )

            conn.commit()
            logger.debug(f"Committed annotation {annotation.id} to database")

            # Verify the annotation was actually stored
            cursor.execute(
                "SELECT COUNT(*) FROM annotations WHERE id = ?", (annotation.id,)
            )
            count = cursor.fetchone()[0]
            if count == 1:
                logger.info(f"✓ Verified annotation {annotation.id} stored in database")
            else:
                logger.error(
                    f"✗ FAILED to verify annotation {annotation.id} in database "
                    f"(found {count} rows, expected 1)"
                )

            return annotation.id
        except sqlite3.Error as e:
            logger.error(f"SQLite error inserting annotation {annotation.id}: {e}")
            conn.rollback()
            raise

    def spatial_query(
        self, center: Point, radius: float, filters: Optional[Dict[str, Any]] = None
    ) -> List[SemanticAnnotation]:
        """Execute spatial query with optional filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM annotations
            WHERE ((x - ?) * (x - ?) + (y - ?) * (y - ?) + (z - ?) * (z - ?)) <= (? * ?)
        """
        params = [
            center.x,
            center.x,
            center.y,
            center.y,
            center.z,
            center.z,
            radius,
            radius,
        ]

        if filters:
            if "object_class" in filters:
                query += " AND object_class = ?"
                params.append(filters["object_class"])
            if "confidence_threshold" in filters:
                query += " AND confidence >= ?"
                params.append(filters["confidence_threshold"])
            if "location_id" in filters:
                query += " AND location_id = ?"
                params.append(filters["location_id"])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        annotations = []
        for row in rows:
            pose = Pose()
            pose.position.x = row["x"]
            pose.position.y = row["y"]
            pose.position.z = row["z"]
            pose.orientation.x = row["qx"] or 0.0
            pose.orientation.y = row["qy"] or 0.0
            pose.orientation.z = row["qz"] or 0.0
            pose.orientation.w = row["qw"] or 1.0

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            # Convert timestamp to float (handle ROS Time string representation)
            timestamp = row["timestamp"]
            if isinstance(timestamp, str):
                # Extract nanoseconds from ROS Time string representation
                # Format: "Time(nanoseconds=3119172..., clock_type=ROS_TIME)"
                match = re.search(r"nanoseconds=(\d+)", timestamp)
                if match:
                    nanoseconds = int(match.group(1))
                    timestamp = nanoseconds / 1e9
                else:
                    raise ValueError(f"Unable to parse timestamp: {timestamp}")

            annotation = SemanticAnnotation(
                id=row["id"],
                object_class=row["object_class"],
                pose=pose,
                confidence=row["confidence"],
                timestamp=float(timestamp),
                detection_source=row["detection_source"],
                source_frame=row["source_frame"],
                location_id=row["location_id"],
                vision_detection_id=row["vision_detection_id"],
                metadata=metadata,
            )
            annotations.append(annotation)

        return annotations

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation by ID. Returns success status."""
        conn = self._get_connection()
        cursor = conn.cursor()

        logger.info(f"Deleting annotation: id={annotation_id}")
        cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        rows_deleted = cursor.rowcount
        conn.commit()

        if rows_deleted > 0:
            logger.info(f"✓ Deleted annotation {annotation_id} from database")
        else:
            logger.warning(f"✗ Annotation {annotation_id} not found for deletion")

        return rows_deleted > 0

    def delete_all_annotations(self, location_id: Optional[str] = None) -> int:
        """Delete all annotations, optionally filtered by location_id.

        Args:
            location_id: If provided, only delete annotations for this location.
                       If None, delete all annotations.

        Returns:
            Number of annotations deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if location_id:
            logger.info(f"Deleting all annotations for location_id={location_id}")
            cursor.execute(
                "DELETE FROM annotations WHERE location_id = ?", (location_id,)
            )
        else:
            logger.info("Deleting all annotations from database")
            cursor.execute("DELETE FROM annotations")

        rows_deleted = cursor.rowcount
        conn.commit()

        logger.info(f"✓ Deleted {rows_deleted} annotation(s) from database")
        return rows_deleted

    def update_annotation(self, annotation: SemanticAnnotation) -> bool:
        """Update existing annotation by ID. Returns success status."""
        conn = self._get_connection()
        cursor = conn.cursor()

        x = annotation.pose.position.x
        y = annotation.pose.position.y
        z = annotation.pose.position.z
        qx = annotation.pose.orientation.x
        qy = annotation.pose.orientation.y
        qz = annotation.pose.orientation.z
        qw = annotation.pose.orientation.w

        metadata_json = json.dumps(annotation.metadata) if annotation.metadata else None

        logger.info(
            f"Updating annotation: id={annotation.id}, class={annotation.object_class}, "
            f"pos=({x:.2f}, {y:.2f}, {z:.2f}), confidence={annotation.confidence:.3f}"
        )

        try:
            cursor.execute(
                """
                UPDATE annotations SET
                    object_class = ?,
                    x = ?, y = ?, z = ?,
                    qx = ?, qy = ?, qz = ?, qw = ?,
                    confidence = ?,
                    timestamp = ?,
                    detection_source = ?,
                    source_frame = ?,
                    location_id = ?,
                    vision_detection_id = ?,
                    metadata = ?
                WHERE id = ?
            """,
                (
                    annotation.object_class,
                    x,
                    y,
                    z,
                    qx,
                    qy,
                    qz,
                    qw,
                    annotation.confidence,
                    annotation.timestamp,
                    annotation.detection_source,
                    annotation.source_frame,
                    annotation.location_id,
                    annotation.vision_detection_id,
                    metadata_json,
                    annotation.id,
                ),
            )

            rows_updated = cursor.rowcount
            conn.commit()

            if rows_updated > 0:
                logger.info(f"✓ Updated annotation {annotation.id} in database")
            else:
                logger.warning(f"✗ Annotation {annotation.id} not found for update")

            return rows_updated > 0
        except sqlite3.Error as e:
            logger.error(f"SQLite error updating annotation {annotation.id}: {e}")
            conn.rollback()
            raise

    def get_distinct_object_classes(
        self, location_id: Optional[str] = None
    ) -> List[str]:
        """Get list of distinct object classes seen in a location."""

    pass
