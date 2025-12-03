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

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from geometry_msgs.msg import Point

from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticMapMemory


def validate_database(database_path: str, location_id: str = "rosbot_xl_demo") -> bool:
    """Validate stored data in semantic map database.

    Args:
        database_path: Path to SQLite database file.
        location_id: Location identifier to query.

    Returns:
        True if validation passes, False otherwise.
    """
    if not Path(database_path).exists():
        print(f"ERROR: Database file not found: {database_path}")
        return False

    try:
        backend = SQLiteBackend(database_path)
        memory = SemanticMapMemory(
            backend=backend,
            location_id=location_id,
            map_frame_id="map",
            resolution=0.05,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize memory: {e}")
        return False

    print(f"Validating semantic map database: {database_path}")
    print(f"Location ID: {location_id}")
    print("-" * 60)

    # Get map metadata
    try:
        metadata = memory.get_map_metadata()
        print(f"Map Frame ID: {metadata.map_frame_id}")
        print(f"Map Resolution: {metadata.resolution} m/pixel")
        print(f"Last Updated: {metadata.last_updated}")
    except Exception as e:
        print(f"WARNING: Failed to get map metadata: {e}")

    print("-" * 60)

    # Query all annotations
    center = Point(x=0.0, y=0.0, z=0.0)
    all_annotations = memory.query_by_location(
        center, radius=1e10, location_id=location_id
    )

    if not all_annotations:
        print("WARNING: No annotations found in database")
        return False

    print(f"Total annotations: {len(all_annotations)}")
    print("-" * 60)

    # Group by object class
    class_counts = defaultdict(int)
    confidence_sum = defaultdict(float)
    detection_sources = defaultdict(set)

    for ann in all_annotations:
        class_counts[ann.object_class] += 1
        confidence_sum[ann.object_class] += ann.confidence
        detection_sources[ann.object_class].add(ann.detection_source)

    print("Annotations by class:")
    for obj_class in sorted(class_counts.keys()):
        count = class_counts[obj_class]
        avg_confidence = confidence_sum[obj_class] / count
        sources = ", ".join(sorted(detection_sources[obj_class]))
        print(
            f"  {obj_class}: {count} annotations, avg confidence: {avg_confidence:.3f}, sources: {sources}"
        )

    print("-" * 60)

    # Check for required fields
    print("Validating annotation fields...")
    all_valid = True

    for ann in all_annotations:
        if not ann.object_class:
            print(f"ERROR: Annotation {ann.id} has empty object_class")
            all_valid = False
        if ann.confidence < 0.0 or ann.confidence > 1.0:
            print(
                f"ERROR: Annotation {ann.id} has invalid confidence: {ann.confidence}"
            )
            all_valid = False
        if not ann.detection_source:
            print(f"WARNING: Annotation {ann.id} has empty detection_source")
        if not ann.source_frame:
            print(f"WARNING: Annotation {ann.id} has empty source_frame")

    if all_valid:
        print("All annotations have valid required fields")

    print("-" * 60)

    # Spatial distribution
    if all_annotations:
        x_coords = [ann.pose.position.x for ann in all_annotations]
        y_coords = [ann.pose.position.y for ann in all_annotations]

        print("Spatial distribution:")
        print(f"  X range: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
        print(f"  Y range: [{min(y_coords):.2f}, {max(y_coords):.2f}]")
        print(
            f"  Mean position: ({sum(x_coords) / len(x_coords):.2f}, {sum(y_coords) / len(y_coords):.2f})"
        )

    print("-" * 60)
    print("Validation complete")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate stored data in semantic map database"
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default="semantic_map.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--location-id",
        type=str,
        default="default_location",
        help="Location identifier to query",
    )

    args = parser.parse_args()

    success = validate_database(args.database_path, args.location_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
