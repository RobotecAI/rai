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

"""Utility script to clear all annotations from semantic map database."""

import argparse
import sys

from geometry_msgs.msg import Point

from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticMapMemory


def main():
    parser = argparse.ArgumentParser(
        description="Clear all annotations from semantic map database"
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default="semantic_map.db",
        help="Path to SQLite database file (default: semantic_map.db)",
    )
    parser.add_argument(
        "--location-id",
        type=str,
        default=None,
        help="If provided, only delete annotations for this location. "
        "If not provided, delete all annotations.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Initialize backend and memory
    backend = SQLiteBackend(args.database_path)
    backend.init_schema()

    # Query to get count before deletion
    if args.location_id:
        center = Point(x=0.0, y=0.0, z=0.0)
        filters = {"location_id": args.location_id}
        existing = backend.spatial_query(center, radius=1e10, filters=filters)
        count = len(existing)
        location_msg = f" for location_id='{args.location_id}'"
    else:
        center = Point(x=0.0, y=0.0, z=0.0)
        existing = backend.spatial_query(center, radius=1e10, filters={})
        count = len(existing)
        location_msg = ""

    if count == 0:
        print(f"No annotations found{location_msg} in database: {args.database_path}")
        return 0

    # Confirmation prompt
    if not args.yes:
        if args.location_id:
            prompt = (
                f"Are you sure you want to delete {count} annotation(s) "
                f"for location_id='{args.location_id}' from {args.database_path}? [y/N]: "
            )
        else:
            prompt = (
                f"Are you sure you want to delete ALL {count} annotation(s) "
                f"from {args.database_path}? [y/N]: "
            )
        response = input(prompt)
        if response.lower() not in ["y", "yes"]:
            print("Cancelled.")
            return 0

    # Delete annotations
    memory = SemanticMapMemory(
        backend=backend,
        location_id=args.location_id or "default_location",
    )

    deleted_count = memory.delete_all_annotations(location_id=args.location_id)

    print(f"✓ Successfully deleted {deleted_count} annotation(s){location_msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
