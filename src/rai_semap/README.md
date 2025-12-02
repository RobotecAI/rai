# RAI Semantic Map Memory

⚠️ **Experimental Module**: This module is in active development. Features may change and some functionality is still in progress.

## Moudle Overview

When a robot explores a new environment, it builds a map of the space using SLAM (Simultaneous Localization and Mapping), which creates a geometric map showing walls, obstacles, and open areas. This module extends that map with semantic information, storing what objects the robot sees and where they are located. It provides persistent storage of semantic annotations (object class, 3D pose, confidence) indexed by both spatial coordinates and semantic labels. Using this semantic map memory, user can do queries like "where did I see a red cup?" and "what objects are within 2m of (x,y)?".

For detailed design and architecture, see [design.md](../design.md).

## End-to-End Validation

Phase 4 validation tests the full pipeline: launching the demo, collecting detections during navigation, and verifying stored data.

### Prerequisites

-   ROS2 environment set up
-   rai_semap package installed, `poetry install --with semap`
-   ROSBot XL demo setup, see general readme.

### Steps

1. Launch rosbot-xl demo with semantic map node:

```bash
ros2 launch examples/rosbot-xl-semap.launch.py \
    game_launcher:=./examples/rosbot-xl.launch.py game_launcher:=demo_assets/rosbot/RAIROSBotXLDemo/RAIROSBotXLDemo.GameLauncher \
    database_path:=semantic_map.db \
    location_id:=rosbot_xl_demo
```

2. Navigate and collect detections:

In a separate terminal, run the navigation script to move the robot through waypoints and collect detections:

```bash
python examples/rosbot-xl-navigate-collect.py \
    --waypoints 2.0 0.0 4.0 0.0 2.0 2.0 \
    --collect-duration 10.0 \
    --use-sim-time
```

The script navigates to each waypoint and waits to allow detections to be collected and stored in the semantic map.

3. Validate stored data:

After navigation completes, run the validation script to verify the database contents:

```bash
python examples/validate-semantic-map.py \
    --database-path semantic_map.db \
    --location-id rosbot_xl_demo
```

The validation script reports:

-   Total annotation count
-   Annotations grouped by object class
-   Average confidence scores per class
-   Detection sources
-   Spatial distribution of annotations
-   Field validation checks

### Query by Timestamp

To query annotations by timestamp, you can filter results after querying. Timestamps are stored as string representations of ROS2 Time objects. Here's an example Python script:

```python
from geometry_msgs.msg import Point
from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticMapMemory

# Initialize memory
backend = SQLiteBackend("semantic_map.db")
memory = SemanticMapMemory(
    backend=backend,
    location_id="rosbot_xl_demo",
    map_frame_id="map",
    resolution=0.05,
)

# Query all annotations
center = Point(x=0.0, y=0.0, z=0.0)
all_annotations = memory.query_by_location(center, radius=1e10)

# Filter by timestamp using string comparison
# Timestamps are stored as strings in format: "Time(nanoseconds=...)"
# Extract nanoseconds from timestamp string for comparison
def parse_timestamp_ns(timestamp_str):
    """Extract nanoseconds from timestamp string like 'Time(nanoseconds=123456789)'"""
    if not timestamp_str or not timestamp_str.startswith("Time(nanoseconds="):
        return None
    try:
        ns_str = timestamp_str.split("nanoseconds=")[1].rstrip(")")
        return int(ns_str)
    except (IndexError, ValueError):
        return None

# Example: Get annotations from last hour
import time
current_ns = int(time.time() * 1e9)
one_hour_ns = int(3600 * 1e9)
one_hour_ago_ns = current_ns - one_hour_ns

recent_annotations = [
    ann for ann in all_annotations
    if ann.timestamp and (ann_ns := parse_timestamp_ns(ann.timestamp)) and ann_ns >= one_hour_ago_ns
]

print(f"Found {len(recent_annotations)} annotations from the last hour")

# Filter by specific time range (using string comparison)
start_time_str = "Time(nanoseconds=1700000000000000000)"  # Example timestamp
end_time_str = "Time(nanoseconds=1700003600000000000)"   # Example timestamp

time_range_annotations = [
    ann for ann in all_annotations
    if ann.timestamp and start_time_str <= ann.timestamp <= end_time_str
]

print(f"Found {len(time_range_annotations)} annotations in time range")
```

For more complex timestamp queries, you can also query the database directly using SQL:

```python
import sqlite3

conn = sqlite3.connect("semantic_map.db")
cursor = conn.cursor()

# Example: Query annotations after a specific timestamp
# Timestamps are stored as strings, so string comparison works for ordering
cursor.execute("""
    SELECT * FROM annotations
    WHERE location_id = ?
    AND timestamp >= ?
    ORDER BY timestamp DESC
""", ("rosbot_xl_demo", "Time(nanoseconds=1700000000000000000)"))

rows = cursor.fetchall()
print(f"Found {len(rows)} annotations after specified timestamp")
```

### Expected Output

The validation script should show annotations stored from detections collected during navigation, with valid confidence scores, detection sources, and spatial coordinates in the map frame.

## Utilities

### Clear All Annotations

To remove all annotations from the semantic map database:

```bash
python -m rai_semap.utils.clear_annotations \
    --database-path semantic_map.db
```

To remove annotations for a specific location only:

```bash
python -m rai_semap.utils.clear_annotations \
    --database-path semantic_map.db \
    --location-id rosbot_xl_demo
```

To skip the confirmation prompt:

```bash
python -m rai_semap.utils.clear_annotations \
    --database-path semantic_map.db \
    --yes
```

The script will:

-   Show the number of annotations that will be deleted
-   Prompt for confirmation (unless `--yes` is used)
-   Delete the annotations and report the count deleted
