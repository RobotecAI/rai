# RAI Semantic Map Memory

⚠️ **Experimental Module**: This module is in active development. Features may change and some functionality is still in progress.

## Overview

Imagine your robot exploring a new warehouse or office building. Using SLAM (Simultaneous Localization and Mapping), it builds a geometric map showing walls and open areas, but it doesn't remember what objects it saw—like that tool cart or equipment in the storage area.

RAI Semantic Map Memory solves this by adding a memory layer. As the robot explores, it remembers not just where walls are, but also what objects it detected and where they were located. Later, you can ask questions like "where did I see a pallet?" or "what objects are near the loading dock?" and the robot can answer using its stored memory.

This module provides persistent storage of semantic annotations—linking object identities (like "shelf", "cart", "pallet") to their 3D locations in the map. It enables spatial-semantic queries that combine "what" and "where" information.

## Some Usage Examples

-   Store object detections with their locations as the robot explores
-   Query objects by location: "what's near point (x, y)?"
-   Visualize stored annotations overlaid on the SLAM map

For detailed design and architecture, see [design.md](../design.md).

## Quick Start

The following examples use the ROSBot XL demo to illustrate how to use rai_semap.

### Prerequisites

-   ROS2 environment set up
-   rai_semap package installed: `poetry install --with semap`
-   ROSBot XL demo setup, see instuctions at [ROSBot XL demo](../../docs/demos/rosbot_xl.md)

### Step 0: Launch the ROSBot XL demo

Follow the instruction from [ROSBot XL demo](../../docs/demos/rosbot_xl.md).

### Step 1: Launch the Semantic Map Node

Start the semantic map node to begin collecting and storing detections:

```bash
ros2 launch src/rai_semap/rai_semap/scripts/semap.launch.py
```

This uses default configuration files from `rai_semap/ros2/config/`. The default configs assume depth topic `/camera/depth/image_rect_raw` and camera info topic `/camera/color/camera_info`. If your topics use different names, create custom config files or override parameters.

To use custom configs:

```bash
ros2 launch src/rai_semap/rai_semap/scripts/semap.launch.py \
    node_config:=/path/to/node.yaml \
    detection_publisher_config:=/path/to/detection_publisher.yaml \
    perception_utils_config:=/path/to/perception_utils.yaml
```

### Step 2: Collect Detections

In a separate terminal, run the navigation script to move the robot through waypoints and collect detections:

```bash
python -m rai_semap.scripts.navigate_collect \
    --waypoints 2.0 0.0 4.0 0.0 2.0 2.0 \
    --collect-duration 10.0 \
    --use-sim-time
```

The script navigates to each waypoint and waits to allow detections to be collected and stored in the semantic map.

### Step 3: Validate Stored Data

After navigation completes, verify what was stored:

```bash
python -m rai_semap.scripts.validate_semap \
    --database-path semantic_map.db \
    --location-id default_location
```

The validation script shows total annotation count, annotations grouped by object class, confidence scores, and spatial distribution.

## Configuration

Configuration parameters (database_path, location_id, topics, etc.) are set in YAML config files. If config files are not provided, default configs in `rai_semap/ros2/config/` are used.

## Visualization

View your semantic map annotations overlaid on the SLAM map in RViz2.

### Start the Visualizer

```bash
python -m rai_semap.ros2.visualizer \
    --ros-args \
    -p database_path:=semantic_map.db \
    -p location_id:=default_location \
    -p update_rate:=1.0 \
    -p marker_scale:=0.3 \
    -p show_text_labels:=true
```

### Setup RViz2

1. Launch RViz2: `rviz2`
2. Add displays:
    - Add "Map" display → subscribe to `/map` topic
    - Add "MarkerArray" display → subscribe to `/semantic_map_markers` topic
3. Set Fixed Frame to `map` (or your map frame ID)

The visualizer shows color-coded markers by object class (bed=blue, chair=green, door=orange, shelf=purple, table=violet). Marker transparency scales with confidence score, and optional text labels show object class names.

## Querying the Semantic Map

Query stored annotations programmatically using the Python API:

```python
from geometry_msgs.msg import Point
from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticMapMemory

# Initialize memory
backend = SQLiteBackend("semantic_map.db")
memory = SemanticMapMemory(
    backend=backend,
    location_id="default_location",
    map_frame_id="map",
    resolution=0.05,
)

# Query annotations near a location
center = Point(x=2.0, y=0.0, z=0.0)
annotations = memory.query_by_location(center, radius=2.0)

for ann in annotations:
    print(f"Found {ann.object_class} at ({ann.pose.position.x}, {ann.pose.position.y})")
```
