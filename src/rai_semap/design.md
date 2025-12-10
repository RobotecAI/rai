# RAI's Agent Memory System

## Table of Contents

-   [Problem Definition](#problem-definition)
    -   [Q & A](#q--a)
-   [Concepts](#concepts)
    -   [High-Level Concepts](#high-level-concepts)
    -   [Data Models](#data-models)
    -   [Relationships](#relationships)
    -   [Non Goals](#non-goals)
-   [Design Proposal](#design-proposal)
    -   [BaseMemory Interface](#basememory-interface)
    -   [SemanticMapMemory Interface](#semanticmapmemory-interface)
    -   [Database Backend Abstraction](#database-backend-abstraction)
    -   [New Component: rai_semap](#new-component-rai_semap)
    -   [Usage Patterns from Other Layers](#usage-patterns-from-other-layers)
    -   [Implementation Phases](#implementation-phases)
-   [Reusability](#reusability)
-   [External Memory Systems](#external-memory-systems)
-   [Appendix](#appendix)

## Problem Definition

[Issue#225 Design SLAM RAI features](https://github.com/RobotecAI/rai/issues/225) presents an explorative SLAM/semantic mapping integration task:

> "Robots often need to find out about their environment first, building a map and localizing themselves on it. During this process, RAI can be used to guide exploration for mapping or to build a semantic map during the SLAM which adds knowledge and memory, which can be used to reason about the area itself and tasks that are to be given in the area. A great start is to design solutions for RAI."

Based on RAI's current capabilities (perception, navigation, multi-modal interaction), this maps to three areas that build on each other: perception feeds the semantic map, exploration uses it to guide decisions, and memory enables task reasoning.

1. Semantic Perception Layer: Built on `rai_perception` (open-set detection) with GroundingDINO integration, creating a semantic annotation pipeline that tags SLAM features/points with object identities during mapping

2. Agent Guided Exploration Strategy: Built on `rai_nomad` (navigation) where the agent decides where to explore based on goals ("find the kitchen", "map storage areas") rather than frontier-based exploration. Frontier-based exploration navigates to boundaries between known and unknown map regions to expand coverage.

3. Spatial Memory System: Provides persistent semantic map storage that agents can query ("where did I see tools?") and reason over ("this room is suitable for assembly tasks"). The word _spatial_ refers to 3D location/position information in map coordinates.

    - The connection between spatial memory and other RAI memory systems (artifact_database.pkl, rai_whoami vector store, StateBasedAgent state) needs exploration: spatial memory could be queried by these systems to provide spatial context, rather than serving as storage. For example, artifacts could be annotated with spatial locations, embodiment docs could reference spatial locations that spatial memory grounds, or recent spatial queries could be included in StateBasedAgent state for spatial awareness during conversations.

### Q & A

Q: why "map storage areas" is not a frontier-based exploration?

"Map storage areas" is goal-based, not frontier-based, because frontier-based chooses the nearest boundary between known and unknown regions, regardless of what might be there. It's geometry-driven. Goal-based ("map storage areas") uses semantic reasoning to prioritize exploration. The agent might:

-   Use partial detections ("I saw a shelf, explore that direction")
-   Reason about room layouts ("storage areas are often in basements or corners")
-   Query the semantic map for hints about where storage might be

Both may explore unknown regions, but the decision differs: frontier-based picks the nearest unknown boundary whereas goal-based uses semantic cues to target likely locations.

## Concepts

### High-Level Concepts

#### Semantic Annotation

**A spatial-semantic record linking an object identity (class label, e.g., "red cup", "tool") to a 3D position (centroid/center) in the map frame, with metadata (timestamp, confidence, detection source).**

Unlike pure geometric SLAM, semantic annotations enable querying "what" and "where" simultaneously, allowing agents to reason about object locations for task planning. The combination of semantic labels and 3D positions bridges perception and spatial memory. The position is stored as a `Pose` object, where `pose.position` is the 3D point (computed from bounding box center or point cloud centroid when available) and `pose.orientation` is typically identity (not meaningful).

```python
# Example: Semantic annotation structure (simplified)
{
    "object_class": "red cup",
    "pose": {"position": {"x": 2.5, "y": 1.3, "z": 0.8}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
    "confidence": 0.92,
    "timestamp": "2025-01-15T10:23:00",
    "detection_source": "GroundingDINO",
    "source_frame": "camera_frame"
}
# Note: pose.position is a 3D point (centroid/center) in map frame, computed from bounding box center
# or point cloud centroid when available. pose.orientation is typically identity (not meaningful).
```

#### Spatial Memory

**A conceptual system that provides persistent storage of semantic annotations indexed by both spatial coordinates (3D: x, y, z) and semantic labels.**

Storage is implemented via database backends (SQLite/PostGIS) accessed through the `SemanticMapMemory` interface. Dual indexing enables efficient queries like "find objects near (x,y)" (2D projection when z is not needed) and "where did I see X?" by combining spatial indexing with semantic search. Without spatial memory, agents cannot recall where objects were seen, limiting task planning.

```python
# Example: Spatial query
from geometry_msgs.msg import Point
center = Point(x=2.5, y=1.3, z=0.0)
results = memory.query_by_location(center, radius=2.0)
# Returns: List[SemanticAnnotation] with objects within radius

# Example: Semantic query
results = memory.query_by_class("red cup")
# Returns: List[SemanticAnnotation] with all "red cup" annotations
```

#### Camera-to-map Transformation

**Converting detections from camera frame to map frame using TF transforms.**

The perception layer provides detections with 3D positions (GroundingDINO provides 2D bounding boxes; we compute 3D poses from bounding box centers using depth images and camera intrinsics). These positions are initially in the camera frame. The system transforms them to the map frame using TF transforms (camera → base_link → map). This is critical for building a consistent spatial-semantic map across robot movements. Without proper frame transformation, detections from different robot positions would be stored in inconsistent coordinate systems, making spatial queries unreliable.

```python
# Example: Frame transformation flow (pseudo code)
# Detection in camera frame
camera_pose = (x=0.3, y=0.1, z=1.2)  # relative to camera

# Transform to map frame via TF
map_pose = transform_pose(
    camera_pose,
    source_frame="camera_frame",
    target_frame="map"
)
# Result: (x=2.5, y=1.3, z=0.8)  # absolute map coordinates

# Same object detected from different angle → same map coordinates
```

#### Temporal Consistency

**Handling multiple detections of the same object instance over time by merging duplicates based on spatial proximity.**

Tracks individual instances (by spatial location), not object classes. Without temporal consistency, repeated detections of the same object would create duplicate records, making queries like "where did I see the red cup?" return multiple locations for the same object, rendering the database inconsistent. Temporal consistency merges repeated detections of the same physical object (same location within a threshold), not different objects even if they share the same class label. A key challenge is distinguishing a moved object (same instance, new location) from a new object instance (different instance, similar appearance).

```python
# Example: Multiple detections of same object
# Detection 1 at t=0: "red cup" at (2.5, 1.3, 0.8), confidence=0.85
# Detection 2 at t=5: "red cup" at (2.52, 1.31, 0.81), confidence=0.92
# → Merged into single annotation with max confidence (0.92) and latest timestamp

# Example: Different objects (same class, different locations)
# Detection 1: "box" at (1.0, 2.0, 0.5)  # Box A
# Detection 2: "box" at (3.0, 4.0, 0.5)  # Box B (different instance)
# → Two separate annotations stored
```

#### Deduplication Strategies

**Multiple techniques work together to prevent duplicate annotations: spatial merging, point cloud-based matching, confidence filtering, and bounding box size filtering.**

These strategies work together to ensure database consistency and query reliability. Confidence and size filtering happen first to remove low-quality detections, then spatial merging with point cloud validation occurs during storage. Without deduplication, the database would be polluted with duplicate entries, making spatial queries return incorrect results and wasting storage.

**1. Spatial Merging**

Detections of the same class within a merge threshold (distance in meters) are merged into a single annotation. The merge threshold is class-specific to handle objects of different sizes. When merging, the system keeps the maximum confidence score and updates the timestamp to the latest detection.

```python
# Example: Class-specific merge thresholds
merge_thresholds = {
    "couch": 2.5,    # Large objects
    "table": 1.5,
    "shelf": 1.5,
    "chair": 0.8,    # Medium objects
    "cup": 0.5       # Small objects (default)
}

# Two "cup" detections within 0.5m → merged
# Two "couch" detections within 2.5m → merged
```

**2. Point Cloud-Based Matching**

When depth images are available, the system extracts 3D point clouds from bounding box regions and uses them for more accurate deduplication. Point cloud centroid is more accurate than bounding box center for spatial matching. Size validation compares 3D point cloud sizes to avoid merging objects of very different sizes. If point cloud sizes differ by >50% and >0.5m, detections are treated as different objects even if spatially close.

```python
# Example: Point cloud validation
detection1 = {
    "centroid": (2.5, 1.3, 0.8),
    "size_3d": 0.15,  # meters
    "point_count": 1250
}
detection2 = {
    "centroid": (2.52, 1.31, 0.81),  # Close spatially
    "size_3d": 0.25,  # 67% larger → different object
    "point_count": 2100
}
# Result: Not merged (size difference >50% and >0.5m)
```

**3. Confidence Filtering**

Only detections above a confidence threshold are stored. The threshold is class-specific to handle high false-positive classes. This prevents low-confidence false positives from polluting the database.

```python
# Example: Class-specific confidence thresholds
confidence_thresholds = {
    "person": 0.7,   # High false-positive rate
    "window": 0.6,
    "door": 0.5,
    "cup": 0.5       # Default threshold
}
```

**4. Bounding Box Size Filtering**

Very small bounding boxes (below minimum area threshold, default: 100 pixels²) are filtered out as they are often false positives from partial occlusions or detection artifacts.

```python
# Example: Size filtering
bbox_area = width * height  # pixels²
if bbox_area < min_bbox_area:  # default: 100 pixels²
    # Filtered out (likely false positive)
    return None
```

#### Query Patterns

**Primary query types: spatial queries (objects near a location), semantic queries (locations of object classes), and hybrid queries (combining both).**

These query patterns enable agents to retrieve spatial-semantic information for task planning. Spatial queries support navigation and proximity-based reasoning. Semantic queries enable object retrieval tasks. Hybrid queries combine both for complex scenarios like "find tools in the workshop."

```python
# Spatial query: "What objects are within 2m of (x,y)?"
from geometry_msgs.msg import Point
center = Point(x=2.5, y=1.3, z=0.0)
results = memory.query_by_location(center, radius=2.0)

# Semantic query: "Where did I see a red cup?"
results = memory.query_by_class("red cup")

# Hybrid query: "Find tools in the workshop" (semantic + spatial region)
results = memory.query_by_region(
    bbox=(x1, y1, x2, y2),  # (min_x, min_y, max_x, max_y)
    object_class="tool"
)
```

### Data Models

#### SemanticAnnotation

**A data structure representing a single semantic annotation with object identity, 3D pose, confidence, and metadata.**

This is the core data model storing all semantic-spatial information. Each annotation links a detected object to its location in the map frame, enabling spatial queries and temporal consistency tracking. The metadata field allows extensibility for point cloud features and other attributes without changing the core schema.

```python
class SemanticAnnotation:
    id: str  # Unique identifier (UUID string)
    object_class: str  # e.g., "red cup", "tool"
    pose: Pose  # 3D pose in map frame (x, y, z, orientation)
    confidence: float  # 0.0-1.0
    timestamp: float  # Unix timestamp in seconds (timezone-naive)
    detection_source: str  # e.g., "GroundingDINO", "GroundedSAM"
    source_frame: str  # Original camera frame_id
    location_id: str  # Identifier for the physical location
    vision_detection_id: Optional[str]  # ID from Detection2D for debugging
    metadata: Dict[str, Any]  # Optional JSON with pointcloud info when available
    # metadata.pointcloud: {centroid, size_3d, point_count}
```

Regarding the `timestamp` field, there is a choice of using Python `datetime`. While it provides type safety and easy comparison, using `datetime` has a few drawbacks. For example:

1. Precision loss: ROS `rclpy.time.Time` has nanosecond precision (`sec` + `nanosec`), while Python `datetime` has microsecond precision.

2. Conversion overhead: We'll need to convert `rclpy.time.Time` → `datetime` at the ROS boundary. Example:

```python
# Conversion needed
ros_time = rclpy.time.Time.from_msg(msg.header.stamp)
# Convert to datetime (loses nanosec precision)
dt = datetime.fromtimestamp(ros_time.nanoseconds / 1e9)
```

3. Database storage: Both SQLite and PostgreSQL still require conversion:
    - SQLite: No native datetime type; stored as TEXT (ISO format) or REAL/INTEGER (Unix timestamp)
    - PostgreSQL: Has TIMESTAMP, but you still need to handle timezone (naive vs aware)

Implementation decision: We use `float` (Unix timestamp in seconds) for timezone-naive, PostgreSQL-compatible storage. Convert from `rclpy.time.Time` using `timestamp.nanoseconds / 1e9`. Nanosecond precision is lost but acceptable for timestamps.

#### SpatialIndex

**Database-level spatial index (R-tree) for efficient spatial queries.**

Spatial indexing is essential for performance when querying large numbers of annotations. Without it, spatial queries would require scanning all records, which becomes prohibitively slow as the map grows. SQLite uses SpatiaLite extension; PostGIS uses native GIST indexes. Both provide sub-linear query performance for spatial operations.

```sql
-- Example: Spatial index creation (SpatiaLite)
CREATE VIRTUAL TABLE annotations_rtree USING rtree(
    id, minx, maxx, miny, maxy
);

-- Example: Efficient spatial query using index
SELECT * FROM annotations
WHERE id IN (
    SELECT id FROM annotations_rtree
    WHERE minx <= x+radius AND maxx >= x-radius
    AND miny <= y+radius AND maxy >= y-radius
);
```

#### MapMetadata

**Metadata about the SLAM map including frame ID, resolution, origin, and last update timestamp.**

This metadata enables correct interpretation of spatial coordinates and consistency with the underlying SLAM map. Resolution and origin convert between map coordinates and pixel coordinates for visualization. The last_updated timestamp tracks map freshness and coordinate system changes.

```python
class MapMetadata:
    location_id: str  # Identifier for the physical location
    map_frame_id: str  # Frame ID of the SLAM map
    resolution: float  # OccupancyGrid resolution (meters/pixel)
    origin: Optional[Pose]  # Optional map origin pose
    last_updated: Optional[float]  # Unix timestamp (seconds) of last annotation
```

### Relationships

#### Perception Layer → Memory System

**`RAIDetectionArray` messages flow from `rai_perception` services (GroundingDINO, GroundedSAM) into `rai_semap`, which projects detections to map frame and stores them.**

The `detection_publisher` node bridges the service-based perception layer to topic-based messaging by subscribing to camera images, calling DINO service, and publishing `RAIDetectionArray` messages to `/detection_array`. This decoupling allows the memory system to work with any perception service that publishes `RAIDetectionArray` messages, not just GroundingDINO. The topic-based interface enables multiple consumers and easier debugging.

```python
# Flow: Camera → detection_publisher → RAIDetectionArray → semantic_map_node
# detection_publisher subscribes to /camera/camera/color/image_raw (configurable)
# Calls /grounding_dino_classify service (configurable)
# Publishes RAIDetectionArray to /detection_array
# semantic_map_node subscribes to /detection_array and stores annotations
```

#### Exploration Layer → Memory System

**Agent-guided exploration uses semantic map queries to find unexplored regions with specific semantic properties.**

The memory system returns candidate locations for exploration goals like "find areas with storage furniture", enabling goal-based exploration rather than purely geometric frontier-based exploration. The exploration layer can query for semantic hints ("I saw a shelf, explore that direction") and use coverage tracking to prioritize unexplored regions.

```python
# Example: Goal-based exploration query
candidates = memory.query_by_class("shelf")
unexplored_regions = exploration.find_unexplored_near(candidates)
# Agent navigates to unexplored regions near detected shelves
```

#### Memory System → Agent Tools

**Agents query semantic map via `QuerySemanticMapTool` to retrieve object locations for task planning.**

This integration enables multi-step task planning: query object locations, navigate to them, verify presence, and manipulate objects. Without this connection, agents would have no persistent spatial memory and would need to re-detect objects every time, limiting task capabilities.

```python
# Example: Agent tool usage
tool = QuerySemanticMapTool(memory=memory)
result = tool.invoke({"query": "red cup in kitchen", "room": "kitchen"})
# Returns: String with object locations
# Agent uses NavigateToPoseTool to go to location
```

### Non Goals

Future Integration Points:

-   `artifact_database.pkl`: Could store semantic annotations alongside multimodal artifacts
-   `rai_whoami` vector store: Could index semantic annotations for LLM-based reasoning
-   `StateBasedAgent` state: Could include recent semantic map queries in conversation context

## Design Proposal

### BaseMemory Interface

**A minimal abstract interface for memory systems that allows future memory systems (conversational, vector-based, etc.) to share a common API.**

Since no `BaseMemory` interface exists in RAI, we define this interface to enable consistent memory system integration across RAI components. It allows `SemanticMapMemory` and future memory systems to share a common API while each extends it with domain-specific methods. See `base_memory.py` for the interface definition.

```python
# Example: BaseMemory interface structure
class BaseMemory(ABC):
    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a value with optional metadata. Returns storage ID."""
        pass

    @abstractmethod
    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Retrieve values matching query and filters.

        Designed for vector database use cases where query is text to embed
        for similarity search, and filters are metadata constraints.
        Not suitable for spatial databases which require concrete query methods
        (e.g., query_by_location, query_by_region).
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a stored value. Returns success status."""
        pass
```

### SemanticMapMemory Interface

**`SemanticMapMemory` extends `BaseMemory` with spatial query capabilities for semantic annotations.**

This interface provides the contract for spatial-semantic memory operations, enabling agent tools and exploration layers to query object locations without depending on the specific database backend implementation. The interface abstracts away backend details (SQLite vs PostGIS) while providing spatial query methods. See `semantic_map_memory.py` for the interface definition.

```python
# Example: SemanticMapMemory interface methods
class SemanticMapMemory(BaseMemory):
    def query_by_location(self, center: Point, radius: float, object_class: Optional[str] = None, location_id: Optional[str] = None) -> List[SemanticAnnotation]:
        """Query annotations within radius of center point."""
        pass

    def query_by_class(self, object_class: str, confidence_threshold: float = 0.5, limit: Optional[int] = None, location_id: Optional[str] = None) -> List[SemanticAnnotation]:
        """Query annotations by object class."""
        pass

    def query_by_region(self, bbox: Tuple[float, float, float, float], object_class: Optional[str] = None, location_id: Optional[str] = None) -> List[SemanticAnnotation]:
        """Query annotations within bounding box region."""
        pass
```

### Database Backend Abstraction

**A backend abstraction layer that supports both SQLite (Phase I) and PostGIS (future) implementations.**

This abstraction enables switching between database backends without changing the `SemanticMapMemory` interface or agent tools. SQLite provides a lightweight, single-file solution for Phase I; PostGIS enables advanced features for future multi-robot deployments. See `spatial_db_backend.py` for the interface definition.

**SQLiteBackend (Phase I):**

Uses SpatiaLite extension for spatial indexing. Single-file database with no external dependencies. Can be deployed on-board the robot (no network or separate server required). Sufficient for single-robot deployments.

```python
# Example: SQLiteBackend usage
backend = SQLiteBackend("semantic_map.db")
backend.init_schema()  # Initialize database schema
memory = SemanticMapMemory(backend, location_id="default_location")
# Single file, no server needed
```

**PostGISBackend (future):**

Full PostgreSQL + PostGIS for advanced spatial operations. Supports multi-robot coordination via shared database (cloud or local network server). Better performance for large-scale maps.

```python
# Example: PostGISBackend usage (future)
backend = PostGISBackend(connection_string="postgresql://...")
backend.init_schema()  # Initialize database schema
memory = SemanticMapMemory(backend, location_id="warehouse_a")
# Shared database for multi-robot coordination
```

Backend selection is configurable via `backend_type` parameter (currently supports "sqlite"; PostGIS backend not yet implemented).

### New Component: `rai_semap`

Architecture:

`rai_semap` consists of a core library, a ROS2 node wrapper and tools.

1. Core Library (`rai_semap.core`):

    - Frame Projection: Transform 3D poses from camera frame to map frame using TF transforms
    - Temporal Filtering: Multi-strategy deduplication (spatial clustering, point cloud-based matching, confidence/size filtering) to merge duplicate detections
    - Storage: Persist annotations to `SemanticMapMemory` backend with point cloud features in metadata
    - Pure Python library with no ROS2 dependencies

2. ROS2 Node Wrapper (`rai_semap.ros2`):

    - `detection_publisher` node: Subscribes to camera images, calls GroundingDINO service, publishes `RAIDetectionArray` messages with configurable throttling
    - `node` (semantic map node): Subscribes to `RAIDetectionArray` and `/map` topics, handles TF transforms, converts ROS2 messages to core library data structures, calls core processing functions
    - `visualizer` node: Publishes semantic map annotations as RViz2 markers for real-time visualization, querying the database at configurable intervals

3. Tools/Services:
    - `QuerySemanticMapTool`: LangChain tool for agent queries
    - ROS2 service for programmatic access (not in current scope)

Dependency Flow:

```
Camera Images → detection_publisher → RAIDetectionArray → semantic_map_node → rai_semap.core → SemanticMapMemory → Agent Tools
     ↓                  ↓                      ↓                    ↓                    ↓              ↓
  /camera/image_raw  DINO Service      Detection2D         ROS2 Wrapper      Frame Transform   SQLite/PostGIS
                     (service call)    (3D pose, class)   (msg conversion)  (TF transform)    Spatial Queries
```

### Usage Patterns from Other Layers

**Perception Layer**: The `detection_publisher` node bridges service-based perception (GroundingDINO) to topic-based messaging, processing camera images and publishing `RAIDetectionArray` messages with configurable confidence filtering and detection rate throttling.

**Exploration Layer** (preliminary): Future integration could support coverage tracking (identifying annotated map regions), goal-based queries (finding unexplored regions with specific semantic properties), and frontier detection (boundaries between mapped/unmapped regions). These features would enable agent-guided exploration beyond geometric frontier-based methods.

Agent Tool Integration:

-   Natural language queries: `QuerySemanticMapTool("red cup in kitchen")` → spatial query
-   Multi-step planning: Query → Navigate → Verify → Manipulate
-   Temporal reasoning: "Where did I see X yesterday?" (requires timestamp filtering)

### Implementation Phases

Phase I (SQLite):

-   Implement `SQLiteBackend` with SpatiaLite
-   Basic `SemanticMapMemory` with spatial queries
-   `rai_semap` node with frame projection
-   `QuerySemanticMapTool` for agent integration
-   Single-robot deployment
-   Validation demo using rosbot-xl: Build semantic map during navigation, query object locations (e.g., "Where did I see the bed?"), verify detections are correctly stored and retrieved

Future direction (PostGIS Migration):

-   Implement `PostGISBackend` with same interface
-   Configuration-based backend switching
-   Multi-robot coordination support
-   Advanced spatial operations (polygon queries, distance calculations)

## Reusability

`rai_semap` may be reusable for object retrieval scenarios where the robot finds and retrieves objects it saw earlier. For example, after initial mapping, user asks: _"Bring me the red cup I saw in the kitchen"_

-   Flow:

1. Agent queries semantic map: `QuerySemanticMapTool.invoke({"query": "red cup", "room": "kitchen"})`
2. Semantic map returns: String with object locations including pose information
3. Agent uses `NavigateToPoseTool` to go to that location
4. Agent uses `GetDetectionTool` to confirm object presence
5. Agent uses manipulation tools to grab and return the cup

-   Benefits:
    -   Persistent memory: remembers objects across sessions
    -   Spatial reasoning: knows where things are, not just what they are
    -   Task planning: can plan multi-step retrieval tasks

More scenarios. These are yet to be explored, listed here just for future revisit of the design.

-   Inventory tracking: "What tools are in the workshop?"
-   Change detection: "Did anything move in the living room?"
-   Multi-robot coordination: share semantic map between robots
-   Long-term monitoring: track object locations over days/weeks

## External Memory Systems

### mem0

[mem0](https://github.com/mem0ai/mem0) is a mature implementation (43.8k stars, production-ready). It's not a good fit for RAI. mem0 targets conversational memory, while RAI needs spatial-semantic storage with pose queries.

### ROS semantic_mapping

C++/ROS1, no Python API or SQLite/PostGIS backend, [source reference](https://github.com/fdayoub/ros-semantic-mapper).

### KnowRob

Knowledge reasoning, not spatial-semantic storage, [source reference](https://github.com/knowrob/knowrob)

### SEGO (Semantic Graph Ontology)

Research framework, no production storage backend, [paper](https://arxiv.org/abs/2506.13149)

### Semantic SLAM projects

Mostly C++ (ORB-SLAM2 etc) not Python with database backends. [orb-slam2 source reference](https://github.com/appliedAI-Initiative/orb_slam_2_ros)
The frame viewer from this [post](https://records.sigmm.org/?open-source-item=openvslam-a-versatile-visual-slam-framework) is fantastic for visualization.

## Appendix

### PostGIS

PostGIS is an extension for PostgreSQL, not a separate database. It adds spatial data types and functions (geometry, geography, spatial indexing, spatial queries). After installing PostgreSQL, then enable the PostGIS extension with `CREATE EXTENSION postgis;`.
