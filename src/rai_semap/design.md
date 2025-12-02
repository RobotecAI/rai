# RAI's Agent Memory System

## Table of Contents

- [Problem Definition](#problem-definition)
  - [Q & A](#q--a)
- [Concepts](#concepts)
  - [High-Level Concepts](#high-level-concepts)
  - [Data Models](#data-models)
  - [Relationships](#relationships)
  - [Non Goals](#non-goals)
- [Design Proposal](#design-proposal)
  - [BaseMemory Interface](#basememory-interface)
  - [SemanticMapMemory Interface](#semanticmapmemory-interface)
  - [Database Backend Abstraction](#database-backend-abstraction)
  - [New Component: rai_semap](#new-component-rai_semap)
  - [Usage Patterns from Other Layers](#usage-patterns-from-other-layers)
  - [Implementation Phases](#implementation-phases)
- [Reusability](#reusability)
- [External Memory Systems](#external-memory-systems)
- [Appendix](#appendix)

## Problem Definition

[Issue#225 Design SLAM RAI features](https://github.com/RobotecAI/rai/issues/225) presents an explorative SLAM/semantic mapping integration task:

> "Robots often need to find out about their environment first, building a map and localizing themselves on it. During this process, RAI can be used to guide exploration for mapping or to build a semantic map during the SLAM which adds knowledge and memory, which can be used to reason about the area itself and tasks that are to be given in the area. A great start is to design solutions for RAI."

Based on RAI's current capabilities (perception, navigation, multi-modal interaction), this roughly maps to three areas:

1. Semantic Perception Layer which may be built on top of `rai_perception` (open-set detection), Grounded SAM 2 integration
   with a new semantic annotation pipeline that tags SLAM features/points with object identities during mapping

2. Agent Guided Exploration Strategy which can be built on existing `rai_nomad` (navigation) where agent decides _where_ to explore based on goals ("find the kitchen", "map storage areas") rather than frontier-based exploration. Frontier-based exploration navigates to boundaries between known and unknown map regions to expand coverage.

3. Spatial Memory System which provides persistent semantic map storage that agents can query ("where did I see tools?") and reason over ("this room is suitable for assembly tasks"). The word _spatial_ refers to 3D location/position information in map coordinates.

    - The connection between spatial memory and other RAI memory systems (artifact_database.pkl, rai_whoami vector store, StateBasedAgent state) needs to be explored: spatial memory could be queried by these systems to provide spatial context, rather than serving as storage for them.
    - For example, could artifacts be annotated with spatial locations queried from spatial memory, could embodiment docs reference spatial locations that spatial memory could ground, or could recent spatial queries be included in StateBasedAgent state to provide spatial awareness during conversations?

For explorative SLAM, these areas build on each other: perception feeds the semantic map, exploration uses it to guide decisions, and memory enables task reasoning.

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

**A spatial-semantic record linking an object identity (class label, e.g., "red cup", "tool") to a 3D pose in the map frame, with metadata (timestamp, confidence, detection source).**

Unlike pure geometric SLAM, semantic annotations enable querying "what" and "where" simultaneously. This allows agents to reason about object locations for task planning and spatial reasoning. The combination of semantic labels and 3D poses creates a rich representation that bridges perception and spatial memory.

```python
# Example: Semantic annotation structure
{
    "object_class": "red cup",
    "pose": {"x": 2.5, "y": 1.3, "z": 0.8, "orientation": {...}},
    "confidence": 0.92,
    "timestamp": "2025-01-15T10:23:00",
    "detection_source": "GroundingDINO",
    "source_frame": "camera_frame"
}
```

#### Spatial Memory

**A conceptual system that provides persistent storage of semantic annotations indexed by both spatial coordinates (3D: x, y, z) and semantic labels.**

The storage can be implemented via database backends (SQLite/PostGIS) accessed through the `SemanticMapMemory` interface. This dual indexing enables efficient queries like "find objects near (x,y)" (2D projection when z is not needed) and "where did I see X?" by combining spatial indexing with semantic search. Without spatial memory, agents cannot recall where objects were seen, limiting task planning capabilities.

```python
# Example: Spatial query
memory.query_nearby_objects(x=2.5, y=1.3, radius=2.0)
# Returns: [{"object": "red cup", "distance": 0.5}, 
#           {"object": "table", "distance": 1.2}]

# Example: Semantic query
memory.query_by_class("red cup")
# Returns: [{"pose": (2.5, 1.3, 0.8), "confidence": 0.92, ...}]
```

#### Camera-to-map Transformation

**Converting detections from camera frame to map frame using TF transforms.**

The perception layer (GroundingDINO, GroundedSAM) already provides 3D poses in `ObjectHypothesisWithPose`, so the system transforms these poses from the camera frame to the map frame using TF transforms (camera → base_link → map). This is critical for building a consistent spatial-semantic map across robot movements. Without proper frame transformation, detections from different robot positions would be stored in inconsistent coordinate systems, making spatial queries unreliable.

```python
# Example: Frame transformation flow
# Detection in camera frame
camera_pose = {"x": 0.3, "y": 0.1, "z": 1.2}  # relative to camera

# Transform to map frame via TF
map_pose = tf_buffer.transform(
    camera_pose, 
    target_frame="map", 
    source_frame="camera_frame"
)
# Result: {"x": 2.5, "y": 1.3, "z": 0.8}  # absolute map coordinates

# Same object detected from different angle → same map coordinates
```

#### Temporal Consistency

**Handling multiple detections of the same object instance over time by merging duplicates based on spatial proximity.**

Tracks individual instances (by spatial location), not object classes. Without temporal consistency, repeated detections of the same object would create duplicate records, making queries like "where did I see the red cup?" return multiple locations for the same object, rendering the database inconsistent and queries unreliable. Temporal consistency merges repeated detections of the same physical object (same location within a threshold), not different objects even if they share the same class label. A key challenge is distinguishing a moved object (same instance, new location) from a new object instance (different instance, similar appearance).

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

Very small bounding boxes (below minimum area threshold, default: 500 pixels²) are filtered out as they are often false positives from partial occlusions or detection artifacts.

```python
# Example: Size filtering
bbox_area = width * height  # pixels²
if bbox_area < 500:
    # Filtered out (likely false positive)
    return None
```

#### Query Patterns

**Primary query types: spatial queries (objects near a location), semantic queries (locations of object classes), and hybrid queries (combining both).**

These query patterns enable agents to retrieve spatial-semantic information for task planning. Spatial queries support navigation and proximity-based reasoning. Semantic queries enable object retrieval tasks. Hybrid queries combine both for complex scenarios like "find tools in the workshop."

```python
# Spatial query: "What objects are within 2m of (x,y)?"
results = memory.query_nearby_objects(x=2.5, y=1.3, radius=2.0)

# Semantic query: "Where did I see a red cup?"
results = memory.query_by_class("red cup")

# Hybrid query: "Find tools in the workshop" (semantic + spatial region)
results = memory.query_by_class_and_region(
    object_class="tool",
    region_polygon=[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
)
```

### Data Models

#### SemanticAnnotation

**A data structure representing a single semantic annotation with object identity, 3D pose, confidence, and metadata.**

This is the core data model that stores all semantic-spatial information in the memory system. Each annotation links a detected object to its location in the map frame, enabling spatial queries and temporal consistency tracking. The metadata field allows extensibility for point cloud features and other attributes without changing the core schema.

```python
@dataclass
class SemanticAnnotation:
    id: UUID  # Unique identifier
    object_class: str  # e.g., "red cup", "tool"
    pose: Pose  # 3D pose in map frame (x, y, z, orientation)
    confidence: float  # 0.0-1.0
    timestamp: Time  # ROS timestamp of detection
    detection_source: str  # e.g., "GroundingDINO", "GroundedSAM"
    source_frame: str  # Original camera frame_id
    vision_detection_id: str  # ID from Detection2D for debugging
    metadata: dict  # Optional JSON with pointcloud info when available
    # metadata.pointcloud: {centroid, size_3d, point_count}
```

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

This metadata enables the system to correctly interpret spatial coordinates and maintain consistency with the underlying SLAM map. The resolution and origin are needed to convert between map coordinates and pixel coordinates for visualization. The last_updated timestamp helps track map freshness and coordinate system changes.

```python
@dataclass
class MapMetadata:
    map_frame_id: str  # Frame ID of the SLAM map
    resolution: float  # OccupancyGrid resolution (meters/pixel)
    origin: Pose  # Map origin pose
    last_updated: Time  # Timestamp of last annotation
```

### Relationships

#### Perception Layer → Memory System

**`RAIDetectionArray` messages flow from `rai_perception` services (GroundingDINO, GroundedSAM) into `rai_semap`, which projects detections to map frame and stores them.**

The `detection_publisher` node bridges the service-based perception layer to topic-based messaging by subscribing to camera images, calling DINO service, and publishing `RAIDetectionArray` messages to `/detection_array` topic. This decoupling allows the memory system to work with any perception service that publishes `RAIDetectionArray` messages, not just GroundingDINO. The topic-based interface enables multiple consumers and easier debugging.

```python
# Flow: Camera → detection_publisher → RAIDetectionArray → semantic_map_node
# detection_publisher subscribes to /camera/image_raw
# Calls /grounding_dino/grounding_dino_classify service
# Publishes RAIDetectionArray to /detection_array
# semantic_map_node subscribes to /detection_array and stores annotations
```

#### Exploration Layer → Memory System

**Agent-guided exploration uses semantic map queries to find unexplored regions with specific semantic properties.**

The memory system returns candidate locations for exploration goals like "find areas with storage furniture". This enables goal-based exploration rather than purely geometric frontier-based exploration. The exploration layer can query for semantic hints ("I saw a shelf, explore that direction") and use coverage tracking to prioritize unexplored regions.

```python
# Example: Goal-based exploration query
candidates = memory.query_by_class("shelf")
unexplored_regions = exploration.find_unexplored_near(candidates)
# Agent navigates to unexplored regions near detected shelves
```

#### Memory System → Agent Tools

**Agents query semantic map via `QuerySemanticMapTool` to retrieve object locations for task planning.**

This integration enables multi-step task planning where agents can query object locations, navigate to them, verify presence, and manipulate objects. Without this connection, agents would have no persistent spatial memory and would need to re-detect objects every time, limiting task capabilities.

```python
# Example: Agent tool usage
tool = QuerySemanticMapTool(memory)
result = tool.invoke("red cup in kitchen")
# Returns: {"object": "red cup", "location": (2.5, 1.3, 0.8), ...}
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

Since no `BaseMemory` interface exists in RAI, we define this interface to enable consistent memory system integration across RAI components. This interface allows `SemanticMapMemory` and future memory systems to share a common API while each extends it with domain-specific methods. See `base_memory.py` for the interface definition.

```python
# Example: BaseMemory interface structure
class BaseMemory(ABC):
    @abstractmethod
    def store(self, data: Any) -> str:
        """Store data and return identifier."""
        pass
    
    @abstractmethod
    def query(self, query: str) -> List[Any]:
        """Query stored data."""
        pass
```

### SemanticMapMemory Interface

**`SemanticMapMemory` extends `BaseMemory` with spatial query capabilities for semantic annotations.**

This interface provides the contract for spatial-semantic memory operations, enabling agent tools and exploration layers to query object locations without depending on the specific database backend implementation. The interface abstracts away backend details (SQLite vs PostGIS) while providing spatial query methods. See `semantic_map_memory.py` for the interface definition.

```python
# Example: SemanticMapMemory interface methods
class SemanticMapMemory(BaseMemory):
    def query_nearby_objects(self, x: float, y: float, radius: float) -> List[SemanticAnnotation]:
        """Query objects within radius of (x, y)."""
        pass
    
    def query_by_class(self, object_class: str) -> List[SemanticAnnotation]:
        """Query all annotations of a given class."""
        pass
```

### Database Backend Abstraction

**A backend abstraction layer that supports both SQLite (Phase I) and PostGIS (future) implementations.**

This abstraction enables switching between database backends without changing the `SemanticMapMemory` interface or agent tools. SQLite provides a lightweight, single-file solution for Phase I, while PostGIS enables advanced features for future multi-robot deployments. See `spatial_db_backend.py` for the interface definition.

**SQLiteBackend (Phase I):**

Uses SpatiaLite extension for spatial indexing. Single-file database with no external dependencies. Can be deployed on-board the robot (no network or separate server required). Sufficient for single-robot deployments.

```python
# Example: SQLiteBackend usage
backend = SQLiteBackend("semantic_map.db")
memory = SemanticMapMemory(backend)
# Single file, no server needed
```

**PostGISBackend (future):**

Full PostgreSQL + PostGIS for advanced spatial operations. Supports multi-robot coordination via shared database (cloud or local network server). Better performance for large-scale maps.

```python
# Example: PostGISBackend usage (future)
backend = PostGISBackend(connection_string="postgresql://...")
memory = SemanticMapMemory(backend)
# Shared database for multi-robot coordination
```

Configuration via ROS2 parameters (set via launch file or command line):

Detection Publisher Node (`rai_semap.ros2.detection_publisher`):

-   `camera_topic`: Camera image topic to subscribe to (default: "/camera/image_raw")
-   `detection_topic`: Topic to publish RAIDetectionArray messages (default: "/detection_array")
-   `dino_service`: GroundingDINO service name (default: "/grounding_dino/grounding_dino_classify")
-   `detection_classes`: Comma-separated list of object classes to detect (default: "person, cup, bottle, box, bag, chair, table, shelf, door, window")
-   `detection_interval`: Minimum time between detections in seconds (default: 2.0)
-   `box_threshold`: DINO box threshold (default: 0.3)
-   `text_threshold`: DINO text threshold (default: 0.25)

Semantic Map Node (`rai_semap.ros2.node`):

-   `database_path`: Path to SQLite database file (default: "semantic_map.db")
-   `confidence_threshold`: Minimum confidence score (0.0-1.0) for storing detections (default: 0.5)
-   `class_confidence_thresholds`: Class-specific confidence thresholds as 'class1:threshold1,class2:threshold2' (e.g., 'person:0.7,window:0.6,door:0.5')
-   `class_merge_thresholds`: Class-specific merge radii (meters) for deduplication as 'class1:radius1,class2:radius2' (e.g., 'couch:2.5,table:1.5,shelf:1.5,chair:0.8')
-   `min_bbox_area`: Minimum bounding box area (pixels²) to filter small false positives (default: 500.0)
-   `use_pointcloud_dedup`: Enable point cloud-based deduplication matching (default: true)
-   `depth_topic`: Depth image topic for point cloud extraction (optional, required if use_pointcloud_dedup=true)
-   `camera_info_topic`: Camera info topic for point cloud extraction (optional, required if use_pointcloud_dedup=true)
-   `detection_topic`: Topic for RAIDetectionArray messages (default: "/detection_array")
-   `map_topic`: Topic for OccupancyGrid map messages (default: "/map")
-   `map_frame_id`: Frame ID of the SLAM map (default: "map")
-   `location_id`: Identifier for the physical location (default: "default_location")
-   `map_resolution`: OccupancyGrid resolution in meters/pixel (default: 0.05)

Future: PostGIS backend selection will be configurable (not yet implemented).

### New Component: `rai_semap`

Architecture:

`rai_semap` consists of a core library, a ROS2 node wrapper and tools.

1. Core Library (`rai_semap.core`):

    - Frame Projection: Transform 3D poses from source frame (camera frame) to map frame using TF transforms. Detections already contain 3D poses in `ObjectHypothesisWithPose`, so depth estimation is handled by the perception layer.
    - Temporal Filtering: Merge duplicate detections of same object using multi-strategy deduplication:
        - Spatial clustering within class-specific merge thresholds
        - Point cloud-based matching when depth data is available (uses 3D centroid and size validation)
        - Confidence filtering with class-specific thresholds
        - Bounding box size filtering to remove small false positives
    - Confidence Aggregation: Update confidence scores for repeated observations (keeps maximum confidence)
    - Storage: Persist annotations to `SemanticMapMemory` backend with point cloud features in metadata
    - No ROS2 dependencies; pure Python library

2. ROS2 Node Wrapper (`rai_semap.ros2`):

    - `detection_publisher` node:
        - Subscribes to camera images (e.g., `/camera/image_raw`)
        - Calls GroundingDINO service with configurable object classes
        - Publishes `RAIDetectionArray` messages to `/detection_array` topic
        - Throttles detections via `detection_interval` parameter to control processing rate
    - `node` (semantic map node):
        - Subscriptions:
            - `RAIDetectionArray` topic (detections from `detection_publisher` or other sources)
            - `/map` (OccupancyGrid from SLAM)
        - TF handling via `tf2_ros.Buffer` and `TransformListener` (automatically subscribes to `/tf` and `/tf_static`)
        - Converts ROS2 messages to core library data structures
        - Calls core library processing functions
        - Handles ROS2-specific concerns (TF lookups, message conversion)

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

Perception Layer Requirements:

-   Service-to-topic bridge: The `detection_publisher` node bridges the service-based perception layer (GroundingDINO, GroundedSAM) to topic-based messaging by continuously processing camera images and publishing detections
-   Topic-based processing: Handle `RAIDetectionArray` messages published to topics (from `detection_publisher` or other sources)
-   Immediate processing: Each detection array is processed immediately upon receipt in the callback
-   Confidence filtering: Only store high-confidence detections (configurable via ROS2 parameter `confidence_threshold`)
-   Configurable detection rate: The `detection_interval` parameter controls how frequently detections are processed to balance accuracy and computational load

Exploration Layer Requirements (prioritized):

1. Coverage tracking: Query which map regions have been semantically annotated (M)
    - Foundation for other exploration features
    - Requires spatial indexing of annotated regions (grid-based or region-based)
    - Grid-based example: Divide map into fixed-size cells (e.g., 0.5m x 0.5m aligned with OccupancyGrid resolution), track which cells contain annotations
    - Region-based example: Use spatial clustering or bounding polygons around annotated areas, define regions dynamically based on annotation density
2. Goal-based queries: "Find unexplored regions with 'kitchen' objects" (M)
    - Depends on: Coverage tracking, semantic queries
    - Combines semantic search with coverage information
3. Frontier detection: Identify boundaries between mapped and unmapped semantic regions (L)
    - Depends on: Coverage tracking
    - Requires boundary detection algorithms and spatial analysis

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

`rai_semap` may be reusable for object retrieval scenario which robot finds and retrieves objects it saw earlier. For example, after initial mapping, user asks: _"Bring me the red cup I saw in the kitchen"_

-   Flow:

1. Agent queries semantic map: `QuerySemanticMapTool("red cup", room="kitchen")`
2. Semantic map returns: `{object: "red cup", location: (x: 2.5, y: 1.3, z: 0.8), timestamp: "2025-01-15T10:23:00", confidence: 0.92}`
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
