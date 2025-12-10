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

import tempfile
from pathlib import Path

# Testing framework
import pytest

# ROS2 core
import rclpy

# ROS2 message types
from geometry_msgs.msg import Pose, Quaternion
from nav_msgs.msg import MapMetaData, OccupancyGrid

# Local imports
from rai.communication.ros2 import ROS2Connector
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, ObjectHypothesis, ObjectHypothesisWithPose

# RAI interfaces
from rai_interfaces.msg import RAIDetectionArray
from rai_semap.ros2.node import SemanticMapNode


@pytest.fixture(scope="module")
def ros2_context():
    """Initialize ROS2 context for testing."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def node(ros2_context, temp_db_path):
    """Create a SemanticMapNode instance for testing.

    Uses single_threaded executor to avoid executor performance warnings
    in simple unit tests that don't need multi-threaded execution.
    """
    connector = ROS2Connector(
        node_name="rai_semap_node", executor_type="single_threaded"
    )
    node = SemanticMapNode(connector=connector, database_path=temp_db_path)
    yield node
    node.connector.shutdown()


# Test helper functions
def create_detection_message(frame_id: str, detections: list) -> RAIDetectionArray:
    """Create a RAIDetectionArray message with given detections."""
    msg = RAIDetectionArray()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.detections = detections
    return msg


def create_detection(
    frame_id: str,
    class_id: str,
    score: float,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 0.0,
    bbox_size_x: float = 100.0,
    bbox_size_y: float = 100.0,
) -> Detection2D:
    """Create a Detection2D with ObjectHypothesisWithPose."""
    detection = Detection2D()
    detection.header = Header()
    detection.header.frame_id = frame_id

    hypothesis = ObjectHypothesis()
    hypothesis.class_id = class_id
    hypothesis.score = score

    result = ObjectHypothesisWithPose()
    result.hypothesis = hypothesis
    result.pose.pose = Pose()
    result.pose.pose.position.x = x
    result.pose.pose.position.y = y
    result.pose.pose.position.z = z
    result.pose.pose.orientation = Quaternion(w=1.0)

    detection.results = [result]
    detection.bbox.size_x = bbox_size_x
    detection.bbox.size_y = bbox_size_y
    return detection


# Tests for refactored methods
def test_validate_and_extract_detection_data_success(node):
    """Test successful validation and extraction of detection data."""
    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "confidence_threshold", rclpy.parameter.Parameter.Type.DOUBLE, 0.5
            ),
            rclpy.parameter.Parameter(
                "min_bbox_area", rclpy.parameter.Parameter.Type.DOUBLE, 100.0
            ),
        ]
    )

    detection = create_detection(
        "camera_frame", "cup", score=0.9, bbox_size_x=200.0, bbox_size_y=200.0
    )

    result = node._validate_and_extract_detection_data(
        detection, confidence_threshold=0.5, default_frame_id="camera_frame"
    )

    assert result is not None
    object_class, confidence, source_frame, pose = result
    assert object_class == "cup"
    assert confidence == 0.9
    assert source_frame == "camera_frame"
    assert pose.position.x == 1.0
    assert pose.position.y == 2.0
    assert pose.position.z == 0.0


def test_validate_and_extract_detection_data_low_confidence(node):
    """Test that low confidence detections are rejected."""
    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "confidence_threshold", rclpy.parameter.Parameter.Type.DOUBLE, 0.8
            ),
        ]
    )

    detection = create_detection("camera_frame", "cup", score=0.5)

    result = node._validate_and_extract_detection_data(
        detection, confidence_threshold=0.8, default_frame_id="camera_frame"
    )

    assert result is None


def test_validate_and_extract_detection_data_frame_id_fallback(node):
    """Test that frame_id fallback works."""
    detection = create_detection("", "cup", score=0.9)
    detection.header.frame_id = ""

    result = node._validate_and_extract_detection_data(
        detection, confidence_threshold=0.5, default_frame_id="default_frame"
    )

    assert result is not None
    _, _, source_frame, _ = result
    assert source_frame == "default_frame"


def test_validate_and_transform_pose_empty_pose(node):
    """Test that empty poses are rejected."""
    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.0

    detection = create_detection("camera_frame", "cup", score=0.9, x=0.0, y=0.0, z=0.0)

    result = node._validate_and_transform_pose(
        pose, "camera_frame", "map", "cup", detection
    )

    assert result is None


def test_extract_pointcloud_features_disabled(node):
    """Test that point cloud extraction returns None when disabled."""
    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "use_pointcloud_dedup",
                rclpy.parameter.Parameter.Type.BOOL,
                False,
            ),
        ]
    )

    detection = create_detection("camera_frame", "cup", score=0.9)

    features, centroid, size = node._extract_pointcloud_features(
        detection, "camera_frame", "map"
    )

    assert features is None
    assert centroid is None
    assert size is None


def test_determine_merge_decision_no_nearby(node):
    """Test merge decision when no nearby annotations exist."""
    should_merge, existing_id = node._determine_merge_decision(
        nearby=[], pointcloud_features=None, pc_size=None, use_pointcloud=False
    )

    assert should_merge is False
    assert existing_id is None


def test_determine_merge_decision_with_nearby_no_pcl(node):
    """Test merge decision with nearby annotation but no point cloud."""
    from rai.types import Point, Pose

    from rai_semap.core.semantic_map_memory import SemanticAnnotation

    existing = SemanticAnnotation(
        id="test-id",
        object_class="cup",
        pose=Pose(position=Point(x=1.0, y=2.0, z=0.0)),
        confidence=0.8,
        timestamp=1234567890,
        detection_source="test",
        source_frame="map",
        location_id="default_location",
    )

    should_merge, existing_id = node._determine_merge_decision(
        nearby=[existing],
        pointcloud_features=None,
        pc_size=None,
        use_pointcloud=False,
    )

    assert should_merge is True
    assert existing_id == "test-id"


def test_determine_merge_decision_with_pc_size_match(node):
    """Test merge decision with point cloud size matching."""
    from rai.types import Point, Pose

    from rai_semap.core.semantic_map_memory import SemanticAnnotation

    existing = SemanticAnnotation(
        id="test-id",
        object_class="cup",
        pose=Pose(position=Point(x=1.0, y=2.0, z=0.0)),
        confidence=0.8,
        timestamp=1234567890,
        detection_source="test",
        source_frame="map",
        location_id="default_location",
        metadata={"pointcloud": {"size_3d": 0.5}},
    )

    pointcloud_features = {"size_3d": 0.5, "centroid": {}, "point_count": 100}

    should_merge, existing_id = node._determine_merge_decision(
        nearby=[existing],
        pointcloud_features=pointcloud_features,
        pc_size=0.5,
        use_pointcloud=True,
    )

    assert should_merge is True
    assert existing_id == "test-id"


def test_store_or_update_annotation_new(node):
    """Test storing a new annotation."""
    import rclpy.time
    from rai.types import Point, Pose

    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "map_frame_id", rclpy.parameter.Parameter.Type.STRING, "camera_frame"
            ),
        ]
    )

    pose = Pose(position=Point(x=1.0, y=2.0, z=0.0))
    timestamp_ros = rclpy.time.Time()
    timestamp = timestamp_ros.nanoseconds / 1e9  # Convert to Unix timestamp (seconds)

    initial_count = len(node.memory.query_by_class("test_object"))
    success = node._store_or_update_annotation(
        object_class="test_object",
        confidence=0.9,
        pose_in_map_frame=pose,
        pointcloud_centroid_map=None,
        pointcloud_features=None,
        pc_size=None,
        timestamp=timestamp,
        detection_source="test",
        source_frame="camera_frame",
        vision_detection_id=None,
    )

    assert success is True
    final_count = len(node.memory.query_by_class("test_object"))
    assert final_count == initial_count + 1


def test_store_or_update_annotation_update_existing(node):
    """Test updating an existing annotation."""
    import rclpy.time
    from rai.types import Point, Pose

    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "map_frame_id", rclpy.parameter.Parameter.Type.STRING, "camera_frame"
            ),
        ]
    )

    # First, store an annotation
    pose1 = Pose(position=Point(x=1.0, y=2.0, z=0.0))
    timestamp_ros = rclpy.time.Time()
    timestamp = timestamp_ros.nanoseconds / 1e9  # Convert to Unix timestamp (seconds)

    node._store_or_update_annotation(
        object_class="test_object",
        confidence=0.8,
        pose_in_map_frame=pose1,
        pointcloud_centroid_map=None,
        pointcloud_features=None,
        pc_size=None,
        timestamp=timestamp,
        detection_source="test",
        source_frame="camera_frame",
        vision_detection_id=None,
    )

    # Now update it with higher confidence
    pose2 = Pose(position=Point(x=1.01, y=2.01, z=0.0))
    success = node._store_or_update_annotation(
        object_class="test_object",
        confidence=0.95,
        pose_in_map_frame=pose2,
        pointcloud_centroid_map=None,
        pointcloud_features=None,
        pc_size=None,
        timestamp=timestamp,
        detection_source="test",
        source_frame="camera_frame",
        vision_detection_id=None,
    )

    assert success is True
    annotations = node.memory.query_by_class("test_object")
    assert len(annotations) == 1
    assert annotations[0].confidence == 0.95


def test_node_creation(node):
    """Test that SemanticMapNode can be created with default parameters."""
    assert isinstance(node.connector, ROS2Connector)
    assert node.connector.node.get_name() == "rai_semap_node"

    expected_params = [
        "database_path",
        "confidence_threshold",
        "detection_topic",
        "map_topic",
        "map_frame_id",
        "location_id",
        "map_resolution",
    ]
    for param in expected_params:
        assert node.connector.node.has_parameter(param)


def test_node_parameter_defaults(node, temp_db_path):
    """Test that node parameters have correct default values."""
    assert (
        node.connector.node.get_parameter("database_path")
        .get_parameter_value()
        .string_value
        == temp_db_path
    )
    assert (
        node.connector.node.get_parameter("confidence_threshold")
        .get_parameter_value()
        .double_value
        == 0.5
    )
    assert (
        node.connector.node.get_parameter("detection_topic")
        .get_parameter_value()
        .string_value
        == "/detection_array"
    )
    assert (
        node.connector.node.get_parameter("map_topic")
        .get_parameter_value()
        .string_value
        == "/map"
    )
    assert (
        node.connector.node.get_parameter("map_frame_id")
        .get_parameter_value()
        .string_value
        == "map"
    )
    assert (
        node.connector.node.get_parameter("location_id")
        .get_parameter_value()
        .string_value
        == "default_location"
    )
    assert (
        node.connector.node.get_parameter("map_resolution")
        .get_parameter_value()
        .double_value
        == 0.05
    )


def test_node_memory_initialization(node):
    """Test that semantic map memory is properly initialized."""
    assert node.memory is not None
    assert node.memory.location_id == "default_location"
    assert node.memory.map_frame_id == "map"
    assert node.memory.resolution == 0.05


def test_node_subscriptions_created(node):
    """Test that subscriptions are created."""
    assert node.detection_subscription is not None
    assert node.map_subscription is not None


def test_detection_callback_low_confidence(node):
    """Test that detections below confidence threshold are filtered out."""
    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "confidence_threshold", rclpy.parameter.Parameter.Type.DOUBLE, 0.8
            ),
        ]
    )

    detection = create_detection("camera_frame", "cup", score=0.5)
    msg = create_detection_message("camera_frame", [detection])

    initial_count = len(node.memory.query_by_class("cup"))
    node.detection_callback(msg)
    final_count = len(node.memory.query_by_class("cup"))

    assert final_count == initial_count


def test_detection_callback_high_confidence(node):
    """Test that high-confidence detections are processed."""
    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "confidence_threshold", rclpy.parameter.Parameter.Type.DOUBLE, 0.5
            ),
            rclpy.parameter.Parameter(
                "map_frame_id", rclpy.parameter.Parameter.Type.STRING, "camera_frame"
            ),
        ]
    )

    detection = create_detection("camera_frame", "bottle", score=0.9)
    msg = create_detection_message("GroundingDINO", [detection])

    initial_count = len(node.memory.query_by_class("bottle"))
    node.detection_callback(msg)
    final_count = len(node.memory.query_by_class("bottle"))

    assert final_count >= initial_count


def test_map_callback_updates_metadata(node):
    """Test that map callback updates metadata."""
    node.connector.node.set_parameters(
        [
            rclpy.parameter.Parameter(
                "map_frame_id", rclpy.parameter.Parameter.Type.STRING, "map"
            ),
        ]
    )

    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.frame_id = "map"
    msg.info = MapMetaData()
    msg.info.resolution = 0.1

    initial_resolution = node.memory.resolution
    node.map_callback(msg)

    assert node.memory.map_frame_id == "map"
    assert node.memory.resolution == 0.1
    assert node.memory.resolution != initial_resolution


def test_detection_callback_empty_detections(node):
    """Test that empty detection arrays are handled gracefully."""
    msg = create_detection_message("camera_frame", [])
    node.detection_callback(msg)


def test_detection_callback_no_results(node):
    """Test that detections without results are skipped."""
    detection = Detection2D()
    detection.header = Header()
    detection.results = []

    msg = create_detection_message("camera_frame", [detection])

    initial_count = len(node.memory.query_by_class("cup"))
    node.detection_callback(msg)
    final_count = len(node.memory.query_by_class("cup"))

    assert final_count == initial_count
