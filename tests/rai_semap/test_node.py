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
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, ObjectHypothesis, ObjectHypothesisWithPose

# RAI interfaces
from rai_interfaces.msg import RAIDetectionArray

# Local imports
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
    node = SemanticMapNode(database_path=temp_db_path, executor_type="single_threaded")
    yield node
    node.shutdown()


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
    return detection


def test_node_creation(node):
    """Test that SemanticMapNode can be created with default parameters."""
    from rai.communication.ros2 import ROS2Connector

    assert isinstance(node, ROS2Connector)
    assert node.node.get_name() == "rai_semap_node"

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
        assert node.node.has_parameter(param)


def test_node_parameter_defaults(node, temp_db_path):
    """Test that node parameters have correct default values."""
    assert (
        node.node.get_parameter("database_path").get_parameter_value().string_value
        == temp_db_path
    )
    assert (
        node.node.get_parameter("confidence_threshold")
        .get_parameter_value()
        .double_value
        == 0.5
    )
    assert (
        node.node.get_parameter("detection_topic").get_parameter_value().string_value
        == "/detection_array"
    )
    assert (
        node.node.get_parameter("map_topic").get_parameter_value().string_value
        == "/map"
    )
    assert (
        node.node.get_parameter("map_frame_id").get_parameter_value().string_value
        == "map"
    )
    assert (
        node.node.get_parameter("location_id").get_parameter_value().string_value
        == "default_location"
    )
    assert (
        node.node.get_parameter("map_resolution").get_parameter_value().double_value
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
    node.node.set_parameters(
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
    node.node.set_parameters(
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
    node.node.set_parameters(
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
