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


import time
import uuid

import pytest
from geometry_msgs.msg import Pose, Quaternion, TransformStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rai.communication.ros2 import QoS, ROS2Connector
from rai.communication.ros2.runtime import ROS2Runtime
from rai.types import Point
from rai.types import Pose as RaiPose
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection2D, ObjectHypothesis, ObjectHypothesisWithPose

from rai_interfaces.msg import RAIDetectionArray
from rai_semap.core.semantic_map_memory import SemanticAnnotation
from rai_semap.ros2.node import SemanticMapConfig, SemanticMapNode


def _name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True, scope="module")
def _rclpy(ros2_context):
    """Connectors here run in a context this module starts and shuts down."""


@pytest.fixture
def node(temp_db_path):
    with ROS2Connector(_name("rai_semap_node")) as connector:
        yield SemanticMapNode(connector=connector, database_path=temp_db_path)


def make_node(temp_db_path, *params: str) -> SemanticMapNode:
    """A node on a private runtime whose --ros-args set params ("name:=value")."""
    args = ["--ros-args", *(a for p in params for a in ("-p", p))]
    runtime = ROS2Runtime(_name("rai_semap_node"), cli_args=args)
    return SemanticMapNode(ROS2Connector(runtime=runtime), database_path=temp_db_path)


def wait_until(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def create_detection_message(frame_id: str, detections: list) -> RAIDetectionArray:
    msg = RAIDetectionArray()
    msg.header = Header(frame_id=frame_id)
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
    detection = Detection2D()
    detection.header = Header(frame_id=frame_id)
    result = ObjectHypothesisWithPose()
    result.hypothesis = ObjectHypothesis(class_id=class_id, score=score)
    result.pose.pose = Pose()
    result.pose.pose.position.x = x
    result.pose.pose.position.y = y
    result.pose.pose.position.z = z
    result.pose.pose.orientation = Quaternion(w=1.0)
    detection.results = [result]
    detection.bbox.size_x = bbox_size_x
    detection.bbox.size_y = bbox_size_y
    return detection


def annotation(**overrides) -> SemanticAnnotation:
    fields = dict(
        id="test-id",
        object_class="cup",
        pose=RaiPose(position=Point(x=1.0, y=2.0, z=0.0)),
        confidence=0.8,
        timestamp=1234567890,
        detection_source="test",
        source_frame="map",
        location_id="default_location",
    )
    return SemanticAnnotation(**(fields | overrides))


def store(node: SemanticMapNode, confidence: float, x: float = 1.0) -> bool:
    return node._store_or_update_annotation(
        object_class="test_object",
        confidence=confidence,
        pose_in_map_frame=RaiPose(position=Point(x=x, y=2.0, z=0.0)),
        pointcloud_centroid_map=None,
        pointcloud_features=None,
        pc_size=None,
        timestamp=1234567890.0,
        detection_source="test",
        source_frame="camera_frame",
        vision_detection_id=None,
    )


def test_validate_and_extract_detection_data_success(node):
    detection = create_detection(
        "camera_frame", "cup", score=0.9, bbox_size_x=200.0, bbox_size_y=200.0
    )
    result = node._validate_and_extract_detection_data(
        detection, confidence_threshold=0.5, default_frame_id="camera_frame"
    )
    assert result is not None
    object_class, confidence, source_frame, pose = result
    assert (object_class, source_frame) == ("cup", "camera_frame")
    assert confidence == pytest.approx(0.9)
    assert (pose.position.x, pose.position.y, pose.position.z) == (1.0, 2.0, 0.0)


def test_validate_and_extract_detection_data_filters(node):
    low = create_detection("camera_frame", "cup", score=0.5)
    assert node._validate_and_extract_detection_data(low, 0.8, "camera_frame") is None
    small = create_detection("camera_frame", "cup", 0.9, bbox_size_x=5, bbox_size_y=5)
    assert node._validate_and_extract_detection_data(small, 0.5, "f") is None
    node.class_thresholds["cup"] = 0.95
    high = create_detection("camera_frame", "cup", score=0.9)
    assert node._validate_and_extract_detection_data(high, 0.5, "f") is None


def test_validate_and_extract_detection_data_frame_id_fallback(node):
    detection = create_detection("", "cup", score=0.9)
    result = node._validate_and_extract_detection_data(detection, 0.5, "default_frame")
    assert result is not None and result[2] == "default_frame"
    assert node._validate_and_extract_detection_data(detection, 0.5, "") is None


def test_validate_and_transform_pose_empty_pose(node):
    detection = create_detection("camera_frame", "cup", score=0.9, x=0.0, y=0.0)
    result = node._validate_and_transform_pose(
        Pose(), "camera_frame", "map", "cup", detection
    )
    assert result is None


def test_validate_and_transform_pose_without_tf(node):
    detection = create_detection("camera_frame", "cup", score=0.9)
    pose = detection.results[0].pose.pose
    assert (
        node._validate_and_transform_pose(pose, "no_frame", "map", "cup", detection)
        is None
    )


def test_extract_pointcloud_features_disabled(node):
    node.config.use_pointcloud_dedup = False
    detection = create_detection("camera_frame", "cup", score=0.9)
    assert node._extract_pointcloud_features(detection, "camera_frame", "map") == (
        None,
        None,
        None,
    )


def test_extract_pointcloud_features_without_depth(node):
    detection = create_detection("camera_frame", "cup", score=0.9)
    assert node._extract_pointcloud_features(detection, "camera_frame", "map") == (
        None,
        None,
        None,
    )


def test_determine_merge_decision(node):
    assert node._determine_merge_decision([], None, None, False) == (False, None)
    assert node._determine_merge_decision([annotation()], None, None, False) == (
        True,
        "test-id",
    )
    sized = annotation(metadata={"pointcloud": {"size_3d": 0.5}})
    features = {"size_3d": 0.5, "centroid": {}, "point_count": 100}
    assert node._determine_merge_decision([sized], features, 0.5, True) == (
        True,
        "test-id",
    )
    features["size_3d"] = 2.0
    assert node._determine_merge_decision([sized], features, 2.0, True) == (
        False,
        None,
    )


def test_store_or_update_annotation(node):
    assert store(node, 0.8)
    assert len(node.memory.query_by_class("test_object")) == 1
    assert store(node, 0.95, x=1.01)  # within the merge radius: an update
    annotations = node.memory.query_by_class("test_object")
    assert len(annotations) == 1
    assert annotations[0].confidence == 0.95


def test_node_config_defaults(node, temp_db_path):
    assert node.config == SemanticMapConfig(
        database_path=temp_db_path,
        depth_topic="/camera/camera/depth/image_rect_raw",
        camera_info_topic="/camera/camera/color/camera_info",
    )
    # Parameters are declared on the node with the YAML defaults.
    assert node.connector.parameter("confidence_threshold", 0.0) == 0.5
    assert node.connector.parameter("map_topic", "") == "/map"
    assert node.memory.location_id == "default_location"
    assert node.memory.map_frame_id == "map"
    assert node.memory.resolution == 0.05
    assert node.detection_subscription is not None
    assert node.map_subscription is not None


def test_node_ros_args_override_yaml(tmp_path, temp_db_path):
    config = tmp_path / "node.yaml"
    config.write_text(
        "rai_semap_node:\n  ros__parameters:\n    storage:\n"
        "      location_id: 'kitchen'\n    detection_filtering:\n"
        "      class_confidence_thresholds: 'person:0.7,bad'\n"
    )
    node = make_node(
        temp_db_path, f"node_config:={config}", "confidence_threshold:=0.8"
    )
    try:
        assert node.config.location_id == "kitchen"
        assert node.config.confidence_threshold == 0.8
        assert node.class_thresholds == {"person": 0.7}
        assert node.memory.location_id == "kitchen"
    finally:
        node.connector.close()


def test_detection_callback_filters_by_confidence(node):
    node.config.confidence_threshold = 0.8
    node.detection_callback(
        create_detection_message(
            "camera_frame", [create_detection("camera_frame", "cup", 0.5)]
        )
    )
    assert node.memory.query_by_class("cup") == []


def test_detection_callback_stores(node):
    node.config.map_frame_id = "camera_frame"
    detection = create_detection("camera_frame", "bottle", score=0.9)
    node.detection_callback(create_detection_message("camera_frame", [detection]))
    assert len(node.memory.query_by_class("bottle")) == 1


def test_detection_callback_empty_and_no_results(node):
    node.detection_callback(create_detection_message("camera_frame", []))
    node.detection_callback(create_detection_message("camera_frame", [Detection2D()]))
    assert node.memory.query_by_class("cup") == []


def test_map_callback_updates_metadata(node):
    msg = OccupancyGrid(header=Header(frame_id="map"), info=MapMetaData(resolution=0.1))
    node.map_callback(msg)
    assert node.memory.map_frame_id == "map"
    assert node.memory.resolution == pytest.approx(0.1)


def test_detections_over_ros_are_transformed_and_stored(temp_db_path):
    """End to end: a detection published on the topic is transformed with TF
    (a static camera -> map transform) and stored in map coordinates."""
    topic = f"/{_name('detections')}"
    node = make_node(temp_db_path, f"detection_topic:={topic}")
    with ROS2Connector() as sender:
        try:
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = "map", "semap_test_cam"
            tf.transform.translation.x = 10.0
            tf.transform.rotation.w = 1.0
            sender.sync.publish(
                "/tf_static", TFMessage(transforms=[tf]), qos=QoS.LATCHED
            )
            msg = create_detection_message(
                "semap_test_cam", [create_detection("semap_test_cam", "chair", 0.9)]
            )

            def stored() -> bool:
                sender.sync.publish(topic, msg, qos=QoS.SENSOR)
                return bool(node.memory.query_by_class("chair"))

            assert wait_until(stored)
            (chair,) = node.memory.query_by_class("chair")
            assert chair.pose.position.x == pytest.approx(11.0)
            assert chair.source_frame == "semap_test_cam"
        finally:
            node.connector.close()
