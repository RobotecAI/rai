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

import numpy as np
import pytest
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose
from rai_perception.components.perception_utils import (
    compute_3d_pose_from_bbox,
    enhance_detection_with_3d_pose,
    extract_pointcloud_from_bbox,
)
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    ObjectHypothesisWithPose,
    Point2D,
)


@pytest.fixture
def bridge():
    """Create a CvBridge instance."""
    return CvBridge()


@pytest.fixture
def camera_info():
    """Create a basic camera info message."""
    info = CameraInfo()
    info.width = 640
    info.height = 480
    info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    return info


@pytest.fixture
def depth_image(bridge):
    """Create a depth image message."""
    depth_array = np.ones((480, 640), dtype=np.uint16) * 1000
    depth_msg = bridge.cv2_to_imgmsg(depth_array, encoding="16UC1")
    depth_msg.header = Header()
    depth_msg.header.frame_id = "camera_frame"
    return depth_msg


@pytest.fixture
def detection2d():
    """Create a Detection2D message."""
    detection = Detection2D()
    detection.bbox = BoundingBox2D()
    detection.bbox.center.position = Point2D(x=320.0, y=240.0)
    detection.bbox.size_x = 100.0
    detection.bbox.size_y = 80.0
    return detection


def test_compute_3d_pose_from_bbox(bridge, camera_info, depth_image):
    """Test computing 3D pose from bounding box center."""
    bbox_center_x = 320.0
    bbox_center_y = 240.0

    pose = compute_3d_pose_from_bbox(
        bbox_center_x, bbox_center_y, depth_image, camera_info, bridge
    )

    assert pose is not None
    assert pose.position.z > 0
    assert pose.orientation.w == 1.0


def test_compute_3d_pose_from_bbox_out_of_bounds(bridge, camera_info, depth_image):
    """Test computing 3D pose with out-of-bounds coordinates."""
    bbox_center_x = 1000.0
    bbox_center_y = 1000.0

    pose = compute_3d_pose_from_bbox(
        bbox_center_x, bbox_center_y, depth_image, camera_info, bridge
    )

    assert pose is None


def test_extract_pointcloud_from_bbox(bridge, camera_info, depth_image, detection2d):
    """Test extracting point cloud from bounding box."""
    result = extract_pointcloud_from_bbox(detection2d, depth_image, camera_info, bridge)

    assert result is not None
    centroid, size, point_count = result
    assert point_count > 0
    assert centroid.x is not None
    assert centroid.y is not None
    assert centroid.z is not None
    assert size >= 0


def test_extract_pointcloud_from_bbox_empty_depth(bridge, camera_info, detection2d):
    """Test extracting point cloud with empty depth image."""
    depth_array = np.zeros((480, 640), dtype=np.uint16)
    depth_msg = bridge.cv2_to_imgmsg(depth_array, encoding="16UC1")
    depth_msg.header = Header()
    depth_msg.header.frame_id = "camera_frame"

    result = extract_pointcloud_from_bbox(detection2d, depth_msg, camera_info, bridge)

    assert result is None


def test_enhance_detection_with_3d_pose_success(
    bridge, camera_info, depth_image, detection2d
):
    """Test enhancing detection with 3D pose when pose is empty."""
    # Add empty result to detection
    result = ObjectHypothesisWithPose()
    result.pose.pose = Pose()
    result.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
    detection2d.results = [result]

    success = enhance_detection_with_3d_pose(
        detection2d, depth_image, camera_info, bridge
    )

    assert success is True
    assert detection2d.results[0].pose.pose.position.z > 0


def test_enhance_detection_with_3d_pose_already_has_pose(
    bridge, camera_info, depth_image, detection2d
):
    """Test that enhancement returns False when pose already exists."""
    result = ObjectHypothesisWithPose()
    result.pose.pose = Pose()
    result.pose.pose.position = Point(x=1.0, y=2.0, z=3.0)
    detection2d.results = [result]

    success = enhance_detection_with_3d_pose(
        detection2d, depth_image, camera_info, bridge
    )

    assert success is False
    assert detection2d.results[0].pose.pose.position.x == 1.0


def test_enhance_detection_with_3d_pose_no_depth(bridge, camera_info, detection2d):
    """Test that enhancement returns False when depth is missing."""
    result = ObjectHypothesisWithPose()
    result.pose.pose = Pose()
    result.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
    detection2d.results = [result]

    success = enhance_detection_with_3d_pose(detection2d, None, camera_info, bridge)

    assert success is False


def test_enhance_detection_with_3d_pose_no_results(
    bridge, camera_info, depth_image, detection2d
):
    """Test that enhancement returns False when detection has no results."""
    detection2d.results = []

    success = enhance_detection_with_3d_pose(
        detection2d, depth_image, camera_info, bridge
    )

    assert success is False
