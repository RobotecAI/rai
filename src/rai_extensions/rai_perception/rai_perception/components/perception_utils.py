# Copyright (C) 2025 Robotech.AI
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

"""Perception utilities for 3D pose computation and point cloud extraction.

This module contains perception layer logic:
- 3D pose computation from 2D bounding boxes using depth images
- Point cloud extraction from bounding box regions
- Detection enhancement (filling empty poses from 2D detections)
"""

from typing import Optional, Tuple

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D


def compute_3d_pose_from_bbox(
    bbox_center_x: float,
    bbox_center_y: float,
    depth_image: Image,
    camera_info: CameraInfo,
    bridge: CvBridge,
    region_size: int = 5,
) -> Optional[Pose]:
    """Compute 3D pose from 2D bounding box center using depth and camera intrinsics.

    This is perception layer logic that converts 2D detections to 3D poses.

    Args:
        bbox_center_x: X coordinate of bounding box center in pixels
        bbox_center_y: Y coordinate of bounding box center in pixels
        depth_image: Depth image message
        camera_info: Camera info message with intrinsics
        bridge: CvBridge instance for image conversion

    Returns:
        Pose in camera frame, or None if computation fails
    """
    try:
        # Convert depth image to numpy array
        depth_array = bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")

        # Get pixel coordinates (round to nearest integer)
        u = int(round(bbox_center_x))
        v = int(round(bbox_center_y))

        # Check bounds
        if u < 0 or u >= depth_array.shape[1] or v < 0 or v >= depth_array.shape[0]:
            return None

        # Get depth value at pixel (in meters, assuming depth is in mm)
        depth_value = float(depth_array[v, u])
        if depth_value <= 0:
            # Try a small region around the center
            y_start = max(0, v - region_size // 2)
            y_end = min(depth_array.shape[0], v + region_size // 2 + 1)
            x_start = max(0, u - region_size // 2)
            x_end = min(depth_array.shape[1], u + region_size // 2 + 1)
            region = depth_array[y_start:y_end, x_start:x_end]
            valid_depths = region[region > 0]
            if len(valid_depths) == 0:
                return None
            depth_value = float(np.median(valid_depths))

        # Convert depth to meters (assuming depth image is in mm, adjust if needed)
        # Common depth encodings: 16UC1 (mm), 32FC1 (m)
        if depth_image.encoding in ["16UC1", "mono16"]:
            depth_value = depth_value / 1000.0  # mm to meters

        # Get camera intrinsics
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        # Project pixel to 3D
        z = depth_value
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Create pose
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # No rotation

        return pose

    except Exception:
        return None


def extract_pointcloud_from_bbox(
    detection,
    depth_image: Image,
    camera_info: CameraInfo,
    bridge: CvBridge,
) -> Optional[Tuple[Point, float, int]]:
    """Extract point cloud from bounding box region and compute features.

    This is perception layer logic that extracts 3D point cloud data from images.

    Args:
        detection: Detection2D message with bounding box
        depth_image: Depth image message
        camera_info: Camera info message with intrinsics
        bridge: CvBridge instance for image conversion

    Returns:
        Tuple of (centroid_3d, pointcloud_size, point_count) or None if extraction fails.
        centroid_3d: 3D centroid of point cloud in camera frame
        pointcloud_size: Approximate 3D size (diagonal of bounding box in meters)
        point_count: Number of valid 3D points
    """
    try:
        # Convert depth image to numpy array
        depth_array = bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")

        # Get bounding box bounds
        bbox_center_x = detection.bbox.center.position.x
        bbox_center_y = detection.bbox.center.position.y
        bbox_size_x = detection.bbox.size_x
        bbox_size_y = detection.bbox.size_y

        # Convert to pixel coordinates
        x_min = int(max(0, bbox_center_x - bbox_size_x / 2))
        x_max = int(min(depth_array.shape[1], bbox_center_x + bbox_size_x / 2))
        y_min = int(max(0, bbox_center_y - bbox_size_y / 2))
        y_max = int(min(depth_array.shape[0], bbox_center_y + bbox_size_y / 2))

        if x_max <= x_min or y_max <= y_min:
            return None

        # Get camera intrinsics
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        # Extract depth region
        depth_region = depth_array[y_min:y_max, x_min:x_max]

        # Convert depth to meters if needed
        if depth_image.encoding in ["16UC1", "mono16"]:
            depth_region = depth_region.astype(np.float32) / 1000.0

        # Extract valid points and convert to 3D
        valid_mask = depth_region > 0
        if not np.any(valid_mask):
            return None

        y_coords, x_coords = np.where(valid_mask)
        depths = depth_region[valid_mask]

        # Convert to 3D points in camera frame
        u_coords = x_coords + x_min
        v_coords = y_coords + y_min

        z = depths
        x = (u_coords - cx) * z / fx
        y = (v_coords - cy) * z / fy

        # Compute centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        centroid_z = np.mean(z)

        # Compute 3D bounding box size (diagonal)
        if len(x) > 0:
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            z_range = np.max(z) - np.min(z)
            size_3d = np.sqrt(x_range**2 + y_range**2 + z_range**2)
        else:
            size_3d = 0.0

        point_count = len(x)

        centroid = Point(x=float(centroid_x), y=float(centroid_y), z=float(centroid_z))

        return (centroid, float(size_3d), point_count)

    except Exception:
        return None


def enhance_detection_with_3d_pose(
    detection: Detection2D,
    depth_image: Optional[Image],
    camera_info: Optional[CameraInfo],
    bridge: CvBridge,
    region_size: int = 5,
) -> bool:
    """Enhance detection with 3D pose if pose is empty and depth data is available.

    This is perception layer logic that handles incomplete detections (2D detections
    without 3D poses) by computing 3D poses from depth images.

    Args:
        detection: Detection2D message to enhance
        depth_image: Optional depth image for 3D pose computation
        camera_info: Optional camera info for 3D pose computation
        bridge: CvBridge instance for image conversion

    Returns:
        True if pose was enhanced, False otherwise
    """
    if not detection.results or len(detection.results) == 0:
        return False

    result = detection.results[0]
    pose = result.pose.pose

    # Check if pose is empty (0,0,0)
    if not (
        pose.position.x == 0.0 and pose.position.y == 0.0 and pose.position.z == 0.0
    ):
        return False  # Pose already exists

    # Compute 3D pose from bounding box if depth and camera info are available
    if depth_image is None or camera_info is None:
        return False

    bbox_center_x = detection.bbox.center.position.x
    bbox_center_y = detection.bbox.center.position.y
    computed_pose = compute_3d_pose_from_bbox(
        bbox_center_x, bbox_center_y, depth_image, camera_info, bridge, region_size
    )

    if computed_pose:
        result.pose.pose = computed_pose
        return True

    return False
