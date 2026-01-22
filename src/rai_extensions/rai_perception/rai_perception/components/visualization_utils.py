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

"""Visualization utilities for gripping points and perception data.

This module provides utilities for visualizing gripping points on camera images
and saving annotated images for debugging and analysis.
"""

from typing import Optional

import cv2
import numpy as np
from cv_bridge import CvBridge
from rai.communication.ros2.connectors import ROS2Connector
from sensor_msgs.msg import CameraInfo, Image

from rai_perception.components.perception_utils import get_camera_intrinsics


def _quaternion_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    """Convert quaternion to rotation matrix (optimized implementation).

    Args:
        qx, qy, qz, qw: Quaternion components

    Returns:
        3x3 rotation matrix as numpy array
    """
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _project_3d_to_2d(
    point_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Optional[tuple[int, int]]:
    """Project a 3D point to 2D image coordinates.

    Args:
        point_3d: 3D point [x, y, z]
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates

    Returns:
        (u, v) pixel coordinates or None if point is behind camera or invalid
    """
    x, y, z = float(point_3d[0]), float(point_3d[1]), float(point_3d[2])

    if z <= 0:
        return None

    u = int((x * fx / z) + cx)
    v = int((y * fy / z) + cy)

    return (u, v)


def _draw_single_gripping_point(image: np.ndarray, u: int, v: int, label: str) -> None:
    """Draw a single gripping point on an image.

    Args:
        image: OpenCV image (BGR format)
        u, v: Pixel coordinates
        label: Text label for the point
    """
    if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
        return

    cv2.circle(image, (u, v), 10, (0, 0, 255), -1)
    cv2.circle(image, (u, v), 15, (0, 255, 0), 2)
    cv2.putText(
        image,
        label,
        (u + 20, v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def draw_gripping_points_on_image(
    image_msg: Image,
    gripping_points_3d: list[np.ndarray],
    camera_info: CameraInfo,
    bridge: Optional[CvBridge] = None,
) -> np.ndarray:
    """Draw gripping points on a camera image by projecting 3D points to 2D.

    Args:
        image_msg: ROS Image message from camera
        gripping_points_3d: List of 3D gripping points in camera frame (Nx3 arrays)
        camera_info: CameraInfo message with intrinsics
        bridge: Optional CvBridge instance (creates new one if not provided)

    Returns:
        Annotated OpenCV image (BGR format) with gripping points drawn
    """
    if bridge is None:
        bridge = CvBridge()

    cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    fx, fy, cx, cy = get_camera_intrinsics(camera_info)

    for i, point in enumerate(gripping_points_3d):
        pixel_coords = _project_3d_to_2d(point, fx, fy, cx, cy)
        if pixel_coords is not None:
            u, v = pixel_coords
            _draw_single_gripping_point(cv_image, u, v, f"GP{i + 1}")

    return cv_image


def transform_points_between_frames(
    connector: ROS2Connector,
    points: list[np.ndarray],
    source_frame: str,
    target_frame: str,
) -> list[np.ndarray]:
    """Transform 3D points from source frame to target frame using TF.

    Args:
        connector: ROS2Connector instance for TF access
        points: List of 3D points (Nx3 arrays) in source frame
        source_frame: Source frame name
        target_frame: Target frame name

    Returns:
        List of transformed 3D points (Nx3 arrays) in target frame.
        Returns original points if transform fails.
    """
    try:
        transform = connector.get_transform(target_frame, source_frame)
        t = transform.transform.translation
        r = transform.transform.rotation

        rotation_matrix = _quaternion_to_rotation_matrix(
            float(r.x), float(r.y), float(r.z), float(r.w)
        )
        translation = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float64)

        transformed_points = []
        for point in points:
            point_3d = point.astype(np.float64)
            transformed = (rotation_matrix @ point_3d + translation).astype(np.float32)
            transformed_points.append(transformed)

        return transformed_points
    except Exception as e:
        connector.node.get_logger().error(
            f"Failed to transform points from {source_frame} to {target_frame}: {e}"
        )
        return points


def save_gripping_points_annotated_image(
    connector: ROS2Connector,
    gripping_points_target_frame: list[np.ndarray],
    camera_topic: str,
    camera_info_topic: str,
    source_frame: str,
    target_frame: str,
    filename: str = "gripping_points_annotated.jpg",
    bridge: Optional[CvBridge] = None,
) -> None:
    """Save an annotated image with gripping points projected onto camera view.

    Transforms gripping points from target frame to camera frame, then projects
    them onto the current camera image and saves the result.

    Args:
        connector: ROS2Connector instance for message and TF access
        gripping_points_target_frame: List of gripping points in target frame (Nx3 arrays)
        camera_topic: ROS2 topic name for camera images
        camera_info_topic: ROS2 topic name for camera info
        source_frame: Camera frame name
        target_frame: Target frame name (where gripping points are expressed)
        filename: Output filename for annotated image
        bridge: Optional CvBridge instance (creates new one if not provided)

    Raises:
        TypeError: If received messages are not of expected types
    """
    if bridge is None:
        bridge = CvBridge()

    # Transform points from target frame to camera frame
    camera_frame_points = transform_points_between_frames(
        connector, gripping_points_target_frame, target_frame, source_frame
    )

    # Receive camera messages
    image_msg = connector.receive_message(camera_topic).payload
    camera_info_msg = connector.receive_message(camera_info_topic).payload

    if not isinstance(image_msg, Image):
        raise TypeError(f"Expected Image message, got {type(image_msg)}")
    if not isinstance(camera_info_msg, CameraInfo):
        raise TypeError(f"Expected CameraInfo message, got {type(camera_info_msg)}")

    # Draw gripping points and save
    annotated_image = draw_gripping_points_on_image(
        image_msg, camera_frame_points, camera_info_msg, bridge
    )
    cv2.imwrite(filename, annotated_image)
