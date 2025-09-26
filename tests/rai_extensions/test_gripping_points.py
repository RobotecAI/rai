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

#!/usr/bin/env python3
"""
Manual test for GetGrippingPointTool with various demo scenarios. Each test:
- Finds gripping points of specified object in the target frame.
- Publishes debug data for visualization.
- Saves annotated image of the gripping points.

The demo app and rivz2 need to be started before running the test. The test will fail if the gripping points are not found.

Usage:
pytest tests/tools/ros2/test_gripping_points.py::test_gripping_points_manipulation_demo -m "manual" -s -v --strategy <strategy>
"""

import time
from typing import List

import cv2
import numpy as np
import pytest
import rclpy
import rclpy.parameter
from cv_bridge import CvBridge
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai_open_set_vision import (
    GetGrippingPointTool,
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
    PointCloudFromSegmentationConfig,
)

# import internal tool
from rai_open_set_vision.tools.pcl_detection import _publish_gripping_point_debug_data


def draw_points_on_image(image_msg, points, camera_info):
    """Draw points on the camera image."""
    # Convert ROS image to OpenCV
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

    # Get camera intrinsics
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    # Project 3D points to 2D
    for i, point in enumerate(points):
        x, y, z = point[0], point[1], point[2]

        # Check if point is in front of camera
        if z <= 0:
            continue

        # Project to pixel coordinates
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)

        # Check if point is within image bounds
        if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
            # Draw circle and label
            cv2.circle(cv_image, (u, v), 10, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(cv_image, (u, v), 15, (0, 255, 0), 2)  # Green outline
            cv2.putText(
                cv_image,
                f"GP{i + 1}",
                (u + 20, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    return cv_image


def extract_gripping_points(result: List[np.ndarray]) -> list[np.ndarray]:
    """Extract gripping points from the result - now returns raw points directly."""
    return result


def transform_points_to_target_frame(connector, points, source_frame, target_frame):
    """Transform points from source frame(e.g. camera frame) to target frame(e.g. robot frame)."""
    try:
        # Get transform from target frame to source frame
        transform = connector.get_transform(source_frame, target_frame)

        # Extract translation and rotation
        t = transform.transform.translation
        r = transform.transform.rotation

        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = float(r.w), float(r.x), float(r.y), float(r.z)

        # Quaternion to rotation matrix conversion
        rotation_matrix = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ]
        )

        translation = np.array([float(t.x), float(t.y), float(t.z)])

        # Transform points: R * point + translation (forward transform)
        transformed_points = []
        for point in points:
            # Apply forward transform: R * point + translation
            transformed_point = rotation_matrix @ point + translation
            transformed_points.append(transformed_point)

        return transformed_points
    except Exception as e:
        print(f"Transform error: {e}")
        return points


def save_annotated_image(
    connector,
    gripping_points,
    camera_topic,
    camera_info_topic,
    source_frame,
    target_frame,
    filename: str = "gripping_points_annotated.jpg",
):
    camera_frame_points = transform_points_to_target_frame(
        connector,
        gripping_points,
        source_frame,
        target_frame,
    )

    # Get current camera image and draw points
    image_msg = connector.receive_message(camera_topic).payload
    camera_info_msg = connector.receive_message(camera_info_topic).payload

    # Draw gripping points on image
    annotated_image = draw_points_on_image(
        image_msg, camera_frame_points, camera_info_msg
    )

    cv2.imwrite(filename, annotated_image)


def main(
    test_object: str = "cube",
    strategy: str = "centroid",
    topics: dict = None,
    frames: dict = None,
    estimator_config: dict = None,
    filter_config: dict = None,
):
    # Default configuration for manipulation-demo
    if topics is None:
        topics = {
            "camera": "/color_image5",
            "depth": "/depth_image5",
            "camera_info": "/color_camera_info5",
        }

    if frames is None:
        frames = {"target": "panda_link0", "source": "RGBDCamera5"}

    if estimator_config is None:
        estimator_config = {"strategy": strategy}

    if filter_config is None:
        filter_config = {
            "strategy": "isolation_forest",
            "if_max_samples": "auto",
            "if_contamination": 0.05,
        }

    services = ["/grounded_sam_segment", "/grounding_dino_classify"]

    # Initialize ROS2
    rclpy.init()
    connector = ROS2Connector(executor_type="single_threaded")

    try:
        # Wait for required services and topics
        print("Waiting for ROS2 services and topics...")
        wait_for_ros2_services(connector, services)
        wait_for_ros2_topics(connector, list(topics.values()))
        print("✅ All services and topics available")

        # Set up conversion ratio parameter
        node = connector.node
        node.declare_parameter("conversion_ratio", 1.0)

        start_time = time.time()

        print(
            f"\nTesting GetGrippingPointTool with object '{test_object}', strategy '{strategy}'"
        )

        # Create the tool with algorithm configurations
        gripping_tool = GetGrippingPointTool(
            connector=connector,
            segmentation_config=PointCloudFromSegmentationConfig(),
            estimator_config=GrippingPointEstimatorConfig(**estimator_config),
            filter_config=PointCloudFilterConfig(**filter_config),
        )

        result = gripping_tool._run(test_object)
        print(f"elapsed time: {time.time() - start_time} seconds")

        # result is now a list of numpy arrays directly
        gripping_points = result
        print(f"\nFound {len(gripping_points)} gripping points in target frame:")

        for i, gp in enumerate(gripping_points):
            print(f"  GP{i + 1}: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]")

        assert len(gripping_points) > 0, "No gripping points found"

        if gripping_points:
            # Call the function in pcl.py to publish the gripping point for visualization
            segmented_clouds = gripping_tool.point_cloud_from_segmentation.run(
                test_object
            )
            filtered_clouds = gripping_tool.point_cloud_filter.run(segmented_clouds)

            print(
                "\nPublishing debug data to /debug_gripping_points_pointcloud and /debug_gripping_points_markerarray"
            )
            _publish_gripping_point_debug_data(
                connector, filtered_clouds, gripping_points, frames["target"]
            )
            print("✅ Debug data published")

            annotated_image_path = f"{test_object}_{strategy}_gripping_points.jpg"
            save_annotated_image(
                connector,
                gripping_points,
                topics["camera"],
                topics["camera_info"],
                frames["source"],
                frames["target"],
                annotated_image_path,
            )
            print(f"✅ Saved annotated image as '{annotated_image_path}'")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if hasattr(connector, "executor"):
            connector.executor.shutdown()
        connector.shutdown()


@pytest.mark.manual
def test_gripping_points_manipulation_demo(strategy):
    """Manual test requiring manipulation-demo app to be started."""
    main("cube", strategy)


@pytest.mark.manual
def test_gripping_points_maciej_demo(strategy):
    """Manual test requiring demo app to be started."""
    main(
        test_object="box",
        strategy=strategy,
        topics={
            "camera": "/rgbd_camera/camera_image_color",
            "depth": "/rgbd_camera/camera_image_depth",
            "camera_info": "/rgbd_camera/camera_info",
        },
        frames={
            "target": "egoarm_base_link",
            "source": "egofront_rgbd_camera_depth_optical_frame",
        },
        estimator_config={
            "strategy": strategy or "biggest_plane",
            "ransac_iterations": 400,
            "distance_threshold_m": 0.008,
        },
        filter_config={
            "strategy": "isolation_forest",
            "if_max_samples": "auto",
            "if_contamination": 0.05,
        },
    )
