import base64
from typing import Type

import cv2
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from nav_msgs.msg import OccupancyGrid, Odometry
from tf_transformations import euler_from_quaternion

from rai.communication.ros_communication import SingleImageGrabber, SingleMessageGrabber

marker_it = 0


class SetWaypointToolInput(BaseModel):
    """Input for the set_waypoint tool."""

    x: float = Field(..., description="X coordinate of the waypoint")
    y: float = Field(..., description="Y coordinate of the waypoint")
    z: float = Field(0.0, description="Z coordinate of the waypoint")
    text: str = Field(
        "", description="Text to display on the waypoint (very short, one or two words)"
    )


class SetWaypointTool(BaseTool):
    """Set a waypoint on the map."""

    name = "SetWaypointTool"
    description: str = (
        "A tool for setting a waypoint on the map. This tool is used for adding information onto the map."
    )

    args_schema: Type[SetWaypointToolInput] = SetWaypointToolInput

    def _run(self, x: float, y: float, z: float = 0.0, text: str = ""):
        global marker_it
        """Sets a waypoint on the map."""
        import rclpy
        from rclpy.node import Node
        from visualization_msgs.msg import Marker

        rclpy.init(args=None)

        class MarkerPublisher(Node):
            def __init__(self):
                super().__init__("marker_publisher")
                self.publisher_ = self.create_publisher(
                    Marker, "/visualization_marker", 10
                )
                self.timer = self.create_timer(0.1, self.publish_marker)
                self.done = False

            def publish_marker(self):
                nonlocal x, y, z, text
                global marker_it
                if self.done:
                    self.timer.cancel()
                    return
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "waypoints"
                marker.id = marker_it
                marker_it += 1
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.pose.orientation.w = 1.0
                marker.scale.z = 0.5
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.text = text
                self.publisher_.publish(marker)
                self.done = True

        marker_publisher = MarkerPublisher()
        rclpy.spin_once(marker_publisher, timeout_sec=1)
        marker_publisher.destroy_node()
        rclpy.shutdown()

        return {"content": "Waypoint set successfully"}


class GetOccupancyGridToolInput(BaseModel):
    """Input for the get_current_map tool."""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")
    odom_topic: str = Field(
        "/odometry/filtered", description="Ros2 odometry topic to subscribe to"
    )


class GetOccupancyGridTool(BaseTool):
    """Get the current map as an image with the robot's position marked on it (red dot)."""

    name: str = "GetOccupancyGridTool"
    description: str = (
        "A tool for getting the current map as an image with the robot's position marked on it."
    )

    args_schema: Type[GetOccupancyGridToolInput] = GetOccupancyGridToolInput

    def _postprocess_msg(self, map_msg: OccupancyGrid, odom_msg: Odometry):
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution
        origin_position = map_msg.info.origin.position
        origin_orientation = map_msg.info.origin.orientation

        data = np.array(map_msg.data).reshape((height, width))

        # Convert the OccupancyGrid values to grayscale image (0-255)
        # the final image shape should be at most (1000x1000), scale to fit
        scale = 1000 / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_NEAREST)
        resolution = resolution / scale
        image = np.zeros_like(data, dtype=np.uint8)
        image[data == -1] = 127  # Unknown space
        image[data == 0] = 255  # Free space
        image[data > 0] = 0  # Occupied space

        # Draw grid lines
        step_size: float = 2.0  # Step size for grid lines in meters, adjust as needed
        step_size_pixels = int(step_size / resolution)
        # print(step_size_pixels, scale)
        for x in range(0, width, step_size_pixels):
            cv2.line(image, (x, 0), (x, height), (200, 200, 200), 1)
            cv2.putText(
                image,
                f"{x * resolution + origin_position.x :.2f}",
                (x, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )
        for y in range(0, height, step_size_pixels):
            cv2.line(image, (0, y), (width, y), (200, 200, 200), 1)
            cv2.putText(
                image,
                f"{y * resolution + origin_position.y:.2f}",
                (5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )

        # Calculate robot's position in the image
        robot_x: float = (
            odom_msg.pose.pose.position.x - origin_position.x
        ) / resolution
        robot_y: float = (
            odom_msg.pose.pose.position.y - origin_position.y
        ) / resolution

        _, _, yaw = euler_from_quaternion(
            [
                origin_orientation.x,
                origin_orientation.y,
                origin_orientation.z,
                origin_orientation.w,
            ]
        )
        # Rotate the robot's position based on the yaw angle
        rotated_x = robot_x * np.cos(yaw) - robot_y * np.sin(yaw)
        rotated_y = robot_x * np.sin(yaw) + robot_y * np.cos(yaw)
        robot_x = int(rotated_x)
        robot_y = int(rotated_y)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Draw the robot's position as a red dot
        if 0 <= robot_x < width and 0 <= robot_y < height:
            cv2.circle(image, (robot_x, robot_y), 5, (0, 0, 255), -1)

        # Encode into PNG base64
        _, buffer = cv2.imencode(".png", image)
        cv2.imwrite("map.png", image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _run(self, topic: str, odom_topic: str = "/odom"):
        """Gets the current map from the specified topic."""
        map_grabber = SingleMessageGrabber(topic, OccupancyGrid, timeout_sec=10)
        odom_grabber = SingleMessageGrabber(odom_topic, Odometry, timeout_sec=10)

        map_msg = map_grabber.get_data()
        odom_msg = odom_grabber.get_data()

        if map_msg is None or odom_msg is None:
            return {"content": "Failed to get the map, wrong topic?"}

        base64_image = self._postprocess_msg(map_msg, odom_msg)
        return {"content": "Map grabbed successfully", "images": [base64_image]}


class GetCurrentPositionToolInput(BaseModel):
    """Input for the get_current_position tool."""

    topic: str = Field(
        "/odometry/filtered", description="Ros2 odometry topic to subscribe to"
    )


class GetCurrentPositionTool(BaseTool):
    """Get the current position of the robot."""

    name = "GetCurrentPositionTool"
    description: str = "A tool for getting the current position of the robot."

    args_schema: Type[GetCurrentPositionToolInput] = GetCurrentPositionToolInput

    def _run(self, topic: str = "/odometry/filtered"):
        """Gets the current position from the specified topic."""
        odom_grabber = SingleMessageGrabber(topic, Odometry, timeout_sec=10)
        odom_msg = odom_grabber.get_data()

        if odom_msg is None:
            return {"content": "Failed to get the position, wrong topic?"}

        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        return {
            "content": str(
                {
                    "x": position.x,
                    "y": position.y,
                    "z": position.z,
                    "yaw": yaw,
                }
            ),
        }


class GetCameraImageToolInput(BaseModel):
    """Input for the get_current_image tool."""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")


class GetCameraImageTool(BaseTool):
    """Get the current image"""

    name = "GetCameraImageTool"
    description: str = "A tool for getting the current image from a ROS2 topic."

    args_schema: Type[GetCameraImageToolInput] = GetCameraImageToolInput

    def _run(self, topic: str):
        """Gets the current image from the specified topic."""
        grabber = SingleImageGrabber(topic, timeout_sec=10)
        base64_image = grabber.get_data()
        return {"content": "Image grabbed successfully", "images": [base64_image]}
