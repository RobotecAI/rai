import base64

import cv2
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field
from nav_msgs.msg import OccupancyGrid, Odometry
from tf_transformations import euler_from_quaternion

from rai.communication.ros_communication import SingleImageGrabber, SingleMessageGrabber


class get_current_map(BaseModel):
    """Get the current map as an image with the robot's position marked on it."""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")

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
        step_size: float = 0.5  # Step size for grid lines in meters, adjust as needed
        step_size_pixels = int(step_size / resolution)
        # print(step_size_pixels, scale)
        for x in range(0, width, step_size_pixels):
            cv2.line(image, (x, 0), (x, height), (200, 200, 200), 1)
            cv2.putText(
                image,
                f"{x * resolution:.2f}",
                (x, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
        for y in range(0, height, step_size_pixels):
            cv2.line(image, (0, y), (width, y), (200, 200, 200), 1)
            cv2.putText(
                image,
                f"{y * resolution:.2f}",
                (5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
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
            cv2.circle(image, (robot_x, height - robot_y), 5, (0, 0, 255), -1)

        # Encode into PNG base64
        _, buffer = cv2.imencode(".png", image)
        cv2.imwrite("map.png", image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def run(self):
        """Gets the current map from the specified topic."""
        grabber = SingleMessageGrabber(self.topic, OccupancyGrid, timeout_sec=10)
        grabber_pose = SingleMessageGrabber("/odom", Odometry, timeout_sec=10)

        map_msg = grabber.get_data()
        odom_msg = grabber_pose.get_data()

        if map_msg is None or odom_msg is None:
            return {"content": "Failed to get the map, wrong topic?"}

        base64_image = self._postprocess_msg(map_msg, odom_msg)
        return {"content": "Map grabbed successfully", "images": [base64_image]}


class get_current_position_relative_to_the_map(BaseModel):
    """Get the current position relative to the map"""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")

    def run(self):
        """Gets the current position relative to the map from the specified topic."""
        grabber = SingleMessageGrabber(self.topic, Odometry, timeout_sec=10)
        msg = grabber.get_data()
        return {"content": msg}


class get_current_image(BaseModel):
    """Get the current image"""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")

    def run(self):
        """Gets the current image from the specified topic."""
        grabber = SingleImageGrabber(self.topic, timeout_sec=10)
        base64_image = grabber.get_data()
        return {"content": "Image grabbed successfully", "images": [base64_image]}
