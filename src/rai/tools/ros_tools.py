import subprocess

from langchain_core.pydantic_v1 import BaseModel, Field
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image

from rai.communication.ros_communication import SingleMessageGrabber


class get_current_map(BaseModel):
    """Get the current map"""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")

    def run(self):
        """Gets the current map from the specified topic."""
        grabber = SingleMessageGrabber(self["topic"], OccupancyGrid, timeout_sec=10)  # type: ignore
        return grabber.get_data()


class get_current_position_relative_to_the_map(BaseModel):
    """Get the current position relative to the map"""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")

    def run(self):
        """Gets the current position relative to the map from the specified topic."""
        grabber = SingleMessageGrabber(self["topic"], Odometry, timeout_sec=10)  # type: ignore
        return grabber.get_data()


class get_current_image(BaseModel):
    """Get the current image"""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")

    def run(self):
        """Gets the current image from the specified topic."""
        grabber = SingleMessageGrabber(self["topic"], Image, timeout_sec=10)  # type: ignore
        return grabber.get_data()


class set_goal_pose_relative_to_the_map(BaseModel):
    """Set the goal pose"""

    topic: str = Field(..., description="Ros2 pose topic to publish to")
    x: float = Field(..., description="X coordinate of the goal pose")
    y: float = Field(..., description="Y coordinate of the goal pose")
    z: float = Field(..., description="Z coordinate of the goal pose")

    def run(self):
        """Sets the goal pose on the specified topic."""
        cmd = (
            f"ros2 topic pub {self.topic} geometry_msgs/PoseStamped "
            f'\'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, '
            f"pose: {{position: {{x: {self.x}, y: {self.y}, z: {self.z}}}}}}}'"
        )
        subprocess.run(cmd, shell=True)
