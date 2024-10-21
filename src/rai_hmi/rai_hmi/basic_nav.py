import itertools
import time
from typing import List

import builtin_interfaces.msg
import geometry_msgs.msg
import numpy as np
import rclpy
import rclpy.action.client
import rclpy.node
import sensor_msgs
import sensor_msgs.msg
from langchain_core.tools import tool
from nav2_msgs.action import DriveOnHeading
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.action.graph import get_action_names_and_types

from rai.tools.ros.utils import get_transform
from rai.utils.ros import NodeDiscovery


class RosbotBasicNavigator(BasicNavigator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive_on_heading_client = rclpy.action.client.ActionClient(
            self, DriveOnHeading, "drive_on_heading"
        )

    def drive_on_heading(self, target: float, speed: float, time_allowance: int):
        self.debug("Waiting for 'DriveOnHeading' action server")
        while not self.drive_on_heading_client.wait_for_server(timeout_sec=1.0):
            self.info("'DriveOnHeading' action server not available, waiting...")
        goal_msg = DriveOnHeading.Goal()
        goal_msg.target = geometry_msgs.msg.Point(x=float(target))
        goal_msg.speed = speed
        goal_msg.time_allowance = builtin_interfaces.msg.Duration(sec=time_allowance)

        self.info(f"DriveOnHeading {goal_msg.target.x} m at {goal_msg.speed} m/s....")
        send_goal_future = self.drive_on_heading_client.send_goal_async(
            goal_msg, self._feedbackCallback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error("DriveOnHeading request was rejected!")
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True


if not rclpy.ok():
    rclpy.init()
navigator = RosbotBasicNavigator()
navigator._waitForNodeToActivate("/bt_navigator")

tool_node = rclpy.create_node(node_name="test")
rclpy.spin_once(tool_node, timeout_sec=5.0)

allowed_interfaces = [
    "/rosout",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/aligned_depth_to_color/camera_info",
    "/camera/camera/color/image_raw",
    "/camera/camera/color/camera_info",
    "/led_strip",
    "/backup",
    "/drive_on_heading",
    "/navigate_through_poses",
    "/navigate_to_pose",
    "/spin",
]


MAP_FRAME = "map"


def wait_for_finish(navigator: BasicNavigator) -> TaskResult:
    while not navigator.isTaskComplete():
        time.sleep(1.0)

    return navigator.getResult()


@tool
def get_ros2_interfaces_tool() -> dict:
    """Get ros2 interfaces"""
    return get_ros2_interfaces()


def get_ros2_interfaces() -> dict:
    """Get ros2 interfaces"""
    return NodeDiscovery(
        topics_and_types=tool_node.get_topic_names_and_types(),
        services_and_types=tool_node.get_service_names_and_types(),
        actions_and_types=get_action_names_and_types(tool_node),
        allow_list=allowed_interfaces,
    ).dict()


@tool
def wait_n_sec_tool(n_sec: int):
    """Wait for given amount of seconds"""
    time.sleep(n_sec)

    return "done"


@tool
def drive_on_heading(dist_to_travel: float, speed: float, time_allowance: int):
    """
    Invokes the DriveOnHeading ROS 2 action server, which causes the robot to drive on the current heading by a specific displacement. It performs a linear translation by a given distance. The nav2_behaviors module implements the DriveOnHeading action server.
    """
    if not navigator.isTaskComplete():
        return "Task is still running"
    navigator.drive_on_heading(dist_to_travel, speed, time_allowance)

    return str(wait_for_finish(navigator))


@tool
def backup(dist_to_travel: float, speed: float, time_allowance: int):
    """
    Invokes the BackUp ROS 2 action server, which causes the robot to back up by a specific displacement. It performs an linear translation by a given distance.
    """
    navigator.backup(dist_to_travel, speed, time_allowance)

    return str(wait_for_finish(navigator))


@tool
def spin(angle_to_spin: float, time_allowance: int):
    """
    Invokes the Spin ROS 2 action server, which is implemented by the nav2_behaviors module. It performs an in-place rotation by a given angle.
    For spinning left use negative angle, for spinning right use positive angle.
    Angle is in radians.
    """
    navigator.spin(angle_to_spin, time_allowance)

    return str(wait_for_finish(navigator))


@tool
def go_to_pose(x: float, y: float, qx: float, qy: float, qz: float, qw: float):
    """
    Navigate to specific pose in the /map variable
    x, y - position
    qx, qy, qz, qw - orientation
    """
    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = MAP_FRAME
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = qw
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    navigator.goToPose(pose)
    return str(wait_for_finish(navigator))


@tool
def get_location():
    """Returns robot's transform in the map frame"""
    tf: geometry_msgs.msg.TransformStamped = get_transform(
        tool_node, "base_link", MAP_FRAME
    )
    return str(tf)


@tool
def led_strip(r: int, g: int, b: int):
    """Sets led strip to specific color"""
    color = (r, g, b)
    led_colors = np.full((1, 18, 3), color, dtype=np.uint8)
    publisher = tool_node.create_publisher(sensor_msgs.msg.Image, "/led_strip", 10)
    msg = sensor_msgs.msg.Image()
    msg.header.stamp = tool_node.get_clock().now().to_msg()
    msg.height = 1
    msg.width = 18
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 18 * 3
    msg.data = led_colors.flatten().tolist()
    publisher.publish(msg)


@tool
def led_strip_array(colors: List[int]):
    """Sets entire led strip to specific pattern. Colors are in RGB order. Exactly 54 values are needed"""

    colors = list(itertools.islice(itertools.cycle(colors), 54))

    publisher = tool_node.create_publisher(sensor_msgs.msg.Image, "/led_strip", 10)
    msg = sensor_msgs.msg.Image()
    msg.header.stamp = tool_node.get_clock().now().to_msg()
    msg.height = 1
    msg.width = 18
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 18 * 3
    msg.data = colors
    publisher.publish(msg)
