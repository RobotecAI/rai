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

import argparse
import time
from typing import List

import rclpy
from nav2_msgs.action import NavigateToPose
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import Nav2Toolkit
from rai.tools.time import WaitForSecondsTool
from rclpy.action import ActionClient


class NavigationCollector:
    """Navigate robot and collect detections for semantic map validation."""

    def __init__(self, connector: ROS2Connector):
        self.connector = connector
        self.nav_toolkit = Nav2Toolkit(connector=connector)
        self.wait_tool = WaitForSecondsTool()
        self._nav_action_ready = False

    def wait_for_nav_action_server(self, timeout_sec: float = 60.0) -> bool:
        """Wait for Nav2 action server to be available.

        Args:
            timeout_sec: Maximum time to wait for server.

        Returns:
            True if server is available, False otherwise.
        """
        if self._nav_action_ready:
            return True

        node = self.connector.node
        node.get_logger().info("Waiting for Nav2 action server to be available...")
        node.get_logger().info("Checking action server at: navigate_to_pose")

        # Try different possible action names
        action_names = ["navigate_to_pose", "/navigate_to_pose"]

        for action_name in action_names:
            node.get_logger().info(f"Trying action name: {action_name}")
            action_client = ActionClient(node, NavigateToPose, action_name)
            start_time = time.time()

            while time.time() - start_time < timeout_sec:
                if action_client.wait_for_server(timeout_sec=2.0):
                    node.get_logger().info(
                        f"Nav2 action server is ready at: {action_name}"
                    )
                    self._nav_action_ready = True
                    return True
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    node.get_logger().info(
                        f"Still waiting for Nav2 action server... ({elapsed:.1f}s)"
                    )

        node.get_logger().error(
            f"Nav2 action server not available after {timeout_sec} seconds"
        )
        node.get_logger().error("Make sure Nav2 is launched and running. Check:")
        node.get_logger().error(
            "  1. Is the launch file running? (ros2 launch examples/rosbot-xl-semap.launch.py ...)"
        )
        node.get_logger().error("  2. Check: ros2 action list | grep navigate")
        node.get_logger().error("  3. Check Nav2 logs for errors")
        return False

    def navigate_to_waypoints(self, waypoints: List[tuple]) -> None:
        """Navigate to a series of waypoints.

        Args:
            waypoints: List of (x, y) or (x, y, yaw) tuples representing waypoints in map frame.
        """
        node = self.connector.node

        # Wait for Nav2 action server to be ready
        if not self.wait_for_nav_action_server():
            node.get_logger().error("Cannot navigate: Nav2 action server not available")
            return

        node.get_logger().info(f"Starting navigation to {len(waypoints)} waypoints")

        for i, waypoint in enumerate(waypoints):
            if len(waypoint) == 2:
                x, y = waypoint
                yaw = 0.0
            elif len(waypoint) == 3:
                x, y, yaw = waypoint
            else:
                node.get_logger().warn(f"Invalid waypoint format: {waypoint}, skipping")
                continue

            node.get_logger().info(
                f"Navigating to waypoint {i + 1}/{len(waypoints)}: ({x}, {y}, yaw={yaw})"
            )

            # Use Nav2Toolkit to navigate
            nav_tools = self.nav_toolkit.get_tools()
            navigate_tool = None
            for tool in nav_tools:
                if "navigate" in tool.name.lower():
                    navigate_tool = tool
                    break

            if navigate_tool:
                try:
                    result = navigate_tool.invoke(
                        {"x": x, "y": y, "z": 0.0, "yaw": yaw}
                    )
                    node.get_logger().info(f"Navigation result: {result}")
                except Exception as e:
                    node.get_logger().warn(f"Navigation failed: {e}")
            else:
                node.get_logger().warn("Navigate tool not found, skipping waypoint")

            # Wait at waypoint to allow detections to be collected
            node.get_logger().info("Waiting at waypoint for detections...")
            self.wait_tool.invoke({"seconds": 5.0})

        node.get_logger().info("Navigation complete")

    def collect_detections(self, duration_seconds: float = 30.0) -> None:
        """Stay in place and collect detections.

        Args:
            duration_seconds: How long to collect detections.
        """
        node = self.connector.node
        node.get_logger().info(
            f"Collecting detections for {duration_seconds} seconds..."
        )

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            self.wait_tool.invoke({"seconds": 2.0})
            elapsed = time.time() - start_time
            node.get_logger().info(
                f"Collecting... {elapsed:.1f}/{duration_seconds} seconds"
            )

        node.get_logger().info("Detection collection complete")


def main():
    parser = argparse.ArgumentParser(
        description="Navigate robot and collect detections for semantic map validation"
    )
    parser.add_argument(
        "--waypoints",
        nargs="+",
        type=float,
        help="Waypoints as x1 y1 x2 y2 ... (in map frame)",
        default=[2.0, 0.0, 4.0, 0.0, 2.0, 2.0],
    )
    parser.add_argument(
        "--collect-duration",
        type=float,
        default=10.0,
        help="Duration to collect detections at each waypoint (seconds)",
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Use simulation time",
    )

    args = parser.parse_args()

    if len(args.waypoints) % 2 != 0:
        parser.error("Waypoints must be pairs of (x, y) coordinates")

    waypoints = [
        (args.waypoints[i], args.waypoints[i + 1])
        for i in range(0, len(args.waypoints), 2)
    ]

    rclpy.init()

    try:
        connector = ROS2Connector(
            executor_type="multi_threaded",
            use_sim_time=args.use_sim_time,
        )

        collector = NavigationCollector(connector)

        # Navigate to waypoints
        collector.navigate_to_waypoints(waypoints)

        # Final collection period
        collector.collect_detections(duration_seconds=args.collect_duration)

        connector.node.get_logger().info("Navigation and collection complete")

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
