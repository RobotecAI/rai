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


"""Drive the robot through waypoints so the semantic map node collects detections."""

import argparse
import logging
import time

from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2.navigation.nav2_blocking import NavigateToPoseBlockingTool

logger = logging.getLogger(__name__)


def wait_for_action(connector: ROS2Connector, action: str, timeout: float) -> bool:
    """Poll the ROS graph until action has a server."""
    name, deadline = connector.resolve(action, "action"), time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(n == name for n, _ in connector.get_actions_names_and_types()):
            return True
        time.sleep(0.1)
    return False


def navigate_to_waypoints(
    connector: ROS2Connector,
    waypoints: list[tuple[float, ...]],
    action: str = "navigate_to_pose",
    dwell_s: float = 5.0,
    server_timeout_s: float = 60.0,
) -> None:
    """Navigate to (x, y) or (x, y, yaw) waypoints in the map frame, pausing at each
    so detections can be collected."""
    if not wait_for_action(connector, action, server_timeout_s):
        logger.error(
            f"Nav2 action server {action} not available after {server_timeout_s}s."
            " Check: ros2 action list | grep navigate"
        )
        return
    tool = NavigateToPoseBlockingTool(connector=connector, action_name=action)
    for i, (x, y, *rest) in enumerate(waypoints):
        logger.info(f"Waypoint {i + 1}/{len(waypoints)}: ({x:.1f}, {y:.1f})")
        args = {"x": x, "y": y, "z": 0.0, "yaw": rest[0] if rest else 0.0}
        logger.info(tool.invoke(args))
        time.sleep(dwell_s)
    logger.info("Navigation complete")


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
        help="Seconds to keep collecting detections at the final location",
    )
    parser.add_argument("--use-sim-time", action="store_true", help="Use sim time")
    args = parser.parse_args()
    if len(args.waypoints) % 2 != 0:
        parser.error("Waypoints must be pairs of (x, y) coordinates")
    waypoints = list(zip(args.waypoints[::2], args.waypoints[1::2]))

    logging.basicConfig(level=logging.INFO)
    with ROS2Connector(use_sim_time=args.use_sim_time) as connector:
        try:
            navigate_to_waypoints(connector, waypoints)
            logger.info(f"Collecting detections for {args.collect_duration}s...")
            time.sleep(args.collect_duration)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
