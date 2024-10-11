# Copyright (C) 2024 Robotec.AI
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
#


import rclpy
import rclpy.executors

from rai.node import RaiStateBasedNode, describe_ros_image
from rai.tools.ros.native import (
    GetCameraImage,
    GetMsgFromTopic,
    Ros2ShowMsgInterfaceTool,
)
from rai.tools.ros.native_actions import Ros2RunActionSync
from rai.tools.ros.tools import GetOccupancyGridTool
from rai.tools.time import WaitForSecondsTool


def main():
    rclpy.init()

    observe_topics = [
        "/camera/camera/color/image_raw",
    ]

    observe_postprocessors = {"/camera/camera/color/image_raw": describe_ros_image}

    topics_whitelist = [
        "/rosout",
        "/camera/camera/color/image_raw",
        "/map",
        "/scan",
        "/diagnostics",
    ]

    actions_whitelist = [
        "/backup",
        "/compute_path_through_poses",
        "/compute_path_to_pose",
        "/dock_robot",
        "/drive_on_heading",
        "/follow_gps_waypoints",
        "/follow_path",
        "/follow_waypoints",
        "/navigate_through_poses",
        "/navigate_to_pose",
        "/smooth_path",
        "/spin",
        "/undock_robot",
        "/wait",
    ]

    SYSTEM_PROMPT = f"""You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate. """

    node = RaiStateBasedNode(
        observe_topics=observe_topics,
        observe_postprocessors=observe_postprocessors,
        whitelist=topics_whitelist + actions_whitelist,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            WaitForSecondsTool,
            GetMsgFromTopic,
            Ros2RunActionSync,
            GetCameraImage,
            Ros2ShowMsgInterfaceTool,
            GetOccupancyGridTool,
        ],
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
