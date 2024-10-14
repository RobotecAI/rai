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

from rai.node import RaiStateBasedLlmNode, describe_ros_image
from rai.tools.ros.native import (
    GetCameraImage,
    GetMsgFromTopic,
    Ros2PubMessageTool,
    Ros2ShowMsgInterfaceTool,
)

# from rai.tools.ros.native_actions import Ros2RunActionSync
from rai.tools.ros.native_actions import (
    Ros2CancelAction,
    Ros2GetActionResult,
    Ros2GetLastActionFeedback,
    Ros2IsActionComplete,
    Ros2RunActionAsync,
)
from rai.tools.ros.tools import GetCurrentPositionTool
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
        # "/cmd_vel",
        "/led_strip",
    ]

    actions_whitelist = [
        # "/backup",
        # "/compute_path_through_poses",
        # "/compute_path_to_pose",
        # "/dock_robot",
        # "/drive_on_heading",
        # "/follow_gps_waypoints",
        # "/follow_path",
        # "/follow_waypoints",
        # "/navigate_through_poses",
        # "/navigate_to_pose",
        # "/smooth_path",
        "/spin",
        # "/undock_robot",
        # "/wait",
    ]

    SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate.

    Navigation tips:
    - for driving forward/backward, if specified, ros2 actions are better.
    - for driving for some specific time or in specific manner (like shaper or turns) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your currect position, then very accurately estimate the goal pose.
    - navigation actions sometimes fail. Theis output can be read from rosout. You can also tell if they partially worked by checking the robot position and rotation.
    - before using any ros2 interfaces, always make sure to check you are usig the right interface

    - you will be given your camera image description. Based on this information you can reason about positions of objects.
    - be caseful and aboid obstacles

    Here are the corners of your environment:
    (-2.76,9.04, 0.0),
    (4.62, 9.07, 0.0),
    (-2.79, -3.83, 0.0),
    (4.59, -3.81, 0.0)

    """

    node = RaiStateBasedLlmNode(
        observe_topics=observe_topics,
        observe_postprocessors=observe_postprocessors,
        whitelist=topics_whitelist + actions_whitelist,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            Ros2PubMessageTool,
            WaitForSecondsTool,
            GetMsgFromTopic,
            Ros2RunActionAsync,
            Ros2IsActionComplete,
            Ros2CancelAction,
            Ros2GetActionResult,
            Ros2GetLastActionFeedback,
            GetCameraImage,
            Ros2ShowMsgInterfaceTool,
            GetCurrentPositionTool,
        ],
    )

    """
        This is location of places:
    Kitchen:
    (2.06, -0.23, 0.0),
    (2.07, -1.43, 0.0),
    (-2.44, -0.38, 0.0),
    (-2.56, -1.47, 0.0)

    Living room:
    (-2.49, 1.87, 0.0),
    (-2.50, 5.49, 0.0),
    (0.79, 5.73, 0.0),
    (0.92, 1.01, 0.0)
    """
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
