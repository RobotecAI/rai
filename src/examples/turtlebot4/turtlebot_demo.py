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


import argparse
from pathlib import Path
from typing import Optional

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.qos
import rclpy.subscription
import rclpy.task

from rai.node import RaiStateBasedLlmNode
from rai.tools.ros.native import (
    GetCameraImage,
    GetMsgFromTopic,
    Ros2PubMessageTool,
    Ros2ShowMsgInterfaceTool,
)
from rai.tools.ros.native_actions import (
    GetTransformTool,
    Ros2CancelAction,
    Ros2GetActionResult,
    Ros2GetLastActionFeedback,
    Ros2IsActionComplete,
    Ros2RunActionAsync,
)
from rai.tools.ros.tools import GetCurrentPositionTool
from rai.tools.time import WaitForSecondsTool

p = argparse.ArgumentParser()
p.add_argument("--allowlist", type=Path, required=False, default=None)


def main(allowlist: Optional[Path] = None):
    rclpy.init()
    ros2_allowlist = allowlist.read_text().splitlines() if allowlist is not None else []

    SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.

    You can use ros2 topics, services and actions to operate.

    Avoid canceling ros2 actions if they don't cause a big danger

    Navigation tips:
    - Always check your transform before and after you perform ros2 actions, so that you can verify if it worked.
    - for driving forward/backward, if specified, ros2 actions are better.
    - for driving for some specific time or in specific manner (like in circle) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your current position, then very accurately estimate the goal pose.
    - to spin right use negative yaw, to spin left use positive yaw
    """

    node = RaiStateBasedLlmNode(
        observe_topics=None,
        observe_postprocessors=None,
        whitelist=ros2_allowlist,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            Ros2PubMessageTool,
            Ros2RunActionAsync,
            Ros2IsActionComplete,
            Ros2CancelAction,
            Ros2GetActionResult,
            Ros2GetLastActionFeedback,
            Ros2ShowMsgInterfaceTool,
            GetCurrentPositionTool,
            WaitForSecondsTool,
            GetMsgFromTopic,
            GetCameraImage,
            GetTransformTool,
        ],
    )
    node.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    args = p.parse_args()
    main(**vars(args))
