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
from rai_open_set_vision.tools import GetDetectionTool

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
        # "/camera/camera/depth/image_rect_raw",
        "/camera/camera/color/camera_info",
        # "/camera/camera/depth/camera_info",
        "/map",
        "/scan",
        "/diagnostics",
        "/cmd_vel",
        "/led_strip",
    ]

    actions_whitelist = [
        "/backup",
        # "/compute_path_through_poses",
        # "/compute_path_to_pose",
        # "/dock_robot",
        # "/drive_on_heading",
        # "/follow_gps_waypoints",
        # "/follow_path",
        # "/follow_waypoints",
        "/navigate_through_poses",
        "/navigate_to_pose",
        # "/smooth_path",
        "/spin",
        # "/undock_robot",
        # "/wait",
    ]

    SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate.

    <rule> use /cmd_vel topic very carefully. Obstacle detection works only with nav2 stack, so be careful when it is not used. </rule>>
    <rule> be patient with running ros2 actions. usually the take some time to run. Refrain from canceling it not dangerous. </rule>

    Navigation tips:
    - it's good to start finding objects by rotating, then navigating to some diverse location with occasional rotations. Remember to frequency detect objects.
    - for driving forward/backward or to some coordinates, ros2 actions are better.
    - drive on heading for driving straight
    - navigate to pose for driving to some specific location if you are sure about target coordinates
    - for driving for some specific time or in specific manner (like shaper or turns) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your currect position, then very accurately estimate the goal pose.
    - it is good to verify using given information if the robot is not stuck
    - navigation actions sometimes fail. Their output can be read from rosout. You can also tell if they partially worked by checking the robot position and rotation.
    - before using any ros2 interfaces, always make sure to check you are usig the right interface
    - processing camera image takes 5-10s. Take it into account that if the robot is moving, the information can be outdated. Handle it by good planning of your movements.
    - you are encouraged to use wait tool in between checking the status of actions
    - to find some object navigate around and check the surrounding area
    - when the goal is accomplished please make sure to cancel running actions
    - when you reach the navigation goal - double check if you reached it by checking the current position

    - you will be given your camera image description. Based on this information you can reason about positions of objects.
    - be careful and aboid obstacles
    - /led_strip is an 3x18 image with uint8 values. 56 values in total are required!
    - use general knowledge about placement of objects in the house

    Example:
    -- Task: Turning led strip to green: ---
    '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "0"}, height: 1, width: 18, step: 56, encoding: "rgb8", is_bigendian: false, data: [0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255]}'

    -- Task: Navigate to the living room. Tell be if you have seen the tv " --
    - tool: Ros2RunAction', 'args': {'action_name': '/navigate_to_pose', 'action_type': 'nav2_msgs/action/NavigateToPose', 'action_goal_args': {'pose': {'header': {'frame_id': 'map'}, 'pose': {'position': {'x': 2.06, 'y': -0.23, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}}, 'behavior_tree': ''}
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the kitchen... }}
    - tool: Ros2IsActionComplete: -> return false
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the bedroom with a lot of plants...}}
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the living room with a sofa and 2 people...}}
    - tool: Ros2IsActionComplete: -> return false
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the door...}}
    - tool: Ros2IsActionComplete: -> return false
    - tool: Ros2IsActionComplete: -> return true
    - agent response: navigated successfully. No tv was noticed on the way.

    -- Task: Navigate to the living room. Tell be if you have seen the tv  --
    - tool: Ros2RunAction', 'args': {'action_name': '/navigate_to_pose', 'action_type': 'nav2_msgs/action/NavigateToPose', 'action_goal_args': {'pose': {'header': {'frame_id': 'map'}, 'pose': {'position': {'x': 2.06, 'y': -0.23, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}}, 'behavior_tree': ''}
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the kitchen... }}
    - tool: Ros2IsActionComplete: -> return false
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the bedroom with a lot of plants...}}
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the living room with a 2 poeple. There is a tv in the background...}}
    - tool: Ros2IsActionComplete: -> return false
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the living room with a sofa and 2 people...}}
    - tool: Ros2IsActionComplete: -> return false
    - 'state_retriever: Retrieved state: {"/camera/camera/color/image_raw": {"camera_image_summary" : The room shows the door...}}
    - tool: Ros2IsActionComplete: -> return false
    - tool: Ros2IsActionComplete: -> return true
    - agent response: navigated successfully. Tv was noticed in the living room.

    This os your locations database:
    Kitchen:
    position:
    x: -0.06767308712005615
    y: -0.8381754159927368
    z: 0.0
    orientation:
    x: 0.0
    y: 0.0
    z: 0.0024492188233537805
    w: 0.9999970006590796

    # Living room:
    position
    x: -1.4494316577911377
    y: 4.2252984046936035
    z: 0.0
    orientation:
    x: 0.0
    y: 0.0
    z: -0.7074458321966236
    w: 0.7067675675267129

    # Bedroom:
    position:
    x: 1.580871820449829
    y: 6.966611385345459
    z: 0.0
    orientation:
    x: 0.0
    y: 0.0
    z: 0.3865577225905555
    w: 0.922265215166225
    """

    node = RaiStateBasedLlmNode(
        observe_topics=observe_topics,
        observe_postprocessors=observe_postprocessors,
        whitelist=topics_whitelist + actions_whitelist,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            Ros2PubMessageTool,
            Ros2RunActionAsync,
            Ros2IsActionComplete,
            Ros2CancelAction,
            # Ros2RunActionSync,
            Ros2GetActionResult,
            Ros2GetLastActionFeedback,
            Ros2ShowMsgInterfaceTool,
            GetCurrentPositionTool,
            WaitForSecondsTool,
            GetMsgFromTopic,
            GetCameraImage,
            GetDetectionTool,
            # GetGrabbingPointTool,
        ],
    )

    node.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
