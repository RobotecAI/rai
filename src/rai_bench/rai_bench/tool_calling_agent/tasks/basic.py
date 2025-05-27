# Copyright (C) 2025 Robotec.AI
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

import logging
from abc import ABC
from typing import Dict, List

from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.mocked_tools import (
    MockGetROS2ImageTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockReceiveROS2MessageTool,
)

loggers_type = logging.Logger
PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT = """You are a ROS 2 expert that want to solve tasks. You have access to various tools that allow you to query the ROS 2 system.
Be proactive and use the tools to answer questions."""

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
    + """
Example of tool calls:
- get_ros2_message_interface, args: {'msg_type': 'geometry_msgs/msg/Twist'}
- publish_ros2_message, args: {'topic': '/cmd_vel', 'message_type': 'geometry_msgs/msg/Twist', 'message': {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}}"""
)

PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT = (
    PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
    + """
- get_ros2_topics_names_and_types, args: {}
- get_ros2_image, args: {'topic': '/camera/image_raw', 'timeout_sec': 10}
- publish_ros2_message, args: {'topic': '/turtle1/teleport_absolute', 'message_type': 'turtlesim/srv/TeleportAbsolute', 'message': {x: 5.0, y: 2.0, theta: 1.57}}"""
)

CAMERA_TOPICS_AND_TYPES = [
    "topic: /color_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
    "topic: /color_image5\ntype: sensor_msgs/msg/Image\n",
    "topic: /depth_camera_info5\ntype: sensor_msgs/msg/CameraInfo\n",
    "topic: /depth_image5\ntype: sensor_msgs/msg/Image\n",
]
CAMERA_TOPICS = [
    "/color_camera_info5",
    "/color_image5",
    "/depth_camera_info5",
    "/depth_image5",
]


TOPICS_AND_TYPES: Dict[str, str] = {
    "/attached_collision_object": "moveit_msgs/msg/AttachedCollisionObject",
    "/clock": "rosgraph_msgs/msg/Clock",
    "/collision_object": "moveit_msgs/msg/CollisionObject",
    "/display_contacts": "visualization_msgs/msg/MarkerArray",
    "/display_planned_path": "moveit_msgs/msg/DisplayTrajectory",
    "/execute_trajectory/_action/feedback": "moveit_msgs/action/ExecuteTrajectory_FeedbackMessage",
    "/execute_trajectory/_action/status": "action_msgs/msg/GoalStatusArray",
    "/joint_states": "sensor_msgs/msg/JointState",
    "/monitored_planning_scene": "moveit_msgs/msg/PlanningScene",
    "/motion_plan_request": "moveit_msgs/msg/MotionPlanRequest",
    "/move_action/_action/feedback": "moveit_msgs/action/MoveGroup_FeedbackMessage",
    "/move_action/_action/status": "action_msgs/msg/GoalStatusArray",
    "/panda_arm_controller/follow_joint_trajectory/_action/feedback": "control_msgs/action/FollowJointTrajectory_FeedbackMessage",
    "/panda_arm_controller/follow_joint_trajectory/_action/status": "action_msgs/msg/GoalStatusArray",
    "/panda_hand_controller/gripper_cmd/_action/feedback": "control_msgs/action/GripperCommand_FeedbackMessage",
    "/panda_hand_controller/gripper_cmd/_action/status": "action_msgs/msg/GoalStatusArray",
    "/parameter_events": "rcl_interfaces/msg/ParameterEvent",
    "/planning_scene": "moveit_msgs/msg/PlanningScene",
    "/planning_scene_world": "moveit_msgs/msg/PlanningSceneWorld",
    "/pointcloud": "sensor_msgs/msg/PointCloud2",
    "/robot_description": "std_msgs/msg/String",
    "/robot_description_semantic": "std_msgs/msg/String",
    "/rosout": "rcl_interfaces/msg/Log",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
    "/trajectory_execution_event": "std_msgs/msg/String",
    # Camera topics
    "/color_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/color_image5": "sensor_msgs/msg/Image",
    "/depth_camera_info5": "sensor_msgs/msg/CameraInfo",
    "/depth_image5": "sensor_msgs/msg/Image",
}

TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {topic_type}\n"
    for topic, topic_type in TOPICS_AND_TYPES.items()
]


class BasicTask(Task, ABC):
    type = "basic"

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            ),
            MockGetROS2ImageTool(available_topics=list(TOPICS_AND_TYPES.keys())),
            MockReceiveROS2MessageTool(available_topics=list(TOPICS_AND_TYPES.keys())),
        ]

    def get_system_prompt(self) -> str:
        if self.n_shots == 0:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_0_SHOT
        elif self.n_shots == 2:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_2_SHOT
        else:
            return PROACTIVE_ROS2_EXPERT_SYSTEM_PROMPT_5_SHOT


class GetROS2TopicsTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return "Get all topics"
        elif self.prompt_detail == "moderate":
            return "Get the names and types of all ROS2 topics"
        else:
            return "Get all ROS2 topics with their names and message types. Use the topics tool to list what's available in the system."


class GetROS2RGBCameraTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return "Get RGB camera image"
        elif self.prompt_detail == "moderate":
            return "Get the RGB image from the camera topic"
        else:
            return "Get the RGB color image from the camera. First check what camera topics are available, then capture the image from the RGB camera topic."


class GetROS2DepthCameraTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return "Get depth camera image"
        elif self.prompt_detail == "moderate":
            return "Get the depth image from the camera sensor"
        else:  # descriptive
            return "Get the depth image from the camera. First check what camera topics are available, then capture the image from the depth camera topic."


class GetPointcloudTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return "Get the pointcloud"
        elif self.prompt_detail == "moderate":
            return "Get the pointcloud data from the topic"
        else:  # descriptive
            return "Get the pointcloud data. First check available topics to find the pointcloud topic, then receive the pointcloud message."


class GetRobotDescriptionTask(BasicTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return "Get robot description"
        elif self.prompt_detail == "moderate":
            return "Get the description of the robot from the topic"
        else:
            return "Get the robot description. First list available topics to find the robot_description topic, then receive the robot description message."


class GetAllROS2CamerasTask(BasicTask):
    complexity = "medium"

    def get_prompt(self) -> str:
        if self.prompt_detail == "brief":
            return "Get all camera images"
        elif self.prompt_detail == "moderate":
            return "Get images from all available cameras in the system, both RGB and depth"
        else:
            return "Get images from all available camera topics in the ROS2 system. This includes both RGB color images and depth images. You should first explore what camera topics are available."


# class GetROS2RGBCameraWithInfoTask(BasicTask):
#     complexity = "easy"

#     def get_prompt(self) -> str:
#         if self.prompt_detail == "brief":
#             return "Get RGB camera image and info"
#         elif self.prompt_detail == "moderate":
#             return "Get the RGB image and camera info from the color camera"
#         else:
#             return "Get the RGB color image and camera calibration info from the camera. First check what camera topics are available, then get both the image and camera info from the color camera."


# class GetROS2DepthWithInfoCameraTask(BasicTask):
#     complexity = "easy"

#     def get_prompt(self) -> str:
#         if self.prompt_detail == "brief":
#             return "Get depth camera image and info"
#         elif self.prompt_detail == "moderate":
#             return "Get the depth image and camera info from the depth sensor"
#         else:
#             return "Get the depth image and camera calibration info from the camera. First check what camera topics are available, then get both the image and camera info from the depth camera."
