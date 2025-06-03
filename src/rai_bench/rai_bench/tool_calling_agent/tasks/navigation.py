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

from typing import List

from langchain_core.tools import BaseTool

from rai_bench.tool_calling_agent.interfaces import Task
from rai_bench.tool_calling_agent.mocked_ros2_interfaces import (
    COMMON_SERVICES_AND_TYPES,
    COMMON_TOPICS_AND_TYPES,
    NAVIGATION_ACTION_MODELS,
    NAVIGATION_ACTIONS_AND_TYPES,
    NAVIGATION_INTERFACES,
    NAVIGATION_SERVICES_AND_TYPES,
    NAVIGATION_TOPICS_AND_TYPES,
)
from rai_bench.tool_calling_agent.mocked_tools import (
    MockActionsToolkit,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2ServicesNamesAndTypesTool,
    MockGetROS2TopicsNamesAndTypesTool,
)

ROBOT_NAVIGATION_SYSTEM_PROMPT_0_SHOT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate.

    <rule> As a first step check transforms by getting 1 message from /tf topic </rule>
    <rule> use /cmd_vel topic very carefully. Obstacle detection works only with nav2 stack, so be careful when it is not used. </rule>>
    <rule> be patient with running ros2 actions. usually the take some time to run. </rule>
    <rule> Always check your transform before and after you perform ros2 actions, so that you can verify if it worked. </rule>

    Navigation tips:
    - it's good to start finding objects by rotating, then navigating to some diverse location with occasional rotations. Remember to frequency detect objects.
    - for driving forward/backward or to some coordinates, ros2 actions are better.
    - for driving for some specific time or in specific manner (like shaper or turns) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your current position, then very accurately estimate the goal pose.
    - it is good to verify using given information if the robot is not stuck
    - navigation actions sometimes fail. Their output can be read from rosout. You can also tell if they partially worked by checking the robot position and rotation.
    - before using any ros2 interfaces, always make sure to check you are using the right interface
    - processing camera image takes 5-10s. Take it into account that if the robot is moving, the information can be outdated. Handle it by good planning of your movements.
    - you are encouraged to use wait tool in between checking the status of actions
    - to find some object navigate around and check the surrounding area
    - when the goal is accomplished please make sure to cancel running actions
    - when you reach the navigation goal - double check if you reached it by checking the current position
    - if you detect collision, please stop operation
    - you will be given your camera image description. Based on this information you can reason about positions of objects.
    - be careful and aboid obstacles

    Here are the corners of your environment:
    (-2.76,9.04, 0.0),
    (4.62, 9.07, 0.0),
    (-2.79, -3.83, 0.0),
    (4.59, -3.81, 0.0)

    This is location of places:
    Kitchen:
    (2.06, -0.23, 0.0),
    (2.07, -1.43, 0.0),
    (-2.44, -0.38, 0.0),
    (-2.56, -1.47, 0.0)

    # Living room:
    (-2.49, 1.87, 0.0),
    (-2.50, 5.49, 0.0),
    (0.79, 5.73, 0.0),
    (0.92, 1.01, 0.0)

    Before starting anything, make sure to load available topics, services and actions."""

ROBOT_NAVIGATION_SYSTEM_PROMPT_2_SHOT = (
    ROBOT_NAVIGATION_SYSTEM_PROMPT_0_SHOT
    + """

    Example tool calls:
    - get_ros2_actions_names_and_types, args: {}
    - start_ros2_action, args: {'action': '/navigate_to_pose', 'action_type': 'nav2_msgs/action/NavigateToPose', 'goal': {'pose': {'header': {'frame_id': 'map'}, 'pose': {'position': {'x': 2.0, 'y': 2.0, 'z': 0.0}}}}}"""
)

ROBOT_NAVIGATION_SYSTEM_PROMPT_5_SHOT = (
    ROBOT_NAVIGATION_SYSTEM_PROMPT_2_SHOT
    + """
    - get_ros2_message_interface, args: {'msg_type': 'nav2_msgs/action/Spin'}
    - start_ros2_action, args: {'action': '/spin', 'action_type': 'nav2_msgs/action/Spin', 'goal': {'target_yaw': 3.14}}
    - start_ros2_action, args: {'action': '/drive_on_heading', 'action_type': 'nav2_msgs/action/DriveOnHeading', 'goal': {'target': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'speed': 0.5}}"""
)
TOPICS_AND_TYPES = COMMON_TOPICS_AND_TYPES | NAVIGATION_TOPICS_AND_TYPES
SERVICES_AND_TYPES = COMMON_SERVICES_AND_TYPES | NAVIGATION_SERVICES_AND_TYPES


TOPIC_STRINGS = [
    f"topic: {topic}\ntype: {topic_type}\n"
    for topic, topic_type in COMMON_TOPICS_AND_TYPES.items()
]

ACTION_STRINGS = [
    f"action: {action}\ntype: {act_type}\n"
    for action, act_type in NAVIGATION_ACTIONS_AND_TYPES.items()
]

SERVICE_STRINGS = [
    f"service: {service}\ntype: {srv_type}\n"
    for service, srv_type in SERVICES_AND_TYPES.items()
]


class NavigationTask(Task):
    type = "navigation"

    @property
    def available_tools(self) -> List[BaseTool]:
        tools = MockActionsToolkit(
            mock_actions_names_and_types=ACTION_STRINGS,
            available_actions=list(NAVIGATION_ACTIONS_AND_TYPES.keys()),
            available_action_types=list(NAVIGATION_ACTIONS_AND_TYPES.values()),
            available_action_models=NAVIGATION_ACTION_MODELS,
        ).get_tools()
        tools.append(
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPIC_STRINGS
            )
        )
        tools.append(
            MockGetROS2ServicesNamesAndTypesTool(
                mock_service_names_and_types=SERVICE_STRINGS
            )
        )
        tools.append(
            MockGetROS2MessageInterfaceTool(mock_interfaces=NAVIGATION_INTERFACES)
        )
        return tools

    def get_system_prompt(self) -> str:
        if self.n_shots == 0:
            return ROBOT_NAVIGATION_SYSTEM_PROMPT_0_SHOT
        elif self.n_shots == 2:
            return ROBOT_NAVIGATION_SYSTEM_PROMPT_2_SHOT
        else:
            return ROBOT_NAVIGATION_SYSTEM_PROMPT_5_SHOT


class NavigateToPointTask(NavigationTask):
    complexity = "easy"

    def get_prompt(self) -> str:
        base_prompt = "Navigate to point (2.0, 2.0, 0.0)"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using start_ros2_action and get_ros2_message_interface tools"
        else:
            return f"{base_prompt}. First call get_ros2_actions_names_and_types to list available actions, then call get_ros2_message_interface with 'nav2_msgs/action/NavigateToPose' to get the interface, and finally call start_ros2_action with the navigation goal."


class SpinAroundTask(NavigationTask):
    recursion_limit = 50
    complexity = "medium"

    def get_prompt(self) -> str:
        base_prompt = "Spin around by 3 radians"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using start_ros2_action tool"
        else:
            return f"{base_prompt}. First call get_ros2_actions_names_and_types to find the spin action, then call start_ros2_action with action='/spin', action_type='nav2_msgs/action/Spin', and target_yaw=3."


class MoveToFrontTask(NavigationTask):
    recursion_limit = 50
    complexity = "medium"

    def get_prompt(self) -> str:
        base_prompt = "Move 2 meters to the front"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using start_ros2_action tool"
        else:
            return f"{base_prompt}. First call get_ros2_actions_names_and_types to find available actions, then call start_ros2_action with action='/drive_on_heading', action_type='nav2_msgs/action/DriveOnHeading', and target with x=2.0, y=0.0, z=0.0."


class MoveToBedTask(NavigationTask):
    recursion_limit = 50
    complexity = "hard"

    def get_prompt(self) -> str:
        base_prompt = "Move closer to the bed eaving 1 meter space"
        if self.prompt_detail == "brief":
            return base_prompt
        elif self.prompt_detail == "moderate":
            return f"{base_prompt} using get_distance_to_objects and start_ros2_action tools."
        else:
            return f"{base_prompt}. First call get_distance_to_objects to locate the bed and measure distance, then call get_ros2_actions_names_and_types to find navigation actions, and finally call start_ros2_action to navigate towards the bed while maintaining 1 meter distance."
