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
from typing import Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from rai_open_set_vision.tools.gdino_tools import (
    DistanceMeasurement,
)

from rai_bench.tool_calling_agent.interfaces import Task
from rai_bench.tool_calling_agent.messages.actions import (
    AssistedTeleopGoal,
    BackUpGoal,
    ComputePathThroughPosesGoal,
    ComputePathToPoseGoal,
    DriveOnHeadingGoal,
    FollowPathGoal,
    FollowWaypointsGoal,
    NavigateThroughPosesGoal,
    NavigateToPoseGoal,
    SmoothPathGoal,
    SpinGoal,
    WaitGoal,
)
from rai_bench.tool_calling_agent.mocked_tools import (
    MockActionsToolkit,
    MockGetDistanceToObjectsTool,
    MockGetROS2ActionFeedbackTool,
    MockGetROS2ActionResultTool,
    MockGetROS2ActionsNamesAndTypesTool,
    MockGetROS2MessageInterfaceTool,
    MockGetROS2TopicsNamesAndTypesTool,
    MockStartROS2ActionTool,
)

loggers_type = logging.Logger

ROBOT_NAVIGATION_SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
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

    Before starting anything, make sure to load available topics, services and actions.
    Example tool calls:
    - get_ros2_message_interface, args: {'msg_type': 'turtlesim/srv/TeleportAbsolute'}
    - publish_ros2_message, args: {'topic': '/cmd_vel', 'message_type': 'geometry_msgs/msg/Twist', 'message': {linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}}
    - start_ros2_action, args: {'action_name': '/dock', 'action_type': 'nav2_msgs/action/Dock', 'action_args': {}}
    """

TOPICS_NAMES_AND_TYPES = [
    "topic: /assisted_teleop/_action/feedback\ntype: nav2_msgs/action/AssistedTeleop_FeedbackMessage\n",
    "topic: /assisted_teleop/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /backup/_action/feedback\ntype: nav2_msgs/action/BackUp_FeedbackMessage\n",
    "topic: /backup/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /behavior_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /behavior_tree_log\ntype: nav2_msgs/msg/BehaviorTreeLog\n",
    "topic: /bond\ntype: bond/msg/Status\n",
    "topic: /bt_navigator/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /camera/camera/color/camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
    "topic: /camera/camera/color/image_raw\ntype: sensor_msgs/msg/Image\n",
    "topic: /camera/camera/depth/camera_info\ntype: sensor_msgs/msg/CameraInfo\n",
    "topic: /camera/camera/depth/image_rect_raw\ntype: sensor_msgs/msg/Image\n",
    "topic: /clock\ntype: rosgraph_msgs/msg/Clock\n",
    "topic: /cmd_vel_nav\ntype: geometry_msgs/msg/Twist\n",
    "topic: /cmd_vel_teleop\ntype: geometry_msgs/msg/Twist\n",
    "topic: /compute_path_through_poses/_action/feedback\ntype: nav2_msgs/action/ComputePathThroughPoses_FeedbackMessage\n",
    "topic: /compute_path_through_poses/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /compute_path_to_pose/_action/feedback\ntype: nav2_msgs/action/ComputePathToPose_FeedbackMessage\n",
    "topic: /compute_path_to_pose/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /controller_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /diagnostics\ntype: diagnostic_msgs/msg/DiagnosticArray\n",
    "topic: /drive_on_heading/_action/feedback\ntype: nav2_msgs/action/DriveOnHeading_FeedbackMessage\n",
    "topic: /drive_on_heading/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /follow_path/_action/feedback\ntype: nav2_msgs/action/FollowPath_FeedbackMessage\n",
    "topic: /follow_path/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /follow_waypoints/_action/feedback\ntype: nav2_msgs/action/FollowWaypoints_FeedbackMessage\n",
    "topic: /follow_waypoints/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /global_costmap/costmap\ntype: nav_msgs/msg/OccupancyGrid\n",
    "topic: /global_costmap/costmap_raw\ntype: nav2_msgs/msg/Costmap\n",
    "topic: /global_costmap/costmap_updates\ntype: map_msgs/msg/OccupancyGridUpdate\n",
    "topic: /global_costmap/footprint\ntype: geometry_msgs/msg/Polygon\n",
    "topic: /global_costmap/global_costmap/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /global_costmap/published_footprint\ntype: geometry_msgs/msg/PolygonStamped\n",
    "topic: /global_costmap/scan\ntype: sensor_msgs/msg/LaserScan\n",
    "topic: /goal_pose\ntype: geometry_msgs/msg/PoseStamped\n",
    "topic: /led_strip\ntype: sensor_msgs/msg/Image\n",
    "topic: /local_costmap/costmap\ntype: nav_msgs/msg/OccupancyGrid\n",
    "topic: /local_costmap/costmap_raw\ntype: nav2_msgs/msg/Costmap\n",
    "topic: /local_costmap/costmap_updates\ntype: map_msgs/msg/OccupancyGridUpdate\n",
    "topic: /local_costmap/footprint\ntype: geometry_msgs/msg/Polygon\n",
    "topic: /local_costmap/local_costmap/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /local_costmap/published_footprint\ntype: geometry_msgs/msg/PolygonStamped\n",
    "topic: /local_costmap/scan\ntype: sensor_msgs/msg/LaserScan\n",
    "topic: /map\ntype: nav_msgs/msg/OccupancyGrid\n",
    "topic: /map_metadata\ntype: nav_msgs/msg/MapMetaData\n",
    "topic: /map_saver/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /navigate_through_poses/_action/feedback\ntype: nav2_msgs/action/NavigateThroughPoses_FeedbackMessage\n",
    "topic: /navigate_through_poses/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /navigate_to_pose/_action/feedback\ntype: nav2_msgs/action/NavigateToPose_FeedbackMessage\n",
    "topic: /navigate_to_pose/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /odom\ntype: nav_msgs/msg/Odometry\n",
    "topic: /odometry/filtered\ntype: nav_msgs/msg/Odometry\n",
    "topic: /parameter_events\ntype: rcl_interfaces/msg/ParameterEvent\n",
    "topic: /plan\ntype: nav_msgs/msg/Path\n",
    "topic: /plan_smoothed\ntype: nav_msgs/msg/Path\n",
    "topic: /planner_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /pose\ntype: geometry_msgs/msg/PoseWithCovarianceStamped\n",
    "topic: /preempt_teleop\ntype: std_msgs/msg/Empty\n",
    "topic: /rosout\ntype: rcl_interfaces/msg/Log\n",
    "topic: /scan\ntype: sensor_msgs/msg/LaserScan\n",
    "topic: /slam_toolbox/feedback\ntype: visualization_msgs/msg/InteractiveMarkerFeedback\n",
    "topic: /slam_toolbox/graph_visualization\ntype: visualization_msgs/msg/MarkerArray\n",
    "topic: /slam_toolbox/scan_visualization\ntype: sensor_msgs/msg/LaserScan\n",
    "topic: /slam_toolbox/update\ntype: visualization_msgs/msg/InteractiveMarkerUpdate\n",
    "topic: /smooth_path/_action/feedback\ntype: nav2_msgs/action/SmoothPath_FeedbackMessage\n",
    "topic: /smooth_path/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /smoother_server/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /speed_limit\ntype: nav2_msgs/msg/SpeedLimit\n",
    "topic: /spin/_action/feedback\ntype: nav2_msgs/action/Spin_FeedbackMessage\n",
    "topic: /spin/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /tf_static\ntype: tf2_msgs/msg/TFMessage\n",
    "topic: /trajectories\ntype: visualization_msgs/msg/MarkerArray\n",
    "topic: /transformed_global_plan\ntype: nav_msgs/msg/Path\n",
    "topic: /unsmoothed_plan\ntype: nav_msgs/msg/Path\n",
    "topic: /velocity_smoother/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
    "topic: /wait/_action/feedback\ntype: nav2_msgs/action/Wait_FeedbackMessage\n",
    "topic: /wait/_action/status\ntype: action_msgs/msg/GoalStatusArray\n",
    "topic: /waypoint_follower/transition_event\ntype: lifecycle_msgs/msg/TransitionEvent\n",
]
ACTIONS_AND_TYPES: Dict[str, str] = {
    "/assisted_teleop": "nav2_msgs/action/AssistedTeleop",
    "/backup": "nav2_msgs/action/BackUp",
    "/compute_path_through_poses": "nav2_msgs/action/ComputePathThroughPoses",
    "/compute_path_to_pose": "nav2_msgs/action/ComputePathToPose",
    "/drive_on_heading": "nav2_msgs/action/DriveOnHeading",
    "/follow_path": "nav2_msgs/action/FollowPath",
    "/follow_waypoints": "nav2_msgs/action/FollowWaypoints",
    "/navigate_through_poses": "nav2_msgs/action/NavigateThroughPoses",
    "/navigate_to_pose": "nav2_msgs/action/NavigateToPose",
    "/smooth_path": "nav2_msgs/action/SmoothPath",
    "/spin": "nav2_msgs/action/Spin",
    "/wait": "nav2_msgs/action/Wait",
}

SERVICES_AND_TYPES: Dict[str, str] = {
    "/assisted_teleop/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/assisted_teleop/_action/get_result": "nav2_msgs/action/AssistedTeleop_GetResult",
    "/assisted_teleop/_action/send_goal": "nav2_msgs/action/AssistedTeleop_SendGoal",
    "/backup/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/backup/_action/get_result": "nav2_msgs/action/BackUp_GetResult",
    "/backup/_action/send_goal": "nav2_msgs/action/BackUp_SendGoal",
    "/behavior_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/behavior_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/behavior_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/behavior_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/behavior_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/behavior_server/get_state": "lifecycle_msgs/srv/GetState",
    "/behavior_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/behavior_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/behavior_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/behavior_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator/change_state": "lifecycle_msgs/srv/ChangeState",
    "/bt_navigator/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/bt_navigator/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator/get_state": "lifecycle_msgs/srv/GetState",
    "/bt_navigator/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/bt_navigator/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_through_poses_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_through_poses_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_through_poses_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/bt_navigator_navigate_to_pose_rclcpp_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/bt_navigator_navigate_to_pose_rclcpp_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/bt_navigator_navigate_to_pose_rclcpp_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/compute_path_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_through_poses/_action/get_result": "nav2_msgs/action/ComputePathThroughPoses_GetResult",
    "/compute_path_through_poses/_action/send_goal": "nav2_msgs/action/ComputePathThroughPoses_SendGoal",
    "/compute_path_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/compute_path_to_pose/_action/get_result": "nav2_msgs/action/ComputePathToPose_GetResult",
    "/compute_path_to_pose/_action/send_goal": "nav2_msgs/action/ComputePathToPose_SendGoal",
    "/controller_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/controller_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/controller_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/controller_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/controller_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/controller_server/get_state": "lifecycle_msgs/srv/GetState",
    "/controller_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/controller_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/controller_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/controller_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/drive_on_heading/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/drive_on_heading/_action/get_result": "nav2_msgs/action/DriveOnHeading_GetResult",
    "/drive_on_heading/_action/send_goal": "nav2_msgs/action/DriveOnHeading_SendGoal",
    "/follow_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_path/_action/get_result": "nav2_msgs/action/FollowPath_GetResult",
    "/follow_path/_action/send_goal": "nav2_msgs/action/FollowPath_SendGoal",
    "/follow_waypoints/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/follow_waypoints/_action/get_result": "nav2_msgs/action/FollowWaypoints_GetResult",
    "/follow_waypoints/_action/send_goal": "nav2_msgs/action/FollowWaypoints_SendGoal",
    "/global_costmap/clear_around_global_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/global_costmap/clear_entirely_global_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/global_costmap/clear_except_global_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/global_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/global_costmap/global_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/global_costmap/global_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/global_costmap/global_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/global_costmap/global_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/global_costmap/global_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/global_costmap/global_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/global_costmap/global_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/global_costmap/global_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/global_costmap/global_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/global_costmap/global_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounded_sam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounded_sam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounded_sam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounded_sam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounded_sam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounded_sam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounded_sam_segment": "rai_interfaces/srv/RAIGroundedSam",
    "/grounding_dino/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/grounding_dino/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/grounding_dino/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/grounding_dino/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/grounding_dino/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/grounding_dino/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/grounding_dino_classify": "rai_interfaces/srv/RAIGroundingDino",
    "/is_path_valid": "nav2_msgs/srv/IsPathValid",
    "/launch_ros_138640/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/launch_ros_138640/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/launch_ros_138640/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/launch_ros_138640/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/launch_ros_138640/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/launch_ros_138640/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_navigation/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_navigation/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_navigation/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_navigation/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_navigation/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_navigation/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_navigation/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_navigation/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/lifecycle_manager_slam/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/lifecycle_manager_slam/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/lifecycle_manager_slam/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/lifecycle_manager_slam/is_active": "std_srvs/srv/Trigger",
    "/lifecycle_manager_slam/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/lifecycle_manager_slam/manage_nodes": "nav2_msgs/srv/ManageLifecycleNodes",
    "/lifecycle_manager_slam/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/lifecycle_manager_slam/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/local_costmap/clear_around_local_costmap": "nav2_msgs/srv/ClearCostmapAroundRobot",
    "/local_costmap/clear_entirely_local_costmap": "nav2_msgs/srv/ClearEntireCostmap",
    "/local_costmap/clear_except_local_costmap": "nav2_msgs/srv/ClearCostmapExceptRegion",
    "/local_costmap/get_costmap": "nav2_msgs/srv/GetCostmap",
    "/local_costmap/local_costmap/change_state": "lifecycle_msgs/srv/ChangeState",
    "/local_costmap/local_costmap/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/local_costmap/local_costmap/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/local_costmap/local_costmap/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/local_costmap/local_costmap/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/local_costmap/local_costmap/get_state": "lifecycle_msgs/srv/GetState",
    "/local_costmap/local_costmap/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/local_costmap/local_costmap/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/local_costmap/local_costmap/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/local_costmap/local_costmap/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/map_saver/change_state": "lifecycle_msgs/srv/ChangeState",
    "/map_saver/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/map_saver/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/map_saver/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/map_saver/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/map_saver/get_state": "lifecycle_msgs/srv/GetState",
    "/map_saver/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/map_saver/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/map_saver/save_map": "nav2_msgs/srv/SaveMap",
    "/map_saver/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/map_saver/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/nav2_container/_container/list_nodes": "composition_interfaces/srv/ListNodes",
    "/nav2_container/_container/load_node": "composition_interfaces/srv/LoadNode",
    "/nav2_container/_container/unload_node": "composition_interfaces/srv/UnloadNode",
    "/navigate_through_poses/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_through_poses/_action/get_result": "nav2_msgs/action/NavigateThroughPoses_GetResult",
    "/navigate_through_poses/_action/send_goal": "nav2_msgs/action/NavigateThroughPoses_SendGoal",
    "/navigate_to_pose/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/navigate_to_pose/_action/get_result": "nav2_msgs/action/NavigateToPose_GetResult",
    "/navigate_to_pose/_action/send_goal": "nav2_msgs/action/NavigateToPose_SendGoal",
    "/o3de_ros2_node/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/o3de_ros2_node/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/o3de_ros2_node/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/o3de_ros2_node/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/o3de_ros2_node/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/o3de_ros2_node/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/planner_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/planner_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/planner_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/planner_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/planner_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/planner_server/get_state": "lifecycle_msgs/srv/GetState",
    "/planner_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/planner_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/planner_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/planner_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/rai_ros2_ari_connector_b6ed00ab6356/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/rai_ros2_ari_connector_b6ed00ab6356/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/rai_ros2_ari_connector_b6ed00ab6356/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/clear_changes": "slam_toolbox/srv/Clear",
    "/slam_toolbox/clear_queue": "slam_toolbox/srv/ClearQueue",
    "/slam_toolbox/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/slam_toolbox/deserialize_map": "slam_toolbox/srv/DeserializePoseGraph",
    "/slam_toolbox/dynamic_map": "nav_msgs/srv/GetMap",
    "/slam_toolbox/get_interactive_markers": "visualization_msgs/srv/GetInteractiveMarkers",
    "/slam_toolbox/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/slam_toolbox/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/slam_toolbox/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/slam_toolbox/manual_loop_closure": "slam_toolbox/srv/LoopClosure",
    "/slam_toolbox/pause_new_measurements": "slam_toolbox/srv/Pause",
    "/slam_toolbox/save_map": "slam_toolbox/srv/SaveMap",
    "/slam_toolbox/serialize_map": "slam_toolbox/srv/SerializePoseGraph",
    "/slam_toolbox/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/slam_toolbox/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/slam_toolbox/toggle_interactive_mode": "slam_toolbox/srv/ToggleInteractive",
    "/smooth_path/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/smooth_path/_action/get_result": "nav2_msgs/action/SmoothPath_GetResult",
    "/smooth_path/_action/send_goal": "nav2_msgs/action/SmoothPath_SendGoal",
    "/smoother_server/change_state": "lifecycle_msgs/srv/ChangeState",
    "/smoother_server/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/smoother_server/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/smoother_server/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/smoother_server/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/smoother_server/get_state": "lifecycle_msgs/srv/GetState",
    "/smoother_server/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/smoother_server/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/smoother_server/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/smoother_server/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/spin/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/spin/_action/get_result": "nav2_msgs/action/Spin_GetResult",
    "/spin/_action/send_goal": "nav2_msgs/action/Spin_SendGoal",
    "/tf2_frames": "tf2_msgs/srv/FrameGraph",
    "/velocity_smoother/change_state": "lifecycle_msgs/srv/ChangeState",
    "/velocity_smoother/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/velocity_smoother/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/velocity_smoother/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/velocity_smoother/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/velocity_smoother/get_state": "lifecycle_msgs/srv/GetState",
    "/velocity_smoother/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/velocity_smoother/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/velocity_smoother/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/velocity_smoother/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
    "/wait/_action/cancel_goal": "action_msgs/srv/CancelGoal",
    "/wait/_action/get_result": "nav2_msgs/action/Wait_GetResult",
    "/wait/_action/send_goal": "nav2_msgs/action/Wait_SendGoal",
    "/waypoint_follower/change_state": "lifecycle_msgs/srv/ChangeState",
    "/waypoint_follower/describe_parameters": "rcl_interfaces/srv/DescribeParameters",
    "/waypoint_follower/get_available_states": "lifecycle_msgs/srv/GetAvailableStates",
    "/waypoint_follower/get_available_transitions": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/get_parameter_types": "rcl_interfaces/srv/GetParameterTypes",
    "/waypoint_follower/get_parameters": "rcl_interfaces/srv/GetParameters",
    "/waypoint_follower/get_state": "lifecycle_msgs/srv/GetState",
    "/waypoint_follower/get_transition_graph": "lifecycle_msgs/srv/GetAvailableTransitions",
    "/waypoint_follower/list_parameters": "rcl_interfaces/srv/ListParameters",
    "/waypoint_follower/set_parameters": "rcl_interfaces/srv/SetParameters",
    "/waypoint_follower/set_parameters_atomically": "rcl_interfaces/srv/SetParametersAtomically",
}
INTERFACES: Dict[str, str] = {
    "nav2_msgs/action/NavigateToPose": """#goal definition
geometry_msgs/PoseStamped pose
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
string behavior_tree
---
#result definition
std_msgs/Empty result
---
#feedback definition
geometry_msgs/PoseStamped current_pose
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
builtin_interfaces/Duration navigation_time
	int32 sec
	uint32 nanosec
builtin_interfaces/Duration estimated_time_remaining
	int32 sec
	uint32 nanosec
int16 number_of_recoveries
float32 distance_remaining
""",
    "nav2_msgs/action/AssistedTeleop": """#goal definition
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback
builtin_interfaces/Duration current_teleop_duration
	int32 sec
	uint32 nanosec""",
    "nav2_msgs/action/BackUp": """#goal definition
geometry_msgs/Point target
	float64 x
	float64 y
	float64 z
float32 speed
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
float32 distance_traveled""",
    "nav2_msgs/action/ComputePathThroughPoses": """#goal definition
geometry_msgs/PoseStamped[] goals
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
geometry_msgs/PoseStamped start
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
string planner_id
bool use_start # If false, use current robot pose as path start, if true, use start above instead
---
#result definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		Pose pose
			Point position
				float64 x
				float64 y
				float64 z
			Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
builtin_interfaces/Duration planning_time
	int32 sec
	uint32 nanosec
---
#feedback definition""",
    "nav2_msgs/action/ComputePathToPose": """#goal definition
geometry_msgs/PoseStamped goal
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
geometry_msgs/PoseStamped start
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
string planner_id
bool use_start # If false, use current robot pose as path start, if true, use start above instead
---
#result definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		Pose pose
			Point position
				float64 x
				float64 y
				float64 z
			Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
builtin_interfaces/Duration planning_time
	int32 sec
	uint32 nanosec
---
#feedback definition""",
    "nav2_msgs/action/DriveOnHeading": """#goal definition
geometry_msgs/Point target
	float64 x
	float64 y
	float64 z
float32 speed
builtin_interfaces/Duration time_allowance
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
float32 distance_traveled""",
    "nav2_msgs/action/FollowPath": """#goal definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		Pose pose
			Point position
				float64 x
				float64 y
				float64 z
			Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
string controller_id
string goal_checker_id
---
#result definition
std_msgs/Empty result
---
#feedback definition
float32 distance_to_goal
float32 speed""",
    "nav2_msgs/action/FollowWaypoints": """#goal definition
geometry_msgs/PoseStamped[] poses
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
---
#result definition
int32[] missed_waypoints
---
#feedback definition
uint32 current_waypoint""",
    "nav2_msgs/action/NavigateThroughPoses": """#goal definition
geometry_msgs/PoseStamped[] poses
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
string behavior_tree
---
#result definition
std_msgs/Empty result
---
#feedback definition
geometry_msgs/PoseStamped current_pose
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	Pose pose
		Point position
			float64 x
			float64 y
			float64 z
		Quaternion orientation
			float64 x 0
			float64 y 0
			float64 z 0
			float64 w 1
builtin_interfaces/Duration navigation_time
	int32 sec
	uint32 nanosec
builtin_interfaces/Duration estimated_time_remaining
	int32 sec
	uint32 nanosec
int16 number_of_recoveries
float32 distance_remaining
int16 number_of_poses_remaining
""",
    "nav2_msgs/action/SmoothPath": """#goal definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		Pose pose
			Point position
				float64 x
				float64 y
				float64 z
			Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
string smoother_id
builtin_interfaces/Duration max_smoothing_duration
	int32 sec
	uint32 nanosec
bool check_for_collisions
---
#result definition
nav_msgs/Path path
	std_msgs/Header header
		builtin_interfaces/Time stamp
			int32 sec
			uint32 nanosec
		string frame_id
	geometry_msgs/PoseStamped[] poses
		std_msgs/Header header
			builtin_interfaces/Time stamp
				int32 sec
				uint32 nanosec
			string frame_id
		Pose pose
			Point position
				float64 x
				float64 y
				float64 z
			Quaternion orientation
				float64 x 0
				float64 y 0
				float64 z 0
				float64 w 1
builtin_interfaces/Duration smoothing_duration
	int32 sec
	uint32 nanosec
bool was_completed
---
#feedback definition
""",
    "nav2_msgs/action/Wait": """#goal definition
builtin_interfaces/Duration time
	int32 sec
	uint32 nanosec
---
#result definition
builtin_interfaces/Duration total_elapsed_time
	int32 sec
	uint32 nanosec
---
#feedback definition
builtin_interfaces/Duration time_left
	int32 sec
	uint32 nanosec""",
}

ACTION_MODELS: Dict[str, Type[BaseModel]] = {
    "nav2_msgs/action/NavigateToPose": NavigateToPoseGoal,
    "nav2_msgs/action/Spin": SpinGoal,
    "nav2_msgs/action/AssistedTeleop": AssistedTeleopGoal,
    "nav2_msgs/action/BackUp": BackUpGoal,
    "nav2_msgs/action/ComputePathThroughPoses": ComputePathThroughPosesGoal,
    "nav2_msgs/action/ComputePathToPose": ComputePathToPoseGoal,
    "nav2_msgs/action/DriveOnHeading": DriveOnHeadingGoal,
    "nav2_msgs/action/FollowPath": FollowPathGoal,
    "nav2_msgs/action/FollowWaypoints": FollowWaypointsGoal,
    "nav2_msgs/action/NavigateThroughPoses": NavigateThroughPosesGoal,
    "nav2_msgs/action/SmoothPath": SmoothPathGoal,
    "nav2_msgs/action/Wait": WaitGoal,
}

ACTION_STRINGS = [
    f"action: {action}\ntype: {act_type}\n"
    for action, act_type in ACTIONS_AND_TYPES.items()
]

SERVICE_STRINGS = [
    f"service: {service}\ntype: {srv_type}\n"
    for service, srv_type in SERVICES_AND_TYPES.items()
]


class NavigationTask(Task):
    @property
    def type(self) -> str:
        return "navigation"

    def get_system_prompt(self) -> str:
        return ROBOT_NAVIGATION_SYSTEM_PROMPT

    @property
    def available_tools(self) -> List[BaseTool]:
        tools = MockActionsToolkit(
            mock_actions_names_and_types=ACTION_STRINGS,
            available_actions=list(ACTIONS_AND_TYPES.keys()),
            available_action_types=list(ACTIONS_AND_TYPES.values()),
            available_action_models=ACTION_MODELS,
        ).get_tools()
        tools.append(MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES))
        return tools


class NavigateToPointTask(NavigationTask):
    complexity = "medium"

    def get_prompt(self) -> str:
        return "Navigate to the point (2.0, 2.0, 0.0). Remember to list actions and get interface"


class SpinAroundTask(NavigationTask):
    recursion_limit = 50
    complexity = "medium"

    def get_prompt(self) -> str:
        return "Spin around by 3 radians."


class MoveToFrontTask(NavigationTask):
    recursion_limit = 50
    complexity = "medium"

    def get_prompt(self) -> str:
        return "Move 2 meters to the front."


class MoveToBedTask(NavigationTask):
    recursion_limit = 50
    complexity = "medium"

    @property
    def available_tools(self) -> List[BaseTool]:
        return [
            MockGetROS2ActionsNamesAndTypesTool(
                mock_actions_names_and_types=ACTION_STRINGS
            ),
            MockStartROS2ActionTool(
                available_actions=list(ACTIONS_AND_TYPES.keys()),
                available_action_types=list(ACTIONS_AND_TYPES.values()),
                available_action_models=ACTION_MODELS,
            ),
            MockGetROS2ActionFeedbackTool(),
            MockGetROS2ActionResultTool(),
            MockGetROS2MessageInterfaceTool(mock_interfaces=INTERFACES),
            MockGetROS2TopicsNamesAndTypesTool(
                mock_topics_names_and_types=TOPICS_NAMES_AND_TYPES
            ),
            MockGetDistanceToObjectsTool(
                available_topics=[
                    "/camera/camera/color/image_raw",
                    "/camera/camera/depth/image_rect_raw",
                ],
                mock_distance_measurements=[
                    DistanceMeasurement(name="bed", distance=5.0)
                ],
            ),
        ]

    def get_prompt(self) -> str:
        return "Move closer to the to the bed. Leave 1 meter of space between the bed and you."
