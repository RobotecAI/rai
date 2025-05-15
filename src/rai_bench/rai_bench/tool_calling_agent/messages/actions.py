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

from typing import Any, Dict, List, Optional

from rai.types import Path, Point, PoseStamped, ROS2BaseModel, Time


class TaskGoal(ROS2BaseModel):
    task: Optional[str] = ""
    description: Optional[str] = ""
    priority: Optional[str] = ""


class TaskResult(ROS2BaseModel):
    success: Optional[bool] = False
    report: Optional[str] = ""


class TaskFeedback(ROS2BaseModel):
    current_status: Optional[str] = ""


class LoadMapRequest(ROS2BaseModel):
    filename: Optional[str] = ""


class LoadMapResponse(ROS2BaseModel):
    success: Optional[bool] = False


class NavigateToPoseGoal(ROS2BaseModel):
    pose: Optional[PoseStamped] = None
    behavior_tree: Optional[str] = None


class ActionResult(ROS2BaseModel):
    result: Optional[Dict[str, Any]] = None


class NavigateToPoseFeedback(ROS2BaseModel):
    current_pose: Optional[PoseStamped] = None
    navigation_time: Optional[Time] = None
    estimated_time_remaining: Optional[Time] = None
    number_of_recoveries: Optional[int] = None
    distance_remaining: Optional[float] = None


class NavigateToPoseAction(ROS2BaseModel):
    goal: Optional[NavigateToPoseGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[NavigateToPoseFeedback] = None


class SpinGoal(ROS2BaseModel):
    target_yaw: Optional[float] = None
    time_allowance: Optional[Time] = None


class SpinFeedback(ROS2BaseModel):
    angle_turned: Optional[float] = None
    remaining_yaw: Optional[float] = None


class SpinAction(ROS2BaseModel):
    goal: Optional[SpinGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[SpinFeedback] = None


class AssistedTeleopGoal(ROS2BaseModel):
    time_allowance: Optional[Time] = None


class AssistedTeleopFeedback(ROS2BaseModel):
    current_teleop_duration: Optional[Time] = None


class AssistedTeleopAction(ROS2BaseModel):
    goal: Optional[AssistedTeleopGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[AssistedTeleopFeedback] = None


class BackUpGoal(ROS2BaseModel):
    target: Optional[Point] = None
    speed: Optional[float] = None
    time_allowance: Optional[Time] = None


class BackUpFeedback(ROS2BaseModel):
    distance_traveled: Optional[float] = None


class BackUpAction(ROS2BaseModel):
    goal: Optional[BackUpGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[BackUpFeedback] = None


class ComputePathThroughPosesGoal(ROS2BaseModel):
    goals: Optional[List[PoseStamped]] = None
    start: Optional[PoseStamped] = None
    planner_id: Optional[str] = None
    use_start: Optional[bool] = None


class PathResult(ROS2BaseModel):
    path: Optional[Path] = None
    planning_time: Optional[Time] = None


class ComputePathThroughPosesAction(ROS2BaseModel):
    goal: Optional[ComputePathThroughPosesGoal] = None
    result: Optional[PathResult] = None
    feedback: Optional[Dict[str, Any]] = None  # Empty feedback


class ComputePathToPoseGoal(ROS2BaseModel):
    goal: Optional[PoseStamped] = None
    start: Optional[PoseStamped] = None
    planner_id: Optional[str] = None
    use_start: Optional[bool] = None


class ComputePathToPoseAction(ROS2BaseModel):
    goal: Optional[ComputePathToPoseGoal] = None
    result: Optional[PathResult] = None
    feedback: Optional[Dict[str, Any]] = None  # Empty feedback


class DriveOnHeadingGoal(ROS2BaseModel):
    target: Optional[Point] = None
    speed: Optional[float] = None
    time_allowance: Optional[Time] = None


class DriveOnHeadingFeedback(ROS2BaseModel):
    distance_traveled: Optional[float] = None


class DriveOnHeadingAction(ROS2BaseModel):
    goal: Optional[DriveOnHeadingGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[DriveOnHeadingFeedback] = None


class FollowPathGoal(ROS2BaseModel):
    path: Optional[Path] = None
    controller_id: Optional[str] = None
    goal_checker_id: Optional[str] = None


class FollowPathFeedback(ROS2BaseModel):
    distance_to_goal: Optional[float] = None
    speed: Optional[float] = None


class FollowPathAction(ROS2BaseModel):
    goal: Optional[FollowPathGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[FollowPathFeedback] = None


class FollowWaypointsGoal(ROS2BaseModel):
    poses: Optional[List[PoseStamped]] = None


class FollowWaypointsResult(ROS2BaseModel):
    missed_waypoints: Optional[List[int]] = None


class FollowWaypointsFeedback(ROS2BaseModel):
    current_waypoint: Optional[int] = None


class FollowWaypointsAction(ROS2BaseModel):
    goal: Optional[FollowWaypointsGoal] = None
    result: Optional[FollowWaypointsResult] = None
    feedback: Optional[FollowWaypointsFeedback] = None


class NavigateThroughPosesGoal(ROS2BaseModel):
    poses: Optional[List[PoseStamped]] = None
    behavior_tree: Optional[str] = None


class NavigateThroughPosesFeedback(ROS2BaseModel):
    current_pose: Optional[PoseStamped] = None
    navigation_time: Optional[Time] = None
    estimated_time_remaining: Optional[Time] = None
    number_of_recoveries: Optional[int] = None
    distance_remaining: Optional[float] = None
    number_of_poses_remaining: Optional[int] = None


class NavigateThroughPosesAction(ROS2BaseModel):
    goal: Optional[NavigateThroughPosesGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[NavigateThroughPosesFeedback] = None


class SmoothPathGoal(ROS2BaseModel):
    path: Optional[Path] = None
    smoother_id: Optional[str] = None
    max_smoothing_duration: Optional[Time] = None
    check_for_collisions: Optional[bool] = None


class SmoothPathResult(ROS2BaseModel):
    path: Optional[Path] = None
    smoothing_duration: Optional[Time] = None
    was_completed: Optional[bool] = None


class SmoothPathAction(ROS2BaseModel):
    goal: Optional[SmoothPathGoal] = None
    result: Optional[SmoothPathResult] = None
    feedback: Optional[Dict[str, Any]] = None  # Empty feedback


class WaitGoal(ROS2BaseModel):
    time: Optional[Time] = None


class WaitFeedback(ROS2BaseModel):
    time_left: Optional[Time] = None


class WaitAction(ROS2BaseModel):
    goal: Optional[WaitGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[WaitFeedback] = None
