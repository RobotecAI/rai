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

from typing import Any, Dict, Optional

from rai.types import PoseStamped, RaiBaseModel, Time


class TaskGoal(RaiBaseModel):
    task: Optional[str] = ""
    description: Optional[str] = ""
    priority: Optional[str] = ""


class TaskResult(RaiBaseModel):
    success: Optional[bool] = False
    report: Optional[str] = ""


class TaskFeedback(RaiBaseModel):
    current_status: Optional[str] = ""


class LoadMapRequest(RaiBaseModel):
    filename: Optional[str] = ""


class LoadMapResponse(RaiBaseModel):
    success: Optional[bool] = False


class NavigateToPoseGoal(RaiBaseModel):
    pose: Optional[PoseStamped] = None
    behavior_tree: Optional[str] = None


class ActionResult(RaiBaseModel):
    result: Optional[Dict[str, Any]] = None


class NavigateToPoseFeedback(RaiBaseModel):
    current_pose: Optional[PoseStamped] = None
    navigation_time: Optional[Time] = None
    estimated_time_remaining: Optional[Time] = None
    number_of_recoveries: Optional[int] = None
    distance_remaining: Optional[float] = None


class NavigateToPoseAction(RaiBaseModel):
    goal: Optional[NavigateToPoseGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[NavigateToPoseFeedback] = None


class SpinGoal(RaiBaseModel):
    target_yaw: Optional[float] = None
    time_allowance: Optional[Time] = None


class SpinFeedback(RaiBaseModel):
    angle_turned: Optional[float] = None
    remaining_yaw: Optional[float] = None


class SpinAction(RaiBaseModel):
    goal: Optional[SpinGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[SpinFeedback] = None
