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

from pydantic import BaseModel

from rai_bench.tool_calling_agent.messages.base import PoseStamped, Time


class TaskGoal(BaseModel):
    task: Optional[str] = ""
    description: Optional[str] = ""
    priority: Optional[str] = ""


class TaskResult(BaseModel):
    success: Optional[bool] = False
    report: Optional[str] = ""


class TaskFeedback(BaseModel):
    current_status: Optional[str] = ""


class LoadMapRequest(BaseModel):
    filename: Optional[str] = ""


class LoadMapResponse(BaseModel):
    success: Optional[bool] = False


class NavigateToPoseGoal(BaseModel):
    pose: Optional[PoseStamped] = None
    behavior_tree: Optional[str] = None


class ActionResult(BaseModel):
    result: Optional[Dict[str, Any]] = None


class NavigateToPoseFeedback(BaseModel):
    current_pose: Optional[PoseStamped] = None
    navigation_time: Optional[Time] = None
    estimated_time_remaining: Optional[Time] = None
    number_of_recoveries: Optional[int] = None
    distance_remaining: Optional[float] = None


class NavigateToPoseAction(BaseModel):
    goal: Optional[NavigateToPoseGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[NavigateToPoseFeedback] = None


class SpinGoal(BaseModel):
    target_yaw: Optional[float] = None
    time_allowance: Optional[Time] = None


class SpinFeedback(BaseModel):
    angle_turned: Optional[float] = None
    remaining_yaw: Optional[float] = None


class SpinAction(BaseModel):
    goal: Optional[SpinGoal] = None
    result: Optional[ActionResult] = None
    feedback: Optional[SpinFeedback] = None
