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

from typing import Optional

from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.actions.action_base_model import ActionBaseModel


class Time(BaseModel):
    sec: Optional[int] = 0
    nanosec: Optional[int] = 0


class Goal(BaseModel):
    target_yaw: Optional[float] = 0.0
    time_allowance: Optional[Time] = Time()


class Result(BaseModel):
    result: dict


class Feedback(BaseModel):
    angle_turned: Optional[float] = 0.0
    remaining_yaw: Optional[float] = 0.0


class SpinAction(ActionBaseModel):
    action_name: str = "/spin"
    action_type: str = "nav2_msgs/action/Spin"
    goal: Goal
    result: Result
    feedback: Feedback
