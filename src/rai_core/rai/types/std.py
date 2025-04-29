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

from pydantic import Field

from rai.types.base import ROS2BaseModel


class BaseStdModel(ROS2BaseModel):
    _prefix: str = "std_msgs/msg"


class Time(BaseStdModel):
    sec: int = 0
    nanosec: int = 0


class Header(BaseStdModel):
    frame_id: str = Field(default="", description="Reference frame of the message")
    stamp: Time = Time()
