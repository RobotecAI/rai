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

from .base import RaiBaseModel
from .std import Header


class Point(RaiBaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Quaternion(RaiBaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.1


class Pose(RaiBaseModel):
    position: Point = Point()
    orientation: Quaternion = Quaternion()


class PoseStamped(RaiBaseModel):
    header: Header = Header()
    pose: Pose = Pose()


class PoseWithCovariance(RaiBaseModel):
    pose: Pose = Pose()
    covariance: List[float] = [0.0] * 36


class Pose2D(RaiBaseModel):
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
