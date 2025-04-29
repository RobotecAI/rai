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

from rai.types.base import ROS2BaseModel
from rai.types.geometry import Pose2D, PoseWithCovariance
from rai.types.std import Header


class BaseVisionModel(ROS2BaseModel):
    _prefix: str = "vision_msgs/msg"


class ObjectHypothesis(BaseVisionModel):
    class_id: str = ""
    score: float = 0.0


class ObjectHypothesisWithPose(BaseVisionModel):
    hypothesis: ObjectHypothesis = ObjectHypothesis()
    pose: PoseWithCovariance = PoseWithCovariance()


class BoundingBox2D(BaseVisionModel):
    center: Pose2D = Pose2D()
    size_x: float = 0.0
    size_y: float = 0.0


class Detection2D(BaseVisionModel):
    header: Header = Header()
    results: List[ObjectHypothesisWithPose] = []
    bbox: BoundingBox2D = BoundingBox2D()
    id: str = ""


class RegionOfInterest(BaseVisionModel):
    x_offset: int = 0
    y_offset: int = 0
    height: int = 0
    width: int = 0
    do_rectify: bool = False
