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

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class Ros2BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Time(Ros2BaseModel):
    sec: Optional[int] = 0
    nanosec: Optional[int] = 0


class Header(Ros2BaseModel):
    stamp: Optional[Time] = Time()
    frame_id: Optional[str] = ""


class RegionOfInterest(Ros2BaseModel):
    x_offset: Optional[int] = 0
    y_offset: Optional[int] = 0
    height: Optional[int] = 0
    width: Optional[int] = 0
    do_rectify: Optional[bool] = False


class Position(Ros2BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0


class Orientation(Ros2BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    w: Optional[float] = 1.0


class Pose(Ros2BaseModel):
    position: Optional[Position] = Position()
    orientation: Optional[Orientation] = Orientation()


class PoseStamped(Ros2BaseModel):
    header: Optional[Header] = Header()
    pose: Optional[Pose] = Pose()


class Clock(Ros2BaseModel):
    clock: Optional[Time] = Time()


class ObjectHypothesis(Ros2BaseModel):
    class_id: Optional[str] = ""
    score: Optional[float] = 0.0


class PoseWithCovariance(Ros2BaseModel):
    pose: Optional[Pose] = Pose()
    covariance: Optional[List[float]] = [0.0] * 36


class ObjectHypothesisWithPose(Ros2BaseModel):
    hypothesis: Optional[ObjectHypothesis] = ObjectHypothesis()
    pose: Optional[PoseWithCovariance] = PoseWithCovariance()


class Point2D(Ros2BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0


class Pose2D(Ros2BaseModel):
    position: Optional[Point2D] = Point2D()
    theta: Optional[float] = 0.0


class BoundingBox2D(Ros2BaseModel):
    center: Optional[Pose2D] = Pose2D()
    size_x: Optional[float] = 0.0
    size_y: Optional[float] = 0.0


class Detection2D(Ros2BaseModel):
    header: Optional[Header] = Header()
    results: Optional[List[ObjectHypothesisWithPose]] = []
    bbox: Optional[BoundingBox2D] = BoundingBox2D()
    id: Optional[str] = ""
