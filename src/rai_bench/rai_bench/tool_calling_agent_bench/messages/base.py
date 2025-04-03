from typing import List, Optional

from pydantic import BaseModel


# TODO (jm) redundant with action models, remove action models later
class Time(BaseModel):
    sec: Optional[int] = 0
    nanosec: Optional[int] = 0


class Header(BaseModel):
    stamp: Optional[Time] = Time()
    frame_id: Optional[str] = ""


class RegionOfInterest(BaseModel):
    x_offset: Optional[int] = 0
    y_offset: Optional[int] = 0
    height: Optional[int] = 0
    width: Optional[int] = 0
    do_rectify: Optional[bool] = False


class Position(BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0


class Orientation(BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    w: Optional[float] = 1.0


class Pose(BaseModel):
    position: Optional[Position] = Position()
    orientation: Optional[Orientation] = Orientation()


class PoseStamped(BaseModel):
    header: Optional[Header] = Header()
    pose: Optional[Pose] = Pose()


class Clock(BaseModel):
    clock: Optional[Time] = Time()


class ObjectHypothesis(BaseModel):
    class_id: Optional[str] = ""
    score: Optional[float] = 0.0


class PoseWithCovariance(BaseModel):
    pose: Optional[Pose] = Pose()
    covariance: Optional[List[float]] = [0.0] * 36


class ObjectHypothesisWithPose(BaseModel):
    hypothesis: Optional[ObjectHypothesis] = ObjectHypothesis()
    pose: Optional[PoseWithCovariance] = PoseWithCovariance()


class Point2D(BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0


class Pose2D(BaseModel):
    position: Optional[Point2D] = Point2D()
    theta: Optional[float] = 0.0


class BoundingBox2D(BaseModel):
    center: Optional[Pose2D] = Pose2D()
    size_x: Optional[float] = 0.0
    size_y: Optional[float] = 0.0


class Detection2D(BaseModel):
    header: Optional[Header] = Header()
    results: Optional[List[ObjectHypothesisWithPose]] = []
    bbox: Optional[BoundingBox2D] = BoundingBox2D()
    id: Optional[str] = ""
