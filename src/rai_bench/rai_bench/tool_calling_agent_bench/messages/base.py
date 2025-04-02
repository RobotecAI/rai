from typing import List, Optional

from pydantic import BaseModel


# TODO (jm) redundant with action models, remove action models later
class Time(BaseModel):
    sec: Optional[int] = 0
    nanosec: Optional[int] = 0


class Header(BaseModel):
    stamp: Optional[Time] = Time()
    frame_id: str = ""


class RegionOfInterest(BaseModel):
    x_offset: int = 0
    y_offset: int = 0
    height: int = 0
    width: int = 0
    do_rectify: bool = False


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Orientation(BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    w: Optional[float] = 1.0


class Pose(BaseModel):
    position: Position = Position()
    orientation: Optional[Orientation] = Orientation()


class PoseStamped(BaseModel):
    header: Header = Header()
    pose: Pose = Pose()


class Clock(BaseModel):
    clock: Time = Time()


class ObjectHypothesis(BaseModel):
    class_id: str = ""
    score: float = 0.0


class PoseWithCovariance(BaseModel):
    pose: Pose = Pose()
    covariance: List[float] = [0.0] * 36


class ObjectHypothesisWithPose(BaseModel):
    hypothesis: ObjectHypothesis = ObjectHypothesis()
    pose: PoseWithCovariance = PoseWithCovariance()


class Point2D(BaseModel):
    x: float = 0.0
    y: float = 0.0


class Pose2D(BaseModel):
    position: Point2D = Point2D()
    theta: float = 0.0


class BoundingBox2D(BaseModel):
    center: Pose2D = Pose2D()
    size_x: float = 0.0
    size_y: float = 0.0


class Detection2D(BaseModel):
    header: Header = Header()
    results: List[ObjectHypothesisWithPose] = []
    bbox: BoundingBox2D = BoundingBox2D()
    id: str = ""
