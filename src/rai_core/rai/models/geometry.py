import time
from dataclasses import dataclass, field


@dataclass
class Point:
    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    x: float = field(default=0)
    y: float = field(default=0)
    z: float = field(default=0)
    w: float = field(default=1)


@dataclass
class Pose:
    position: Point
    orientation: Quaternion


@dataclass
class Header:
    frame_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PoseStamped:
    header: Header
    pose: Pose
