from typing import Optional

from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.actions.action_base_model import ActionBaseModel


class Time(BaseModel):
    sec: Optional[int] = 0
    nanosec: Optional[int] = 0


class Header(BaseModel):
    stamp: Optional[Time] = Time()
    frame_id: str


class Position(BaseModel):
    x: float
    y: float
    z: float


class Orientation(BaseModel):
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    w: Optional[float] = 1.0


class Pose(BaseModel):
    position: Position
    orientation: Optional[Orientation] = Orientation()


class PoseStamped(BaseModel):
    header: Header
    pose: Pose


class Goal(BaseModel):
    pose: PoseStamped
    behavior_tree: Optional[str] = ""


class Result(BaseModel):
    result: dict


class Feedback(BaseModel):
    current_pose: PoseStamped
    navigation_time: Time
    estimated_time_remaining: Time
    number_of_recoveries: int
    distance_remaining: float


class NavigateToPoseAction(ActionBaseModel):
    action_name: str = "/navigate_to_pose"
    action_type: str = "nav2_msgs/action/NavigateToPose"
    goal: Goal
    result: Result
    feedback: Feedback


# TODO (mkotynia): create init for actions
