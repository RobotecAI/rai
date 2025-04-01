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
