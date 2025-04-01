from typing import Any

from pydantic import BaseModel


class ActionBaseModel(BaseModel):
    action_name: str
    action_type: str
    goal: Any
    result: Any
    feedback: Any
