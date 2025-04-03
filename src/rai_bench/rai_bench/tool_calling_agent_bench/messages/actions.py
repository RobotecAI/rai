from typing import Optional

from pydantic import BaseModel


class TaskGoal(BaseModel):
    task: Optional[str] = ""
    description: Optional[str] = ""
    priority: Optional[str] = ""


class TaskResult(BaseModel):
    success: Optional[bool] = False
    report: Optional[str] = ""


class TaskFeedback(BaseModel):
    current_status: Optional[str] = ""


class LoadMapRequest(BaseModel):
    filename: Optional[str] = ""


class LoadMapResponse(BaseModel):
    success: Optional[bool] = False
