from pydantic import BaseModel


class TaskGoal(BaseModel):
    task: str = ""
    description: str = ""
    priority: str = ""


class TaskResult(BaseModel):
    success: bool = False
    report: str = ""


class TaskFeedback(BaseModel):
    current_status: str = ""


class LoadMapRequest(BaseModel):
    filename: str = ""


class LoadMapResponse(BaseModel):
    success: bool = False
