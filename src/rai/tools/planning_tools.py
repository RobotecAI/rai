from queue import Queue
from typing import Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool


class AddTaskToolInput(BaseModel):
    task: str = Field(..., title="Task to be added into the task list.")


class AddTaskTool(BaseTool):
    name: str = "AddTaskTool"
    description: str = "Add a task to the task list for later execution."
    args_schema: Type[AddTaskToolInput] = AddTaskToolInput

    queue: Queue[str]

    def _run(self, task: str):
        self.queue.put(task)
        return "Task added to the task list."


class GetNewTaskToolInput(BaseModel):
    pass


class GetNewTaskTool(BaseTool):
    name: str = "GetNewTaskTool"
    description: str = "Get a new task from the task list."
    args_schema: Type[GetNewTaskToolInput] = GetNewTaskToolInput

    queue: Queue[str]

    def _run(self):
        if self.queue.empty():
            return "Task list is empty."
        else:
            task = self.queue.get()
            return "Retrieved task: " + task
