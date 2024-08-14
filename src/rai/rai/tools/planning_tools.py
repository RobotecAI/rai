# Copyright (C) 2024 Robotec.AI
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
#

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
