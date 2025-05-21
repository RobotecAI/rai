# Copyright (C) 2025 Robotec.AI
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


from typing import Any, Dict, List
from uuid import UUID

from pydantic import BaseModel, Field

from rai_bench.base_benchmark import RunSummary


class ToolCallingAgentRunSummary(RunSummary):
    total_extra_tool_calls_used: int = Field(
        ..., description="Total number of extra tool calls used in this Task"
    )


class SubTaskResult(BaseModel):
    args: Dict[str, Any]
    errors: List[str]
    passed: bool


class ValidatorResult(BaseModel):
    type: str
    subtasks: List[SubTaskResult]
    extra_tool_calls_used: int
    passed: bool


class TaskResult(BaseModel):
    task_prompt: str = Field(..., description="The task prompt.")
    system_prompt: str = Field(..., description="The system prompt.")
    complexity: str = Field(..., description="Complexity of the task.")
    type: str = Field(..., description="Type of task, for example: manipulation")
    model_name: str = Field(..., description="Name of the LLM.")
    validation_info: List[ValidatorResult] = Field(
        ..., description="Validation structure, errors, etc."
    )
    extra_tool_calls: int = Field(
        ...,
        description="Maximum number of extra tool calls agent can make and still pass a task",
    )
    extra_tool_calls_used: int = Field(
        ..., description="Total number of extra tool calls used in this Task"
    )
    score: float = Field(
        ...,
        description="Value between 0 and 1, describing how many validation setps passed",
    )

    total_time: float = Field(..., description="Total time taken to complete the task.")
    run_id: UUID = Field(..., description="UUID of the task run.")
