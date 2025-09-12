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


from uuid import UUID

from pydantic import BaseModel, Field


class TaskResult(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task object.")
    task_prompt: str = Field(..., description="The task prompt.")
    system_prompt: str = Field(..., description="The system prompt.")
    complexity: str = Field(..., description="Complexity of the task.")
    type: str = Field(
        ..., description="Type of task, for example: bool_response_image_task"
    )
    model_name: str = Field(..., description="Name of the LLM.")
    score: float = Field(
        ...,
        description="Value between 0 and 1.",
    )

    total_time: float = Field(..., description="Total time taken to complete the task.")
    run_id: UUID = Field(..., description="UUID of the task run.")
