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

from pydantic import BaseModel, Field


class ScenarioResult(BaseModel):
    """Result of a single scenario execution."""

    task_prompt: str = Field(..., description="The task prompt.")
    system_prompt: str = Field(..., description="The system prompt.")
    model_name: str = Field(..., description="Name of the LLM.")
    scene_config_path: str = Field(
        ..., description="Path to the scene configuration file."
    )
    score: float = Field(
        ..., description="Value between 0 and 1, describing the task score."
    )
    level: str = Field(..., description="Difficulty of the scenario")
    total_time: float = Field(..., description="Total time taken to complete the task.")
    number_of_tool_calls: int = Field(
        ..., description="Number of tool calls made during the task."
    )
