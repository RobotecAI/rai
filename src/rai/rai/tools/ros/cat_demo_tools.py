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

import subprocess
from typing import Type

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel


class UseLightsToolInput(BaseModel):
    """Input for the use lights tool."""


class UseLightsTool(BaseTool):
    """Use the lights"""

    name: str = "UseLightsTool"
    description: str = "Turns on the lights."
    args_schema: Type[UseLightsToolInput] = UseLightsToolInput

    def _run(self):
        """Turns on the lights."""
        result = subprocess.run(
            "echo 'Lights have been turned on'",
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


class UseHonkToolInput(BaseModel):
    """Input for the use honk tool."""


class UseHonkTool(BaseTool):
    """Use the honk"""

    name: str = "UseHonkTool"
    description: str = "Activates the honk."
    args_schema: Type[UseHonkToolInput] = UseHonkToolInput

    def _run(self):
        """Activates the honk."""
        result = subprocess.run(
            "echo 'Activating honk'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()


class ReplanWithoutCurrentPathToolInput(BaseModel):
    """Input for the replan without current path tool."""


class ReplanWithoutCurrentPathTool(BaseTool):
    """Replan without current path"""

    name: str = "ReplanWithoutCurrentPathTool"
    description: str = "Replans without the current path."
    args_schema: Type[ReplanWithoutCurrentPathToolInput] = (
        ReplanWithoutCurrentPathToolInput
    )

    def _run(self):
        """Replans without the current path."""
        result = subprocess.run(
            "echo 'Replanning without current path'",
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


class ContinueActionToolInput(BaseModel):
    """Input for the continue action tool."""


class ContinueActionTool(BaseTool):
    """Continue action"""

    name: str = "ContinueActionTool"
    description: str = "Continues the current operation."
    args_schema: Type[ContinueActionToolInput] = ContinueActionToolInput

    def _run(self):
        """Continues the current operation."""
        result = subprocess.run(
            "echo 'Continuing'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()


class StopToolInput(BaseModel):
    """Input for the stop tool."""


class StopTool(BaseTool):
    """Stop action"""

    name: str = "StopTool"
    description: str = "Stops the current operation."
    args_schema: Type[StopToolInput] = StopToolInput

    def _run(self):
        """Stops the current operation."""
        result = subprocess.run(
            "echo 'Stopping'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()


class FinishToolInput(BaseModel):
    """Input for the finish tool."""


class FinishTool(BaseTool):
    """Finish the conversation. Does not impact the actual mission."""

    name: str = "FinishTool"
    description: str = "Ends the conversation."
    args_schema: Type[FinishToolInput] = FinishToolInput

    def _run(self):
        """Ends the conversation."""
        return "Conversation finished."
