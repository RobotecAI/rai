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

from typing import List, Type

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field


class OpenSetSegmentationToolInput(BaseModel):
    """Input for the open set segmentation tool."""

    topic: str = Field(..., description="ROS2 image topic to subscribe to")
    classes: List[str] = Field(..., description="Classes to segment")


class OpenSetSegmentationTool(BaseTool):
    """Get the segmentation of an image into any list of classes"""

    name: str = "OpenSetSegmentationTool"
    description: str = (
        "Segments an image into specified classes from a given ROS2 topic."
    )
    args_schema: Type[OpenSetSegmentationToolInput] = OpenSetSegmentationToolInput

    def _run(self, topic: str, classes: List[str]):
        """Implements the segmentation logic for the specified classes on the given topic."""
        return f"Segmentation on topic {topic} for classes {classes} started."


class VisualQuestionAnsweringToolInput(BaseModel):
    """Input for the visual question answering tool."""

    topic: str = Field(..., description="ROS2 image topic to subscribe to")
    question: str = Field(..., description="Question about the image")


class VisualQuestionAnsweringTool(BaseTool):
    """Ask a question about an image"""

    name: str = "VisualQuestionAnsweringTool"
    description: str = (
        "Processes an image from a ROS2 topic and answers a specified question."
    )
    args_schema: Type[VisualQuestionAnsweringToolInput] = (
        VisualQuestionAnsweringToolInput
    )

    def _run(self, topic: str, question: str):
        """Processes the image from the specified topic and answers the given question."""
        return f"Processing and answering question about {topic}: {question}"


class ObserveSurroundingsToolInput(BaseModel):
    """Input for the observe surroundings tool."""

    topic: str = Field(..., description="ROS2 image topic to subscribe to")


class ObserveSurroundingsTool(BaseTool):
    """Observe the surroundings"""

    name: str = "ObserveSurroundingsTool"
    description: str = "Observes and processes data from a given ROS2 topic."
    args_schema: Type[ObserveSurroundingsToolInput] = ObserveSurroundingsToolInput

    def _run(self, topic: str):
        """Observes and processes data from the given ROS2 topic."""
        return f"Observing surroundings using topic {topic}"
