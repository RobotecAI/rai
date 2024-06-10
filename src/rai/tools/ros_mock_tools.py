import subprocess
from typing import List, Type

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field


def check_topic_exists(v):
    output = subprocess.run(
        "ros2 topic list", capture_output=True, text=True, shell=True
    )
    topics = output.stdout.strip().split("\n")
    return v in topics


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
    args_schema: Type[
        VisualQuestionAnsweringToolInput
    ] = VisualQuestionAnsweringToolInput

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


class StartCameraNodeToolInput(BaseModel):
    """Input for the start camera node tool."""


class StartCameraNodeTool(BaseTool):
    """Start the camera node"""

    name: str = "StartCameraNodeTool"
    description: str = "Starts the camera node."
    args_schema: Type[StartCameraNodeToolInput] = StartCameraNodeToolInput

    def _run(self):
        """Starts the camera node."""
        subprocess.Popen(
            "python /home/mmajek/projects/internal/rai-private/service.py", shell=True
        )
        return "Camera node started."


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
        subprocess.check_output("play outputs/car-honk-1-short.wav", shell=True)
        result = subprocess.run(
            "ros2 topic pub --once /tractor1/alert/flash std_msgs/Bool \"data: 'true'\"",
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


class ReplanWithoutCurrentPathToolInput(BaseModel):
    """Input for the replan without current path tool."""


class ReplanWithoutCurrentPathTool(BaseTool):
    """Replan without current path"""

    name: str = "ReplanWithoutCurrentPathTool"
    description: str = "Replans without the current path."
    args_schema: Type[
        ReplanWithoutCurrentPathToolInput
    ] = ReplanWithoutCurrentPathToolInput

    def _run(self):
        """Replans without the current path."""
        command = "ros2 topic pub /tractor1/control/move std_msgs/msg/String \"{data: 'REPLAN'}\" --once"
        subprocess.run(command, shell=True)
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
        command = "ros2 topic pub /tractor1/control/move std_msgs/msg/String \"{data: 'CONTINUE'}\" --once"
        subprocess.run(command, shell=True)
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
