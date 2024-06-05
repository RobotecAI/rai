import subprocess
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


def check_topic_exists(v):
    output = subprocess.run(
        "ros2 topic list", capture_output=True, text=True, shell=True
    )
    topics = output.stdout.strip().split("\n")
    return v in topics


class ros2_topic_list(BaseModel):
    """Get a list of ros2 topics"""

    def run(self):
        """Executes the listing of ROS2 topics and returns them."""
        result = subprocess.check_output("ros2 topic list", shell=True)
        return result


class open_set_segmentation(BaseModel):
    """Get the segmentation of an image into any list of classes"""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")
    classes: List[str] = Field(..., description="Classes to segment")

    def run(self):
        """Implements the segmentation logic for the specified classes on the given topic."""
        # Placeholder for actual segmentation logic
        return f"Segmentation on topic {self['topic']} for classes {self['classes']} started."


class visual_question_answering(BaseModel):
    """Ask a question about an image"""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")
    question: str = Field(..., description="Question about the image")

    def run(self):
        """Processes the image from the specified topic and answers the given question."""
        # Placeholder for actual image processing and question answering logic
        return f"Processing and answering question about {self.topic}: {self.question}"


class observe_surroundings(BaseModel):
    """Observe the surroundings"""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")

    def run(self):
        """Observes and processes data from the given ROS2 topic."""
        # Placeholder for actual observation logic
        return f"Observing surroundings using topic {self['topic']}"


class start_camera_node(BaseModel):
    """Start the camera node"""

    def run(self):
        """Starts the camera node."""
        # Placeholder for actual camera node starting logic
        import subprocess

        subprocess.Popen(
            "python /home/mmajek/projects/internal/rai-private/service.py", shell=True
        )
        return "Camera node started."


class use_lights(BaseModel):
    """Use the lights"""

    def run(self):
        """Turns on the lights."""
        result = subprocess.run(
            "echo 'Lights have been turned on'",
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


class use_honk(BaseModel):
    """Use the honk"""

    def run(self):
        """Activates the honk."""
        result = subprocess.run(
            "echo 'Honk has been activated'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()


class replan_without_current_path(BaseModel):
    """Replan without current path"""

    def run(self):
        """Replans without the current path."""
        result = subprocess.run(
            "echo 'Replanning without current path'",
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


class continue_action(BaseModel):
    """Continue action"""

    def run(self):
        """Continues the current operation."""
        result = subprocess.run(
            "echo 'Continuing'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()


class stop(BaseModel):
    """Stop action"""

    def run(self):
        """Stops the current operation."""
        result = subprocess.run(
            "echo 'Stopping'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()


class finish(BaseModel):
    """Finish the conversation. Does not impact the actual mission."""

    def run(self):
        """Ends the conversation."""
        return "Conversation finished."
