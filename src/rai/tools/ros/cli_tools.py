import subprocess
from typing import Type

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field


class Ros2TopicToolInput(BaseModel):
    """Input for the ros2_topic tool."""

    command: str = Field(..., description="The command to run")


class Ros2TopicTool(BaseTool):
    """Tool for interacting with ROS2 topics."""

    name: str = "Ros2TopicTool"
    description: str = """
        usage: ros2 topic [-h] [--include-hidden-topics] Call `ros2 topic <command> -h` for more detailed usage. ...

    Various topic related sub-commands

    options:
      -h, --help            show this help message and exit
      --include-hidden-topics
                            Consider hidden topics as well

    Commands:
      bw     Display bandwidth used by topic
      delay  Display delay of topic from timestamp in header
      echo   Output messages from a topic
      find   Output a list of available topics of a given type
      hz     Print the average publishing rate to screen
      info   Print information about a topic
      list   Output a list of available topics
      pub    Publish a message to a topic
      type   Print a topic's type

      Call `ros2 topic <command> -h` for more detailed usage.
    """
    args_schema: Type[Ros2TopicToolInput] = Ros2TopicToolInput

    def _run(self, command: str):
        """Executes the specified ROS2 topic command."""
        result = subprocess.run(
            f"ros2 topic {command} -v", shell=True, capture_output=True
        )
        return result


class Ros2InterafaceToolInput(BaseModel):
    """Input for the ros2_interface tool."""

    command: str = Field(..., description="The command to run")


class Ros2InterfaceTool(BaseTool):

    name: str = "Ros2InterfaceTool"

    description: str = """
    usage: ros2 interface [-h] Call `ros2 interface <command> -h` for more detailed usage. ...

    Show information about ROS interfaces

    options:
      -h, --help            show this help message and exit

    Commands:
      list      List all interface types available
      package   Output a list of available interface types within one package
      packages  Output a list of packages that provide interfaces
      proto     Output an interface prototype
      show      Output the interface definition

      Call `ros2 interface <command> -h` for more detailed usage.
    """

    args_schema: Type[Ros2InterafaceToolInput] = Ros2InterafaceToolInput

    def _run(self, command: str):
        command = f"ros2 interface {command}"
        result = subprocess.run(command, shell=True, capture_output=True)
        return result


class Ros2ServiceToolInput(BaseModel):
    """Input for the ros2_service tool."""

    command: str = Field(..., description="The command to run")


class Ros2ServiceTool(BaseTool):
    name: str = "Ros2ServiceTool"

    description: str = """
    usage: ros2 service [-h] [--include-hidden-services] Call `ros2 service <command> -h` for more detailed usage. ...

    Various service related sub-commands

    options:
      -h, --help            show this help message and exit
      --include-hidden-services
                            Consider hidden services as well

    Commands:
      call  Call a service
      find  Output a list of available services of a given type
      list  Output a list of available services
      type  Output a service's type
    """

    args_schema: Type[Ros2ServiceToolInput] = Ros2ServiceToolInput

    def _run(self, command: str):
        command = f"ros2 service {command}"
        result = subprocess.run(command, shell=True, capture_output=True)
        return result


class SetGoalPoseToolInput(BaseModel):
    """Input for the set_goal_pose tool."""

    topic: str = Field(
        "/goal_pose", description="Ros2 topic to publish the goal pose to"
    )
    x: float = Field(..., description="The x coordinate of the goal pose")
    y: float = Field(..., description="The y coordinate of the goal pose")


class SetGoalPoseTool(BaseTool):
    """Set the goal pose for the robot"""

    name = "SetGoalPoseTool"
    description: str = "A tool for setting the goal pose for the robot."

    args_schema: Type[SetGoalPoseToolInput] = SetGoalPoseToolInput

    def _run(self, topic: str, x: float, y: float):
        """Sets the goal pose for the robot."""

        _ = (
            f"ros2 topic pub {topic} geometry_msgs/PoseStamped "
            f'\'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, '
            f"pose: {{position: {{x: {x}, y: {y}, z: {0.0}}}}}}}' --once"
        )
        return "done"  # subprocess.run(cmd, shell=True)
