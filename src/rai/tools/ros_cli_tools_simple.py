import subprocess

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field


class Ros2TopicToolInput(BaseModel):
    """Input for the ros2_topic tool."""

    command: str = Field(..., description="The command to run")


class Ros2TopicTool(BaseTool):
    """Tool for interacting with ROS2 topics."""

    name: str = "ros2_topic"
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
    args_schema = Ros2TopicToolInput

    def _run(self, command: str):
        """Executes the specified ROS2 topic command."""
        result = subprocess.run(
            f"ros2 topic {command}", shell=True, capture_output=True
        )
        return result


class Ros2InterafaceToolInput(BaseModel):
    """Input for the ros2_interface tool."""

    command: str = Field(..., description="The command to run")


class Ros2InterfaceTool(BaseTool):

    name: str = "ros2 interace tool"

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

    args_schema = Ros2InterafaceToolInput

    def _run(self, command: str):
        command = f"ros2 interface {command}"
        result = subprocess.run(command, shell=True, capture_output=True)
        return result


class Ros2ServiceToolInput(BaseModel):
    """Input for the ros2_service tool."""

    command: str = Field(..., description="The command to run")


class Ros2ServiceTool(BaseModel):
    name: str = "ros2 service tool"

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

    usage: ros2 service call [-h] [-r N] service_name service_type [values]
    ros2 service call: error: the following arguments are required: service_name, service_type

    usage: ros2 service find [-h] [-c] [--include-hidden-services] service_type
    ros2 service find: error: the following arguments are required: service_type
    usage: ros2 service type [-h] service_name
    ros2 service type: error: the following arguments are required: service_name
    """

    args_schema = Ros2ServiceToolInput

    def _run(self, command: str):
        command = f"ros2 service {command}"
        result = subprocess.run(command, shell=True, capture_output=True)
        return result
