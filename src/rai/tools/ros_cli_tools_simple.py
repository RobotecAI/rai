import subprocess

from langchain_core.pydantic_v1 import BaseModel, Field


class ros2_generic_cli_call(BaseModel):
    """
    usage: ros2 [-h] Call `ros2 <command> -h` for more detailed usage.
    ros2 is an extensible command-line tool for ROS 2.

    Commands:
      ros2 action
      ros2 bag
      ros2 component
      ros2 daemon
      ros2 doctor
      ros2 interface
      ros2 launch
      ros2 lifecycle
      ros2 multicast
      ros2 node
      ros2 param
      ros2 pkg
      ros2 run
      ros2 security
      ros2 service
      ros2 topic
      ros2 wtf
      Call `ros2 <command> -h` for more detailed usage."""

    command: str = Field(..., description="A ros2 command to execute.")

    def run(self):
        """Executes the specified ROS2 command."""
        result = subprocess.run(self.command, shell=True)
        return result


class ros2_interface(BaseModel):
    """
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

    command: str = Field(..., description="The command to run")

    def run(self):
        command = f"ros2 interface {self.command}"
        result = subprocess.run(command, shell=True, capture_output=True)
        return result


class ros2_service(BaseModel):
    """
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

    command: str = Field(..., description="The command to run")

    def run(self):
        command = f"ros2 service {self.command}"
        result = subprocess.run(command, shell=True, capture_output=True)
        return result
