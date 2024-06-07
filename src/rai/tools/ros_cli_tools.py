import subprocess

from langchain_core.pydantic_v1 import BaseModel, Field


class set_goal_pose_relative_to_the_map(BaseModel):
    """Set the goal pose"""

    topic: str = Field(..., description="Ros2 pose topic to publish to")
    x: float = Field(..., description="X coordinate of the goal pose")
    y: float = Field(..., description="Y coordinate of the goal pose")
    z: float = Field(..., description="Z coordinate of the goal pose")

    def run(self):
        """Sets the goal pose on the specified topic."""
        cmd = (
            f"ros2 topic pub {self.topic} geometry_msgs/PoseStamped "
            f'\'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, '
            f"pose: {{position: {{x: {self.x}, y: {self.y}, z: {self.z}}}}}}}' --once"
        )
        subprocess.run(cmd, shell=True)


ros2_action_doc = """
usage: ros2 action [-h]
                   Call `ros2 action <command> -h` for more detailed usage.
                   ...

Various action related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  info       Print information about an action
  list       Output a list of action names
  send_goal  Send an action goal

  Call `ros2 action <command> -h` for more detailed usage.

"""
ros2_bag_doc = """
usage: ros2 bag [-h] Call `ros2 bag <command> -h` for more detailed usage. ...

Various rosbag related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  convert  Given an input bag, write out a new bag with different settings
  info     Print information about a bag to the screen
  list     Print information about available plugins to the screen
  play     Play back ROS data from a bag
  record   Record ROS data to a bag
  reindex  Reconstruct metadata file for a bag

  Call `ros2 bag <command> -h` for more detailed usage.

"""
ros2_component_doc = """
usage: ros2 component [-h]
                      Call `ros2 component <command> -h` for more detailed
                      usage. ...

Various component related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  list        Output a list of running containers and components
  load        Load a component into a container node
  standalone  Run a component into its own standalone container node
  types       Output a list of components registered in the ament index
  unload      Unload a component from a container node

  Call `ros2 component <command> -h` for more detailed usage.

"""
ros2_daemon_doc = """
usage: ros2 daemon [-h]
                   Call `ros2 daemon <command> -h` for more detailed usage.
                   ...

Various daemon related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  start   Start the daemon if it isn't running
  status  Output the status of the daemon
  stop    Stop the daemon if it is running

  Call `ros2 daemon <command> -h` for more detailed usage.

"""
ros2_doctor_doc = """
usage: ros2 doctor [-h] [--report | --report-failed] [--include-warnings]
                   Call `ros2 doctor <command> -h` for more detailed usage.
                   ...

Check ROS setup and other potential issues

options:
  -h, --help            show this help message and exit
  --report, -r          Print all reports.
  --report-failed, -rf  Print reports of failed checks only.
  --include-warnings, -iw
                        Include warnings as failed checks. Warnings are
                        ignored by default.

Commands:
  hello  Check network connectivity between multiple hosts

  Call `ros2 doctor <command> -h` for more detailed usage.

"""
ros2_interface_doc = """
usage: ros2 interface [-h]
                      Call `ros2 interface <command> -h` for more detailed
                      usage. ...

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
ros2_launch_doc = """
usage: ros2 launch [-h] [-n] [-d] [-p | -s] [-a]
                   [--launch-prefix LAUNCH_PREFIX]
                   [--launch-prefix-filter LAUNCH_PREFIX_FILTER]
                   package_name [launch_file_name] [launch_arguments ...]

Run a launch file

positional arguments:
  package_name          Name of the ROS package which contains the launch file
  launch_file_name      Name of the launch file
  launch_arguments      Arguments to the launch file; '<name>:=<value>' (for
                        duplicates, last one wins)

"""  # this is incomplete, was too long for the openai tool description

ros2_lifecycle_doc = """
usage: ros2 lifecycle [-h]
                      Call `ros2 lifecycle <command> -h` for more detailed
                      usage. ...

Various lifecycle related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  get    Get lifecycle state for one or more nodes
  list   Output a list of available transitions
  nodes  Output a list of nodes with lifecycle
  set    Trigger lifecycle state transition

  Call `ros2 lifecycle <command> -h` for more detailed usage.

"""
ros2_multicast_doc = """
usage: ros2 multicast [-h]
                      Call `ros2 multicast <command> -h` for more detailed
                      usage. ...

Various multicast related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  receive  Receive a single UDP multicast packet
  send     Send a single UDP multicast packet

  Call `ros2 multicast <command> -h` for more detailed usage.

"""
ros2_node_doc = """
usage: ros2 node [-h]
                 Call `ros2 node <command> -h` for more detailed usage. ...

Various node related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  info  Output information about a node
  list  Output a list of available nodes

  Call `ros2 node <command> -h` for more detailed usage.

"""
ros2_param_doc = """
usage: ros2 param [-h]
                  Call `ros2 param <command> -h` for more detailed usage. ...

Various param related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  delete    Delete parameter
  describe  Show descriptive information about declared parameters
  dump      Show all of the parameters of a node in a YAML file format
  get       Get parameter
  list      Output a list of available parameters
  load      Load parameter file for a node
  set       Set parameter

  Call `ros2 param <command> -h` for more detailed usage.

"""
ros2_pkg_doc = """
usage: ros2 pkg [-h] Call `ros2 pkg <command> -h` for more detailed usage. ...

Various package related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  create       Create a new ROS 2 package
  executables  Output a list of package specific executables
  list         Output a list of available packages
  prefix       Output the prefix path of a package
  xml          Output the XML of the package manifest or a specific tag

  Call `ros2 pkg <command> -h` for more detailed usage.

"""
ros2_run_doc = """
usage: ros2 run [-h] [--prefix PREFIX] package_name executable_name ...

Run a package specific executable

positional arguments:
  package_name     Name of the ROS package
  executable_name  Name of the executable
  argv             Pass arbitrary arguments to the executable

options:
  -h, --help       show this help message and exit
  --prefix PREFIX  Prefix command, which should go before the executable.
                   Command must be wrapped in quotes if it contains spaces
                   (e.g. --prefix 'gdb -ex run --args').

"""
ros2_security_doc = """
usage: ros2 security [-h]
                     Call `ros2 security <command> -h` for more detailed
                     usage. ...

Various security related sub-commands

options:
  -h, --help            show this help message and exit

Commands:
  create_enclave      Create enclave
  create_key          DEPRECATED: Create enclave. Use create_enclave instead
  create_keystore     Create keystore
  create_permission   Create permission
  generate_artifacts  Generate keys and permission files from a list of identities and policy files
  generate_policy     Generate XML policy file from ROS graph data
  list_enclaves       List enclaves in keystore
  list_keys           DEPRECATED: List enclaves in keystore. Use list_enclaves instead

  Call `ros2 security <command> -h` for more detailed usage.

"""
ros2_service_doc = """
usage: ros2 service [-h] [--include-hidden-services]
                    Call `ros2 service <command> -h` for more detailed usage.
                    ...

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

  Call `ros2 service <command> -h` for more detailed usage.

"""
ros2_topic_doc = """
usage: ros2 topic [-h] [--include-hidden-topics]
                  Call `ros2 topic <command> -h` for more detailed usage. ...

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
ros2_wtf_doc = """
usage: ros2 wtf [-h] [--report | --report-failed] [--include-warnings]
                Call `ros2 wtf <command> -h` for more detailed usage. ...

Use `wtf` as alias to `doctor`

options:
  -h, --help            show this help message and exit
  --report, -r          Print all reports.
  --report-failed, -rf  Print reports of failed checks only.
  --include-warnings, -iw
                        Include warnings as failed checks. Warnings are
                        ignored by default.

Commands:
  hello  Check network connectivity between multiple hosts

  Call `ros2 wtf <command> -h` for more detailed usage.

"""


class RosCLIGenericTool(BaseModel):
    command: str = Field(
        ..., description="Command to be executed. Usage: ros2 <command>"
    )
    run = lambda self: run_ros2_cli_tool(f"ros2 {self.command}")


class ros2_action(RosCLIGenericTool):
    __doc__ = ros2_action_doc


class ros2_bag(RosCLIGenericTool):
    __doc__ = ros2_bag_doc


class ros2_component(RosCLIGenericTool):
    __doc__ = ros2_component_doc


class ros2_daemon(RosCLIGenericTool):
    __doc__ = ros2_daemon_doc


class ros2_doctor(RosCLIGenericTool):
    __doc__ = ros2_doctor_doc


class ros2_interface(RosCLIGenericTool):
    __doc__ = ros2_interface_doc


class ros2_launch(RosCLIGenericTool):
    __doc__ = ros2_launch_doc


class ros2_lifecycle(RosCLIGenericTool):
    __doc__ = ros2_lifecycle_doc


class ros2_multicast(RosCLIGenericTool):
    __doc__ = ros2_multicast_doc


class ros2_node(RosCLIGenericTool):
    __doc__ = ros2_node_doc


class ros2_param(RosCLIGenericTool):
    __doc__ = ros2_param_doc


class ros2_pkg(RosCLIGenericTool):
    __doc__ = ros2_pkg_doc


class ros2_run(RosCLIGenericTool):
    __doc__ = ros2_run_doc


class ros2_security(RosCLIGenericTool):
    __doc__ = ros2_security_doc


class ros2_service(RosCLIGenericTool):
    __doc__ = ros2_service_doc


class ros2_topic(RosCLIGenericTool):
    __doc__ = ros2_topic_doc


class ros2_wtf(RosCLIGenericTool):
    __doc__ = ros2_wtf_doc


def run_ros2_cli_tool(cmd):
    print(cmd)
    return subprocess.run(cmd, shell=True)


ros2_cli_tools = [
    ros2_action,
    ros2_bag,
    ros2_component,
    ros2_daemon,
    ros2_doctor,
    ros2_interface,
    ros2_launch,
    ros2_lifecycle,
    ros2_multicast,
    ros2_node,
    ros2_param,
    ros2_pkg,
    ros2_run,
    ros2_security,
    ros2_service,
    ros2_topic,
    ros2_wtf,
]
