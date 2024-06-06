import subprocess

from langchain_core.pydantic_v1 import BaseModel, Field
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image

from rai.communication.ros_communication import SingleMessageGrabber


class get_current_map(BaseModel):
    """Get the current map"""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")

    def run(self):
        """Gets the current map from the specified topic."""
        grabber = SingleMessageGrabber(self["topic"], OccupancyGrid, timeout_sec=10)  # type: ignore
        return grabber.get_data()


class get_current_position_relative_to_the_map(BaseModel):
    """Get the current position relative to the map"""

    topic: str = Field(..., description="Ros2 occupancy grid topic to subscribe to")

    def run(self):
        """Gets the current position relative to the map from the specified topic."""
        grabber = SingleMessageGrabber(self["topic"], Odometry, timeout_sec=10)  # type: ignore
        return grabber.get_data()


class get_current_image(BaseModel):
    """Get the current image"""

    topic: str = Field(..., description="Ros2 image topic to subscribe to")

    def run(self):
        """Gets the current image from the specified topic."""
        grabber = SingleMessageGrabber(self["topic"], Image, timeout_sec=10)  # type: ignore
        return grabber.get_data()


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
            f"pose: {{position: {{x: {self.x}, y: {self.y}, z: {self.z}}}}}}}'"
        )
        subprocess.run(cmd, shell=True)


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
    usage: ros2 interface list [-h] [-m] [-s] [-a]

    List all interface types available

    options:
      -h, --help          show this help message and exit
      -m, --only-msgs     Print out only the message types
      -s, --only-srvs     Print out only the service types
      -a, --only-actions  Print out only the action types
    usage: ros2 interface package [-h] package_name

    Output a list of available interface types within one package

    positional arguments:
      package_name  Name of the ROS package (e.g. 'example_interfaces')

    options:
      -h, --help    show this help message and exit
    usage: ros2 interface packages [-h] [-m] [-s] [-a]

    Output a list of packages that provide interfaces

    options:
      -h, --help          show this help message and exit
      -m, --only-msgs     Only list packages that generate messages
      -s, --only-srvs     Only list packages that generate services
      -a, --only-actions  Only list packages that generate actions
    usage: ros2 interface proto [-h] [--no-quotes] type

    Output an interface prototype

    positional arguments:
      type         Show an interface definition (e.g. 'example_interfaces/msg/String')

    options:
      -h, --help   show this help message and exit
      --no-quotes  if true output has no outer quotes.
    usage: ros2 interface show [-h] [--all-comments | --no-comments] type

    Output the interface definition

    positional arguments:
      type            Show an interface definition (e.g. 'example_interfaces/msg/String'). Passing '-' reads the argument from stdin (e.g. 'ros2 topic type /chatter | ros2 interface show -').

    options:
      -h, --help      show this help message and exit
      --all-comments  Show all comments, including for nested interface definitions
      --no-comments   Show no comments or whitespace
    """

    command: str = Field(..., description="The command to run")

    def run(self):
        command = f'ros2 interface {self["command"]}'
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

      Call `ros2 service <command> -h` for more detailed usage.
    usage: ros2 service call [-h] [-r N] service_name service_type [values]

    Call a service

    positional arguments:
      service_name    Name of the ROS service to call to (e.g. '/add_two_ints')
      service_type    Type of the ROS service (e.g. 'std_srvs/srv/Empty')
      values          Values to fill the service request with in YAML format (e.g. '{a: 1, b: 2}'), otherwise the service request will be published with default values

    options:
      -h, --help      show this help message and exit
      -r N, --rate N  Repeat the call at a specific rate in Hz
    usage: ros2 service find [-h] [-c] [--include-hidden-services] service_type

    Output a list of available services of a given type

    positional arguments:
      service_type          Name of the ROS service type to filter for (e.g. 'rcl_interfaces/srv/ListParameters')

    options:
      -h, --help            show this help message and exit
      -c, --count-services  Only display the number of services discovered
      --include-hidden-services
                            Consider hidden services as well
    usage: ros2 service list [-h] [--spin-time SPIN_TIME] [-s] [--no-daemon] [-t] [-c] [--include-hidden-services]

    Output a list of available services

    options:
      -h, --help            show this help message and exit
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
      --no-daemon           Do not spawn nor use an already running daemon
      -t, --show-types      Additionally show the service type
      -c, --count-services  Only display the number of services discovered
      --include-hidden-services
                            Consider hidden services as well
    usage: ros2 service type [-h] service_name

    Output a service's type

    positional arguments:
      service_name  Name of the ROS service to get type (e.g. '/talker/list_parameters')

    options:
      -h, --help    show this help message and exit
    """

    command: str = Field(..., description="The command to run")

    def run(self):
        command = f'ros2 service {self["command"]}'
        result = subprocess.run(command, shell=True, capture_output=True)
        return result


class ros2_topic(BaseModel):
    """
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
    usage: ros2 topic list [-h] [--spin-time SPIN_TIME] [-s] [--no-daemon] [-t] [-c] [--include-hidden-topics] [-v]

    Output a list of available topics

    options:
      -h, --help            show this help message and exit
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
      --no-daemon           Do not spawn nor use an already running daemon
      -t, --show-types      Additionally show the topic type
      -c, --count-topics    Only display the number of topics discovered
      --include-hidden-topics
                            Consider hidden topics as well
      -v, --verbose         List full details about each topic
    usage: ros2 topic echo [-h] [--spin-time SPIN_TIME] [-s] [--no-daemon] [--qos-profile {unknown,system_default,sensor_data,services_default,parameters,parameter_events,action_status_default}] [--qos-depth N]
                           [--qos-history {system_default,keep_last,keep_all,unknown}] [--qos-reliability {system_default,reliable,best_effort,unknown}]
                           [--qos-durability {system_default,transient_local,volatile,unknown}] [--csv] [--field FIELD] [--full-length] [--truncate-length TRUNCATE_LENGTH] [--no-arr] [--no-str] [--flow-style]
                           [--lost-messages] [--no-lost-messages] [--raw] [--filter FILTER_EXPR] [--once]
                           topic_name [message_type]

    Output messages from a topic

    positional arguments:
      topic_name            Name of the ROS topic to listen to (e.g. '/chatter')
      message_type          Type of the ROS message (e.g. 'std_msgs/msg/String')

    options:
      -h, --help            show this help message and exit
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
      --no-daemon           Do not spawn nor use an already running daemon
      --qos-profile {unknown,system_default,sensor_data,services_default,parameters,parameter_events,action_status_default}
                            Quality of service preset profile to subscribe with (default: sensor_data)
      --qos-depth N         Queue size setting to subscribe with (overrides depth value of --qos-profile option)
      --qos-history {system_default,keep_last,keep_all,unknown}
                            History of samples setting to subscribe with (overrides history value of --qos-profile option, default: keep_last)
      --qos-reliability {system_default,reliable,best_effort,unknown}
                            Quality of service reliability setting to subscribe with (overrides reliability value of --qos-profile option, default: Automatically match existing publishers )
      --qos-durability {system_default,transient_local,volatile,unknown}
                            Quality of service durability setting to subscribe with (overrides durability value of --qos-profile option, default: Automatically match existing publishers )
      --csv                 Output all recursive fields separated by commas (e.g. for plotting)
      --field FIELD         Echo a selected field of a message. Use '.' to select sub-fields. For example, to echo the position field of a nav_msgs/msg/Odometry message: 'ros2 topic echo /odom --field
                            pose.pose.position'
      --full-length, -f     Output all elements for arrays, bytes, and string with a length > '--truncate-length', by default they are truncated after '--truncate-length' elements with '...''
      --truncate-length TRUNCATE_LENGTH, -l TRUNCATE_LENGTH
                            The length to truncate arrays, bytes, and string to (default: 128)
      --no-arr              Don't print array fields of messages
      --no-str              Don't print string fields of messages
      --flow-style          Print collections in the block style (not available with csv format)
      --lost-messages       DEPRECATED: Does nothing
      --no-lost-messages    Don't report when a message is lost
      --raw                 Echo the raw binary representation
      --filter FILTER_EXPR  Python expression to filter messages that are printed. Expression can use Python builtins as well as m (the message).
      --once                Print the first message received and then exit.
    usage: ros2 topic pub [-h] [-r N] [-p N] [-1 | -t TIMES] [-w WAIT_MATCHING_SUBSCRIPTIONS] [--keep-alive N] [-n NODE_NAME]
                          [--qos-profile {unknown,system_default,sensor_data,services_default,parameters,parameter_events,action_status_default}] [--qos-depth N]
                          [--qos-history {system_default,keep_last,keep_all,unknown}] [--qos-reliability {system_default,reliable,best_effort,unknown}]
                          [--qos-durability {system_default,transient_local,volatile,unknown}] [--spin-time SPIN_TIME] [-s]
                          topic_name message_type [values]

    Publish a message to a topic

    positional arguments:
      topic_name            Name of the ROS topic to publish to (e.g. '/chatter')
      message_type          Type of the ROS message (e.g. 'std_msgs/String')
      values                Values to fill the message with in YAML format (e.g. 'data: Hello World'), otherwise the message will be published with default values

    options:
      -h, --help            show this help message and exit
      -r N, --rate N        Publishing rate in Hz (default: 1)
      -p N, --print N       Only print every N-th published message (default: 1)
      -1, --once            Publish one message and exit
      -t TIMES, --times TIMES
                            Publish this number of times and then exit
      -w WAIT_MATCHING_SUBSCRIPTIONS, --wait-matching-subscriptions WAIT_MATCHING_SUBSCRIPTIONS
                            Wait until finding the specified number of matching subscriptions. Defaults to 1 when using "-1"/"--once"/"--times", otherwise defaults to 0.
      --keep-alive N        Keep publishing node alive for N seconds after the last msg (default: 0.1)
      -n NODE_NAME, --node-name NODE_NAME
                            Name of the created publishing node
      --qos-profile {unknown,system_default,sensor_data,services_default,parameters,parameter_events,action_status_default}
                            Quality of service preset profile to publish)
      --qos-depth N         Queue size setting to publish with (overrides depth value of --qos-profile option)
      --qos-history {system_default,keep_last,keep_all,unknown}
                            History of samples setting to publish with (overrides history value of --qos-profile option, default: keep_last)
      --qos-reliability {system_default,reliable,best_effort,unknown}
                            Quality of service reliability setting to publish with (overrides reliability value of --qos-profile option, default: reliable)
      --qos-durability {system_default,transient_local,volatile,unknown}
                            Quality of service durability setting to publish with (overrides durability value of --qos-profile option, default: transient_local)
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
    usage: ros2 topic bw [-h] [--window WINDOW] [--spin-time SPIN_TIME] [-s] topic

    Display bandwidth used by topic

    positional arguments:
      topic                 Topic name to monitor for bandwidth utilization

    options:
      -h, --help            show this help message and exit
      --window WINDOW, -w WINDOW
                            maximum window size, in # of messages, for calculating rate (default: 100)
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
    usage: ros2 topic delay [-h] [--window WINDOW] [--spin-time SPIN_TIME] [-s] topic

    Display delay of topic from timestamp in header

    positional arguments:
      topic                 Topic name to calculate the delay for

    options:
      -h, --help            show this help message and exit
      --window WINDOW, -w WINDOW
                            window size, in # of messages, for calculating rate, string to (default: 10000)
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
    usage: ros2 topic hz [-h] [--window WINDOW] [--filter EXPR] [--wall-time] [--spin-time SPIN_TIME] [-s] topic_name

    Print the average publishing rate to screen

    positional arguments:
      topic_name            Name of the ROS topic to listen to (e.g. '/chatter')

    options:
      -h, --help            show this help message and exit
      --window WINDOW, -w WINDOW
                            window size, in # of messages, for calculating rate (default: 10000)
      --filter EXPR         only measure messages matching the specified Python expression
      --wall-time           calculates rate using wall time which can be helpful when clock is not published during simulation
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
    usage: ros2 topic type [-h] [--spin-time SPIN_TIME] [-s] [--no-daemon] topic_name

    Print a topic's type

    positional arguments:
      topic_name            Name of the ROS topic to get type (e.g. '/chatter')

    options:
      -h, --help            show this help message and exit
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
      --no-daemon           Do not spawn nor use an already running daemon
    usage: ros2 topic info [-h] [--spin-time SPIN_TIME] [-s] [--no-daemon] [--verbose] topic_name

    Print information about a topic

    positional arguments:
      topic_name            Name of the ROS topic to get info (e.g. '/chatter')

    options:
      -h, --help            show this help message and exit
      --spin-time SPIN_TIME
                            Spin time in seconds to wait for discovery (only applies when not using an already running daemon)
      -s, --use-sim-time    Enable ROS simulation time
      --no-daemon           Do not spawn nor use an already running daemon
      --verbose, -v         Prints detailed information like the node name, node namespace, topic type, GUID and QoS Profile of the publishers and subscribers to this topic
    """

    command: str = Field(..., description="The command to run")

    def run(self):
        command = f'ros2 topic {self["command"]}'
        result = subprocess.run(command, shell=True, capture_output=True)
        return result
