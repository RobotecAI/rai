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

from subprocess import PIPE, Popen
from threading import Timer
from typing import List, Literal

from langchain_core.tools import tool

FORBIDDEN_CHARACTERS = ["&", ";", "|", "&&", "||", "(", ")", "<", ">", ">>", "<<"]


def run_with_timeout(cmd: List[str], timeout_sec: int):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        return stdout, stderr
    finally:
        timer.cancel()


def run_command(cmd: List[str], timeout: int = 5):
    # Validate command safety by checking for shell operators
    # Block potentially dangerous characters
    if any(char in " ".join(cmd) for char in FORBIDDEN_CHARACTERS):
        raise ValueError(
            "Command is not safe to run. The command contains forbidden characters."
        )
    stdout, stderr = run_with_timeout(cmd, timeout)
    output = {}
    if stdout:
        output["stdout"] = stdout.decode("utf-8")
    else:
        output["stdout"] = "Command returned no stdout output"
    if stderr:
        output["stderr"] = stderr.decode("utf-8")
    else:
        output["stderr"] = "Command returned no stderr output"
    return str(output)


@tool
def ros2_action(
    command: Literal["info", "list", "type"],
    action_name: str = "",
    timeout: int = 5,
):
    """Run a ROS2 action command
    Args:
        command: The action command to run (info/list/type)
        action_name: Name of the action (required for info and type)
        timeout: Command timeout in seconds
    """
    if command in ["info", "type"]:
        if not action_name:
            raise ValueError("Action name is required for info and type commands")

    cmd = ["ros2", "action", command]
    if action_name:
        cmd.append(action_name)
    return run_command(cmd, timeout)


@tool
def ros2_service(
    command: Literal["call", "find", "info", "list", "type"],
    service_name: str = "",
    timeout: int = 5,
):
    """Run a ROS2 service command
    Args:
        command: The service command to run
        service_name: Name of the service (required for call, info, and type)
        timeout: Command timeout in seconds
    """
    if command in ["call", "info", "type"]:
        if not service_name:
            raise ValueError(
                "Service name is required for call, info, and type commands"
            )

    cmd = ["ros2", "service", command]
    if service_name:
        cmd.append(service_name)
    return run_command(cmd, timeout)


@tool
def ros2_node(command: Literal["info", "list"], node_name: str = "", timeout: int = 5):
    """Run a ROS2 node command
    Args:
        command: The node command to run
        node_name: Name of the node (required for info)
        timeout: Command timeout in seconds
    """
    if command == "info":
        if not node_name:
            raise ValueError("Node name is required for info command")

    cmd = ["ros2", "node", command]
    if node_name:
        cmd.append(node_name)
    return run_command(cmd, timeout)


@tool
def ros2_param(
    command: Literal["delete", "describe", "dump", "get", "list", "set"],
    node_name: str = "",
    param_name: str = "",
    timeout: int = 5,
):
    """Run a ROS2 parameter command
    Args:
        command: The parameter command to run
        node_name: Name of the node
        param_name: Name of the parameter (required for get, set, delete)
        timeout: Command timeout in seconds
    """
    if command in ["get", "set", "delete"]:
        if not param_name:
            raise ValueError(
                "Parameter name is required for get, set, and delete commands"
            )

    cmd = ["ros2", "param", command]
    if node_name:
        cmd.append(node_name)
    if param_name:
        cmd.append(param_name)
    return run_command(cmd, timeout)


@tool
def ros2_interface(
    command: Literal["list", "package", "packages", "proto", "show"],
    interface_name: str = "",
    timeout: int = 5,
):
    """Run a ROS2 interface command
    Args:
        command: The interface command to run
        interface_name: Name of the interface (required for show, proto)
        timeout: Command timeout in seconds
    """
    if command in ["show", "proto"]:
        if not interface_name:
            raise ValueError("Interface name is required for show and proto commands")

    cmd = ["ros2", "interface", command]
    if interface_name:
        cmd.append(interface_name)
    return run_command(cmd, timeout)


@tool
def ros2_topic(
    command: Literal[
        "bw", "delay", "echo", "find", "hz", "info", "list", "pub", "type"
    ],
    topic_name: str = "",
    timeout: int = 5,
):
    """Run a ROS2 topic command
    Args:
        command: The topic command to run:
            - bw: Display bandwidth used by topic
            - delay: Display delay of topic from timestamp in header
            - echo: Output messages from a topic
            - find: Output a list of available topics of a given type
            - hz: Print the average publishing rate to screen
            - info: Print information about a topic
            - list: Output a list of available topics
            - pub: Publish a message to a topic
            - type: Print a topic's type
        topic_name: Name of the topic (required for all commands except list)
        timeout: Command timeout in seconds
    """
    if command in ["bw", "delay", "echo", "hz", "info", "pub", "type"]:
        if not topic_name:
            raise ValueError("Topic name is required for all commands except list")

    cmd = ["ros2", "topic", command]
    if topic_name:
        cmd.append(topic_name)
    return run_command(cmd, timeout)
