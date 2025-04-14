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
from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, BaseToolkit, tool

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


class ROS2CLIToolkit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        return [
            ros2_action,
            ros2_service,
            ros2_node,
            ros2_param,
            ros2_interface,
            ros2_topic,
        ]


@tool
def ros2_action(
    command: Literal["info", "list", "type", "send_goal"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
):
    """Run a ROS2 action command
    Args:
        command: The action command to run (info/list/type)
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    cmd = ["ros2", "action", command]
    if arguments:
        cmd.extend(arguments)
    return run_command(cmd, timeout)


@tool
def ros2_service(
    command: Literal["call", "find", "info", "list", "type"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
):
    """Run a ROS2 service command
    Args:
        command: The service command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    cmd = ["ros2", "service", command]
    if arguments:
        cmd.extend(arguments)
    return run_command(cmd, timeout)


@tool
def ros2_node(
    command: Literal["info", "list"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
):
    """Run a ROS2 node command
    Args:
        command: The node command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    cmd = ["ros2", "node", command]
    if arguments:
        cmd.extend(arguments)
    return run_command(cmd, timeout)


@tool
def ros2_param(
    command: Literal["delete", "describe", "dump", "get", "list", "set"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
):
    """Run a ROS2 parameter command
    Args:
        command: The parameter command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    cmd = ["ros2", "param", command]
    if arguments:
        cmd.extend(arguments)
    return run_command(cmd, timeout)


@tool
def ros2_interface(
    command: Literal["list", "package", "packages", "proto", "show"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
):
    """Run a ROS2 interface command
    Args:
        command: The interface command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    cmd = ["ros2", "interface", command]
    if arguments:
        cmd.extend(arguments)
    return run_command(cmd, timeout)


@tool
def ros2_topic(
    command: Literal[
        "bw", "delay", "echo", "find", "hz", "info", "list", "pub", "type"
    ],
    arguments: Optional[List[str]] = None,
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
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    cmd = ["ros2", "topic", command]
    if arguments:
        cmd.extend(arguments)
    return run_command(cmd, timeout)
