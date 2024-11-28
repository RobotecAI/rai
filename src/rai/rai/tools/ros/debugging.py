import shlex
from subprocess import PIPE, Popen
from threading import Timer
from typing import Literal

from langchain_core.tools import tool

FORBIDDEN_CHARACTERS = ["&", ";", "|", "&&", "||", "(", ")", "<", ">", ">>", "<<"]


def run_with_timeout(cmd: str, timeout_sec: int):
    command = shlex.split(cmd)
    proc = Popen(command, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        return stdout, stderr
    finally:
        timer.cancel()


def run_command(cmd: str, timeout: int = 5):
    # Validate command safety by checking for shell operators
    # Block potentially dangerous characters
    if any(char in cmd for char in FORBIDDEN_CHARACTERS):
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
    command: Literal["info", "list", "send_goal", "type"],
    action_name: str = "",
    timeout: int = 5,
):
    """Run a ROS2 action command
    Args:
        command: The action command to run (info/list/send_goal/type)
        action_name: Name of the action (required for info, send_goal, and type)
        timeout: Command timeout in seconds
    """
    cmd = f"ros2 action {command}"
    if action_name:
        cmd += f" {action_name}"
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
    cmd = f"ros2 service {command}"
    if service_name:
        cmd += f" {service_name}"
    return run_command(cmd, timeout)


@tool
def ros2_node(command: Literal["info", "list"], node_name: str = "", timeout: int = 5):
    """Run a ROS2 node command
    Args:
        command: The node command to run
        node_name: Name of the node (required for info)
        timeout: Command timeout in seconds
    """
    cmd = f"ros2 node {command}"
    if node_name:
        cmd += f" {node_name}"
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
    cmd = f"ros2 param {command}"
    if node_name:
        cmd += f" {node_name}"
        if param_name:
            cmd += f" {param_name}"
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
    cmd = f"ros2 interface {command}"
    if interface_name:
        cmd += f" {interface_name}"
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
    cmd = f"ros2 topic {command}"
    if topic_name:
        cmd += f" {topic_name}"
    return run_command(cmd, timeout)
