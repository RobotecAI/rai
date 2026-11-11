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

"""Read-only `ros2` CLI tools. Write commands (topic pub, service call, action
send_goal, param set/delete/load) are not offered: they would bypass the access
lists and the connector's egress hooks."""

import asyncio
from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, BaseToolkit

from rai.tools.base import rai_tool
from rai.tools.hooks import Effect

FORBIDDEN_CHARACTERS = ["&", ";", "|", "&&", "||", "(", ")", "<", ">", ">>", "<<"]


async def run_command(cmd: List[str], timeout: float = 5) -> str:
    """Run cmd (no shell) and return its output; the process is killed after timeout."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be a positive number, got {timeout!r}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    if any(char in " ".join(cmd) for char in FORBIDDEN_CHARACTERS):
        raise ValueError(
            "Command is not safe to run. The command contains forbidden characters."
        )
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await proc.communicate()
    except TimeoutError:  # streaming commands (echo, hz) run until killed
        proc.kill()
        stdout, stderr = await proc.communicate()
    return str(
        {
            "stdout": stdout.decode() or "Command returned no stdout output",
            "stderr": stderr.decode() or "Command returned no stderr output",
        }
    )


async def _ros2(
    group: str, command: str, arguments: Optional[List[str]], timeout: float
) -> str:
    return await run_command(["ros2", group, command, *(arguments or [])], timeout)


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


@rai_tool(Effect.READ)
async def ros2_action(
    command: Literal["info", "list", "type"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
) -> str:
    """Run a read-only ROS2 action command
    Args:
        command: The action command to run (info/list/type)
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    return await _ros2("action", command, arguments, timeout)


@rai_tool(Effect.READ)
async def ros2_service(
    command: Literal["find", "info", "list", "type"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
) -> str:
    """Run a read-only ROS2 service command
    Args:
        command: The service command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    return await _ros2("service", command, arguments, timeout)


@rai_tool(Effect.READ)
async def ros2_node(
    command: Literal["info", "list"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
) -> str:
    """Run a ROS2 node command
    Args:
        command: The node command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    return await _ros2("node", command, arguments, timeout)


@rai_tool(Effect.READ)
async def ros2_param(
    command: Literal["describe", "get", "list"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
) -> str:
    """Run a read-only ROS2 parameter command
    Args:
        command: The parameter command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    return await _ros2("param", command, arguments, timeout)


@rai_tool(Effect.READ)
async def ros2_interface(
    command: Literal["list", "package", "packages", "proto", "show"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
) -> str:
    """Run a ROS2 interface command
    Args:
        command: The interface command to run
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    return await _ros2("interface", command, arguments, timeout)


@rai_tool(Effect.READ)
async def ros2_topic(
    command: Literal["bw", "delay", "echo", "find", "hz", "info", "list", "type"],
    arguments: Optional[List[str]] = None,
    timeout: int = 5,
) -> str:
    """Run a read-only ROS2 topic command
    Args:
        command: The topic command to run:
            - bw: Display bandwidth used by topic
            - delay: Display delay of topic from timestamp in header
            - echo: Output one message from a topic
            - find: Output a list of available topics of a given type
            - hz: Print the average publishing rate to screen
            - info: Print information about a topic
            - list: Output a list of available topics
            - type: Print a topic's type
        arguments: Additional arguments for the command as a list of strings
        timeout: Command timeout in seconds
    """
    if command == "echo" and "--once" not in (arguments or []):
        arguments = [*(arguments or []), "--once"]
    return await _ros2("topic", command, arguments, timeout)
