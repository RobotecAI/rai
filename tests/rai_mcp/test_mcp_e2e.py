# Copyright (C) 2025 Robotec.AI
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

"""`rai mcp serve` in a subprocess over stdio, against real ROS 2 topics."""

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("mcp")
pytest.importorskip("langchain_mcp_adapters")

import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path

import rai.mcp
from langchain_core.messages import ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.mcp import ros2_tools
from rai.tools import Effect
from std_msgs.msg import String

from tests.rai_mcp.scripted import ScriptedChatModel

RAI = str(Path(sys.executable).with_name("rai"))
DENY_MOTION = f"{Path(__file__).with_name('scripted.py')}:deny_motion"
NS = f"/rai_mcp_test_{uuid.uuid4().hex[:8]}"
STATUS = f"{NS}/status"
CMD_VEL = f"{NS}/cmd_vel"
TWIST = {
    "topic": CMD_VEL,
    "message": {"linear": {"x": 0.1}},
    "message_type": "geometry_msgs/msg/Twist",
}


def serve_args(*args: str) -> list[str]:
    return ["mcp", "serve", "--node-name", f"rai_mcp_{uuid.uuid4().hex[:8]}", *args]


@pytest.fixture(scope="module")
def connector():
    with ROS2Connector(f"rai_mcp_test_{uuid.uuid4().hex[:8]}") as connector:
        yield connector


@pytest.fixture(scope="module")
def status_publisher(connector):
    """Publishes "hello" on STATUS at 20 Hz while the module runs."""
    stop = threading.Event()

    def publish():
        while not stop.wait(0.05):
            connector.sync.publish(STATUS, String(data="hello"))

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join()


async def until_discovered(list_topics) -> None:
    """A fresh node takes up to a few seconds to discover STATUS (Fast DDS)."""
    async with asyncio.timeout(15):
        while STATUS not in await list_topics():
            await asyncio.sleep(0.2)


async def drive(*args: str):
    params = StdioServerParameters(
        command=RAI, args=serve_args(*args), env=dict(os.environ)
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        tools = {t.name: t for t in (await session.list_tools()).tools}

        async def list_topics() -> str:
            result = await session.call_tool("get_ros2_topics_names_and_types", {})
            return result.content[0].text if result.content else ""

        await until_discovered(list_topics)
        read = await session.call_tool(
            "receive_ros2_message", {"topic": STATUS, "timeout_sec": 10.0}
        )
        publish = await session.call_tool("publish_ros2_message", TWIST)
    return tools, read, publish


def test_ros2_tools_groups_and_read_only(connector):
    names = {t.name for t in ros2_tools(connector, ["topics"])}
    assert "publish_ros2_message" in names and "ros2_topic" not in names
    read_only = ros2_tools(connector, read_only=True)
    assert read_only and all(t.effect is Effect.READ for t in read_only)
    assert {"ros2_topic", "receive_ros2_message"} <= {t.name for t in read_only}
    with pytest.raises(ValueError, match="teleport"):
        ros2_tools(connector, ["teleport"])


def test_serve_lists_annotated_tools_and_reads_a_real_topic(status_publisher):
    tools, read, publish = asyncio.run(drive())
    hints = {
        name: (
            tools[name].annotations.readOnlyHint,
            tools[name].annotations.destructiveHint,
        )
        for name in (
            "receive_ros2_message",
            "call_ros2_service",
            "publish_ros2_message",
        )
    }
    assert hints == {
        "receive_ros2_message": (True, False),
        "call_ros2_service": (False, False),
        "publish_ros2_message": (False, True),
    }
    assert {"navigate_to_pose", "ros2_topic", "start_ros2_action"} <= set(tools)
    assert not read.isError and "hello" in read.content[0].text
    assert not publish.isError, publish.content


def test_egress_hooks_deny_actuation_over_mcp(status_publisher):
    _, read, publish = asyncio.run(drive("--setup", DENY_MOTION))
    assert not read.isError
    assert publish.isError
    assert "denied by no_motion: motion is off in this test" in publish.content[0].text


def test_read_only_server_has_no_write_tools():
    async def listed():
        params = StdioServerParameters(
            command=RAI, args=serve_args("--read-only"), env=dict(os.environ)
        )
        async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            return (await session.list_tools()).tools

    tools = asyncio.run(listed())
    assert tools and all(t.annotations.readOnlyHint for t in tools)
    assert "publish_ros2_message" not in {t.name for t in tools}


def test_agent_uses_the_served_tools(status_publisher):
    servers = {
        "robot": {
            "transport": "stdio",
            "command": RAI,
            "args": serve_args("--tools", "topics,actions", "--setup", DENY_MOTION),
            "env": dict(os.environ),
        }
    }

    async def run():
        async with rai.mcp.connect(servers) as tools:
            by_name = {t.name: t for t in tools}
            list_tool = by_name["get_ros2_topics_names_and_types"]

            async def list_topics() -> str:
                return str(await list_tool.ainvoke({}))

            await until_discovered(list_topics)
            agent = ReActAgent(
                ScriptedChatModel(
                    steps=[
                        [
                            (
                                "receive_ros2_message",
                                {"topic": STATUS, "timeout_sec": 10.0},
                            )
                        ],
                        [("publish_ros2_message", TWIST)],
                        "status is hello, driving is off",
                    ]
                ),
                tools,
            )
            answer = await agent.ainvoke("check the status, then drive")
            return {t.name: t.effect for t in tools}, answer, agent.history

    effects, answer, history = asyncio.run(run())
    assert effects["receive_ros2_message"] is Effect.READ
    assert effects["cancel_ros2_action"] is Effect.WRITE
    assert effects["publish_ros2_message"] is Effect.ACTUATE
    assert answer.text == "status is hello, driving is off"
    read, publish = [m for m in history if isinstance(m, ToolMessage)]
    assert read.status == "success" and "hello" in str(read.content)
    assert publish.status == "error" and "motion is off" in str(publish.content)
