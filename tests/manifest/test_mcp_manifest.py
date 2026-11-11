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

"""`rai mcp serve --setup rai.manifest.mcp:setup` over stdio: the manifest
resources and tool, against a live graph in this process."""

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("mcp")

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from geometry_msgs.msg import Twist
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rai.communication.ros2 import ROS2Connector
from std_msgs.msg import String

RAI = str(Path(sys.executable).with_name("rai"))
TOKEN = f"mfmcp{uuid.uuid4().hex[:8]}"
NS = f"/{TOKEN}"
STATUS, CMD_VEL = f"{NS}/status", f"{NS}/cmd_vel"


@pytest.fixture(scope="module")
def robot():
    with ROS2Connector(f"{TOKEN}_robot") as robot:
        robot.sync.publish(STATUS, String(data="ok"))
        robot.sync.publish(CMD_VEL, Twist())
        yield robot


@pytest.fixture
def config(tmp_path):
    spec = tmp_path / "robot.toml"
    spec.write_text(
        'name = "mcpbot"\nsafety = ["Stop when asked."]\n'
        f'[[capabilities]]\nname = "drive"\ntopics = ["{CMD_VEL}"]\n'
    )
    config = tmp_path / "rai.toml"
    config.write_text(f'[manifest]\nspec = "{spec}"\ninclude = ["*{TOKEN}*"]\n')
    return config


async def read_manifest(config: Path, *args: str):
    params = StdioServerParameters(
        command=RAI,
        args=[
            "mcp",
            "serve",
            "--node-name",
            f"rai_mcp_{uuid.uuid4().hex[:8]}",
            "--setup",
            "rai.manifest.mcp:setup",
            *args,
        ],
        env={**os.environ, "RAI_CONFIG": str(config)},
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        resources = {str(r.uri) for r in (await session.list_resources()).resources}
        tools = {t.name: t for t in (await session.list_tools()).tools}
        async with asyncio.timeout(15):  # discovery of the robot node
            while True:
                result = await session.read_resource("rai://manifest")  # type: ignore[arg-type]
                data = json.loads(result.contents[0].text)  # type: ignore[union-attr]
                if {STATUS, CMD_VEL} <= {t["name"] for t in data["topics"]}:
                    break
                await asyncio.sleep(0.2)
        text = await session.read_resource("rai://manifest/text")  # type: ignore[arg-type]
        called = await session.call_tool("get_robot_manifest", {"format": "text"})
    return resources, tools, data, text.contents[0].text, called  # type: ignore[union-attr]


def test_manifest_resource_and_tool(robot, config):
    resources, tools, data, text, called = asyncio.run(read_manifest(config))
    assert {"rai://manifest", "rai://manifest/text"} <= resources
    assert tools["get_robot_manifest"].annotations.readOnlyHint
    assert data["name"] == "mcpbot"
    topics = {t["name"]: t for t in data["topics"]}
    assert topics[CMD_VEL]["motion"] and topics[CMD_VEL]["capabilities"] == ["drive"]
    assert topics[CMD_VEL]["access"] == "rw"  # the served topic tools apply
    assert "publish_ros2_message" in data["generic_tools"]["topic"]
    assert f"{CMD_VEL} geometry_msgs/msg/Twist [MOTION, rw, cap:drive]" in text
    assert not called.isError and "# Robot manifest: mcpbot" in called.content[0].text


def test_read_only_server_annotates_read_access_only(robot, config):
    _, tools, data, _, _ = asyncio.run(read_manifest(config, "--read-only"))
    assert "publish_ros2_message" not in tools
    topics = {t["name"]: t for t in data["topics"]}
    assert topics[CMD_VEL]["access"] == "r" and topics[CMD_VEL]["motion"]
