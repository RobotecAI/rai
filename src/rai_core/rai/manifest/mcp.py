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

"""The manifest on an MCP server: `rai mcp serve --setup rai.manifest.mcp:setup`.

Registers the resources `rai://manifest` (JSON) and `rai://manifest/text`
(the prompt rendering) and the `get_robot_manifest` tool. Each read rebuilds
the manifest only if the graph changed. The spec file and rai_whoami path come
from the `[manifest]` table of `rai.toml` (or `RAI_MANIFEST__SPEC`, ...).
"""

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from rai.mcp.server import add_tool, rai_tools

from .agent import GetRobotManifestTool
from .manifest import ManifestSource

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

MANIFEST_URI = "rai://manifest"
MANIFEST_TEXT_URI = "rai://manifest/text"


def register(server: FastMCP, source: ManifestSource) -> None:
    """Add the manifest resources and tool to `server`."""

    @server.resource(
        MANIFEST_URI,
        name="robot_manifest",
        description="The robot manifest: spec, rai_whoami and the live ROS 2 graph",
        mime_type="application/json",
    )
    def manifest_json() -> str:
        return source.current().model_dump_json(indent=1)

    @server.resource(
        MANIFEST_TEXT_URI,
        name="robot_manifest_text",
        description="The robot manifest as compact text for a prompt",
        mime_type="text/plain",
    )
    def manifest_text() -> str:
        return source.render()

    add_tool(server, GetRobotManifestTool(source=source))


def setup(server: FastMCP, connector: "ROS2Connector") -> None:
    """A `rai mcp serve --setup` function. The manifest annotates the graph
    with the tools the server serves, read at each build."""
    register(
        server, ManifestSource.from_settings(connector, tools=lambda: rai_tools(server))
    )
