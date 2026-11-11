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

"""The `[mcp]` table of `rai.toml`: `rai mcp serve` options and the client's servers."""

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from rai.settings import section

ToolGroup = Literal["topics", "services", "actions", "nav2", "diagnostics"]
TOOL_GROUPS: tuple[ToolGroup, ...] = (
    "topics",
    "services",
    "actions",
    "nav2",
    "diagnostics",
)


class MCPSettings(BaseModel):
    """`rai mcp serve` reads every field but `servers`; `rai.mcp.connect` and
    `rai.mcp.load_tools` read `servers` (`langchain-mcp-adapters` connections)."""

    model_config = ConfigDict(extra="forbid")

    name: str = "rai"
    groups: list[ToolGroup] = list(TOOL_GROUPS)
    read_only: bool = False  # expose READ tools only
    node_name: str = "rai_mcp"
    readable: list[str] | None = None
    writable: list[str] | None = None
    forbidden: list[str] | None = None
    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    # "module:function" or "path/to/file.py:function", called as fn(server, connector)
    setup: list[str] = []
    # e.g. {"robot": {"transport": "stdio", "command": "rai", "args": ["mcp", "serve"]}}
    servers: dict[str, dict[str, Any]] = {}


def mcp_settings(path: str | os.PathLike[str] | None = None) -> MCPSettings:
    """The `[mcp]` table of the settings file; `RAI_MCP__<FIELD>` env vars override it."""
    return section("mcp", MCPSettings, path)
