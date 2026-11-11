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

"""MCP as server (`rai mcp serve`) and client. Needs the `rai-core[mcp]` extra."""

import importlib

_LAZY = {
    "MCPSettings": "rai.mcp.settings",
    "mcp_settings": "rai.mcp.settings",
    "TOOL_GROUPS": "rai.mcp.settings",
    "add_tool": "rai.mcp.server",
    "annotations": "rai.mcp.server",
    "build_server": "rai.mcp.server",
    "call_tool": "rai.mcp.server",
    "mcp_tool": "rai.mcp.server",
    "ros2_tools": "rai.mcp.server",
    "serve": "rai.mcp.server",
    "connect": "rai.mcp.client",
    "load_tools": "rai.mcp.client",
}

__all__ = [*_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
