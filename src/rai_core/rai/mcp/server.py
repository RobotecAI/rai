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

"""RAI tools on an MCP server (`rai mcp serve`).

Tools are registered on the `mcp` SDK's FastMCP directly, not through
`langchain_mcp_adapters.to_fastmcp`, which drops annotations and reports a
handled tool error as a success. Every call goes through the `RaiTool`, so tool
hooks run; connector egress hooks run in the ROS runtime below it.
"""

import importlib
import importlib.util
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import (
    CallToolResult,
    ContentBlock,
    ImageContent,
    TextContent,
    ToolAnnotations,
)
from pydantic import ConfigDict

from rai.tools import Effect, RaiTool, from_langchain

from .settings import TOOL_GROUPS, MCPSettings, ToolGroup

if TYPE_CHECKING:
    from rai.communication.ros2 import ROS2Connector

logger = logging.getLogger(__name__)

Setup = Callable[[FastMCP, "ROS2Connector"], None]


def annotations(effect: Effect) -> ToolAnnotations:
    """MCP hints for an effect; `rai.tools.effect_from_mcp_annotations` inverts it."""
    return ToolAnnotations(
        readOnlyHint=effect is Effect.READ,
        destructiveHint=effect is Effect.ACTUATE,
    )


class _Arguments(ArgModelBase):
    """Pass arguments through: the RaiTool validates them and reports errors."""

    model_config = ConfigDict(extra="allow")

    def model_dump_one_level(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


def mcp_tool(tool: BaseTool) -> Tool:
    """A FastMCP tool that calls `tool`. Non-RAI tools are wrapped with
    `from_langchain` (effect from their MCP annotations, else ACTUATE), so hooks
    always run."""
    rai_tool = tool if isinstance(tool, RaiTool) else from_langchain(tool)
    schema = rai_tool.tool_call_schema
    parameters = schema if isinstance(schema, dict) else schema.model_json_schema()

    async def call(**arguments: Any) -> CallToolResult:
        return await call_tool(rai_tool, arguments)

    return Tool(
        fn=call,
        name=rai_tool.name,
        description=rai_tool.description,
        parameters=parameters,
        fn_metadata=FuncMetadata(arg_model=_Arguments),
        is_async=True,
        annotations=annotations(rai_tool.effect),
        meta={"rai": {"effect": str(rai_tool.effect), "targets": rai_tool.targets}},
    )


def add_tool(server: FastMCP, tool: BaseTool) -> None:
    """Register `tool` on a running or built server (e.g. from a setup function)."""
    mcp = mcp_tool(tool)
    server._tool_manager._tools[mcp.name] = mcp  # FastMCP only adds functions


async def call_tool(tool: RaiTool, arguments: dict[str, Any]) -> CallToolResult:
    """Run `tool` as a tool call; an error tool message becomes `isError`."""
    call = {"type": "tool_call", "name": tool.name, "args": arguments, "id": "mcp"}
    message = await tool.ainvoke(call)
    content = _content(message.content) + _artifact(message.artifact)
    return CallToolResult(content=content, isError=message.status == "error")


def _content(content: Any) -> list[ContentBlock]:
    if isinstance(content, str):
        return [TextContent(type="text", text=content)] if content else []
    blocks: list[ContentBlock] = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, str):
            blocks.append(TextContent(type="text", text=block))
        elif isinstance(block, dict) and block.get("type") == "text":
            blocks.append(TextContent(type="text", text=block["text"]))
        elif (
            isinstance(block, dict)
            and block.get("type") == "image"
            and "base64" in block
        ):
            blocks.append(_image(block["base64"], block.get("mime_type")))
        else:
            blocks.append(TextContent(type="text", text=json.dumps(block, default=str)))
    return blocks


def _artifact(artifact: Any) -> list[ContentBlock]:
    """RAI image tools return {"images": [base64, ...]} as their artifact."""
    if not isinstance(artifact, dict):
        return []
    return [_image(data) for data in artifact.get("images", ())]


def _image(data: str, mime_type: str | None = None) -> ImageContent:
    if mime_type is None:
        mime_type = "image/png" if data.startswith("iVBOR") else "image/jpeg"
    return ImageContent(type="image", data=data, mimeType=mime_type)


def build_server(
    tools: Iterable[BaseTool], *, name: str = "rai", **fastmcp: Any
) -> FastMCP:
    """A FastMCP server with `tools`; `fastmcp` goes to `FastMCP` (host, port, ...)."""
    return FastMCP(name, tools=[mcp_tool(t) for t in tools], **fastmcp)


def ros2_tools(
    connector: "ROS2Connector",
    groups: Iterable[ToolGroup] = TOOL_GROUPS,
    *,
    read_only: bool = False,
    readable: list[str] | None = None,
    writable: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> list[RaiTool]:
    """The RAI ROS 2 tools of `groups` on `connector`; with read_only, READ tools only."""
    from rai.tools.ros2 import (
        Nav2Toolkit,
        ROS2ActionToolkit,
        ROS2DiagnosticsToolkit,
        ROS2ServicesToolkit,
        ROS2TopicsToolkit,
    )

    access = dict(
        connector=connector, readable=readable, writable=writable, forbidden=forbidden
    )
    toolkits = {
        "topics": lambda: ROS2TopicsToolkit(**access),
        "services": lambda: ROS2ServicesToolkit(**access),
        "actions": lambda: ROS2ActionToolkit(**access),
        "nav2": lambda: Nav2Toolkit(connector=connector),
        "diagnostics": lambda: ROS2DiagnosticsToolkit(**access),
    }
    unknown = set(groups) - set(toolkits)
    if unknown:
        raise ValueError(
            f"Unknown tool groups {sorted(unknown)}, expected {TOOL_GROUPS}"
        )
    tools = [t for g in dict.fromkeys(groups) for t in toolkits[g]().get_tools()]
    rai_tools = [t if isinstance(t, RaiTool) else from_langchain(t) for t in tools]
    return [t for t in rai_tools if not read_only or t.effect is Effect.READ]


def load_setup(spec: str) -> Setup:
    """`"module:function"` or `"path/to/file.py:function"`."""
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise ValueError(f"Setup must be 'module:function', got {spec!r}")
    if target.endswith(".py"):
        file_spec = importlib.util.spec_from_file_location(Path(target).stem, target)
        if file_spec is None or file_spec.loader is None:
            raise ImportError(f"Cannot load {target}")
        module = importlib.util.module_from_spec(file_spec)
        file_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


def serve(settings: MCPSettings | None = None) -> None:
    """Serve the RAI ROS 2 tools until the client disconnects (stdio) or Ctrl-C."""
    from rai.communication.ros2 import ROS2Connector

    settings = settings or MCPSettings()
    connector = ROS2Connector(settings.node_name)
    try:
        tools = ros2_tools(
            connector,
            settings.groups,
            read_only=settings.read_only,
            readable=settings.readable,
            writable=settings.writable,
            forbidden=settings.forbidden,
        )
        server = build_server(
            tools, name=settings.name, host=settings.host, port=settings.port
        )
        for spec in settings.setup:
            load_setup(spec)(server, connector)
        logger.info("Serving %d RAI tools over %s", len(tools), settings.transport)
        server.run(settings.transport)
    finally:
        connector.close()
