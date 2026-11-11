"""6. MCP on LangChain: server via langchain_mcp_adapters.to_fastmcp, client via MultiServerMCPClient."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..common.guard import SafetyHook
from ..common.robot import Effect
from .slice import effect_of, guarded


def build_server(
    tools: Sequence[BaseTool], hooks: Sequence[SafetyHook] = ()
) -> FastMCP:
    mcp_tools = []
    for tool in tools:
        mcp_tool = to_fastmcp(
            guarded(tool, hooks, raise_denial=True)
        )  # middleware is not on this path
        effect = effect_of(tool)
        mcp_tool.annotations = ToolAnnotations(
            readOnlyHint=effect is Effect.READ, destructiveHint=effect is Effect.ACTUATE
        )
        mcp_tools.append(mcp_tool)
    return FastMCP("rai-d1-langchain", tools=mcp_tools)


async def load_client_tools(script_path: str) -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "ext": {
                "command": sys.executable,
                "args": [script_path],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()
