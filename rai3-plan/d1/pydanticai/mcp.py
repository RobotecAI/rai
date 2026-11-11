"""6. MCP on PydanticAI: client via MCPToolset; server is a small toolset-to-MCP adapter
(PydanticAI ships no server side), so every toolset wrapper (FailClosed, approval) applies."""

from __future__ import annotations

import sys
from dataclasses import replace

import mcp.types as types
from fastmcp.client.transports import StdioTransport
from mcp.server.lowlevel import Server
from pydantic_ai import ApprovalRequired, RunContext, ToolFailed
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from ..common.robot import Effect
from .slice import effect_of


def build_server(toolset: AbstractToolset, name: str = "rai-d1-pydanticai") -> Server:
    server = Server(name)
    ctx = RunContext(
        deps=None, model=TestModel(), usage=RunUsage()
    )  # no agent run here

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=n,
                description=t.tool_def.description,
                inputSchema=t.tool_def.parameters_json_schema,
                annotations=types.ToolAnnotations(
                    readOnlyHint=effect_of(t.tool_def) is Effect.READ,
                    destructiveHint=effect_of(t.tool_def) is Effect.ACTUATE,
                ),
            )
            for n, t in (await toolset.get_tools(ctx)).items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        tool = (await toolset.get_tools(ctx))[name]
        args = tool.args_validator.validate_python(arguments)
        try:
            result = await toolset.call_tool(
                name, args, replace(ctx, tool_name=name), tool
            )
        except ToolFailed as e:
            raise RuntimeError(e.message) from None  # becomes isError=True
        except ApprovalRequired:
            raise RuntimeError(
                f"{name} needs human approval, not available over MCP"
            ) from None
        return [types.TextContent(type="text", text=str(result))]

    return server


async def serve_stdio(toolset: AbstractToolset) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(toolset)
    async with toolset, stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def client_toolset(script_path: str) -> MCPToolset:
    return MCPToolset(StdioTransport(sys.executable, [script_path]))
