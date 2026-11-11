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

"""MCP servers' tools as `RaiTool`s, through `langchain-mcp-adapters`.

Effects come from the servers' annotations (`readOnlyHint` -> READ,
`destructiveHint=False` -> WRITE, anything else -> ACTUATE). They are hints from
the server: pass `effects` to override them for servers you do not trust.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool

from rai.tools import Effect, RaiTool, from_langchain

from .settings import mcp_settings

Servers = Mapping[str, Mapping[str, Any]]


def _servers(servers: Servers | None) -> dict[str, Any]:
    servers = mcp_settings().servers if servers is None else servers
    if not servers:
        raise ValueError(
            "No MCP servers: pass servers or set [mcp.servers] in rai.toml"
        )
    return {name: dict(conn) for name, conn in servers.items()}


def _wrap(tools: list[BaseTool], effects: Mapping[str, Effect]) -> list[RaiTool]:
    return [from_langchain(t, effect=effects.get(t.name)) for t in tools]


async def load_tools(
    servers: Servers | None = None,
    *,
    effects: Mapping[str, Effect] = {},
    prefix: bool = False,
) -> list[RaiTool]:
    """Tools of `servers` (default: `[mcp.servers]`). Every call opens a new
    session, so the tools work from any event loop but keep no server state
    (a stdio server is started per call); prefer `connect` for agents."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(_servers(servers), tool_name_prefix=prefix)
    return _wrap(await client.get_tools(), effects)


@asynccontextmanager
async def connect(
    servers: Servers | None = None,
    *,
    effects: Mapping[str, Effect] = {},
    prefix: bool = False,
) -> AsyncIterator[list[RaiTool]]:
    """Open one session per server for the block and yield their tools.

    The tools must be called on the loop that entered the block.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    connections = _servers(servers)
    client = MultiServerMCPClient(connections)
    async with AsyncExitStack() as stack:
        tools: list[BaseTool] = []
        for name in connections:
            session = await stack.enter_async_context(client.session(name))
            tools += await load_mcp_tools(
                session, server_name=name, tool_name_prefix=prefix
            )
        yield _wrap(tools, effects)
