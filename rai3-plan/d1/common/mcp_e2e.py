"""Shared MCP stdio end-to-end check: start a slice's server as a subprocess and drive it."""

import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PLAN_DIR = Path(__file__).resolve().parents[2]
EXT_SERVER = str(Path(__file__).with_name("ext_mcp_server.py"))


async def drive_server(framework: str, robot: str, hook: str = "") -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "d1.common.serve", framework, robot, *([hook] if hook else [])],
        cwd=str(PLAN_DIR),
        env=dict(os.environ),
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = {t.name: t for t in (await s.list_tools()).tools}
        read = await s.call_tool("get_latest_message", {"topic": "/d1/status"})
        drive = await s.call_tool(
            "publish_twist", {"linear_x": 0.2, "angular_z": 0.0, "duration_s": 0.5}
        )
    return {"tools": tools, "read": read, "drive": drive}
