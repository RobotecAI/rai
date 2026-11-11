"""An external MCP server (official mcp SDK, stdio) that the slices consume as clients."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("ext")


@server.tool()
def robot_name() -> str:
    """Return the name of the robot."""
    return "rosbot-xl-7"


if __name__ == "__main__":
    server.run("stdio")
