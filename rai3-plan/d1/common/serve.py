"""MCP server entry point: python -m d1.common.serve {langchain|pydanticai} {fake|ros} [deny|crash]"""

import asyncio
import sys

from .guard import crashing_hook, deny_actuate
from .robot import ConnectorRobot, FakeRobot


def main() -> None:
    framework, robot_kind = sys.argv[1], sys.argv[2]
    hooks = {"deny": [deny_actuate], "crash": [crashing_hook]}.get(
        sys.argv[3] if len(sys.argv) > 3 else "", []
    )
    if robot_kind == "ros":
        import rclpy
        from rai.communication.ros2.connectors import ROS2Connector

        rclpy.init()
        io = ConnectorRobot(ROS2Connector(executor_type="single_threaded"))
    else:
        io = FakeRobot()
    if framework == "langchain":
        from ..langchain.mcp import build_server
        from ..langchain.slice import make_tools

        build_server(make_tools(io), hooks).run("stdio")
    else:
        from ..pydanticai.mcp import serve_stdio
        from ..pydanticai.slice import guard, make_tools

        asyncio.run(
            serve_stdio(guard([make_tools(io)], hooks, approve_actuation=False))
        )


if __name__ == "__main__":
    main()
