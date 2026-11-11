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

"""The `rai` command. `rai mcp serve` is its only command so far."""

import argparse
import logging
import sys
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rai")
    commands = parser.add_subparsers(dest="command", required=True)
    mcp = commands.add_parser("mcp", help="MCP server").add_subparsers(
        dest="mcp_command", required=True
    )
    serve = mcp.add_parser(
        "serve",
        help="Serve the RAI ROS 2 tools over MCP",
        description="Options override the [mcp] table of the settings file.",
    )
    serve.add_argument(
        "--config", help="settings file (default: $RAI_CONFIG, ./rai.toml)"
    )
    serve.add_argument(
        "--tools",
        help="comma-separated tool groups: topics,services,actions,nav2,diagnostics (default: all)",
    )
    serve.add_argument(
        "--read-only", action="store_true", default=None, help="expose READ tools only"
    )
    serve.add_argument("--node-name", help="ROS 2 node name (default: rai_mcp)")
    serve.add_argument("--transport", choices=["stdio", "streamable-http"])
    serve.add_argument("--host", help="streamable-http host (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, help="streamable-http port (default: 8000)")
    serve.add_argument(
        "--setup",
        action="append",
        metavar="MODULE:FN",
        help="call fn(server, connector) before serving, e.g. to add egress hooks"
        " (module or path/to/file.py; repeatable)",
    )
    return parser


def _serve(args: argparse.Namespace) -> None:
    from pydantic import ValidationError

    from rai.mcp.server import serve
    from rai.mcp.settings import MCPSettings, mcp_settings

    settings = mcp_settings(args.config)
    overrides = {
        "groups": args.tools and [g.strip() for g in args.tools.split(",") if g],
        "read_only": args.read_only,
        "node_name": args.node_name,
        "transport": args.transport,
        "host": args.host,
        "port": args.port,
        "setup": args.setup and [*settings.setup, *args.setup],
    }
    try:
        settings = MCPSettings.model_validate(
            settings.model_dump()
            | {k: v for k, v in overrides.items() if v is not None}
        )
    except ValidationError as e:
        sys.exit(f"rai mcp serve: {e}")
    serve(settings)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    # stdout carries the MCP protocol on stdio: logs go to stderr
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if args.command == "mcp" and args.mcp_command == "serve":
        _serve(args)


if __name__ == "__main__":
    main()
