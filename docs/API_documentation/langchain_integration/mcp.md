# MCP

RAI is both an MCP server and an MCP client. Install the extra: `pip install "rai-core[mcp]"`.

## Server: `rai mcp serve`

`rai mcp serve` exposes the RAI ROS 2 tools over MCP (stdio by default), so any MCP client (Claude Desktop, Claude Code, an IDE, another agent) can use the robot.

```bash
rai mcp serve                                  # every tool group, stdio
rai mcp serve --read-only                      # READ tools only
rai mcp serve --tools topics,cli               # some groups
rai mcp serve --transport streamable-http --port 8000
rai mcp serve --setup my_robot.safety:install  # add hooks before serving
```

| Option                            | Meaning                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| `--tools`                         | Comma-separated groups: `topics`, `services`, `actions`, `nav2`, `cli`. Default: all.       |
| `--read-only`                     | Expose only tools whose effect is READ.                                                     |
| `--node-name`                     | ROS 2 node name of the server (default `rai_mcp`).                                          |
| `--transport`, `--host`, `--port` | `stdio` (default) or `streamable-http`.                                                     |
| `--setup MODULE:FN`               | Call `fn(server, connector)` before serving. `MODULE` may be `path/to/file.py`. Repeatable. |
| `--config`                        | Settings file (default `$RAI_CONFIG`, then `./rai.toml`).                                   |

Flags override the `[mcp]` table of the settings file:

```toml
[mcp]
groups = ["topics", "services", "actions"]
read_only = false
forbidden = ["/cmd_vel"]          # also readable, writable (see ROS 2 Tools)
setup = ["my_robot.safety:install"]
```

What a client sees:

-   **Annotations from effects.** READ tools have `readOnlyHint: true`, WRITE tools `destructiveHint: false`, ACTUATE tools `destructiveHint: true`. The effect and the ROS targets are also in the tool's `_meta.rai`.
-   **Errors are errors.** A tool failure, a denied hook or invalid arguments return `isError: true` with the reason as text. Image tools return MCP image content.
-   **Safety still applies.** Each call runs the `RaiTool`, so its tool hooks run, and the connector's egress hooks run under it. A setup function is the place to install them:

```python
# my_robot/safety.py
from rai.communication.ros2.egress import MOTION, Deny


def install(server, connector):
    connector.egress.add(lambda egress: Deny("motion is disabled"), types=MOTION)
```

The `server` argument is the `mcp.server.fastmcp.FastMCP` instance, so a setup function can also add resources, prompts or tools (`rai.mcp.add_tool(server, tool)`).

!!! note "ROS environment"

    MCP clients usually start stdio servers with a minimal environment. Source ROS 2 in the command, e.g. `"command": "bash", "args": ["-c", "source /opt/ros/jazzy/setup.bash && exec rai mcp serve"]`.

To serve your own tools, build the server in Python:

```python
from rai.mcp import build_server, ros2_tools

server = build_server([*ros2_tools(connector, ["topics"]), my_tool], name="my-robot")
server.run("stdio")
```

## Client: MCP tools in RAI agents

`rai.mcp.connect` opens one session per server and yields its tools as `RaiTool`s, so agents use them like any other tool and RAI tool hooks apply to them. Servers are `langchain-mcp-adapters` connections.

```python
import rai.mcp
from rai.agents import ReActAgent

servers = {"robot": {"transport": "stdio", "command": "rai", "args": ["mcp", "serve"]}}

async with rai.mcp.connect(servers) as tools:
    agent = ReActAgent("complex", tools)
    await agent.ainvoke("What is on /battery_state?")
```

-   **Effects** come from the server's annotations: `readOnlyHint` gives READ, `destructiveHint: false` gives WRITE, anything else (including no annotations) gives ACTUATE. They are hints from the server; for servers you do not trust, override them with `effects={"tool_name": Effect.ACTUATE}`.
-   **`load_tools(servers)`** returns tools that open a new session per call. They work from any event loop but keep no server state between calls (action goal ids, for example), and a stdio server is started per call. Prefer `connect` for agents.
-   Without `servers`, both read `[mcp.servers]` from `rai.toml`:

```toml
[mcp.servers.robot]
transport = "stdio"
command = "rai"
args = ["mcp", "serve", "--read-only"]
```

## API

::: rai.mcp.server.build_server

::: rai.mcp.server.ros2_tools

::: rai.mcp.server.add_tool

::: rai.mcp.client.connect

::: rai.mcp.client.load_tools
