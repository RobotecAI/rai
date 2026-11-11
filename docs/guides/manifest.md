# Robot manifest

`rai.manifest` tells the model what the robot is and what its running ROS 2 graph offers, so it stops guessing topic names and types. A `RobotManifest` merges three sources:

1. **The live ROS 2 graph**: nodes, topics, services and actions with their types. Noise is left out (`/rosout`, `/parameter_events`, parameter services, hidden names such as action internals). Each entry is annotated with what RAI can do there: `MOTION` if its type moves hardware (the egress `MOTION` set plus the spec's `motion_types`), `r`/`w` if the generic tools you give it may read or write it (their `readable`/`writable`/`forbidden` lists apply), and the configured tools that target it.
2. **A per-robot spec file** (TOML): name, description, frames, limits, safety notes, capabilities with the interfaces they map to, names to hide and names the model must not use.
3. **rai_whoami's description**, when there is one (see [Robot identity](whoami.md)). rai_whoami need not be installed: its `info.json` is read directly, and a missing or broken one only logs a warning.

## Spec file

Every field is optional; unknown keys, invalid ROS names and duplicate capabilities are errors.

```toml
name = "turtlebot3"
description = "Differential drive robot with a 360 degree lidar and a camera."
safety = ["Stop before any person closer than 0.5 m.", "Never drive in reverse."]
hide = ["*/debug*"]              # left out of the manifest (fnmatch, resolved names)
forbidden = ["*/factory_reset"]  # shown as FORBIDDEN; enforce with tool lists or egress hooks
motion_types = ["my_msgs/msg/WheelSpeeds"]  # added to egress.MOTION

[frames]
base = "base_link"
map = "map"

[limits]
max_linear_speed_mps = 0.22      # put the unit in the key
max_angular_speed_radps = 2.84

[[capabilities]]
name = "drive"
description = "Velocity control."
topics = ["cmd_vel"]             # resolved like tool names: "cmd_vel" -> "/cmd_vel"
examples = ["publish geometry_msgs/msg/Twist with linear.x 0.1 to creep forward"]

[[capabilities]]
name = "navigate"
actions = ["navigate_to_pose"]
```

A capability whose interfaces are not in the graph is marked `(missing)`.

## Building and refreshing

```python
from rai.communication.ros2 import ROS2Connector
from rai.manifest import ManifestSource
from rai.mcp import ros2_tools

connector = ROS2Connector()
tools = ros2_tools(connector, ["topics", "services", "actions"])
source = ManifestSource(
    connector,
    spec="robot.toml",
    whoami="path/to/whoami_dir",   # or info.json, or an EmbodimentInfo
    tools=tools,                   # or a callable returning them
)
manifest = source.current()        # RobotManifest
print(manifest.render(max_chars=4000))
manifest.model_dump_json()         # the JSON form
```

There is no background polling. `source.changed()` re-reads the graph from rmw's local discovery cache (no traffic) and compares a fingerprint of the filtered graph; `source.refresh()` returns a new manifest only when it changed; `source.current()` returns the manifest of the graph as it is now. Graph noise and hidden names do not count as changes.

`render(max_chars)` is deterministic and never longer than `max_chars` (about 4 characters per token). The description, safety notes, limits, frames and capabilities come first; then graph entries that matter (MOTION, forbidden, used by a capability or a configured tool) are kept before the rest, and whatever does not fit is counted in an "Omitted for length" line:

```text
# Robot manifest: testbot
A differential drive test robot with a camera.
## Safety
- Stop before any person closer than 0.5 m.
## Limits
- max_linear_speed_mps: 0.5
## Capabilities
- drive: Velocity control. Uses topics /cmd_vel. Examples: publish geometry_msgs/msg/Twist ...
- dock: Docking station. Uses actions /dock (missing).
## ROS 2 graph
Flags: MOTION moves hardware; FORBIDDEN do not use; r/w readable/writable with the generic tools.
Generic tools: actions: start_ros2_action; services: call_ros2_service; topics: publish_ros2_message, receive_ros2_message, get_ros2_image
### Actions (1)
/fibonacci example_interfaces/action/Fibonacci [w, cap:count]
### Services (1)
/factory_reset std_srvs/srv/Trigger [FORBIDDEN, w]
### Topics (3)
/camera sensor_msgs/msg/Image [tool:get_ros2_camera_image]
/cmd_vel geometry_msgs/msg/Twist [MOTION, rw, cap:drive]
/status std_msgs/msg/String [rw]
### Nodes (1)
/robot_driver
```

## In an agent

`ManifestPrompt` appends the manifest to the system prompt of every model call. It is never written into the history, so a refresh replaces it instead of growing the conversation. The graph is checked at most every `min_interval_s` (default 1 s) and the manifest is rebuilt only when it changed.

```python
from rai.agents import ReActAgent
from rai.manifest import GetRobotManifestTool, ManifestPrompt

agent = ReActAgent(
    "complex",
    [*tools, GetRobotManifestTool(source=source)],
    system_prompt="You operate the robot.",
    middleware=[ManifestPrompt(source)],
)
```

`GetRobotManifestTool` (`get_robot_manifest`, effect READ) lets an agent pull the manifest itself: `format="text"` for the rendering, `format="json"` for every entry.

## Over MCP

```bash
rai mcp serve --setup rai.manifest.mcp:setup
```

registers the resources `rai://manifest` (JSON) and `rai://manifest/text` and the `get_robot_manifest` tool. The graph is annotated with the tools the server serves, so `--read-only` gives `r` access only. From your own setup function, call `rai.manifest.mcp.register(server, source)` with a `ManifestSource` you configured.

## Settings

The `[manifest]` table of `rai.toml` configures `ManifestSource.from_settings` and the MCP setup (`RAI_MANIFEST__<FIELD>` env vars override it):

```toml
[manifest]
spec = "robot.toml"
whoami = "whoami/turtlebot3"
max_chars = 6000
include = []        # keep only matching names (fnmatch); empty: all
hide = ["/tf*"]     # added to the spec's hide
nodes = true        # list node names
```

See the [API reference](../api/manifest.md) for every parameter.
