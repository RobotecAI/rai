# Safety

RAI checks robot actions at two hook points, and both fail closed (a hook that denies, crashes or times out blocks the action):

-   **Agent side**: every `RaiTool` call runs `rai.tools.global_tool_hooks` and the tool's own hooks first. This covers agents, the MCP server and direct calls. A denial reaches the LLM as an error tool message, and the run continues.
-   **Connector side**: every publish, service call, goal and cancel runs the egress hooks (`rai.communication.ros2.egress.global_egress`, then the runtime's). This is the guarantee: it also covers code that never goes through a tool.

`rai.safety` ships four built-in hooks that work on both hook points:

| Hook               | What it does                                                                                                                                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EStop`            | A latch. While engaged it denies MOTION egress and ACTUATE tool calls. Engaging it cancels every active MOTION goal, including goals of a tool that is already running. Cancels are never gated.                       |
| `ConfirmActuation` | Opt-in. Asks a human before ACTUATE tool calls and MOTION egress. An approval leases the stream (`lease_s`), so a 30 Hz `cmd_vel` stream is approved once. If nobody answers before the timeout, the action is denied. |
| `RateLimit`        | A token bucket per tool name and per topic, service or action. Anything over the rate is denied, streams included.                                                                                                     |
| `AuditLog`         | An observer: it can never deny or block. Writes every decision as JSON lines and passes `AuditRecord`s to subscribers (the [flight recorder](../recorder/overview.md)).                                                |

## Usage

```python
from rai import safety

guard = safety.install(audit_path="audit.jsonl")  # EStop + ROS trigger + audit
guard.estop.engage("operator")
guard.estop.release()
guard.uninstall()
```

With confirmation and a rate limit:

```python
from rai.safety import HRIAsker, RateLimit, install

asker = HRIAsker(hri_connector, "/to_human", "/from_human")
guard = install(confirm=asker, rate_limits=[RateLimit(20, types={"geometry_msgs/msg/Twist"})])
```

`confirm=True` asks on the terminal (`cli_asker`). With no targets, `install` guards every runtime and every RaiTool. Pass connectors to guard only their runtimes.

## E-stop from ROS

| Interface                | Type                   | Effect                                                                                            |
| ------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------- |
| `/rai/estop` topic       | `std_msgs/msg/Bool`    | `true` engages, `false` releases                                                                  |
| `/rai/estop/set` service | `std_srvs/srv/SetBool` | `data: true` engages, `data: false` releases; the response's `message` is `engaged` or `released` |

```bash
ros2 topic pub --once /rai/estop std_msgs/msg/Bool "{data: true}"
ros2 service call /rai/estop/set std_srvs/srv/SetBool "{data: false}"
```

The e-stop is a software stop. It does not replace a hardware e-stop.

To stop the agent's turn as well, use `estop.on_engage(lambda _: agent.interrupt())`.

## Bypasses

Connectors do not expose their ROS 2 node, so every publish, service call, goal and cancel goes through the egress hooks. The ROS 2 CLI toolkit only offers read commands.

## API

::: rai.safety.install

::: rai.safety.EStop

::: rai.safety.ConfirmActuation

::: rai.safety.RateLimit

::: rai.safety.AuditLog

::: rai.safety.AuditRecord
