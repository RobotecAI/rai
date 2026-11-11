# Safety

RAI checks robot actions at two hook points. Both fail closed: a hook that denies, raises or times out blocks the action.

| Hook point         | Runs on                                                                                          | Covers                                                       | A denial                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------- |
| **Agent side**     | every `RaiTool` call: `rai.tools.global_tool_hooks`, then `tool.hooks`                           | agents, the MCP server, direct tool calls                    | becomes an error tool message; the run continues                          |
| **Connector side** | every publish, service call, goal and cancel: `global_egress`, then the runtime's `egress` chain | everything that reaches ROS through a connector, tool or not | raises `EgressDenied` in the caller (an error tool message inside a tool) |

The connector side is the guarantee: it also covers code that never goes through a tool, and it matches on message types, which remapping does not change. The agent side gives the model an early, readable denial and can decide on tool names and arguments. Connectors do not expose their ROS 2 node, so there is no path to ROS around the egress hooks.

The built-in hooks below install on both points. This page covers what they do; [Tools and effects](tools.md#tool-hooks) covers writing your own tool hooks.

## Built-in hooks

| Hook               | What it does                                                                                                                                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EStop`            | A latch. While engaged it denies MOTION egress and ACTUATE tool calls. Engaging it cancels every active MOTION goal, including goals of a tool that is already running. Cancels are never gated.                       |
| `ConfirmActuation` | Opt-in. Asks a human before ACTUATE tool calls and MOTION egress. An approval leases the stream (`lease_s`), so a 30 Hz `cmd_vel` stream is approved once. If nobody answers before `timeout_s`, the action is denied. |
| `RateLimit`        | A token bucket per tool name and per topic, service or action. Anything over the rate is denied, streams included.                                                                                                     |
| `AuditLog`         | An observer: it never denies or blocks. Writes every decision of both hook points as JSON lines and passes `AuditRecord`s to subscribers such as the [flight recorder](../guides/recorder.md).                         |

`rai.safety.install` sets up a standard set in one call:

<!-- test: run -->

```python
from rai import safety

guard = safety.install(audit_path="audit.jsonl")  # EStop with its ROS trigger, and an audit log
guard.estop.engage("operator")
print(guard.estop.engaged)
guard.estop.release()
guard.uninstall()
```

With confirmation over HRI topics and a rate limit on velocity commands:

```python
from rai.communication.ros2 import ROS2HRIConnector
from rai.safety import HRIAsker, RateLimit, install

asker = HRIAsker(ROS2HRIConnector(), "/to_human", "/from_human")  # "yes" approves
guard = install(confirm=asker, rate_limits=[RateLimit(20, types={"geometry_msgs/msg/Twist"})])
```

-   `confirm=True` asks on the terminal (`cli_asker`).
-   With no targets, `install` guards every runtime and every `RaiTool`. Pass connectors (`install(connector)`) to guard only their runtimes.
-   The order is EStop, rate limits, ConfirmActuation, then EStop again, so nobody is asked about an action that would be denied anyway, and an e-stop engaged while a question is open still wins.
-   `estop.on_engage(lambda reason: agent.interrupt())` also stops the agent's current turn.

### E-stop from ROS

| Interface                | Type                   | Effect                                                                                            |
| ------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------- |
| `/rai/estop` topic       | `std_msgs/msg/Bool`    | `true` engages, `false` releases                                                                  |
| `/rai/estop/set` service | `std_srvs/srv/SetBool` | `data: true` engages, `data: false` releases; the response's `message` is `engaged` or `released` |

```bash
ros2 topic pub --once /rai/estop std_msgs/msg/Bool "{data: true}"
ros2 service call /rai/estop/set std_srvs/srv/SetBool "{data: false}"
```

The e-stop is a software stop. It does not replace a hardware e-stop.

## The MOTION set

`rai.communication.ros2.MOTION` names the message, service and action types that move hardware: `geometry_msgs` `Twist` and `TwistStamped`, `trajectory_msgs/msg/JointTrajectory`, `FollowJointTrajectory` and `GripperCommand`, the Nav2 actions (`NavigateToPose`, `NavigateThroughPoses`, `FollowPath`, `FollowWaypoints`, `Spin`, `BackUp`, ...), MoveIt's `MoveGroup` and `ExecuteTrajectory`, and `rai_interfaces/srv/ManipulatorMoveTo`. Extend it with a set union for your robot's own types: `MOTION | {"my_msgs/msg/WheelSpeeds"}`. The [robot manifest](../guides/manifest.md) spec can add types too.

## Connector side: egress hooks

An egress hook is `(Egress) -> Allow | Deny`, sync or async. `Egress` has the `kind` (`publish`, `service`, `goal` or `cancel`), the resolved `name`, the `type` (`"pkg/msg/Name"`), the read-only `payload` and the `origin` (the tool that sent it, if any).

<!-- test: run -->

```python
from geometry_msgs.msg import Twist, Vector3

from rai.communication.ros2 import MOTION, EgressDenied, ROS2Connector
from rai.communication.ros2.egress import Allow, Deny, Egress


def speed_limit(egress: Egress) -> Allow | Deny:
    if abs(egress.payload.linear.x) > 0.5:
        return Deny("faster than 0.5 m/s")
    return Allow()


with ROS2Connector() as connector:
    connector.egress.add(speed_limit, types={"geometry_msgs/msg/Twist"})
    connector.sync.publish("/cmd_vel", Twist())  # allowed
    try:
        connector.sync.publish("/cmd_vel", Twist(linear=Vector3(x=1.0)))
    except EgressDenied as e:
        print(e)
```

-   **Matching.** `add(hook, types=..., names=..., kinds=...)` runs the hook only on egress whose type is in `types` or whose resolved name matches a pattern in `names` (fnmatch, such as `"/*/cmd_vel"`), restricted to `kinds`. Match on types first: names get remapped, types do not. With neither, the hook sees every egress of its kinds. Egress no hook matches pays nothing.
-   **Cancels.** The default kinds are `publish`, `service` and `goal`. Cancels stop motion, so a hook gates them only if it asks for `kinds=ALL_KINDS`.
-   **Leases.** `Allow(lease_s=2.0)` approves the same kind, name and type for that hook for 2 s, so a stream is not re-approved per message. `chain.revoke()` drops leases.
-   **Chains.** `connector.egress` is the chain of the connector's runtime, shared by every connector on it. `rai.communication.ros2.egress.global_egress` runs before the chain of every runtime and survives runtimes being closed and re-created.
-   **Failing closed.** A hook that raises, times out (`timeout_s`, 2 s) or returns something else denies. Sync hooks run inline and cannot be timed out, so keep them fast or make them async. Hooks never run on the executor thread.
-   **Observers.** `chain.observe(observer)` sees every decision (`EgressDecision`), leases included.

## Remaining bypasses

-   The ROS 2 tools have no CLI write commands (`ros2 topic pub`, `ros2 service call`, ...): diagnostics read the graph through the connector.
-   Code that creates its own rclpy node, or another process on the ROS graph, is not covered by egress hooks. Guard the robot's own controllers for those.

See the [API reference](../api/safety.md) for every parameter.
