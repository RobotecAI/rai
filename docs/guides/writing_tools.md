# Writing tools

RAI's [ROS 2 tools](ros2_tools.md) cover generic topics, services and actions. Robot-specific skills (dock, open the gripper, read the battery the way your robot reports it) are better as their own tools: the model gets one clear action instead of composing raw messages, and safety hooks see a precise effect and target.

Read [Tools and effects](../concepts/tools.md) first for `RaiTool`, effects and hooks.

## What the model sees

The model chooses tools by their `name`, `description` and argument schema. Write them for the model:

-   Name the action: `dock`, `get_battery_state`, not `DockTool`.
-   Say in the description what the tool does, when to use it and what it returns. Mention units.
-   Describe every argument with `pydantic.Field(description=...)` and give units in the description.
-   Return short text with units. Raise an exception for failures: it becomes an error tool message with the exception text, and the model can try something else.

## A function tool

<!-- test: run -->

```python
from rai.tools import Effect, rai_tool


@rai_tool(Effect.READ, targets=["/battery_state"])
async def get_battery_state() -> str:
    """The battery charge in percent and whether the robot is charging."""
    return "87%, not charging"


print(get_battery_state.invoke({}))
```

`rai_tool` accepts sync functions too; they run in a worker thread.

## A ROS 2 tool

Subclass `BaseROS2Tool` for tools that use a connector. It adds the `connector` field and the `readable`, `writable` and `forbidden` [access lists](ros2_tools.md#access-lists), and resolves ROS names into the tool's targets:

-   `target_fields` maps configuration fields to the kind of name they hold (`"topic"`, `"service"` or `"action"`): fixed targets, known before any call.
-   `target_args` maps call arguments to their kind: targets chosen by the model, resolved per call.

<!-- test: run -->

```python
from pydantic import BaseModel, Field
from std_srvs.srv import SetBool

from rai.tools import Effect
from rai.tools.ros2.base import BaseROS2Tool


class DockInput(BaseModel):
    dock: bool = Field(description="True to dock and charge, False to undock")


class DockTool(BaseROS2Tool):
    default_effect = Effect.ACTUATE  # it moves the robot
    target_fields = {"service": "service"}
    name: str = "dock"
    description: str = "Drive onto the charging dock, or leave it. Takes up to a minute."
    args_schema: type[BaseModel] = DockInput

    service: str = "/dock"

    async def _acall(self, dock: bool) -> str:
        self.check_writable(self.service, "service")
        response = await self.connector.call(self.service, SetBool.Request(data=dock), timeout=60.0)
        if not response.success:
            raise RuntimeError(f"Docking failed: {response.message}")
        return "Docked" if dock else "Undocked"
```

The connector call raises `ROS2TimeoutError` if the service does not answer and `EgressDenied` if an [egress hook](../concepts/safety.md#connector-side-egress-hooks) refuses; both reach the model as error tool messages.

Use the tool like any other:

```python
from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector

connector = ROS2Connector()
agent = ReActAgent("complex", [DockTool(connector=connector), get_battery_state])
```

## Reading sensors

Read the latest message with `receive`. Pass `max_age` when stale data would mislead the model:

```python
from sensor_msgs.msg import LaserScan

from rai.tools import Effect
from rai.tools.ros2.base import BaseROS2Tool


class NearestObstacleTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_fields = {"topic": "topic"}
    name: str = "nearest_obstacle"
    description: str = "Distance to the nearest obstacle seen by the lidar, in meters"
    topic: str = "/scan"

    async def _acall(self) -> str:
        self.check_readable(self.topic)
        scan = await self.connector.receive(self.topic, msg_type=LaserScan, timeout=2.0, max_age=1.0)
        ranges = [r for r in scan.ranges if scan.range_min < r < scan.range_max]
        return f"{min(ranges):.2f} m" if ranges else "no obstacle in range"
```

## Returning images

Set `response_format = "content_and_artifact"` and return the text and `{"images": [...]}`. RAI agents show the images to the model with the next request.

```python
from typing import Any

from sensor_msgs.msg import Image as ImageMsg

from rai.communication.ros2 import encode_image
from rai.tools import Effect
from rai.tools.ros2.base import BaseROS2Tool


class WristCameraTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_fields = {"topic": "topic"}
    name: str = "look_with_wrist_camera"
    description: str = "A picture from the camera on the gripper"
    response_format: str = "content_and_artifact"
    topic: str = "/wrist/image_raw"

    async def _acall(self) -> tuple[str, dict[str, Any]]:
        msg = await self.connector.receive(self.topic, msg_type=ImageMsg, timeout=2.0, max_age=0.5)
        return "The wrist camera image", {"images": [encode_image(msg)]}
```

## Long-running actions

A tool may await a long action: cancelling the tool (the agent's `stop()` or `interrupt()`) cancels the coroutine, so cancel the goal in `finally`:

```python
from nav2_msgs.action import NavigateToPose


async def navigate(connector, goal: NavigateToPose.Goal) -> str:
    handle = await connector.send_goal("/navigate_to_pose", goal)
    try:
        await handle.result(timeout=300.0)
        return f"Navigation finished: {handle.status.name}"
    finally:
        if not handle.done:
            await handle.cancel()
```

For tasks the model should monitor while they run, split them into start, feedback, result and cancel tools that share a goal registry, like `Nav2Toolkit` and `ROS2ActionToolkit`.

## Wrapping existing LangChain tools

Tools from other libraries or MCP servers work as they are: `ReActAgent` wraps every tool that is not a `RaiTool` with `rai.tools.from_langchain`, so hooks apply. Wrap them yourself to set the effect, since an unknown effect is treated as ACTUATE:

```python
from langchain_core.tools import tool

from rai.tools import Effect, from_langchain


@tool
def lookup_manual(query: str) -> str:
    """Search the robot's manual."""
    return "..."


manual = from_langchain(lookup_manual, effect=Effect.READ)
```
