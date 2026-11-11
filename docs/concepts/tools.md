# Tools and effects

A tool is what an agent can do. RAI tools are LangChain tools (`BaseTool`), so any LangChain agent can use them, with three additions in `rai.tools.RaiTool`:

-   an **effect**: what the tool does to the world;
-   **targets**: the ROS names it touches;
-   **hooks**: a fail-closed check before every call.

## Effects

| Effect           | Meaning                                        | Examples                                    |
| ---------------- | ---------------------------------------------- | ------------------------------------------- |
| `Effect.READ`    | Observes only                                  | Read a topic, list nodes, get an image      |
| `Effect.WRITE`   | Changes software state, does not move hardware | Set a parameter, publish to a UI topic      |
| `Effect.ACTUATE` | May move hardware                              | Publish a velocity, start a navigation goal |

A tool without a declared effect is ACTUATE, the most restricted. Hooks, the MCP server (`readOnlyHint` and `destructiveHint`), the UI and the recorder all read the effect.

## Defining tools

The `rai_tool` decorator turns a function into a `RaiTool`. The docstring is the description and the signature is the schema. Async functions are preferred; sync functions run in a thread.

<!-- test: run -->

```python
from rai.tools import Effect, rai_tool


@rai_tool(Effect.ACTUATE, targets=["/cmd_vel"], timeout_s=10.0)
async def drive(speed: float) -> str:
    """Drive forward at speed (m/s)."""
    return f"driving at {speed} m/s"


print(drive.name, drive.effect, drive.targets)
print(drive.invoke({"speed": 0.5}))
```

For tools with configuration or state, subclass `RaiTool` and implement `async _acall`:

<!-- test: run -->

```python
from pydantic import BaseModel, Field

from rai.tools import Effect, RaiTool


class SayInput(BaseModel):
    text: str = Field(description="What to say")


class SayTool(RaiTool):
    default_effect = Effect.WRITE
    name: str = "say"
    description: str = "Say something through the robot's speaker"
    args_schema: type[BaseModel] = SayInput
    volume: float = 0.8  # configuration

    async def _acall(self, text: str) -> str:
        return f"said {text!r} at volume {self.volume}"


print(SayTool(volume=0.5).invoke({"text": "hello"}))
```

Implement `_acall`, never `_run` or `_arun`: `RaiTool` owns those to run the hooks. The sync `invoke` works from any thread.

-   **Errors** go back to the model. An exception in `_acall` becomes an error tool message (`"ValueError: ..."`) and the agent continues. Cancellation (the agent's `stop()` or `interrupt()`) propagates into the tool, so `finally` blocks run.
-   **Timeouts.** `timeout_s` bounds each call.
-   **Images.** Set `response_format = "content_and_artifact"` and return `(text, {"images": [...]})` with `rai.content.Image`s or base64 strings. RAI agents show the images to the model.
-   **Other tools.** `rai.tools.from_langchain(tool, effect=...)` wraps any LangChain tool (MCP client tools included) so the hooks apply to it. `ReActAgent` does this for every tool that is not a `RaiTool`, with the ACTUATE default unless MCP annotations say otherwise.

## Targets

Targets are the resolved ROS names a call touches. They let hooks decide on names rather than tool names. A tool's fixed targets come from `targets=` or `default_targets`. ROS 2 tools add the names they are configured with or called with, resolved through the connector (`"cmd_vel"`, `"/cmd_vel"` and remapped forms all match), see [ROS 2 tools](../guides/ros2_tools.md).

## Tool hooks

A tool hook is `async (ToolCallContext) -> Allow | Deny`. The context has the tool's `name`, `effect`, `targets` and the call's `args`. Every `RaiTool` call runs `rai.tools.global_tool_hooks` first, then the tool's own `tool.hooks`, in insertion order; the first denial wins.

<!-- test: run -->

```python
from rai.tools import Allow, Deny, Effect, ToolCallContext, global_tool_hooks


async def no_actuation(call: ToolCallContext) -> Allow | Deny:
    if call.effect is Effect.ACTUATE:
        return Deny("the robot is in read-only mode")
    return Allow()


global_tool_hooks.add(no_actuation, timeout_s=2.0)
print(drive.invoke({"speed": 0.5}))  # Denied by no_actuation: the robot is in read-only mode
global_tool_hooks.remove(no_actuation)
```

The chain fails closed: a hook that raises, times out (`timeout_s`, 2 s by default) or returns anything but `Allow` or `Deny` denies the call. A denial reaches the model as an error tool message and the `ToolEnd` event has status `denied`. With no hooks registered the check costs nothing.

`hooks.observe(observer)` registers a function that sees every hook's verdict (`ToolDecision`) without being able to change it; `rai.safety.AuditLog` uses it.

Tool hooks run inside the tool, so they cover every way a tool is called: agents, the MCP server and direct calls. They are one of two hook points; the other guards every ROS publish, call and goal. See [Safety](safety.md).

See the [API reference](../api/tools.md) for every parameter, and [Writing tools](../guides/writing_tools.md) for ROS 2 tools.
