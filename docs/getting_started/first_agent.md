# Your first agent

This page builds an agent with one tool, gives it ROS 2 tools, runs it as a service the robot's users can talk to, and adds an e-stop. It assumes rai_core is [installed](install.md) with a model provider and its API key is set (`OPENAI_API_KEY` for the defaults).

## An agent in 10 lines

<!-- test: run -->

```python
import asyncio

from rai.agents import ReActAgent
from rai.tools import Effect, rai_tool


@rai_tool(Effect.READ)
async def battery_level() -> str:
    """The robot's battery charge."""
    return "87%"


agent = ReActAgent("complex", [battery_level], system_prompt="You are a robot assistant.")
answer = asyncio.run(agent.ainvoke("How much battery is left?"))
print(answer.text)
```

-   `"complex"` is a model role from the [settings](configuration.md). A `"provider:model"` string or any LangChain chat model works too.
-   `rai_tool` turns a function into a `RaiTool`. Its docstring is the description the model sees, and its `Effect` (READ, WRITE or ACTUATE) tells safety hooks what the tool does to the world. See [Tools and effects](../concepts/tools.md).
-   `ainvoke` runs one turn: the model calls tools until it can answer. The agent keeps the conversation in `agent.history`, so the next `ainvoke` continues it.

## Give it the robot

With ROS 2 sourced, a `ROS2Connector` gives tools access to topics, services and actions. `ROS2Toolkit` is a generic set of tools that lists, reads and publishes topics, calls services and runs actions:

<!-- test: run -->

```python
import asyncio

from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import ROS2Toolkit

with ROS2Connector() as connector:
    tools = ROS2Toolkit(connector=connector, forbidden=["/cmd_vel"]).get_tools()
    agent = ReActAgent("complex", tools)
    print(asyncio.run(agent.ainvoke("Which topics are available?")).text)
```

`forbidden` (and the `readable` and `writable` allowlists) limit what the generic tools may touch. See [ROS 2 tools](../guides/ros2_tools.md) for the toolkits, including Nav2 and diagnostics.

## Run it as a service

To let people talk to the robot, run the agent until Ctrl+C and connect it to HRI topics. `ROS2HRIConnector` carries `rai_interfaces/msg/HRIMessage` (text, images, audio), so it needs [rai_interfaces](install.md#3-install-rai_interfaces).

```python
import rai
from rai.agents import HRIOutput, ReActAgent, hri_connector_sender
from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector
from rai.tools.ros2 import ROS2Toolkit

connector = ROS2Connector()
hri = ROS2HRIConnector()

agent = ReActAgent("complex", ROS2Toolkit(connector=connector).get_tools())
agent.subscribe(HRIOutput(hri_connector_sender(hri, "/to_human")))  # what the agent says
hri.subscribe("/from_human", agent.submit)  # what people say
rai.run(agent)  # blocks until Ctrl+C or SIGTERM
```

Talk to it from another terminal:

```bash
ros2 topic pub --once /from_human rai_interfaces/msg/HRIMessage "{text: 'What do you see?'}"
ros2 topic echo /to_human rai_interfaces/msg/HRIMessage
```

A new message interrupts the running turn (`on_busy="interrupt"`, the default) and the next turn sees every pending message. Pass `on_busy="queue"` to finish the turn first.

## Add an e-stop

Add one line before `rai.run(agent)`:

```python
from rai import safety

guard = safety.install()
```

`install()` registers an `EStop` on both hook points: tool calls with the ACTUATE effect and every publish, service call or goal of a motion type (`Twist`, Nav2 and MoveIt actions, ...) are denied while it is engaged, and engaging it cancels running motion goals. Engage it from ROS:

```bash
ros2 topic pub --once /rai/estop std_msgs/msg/Bool "{data: true}"
```

The e-stop is a software stop. It does not replace a hardware e-stop. See [Safety](../concepts/safety.md) for confirmation, rate limits and audit logs.

## Next steps

-   [Configuration](configuration.md): models, providers, local models and tracing.
-   [Agents and lifecycle](../concepts/agents.md): `StateBasedAgent`, the event stream and how agents start and stop.
-   [Writing tools](../guides/writing_tools.md): tools for your robot.
-   [Build an agent for your robot](../guides/walkthrough.md): a custom agent, a robot-specific tool and a whoami system prompt.
