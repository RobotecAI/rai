# Agents and lifecycle

An agent is anything with a start, a body and a stop: a ReAct loop driven by an LLM, a speech recognizer, a node that watches a topic. They share one base class, `rai.agents.BaseAgent`, and one way to run.

## Lifecycle

```text
CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED
              |           |          ^
              +--------> FAILED -----+
```

A subclass overrides any of three async hooks:

-   `on_start()`: acquire resources (subscribe, create services). If it raises, `start()` raises.
-   `run()`: the body. The default waits until the agent is stopped. If `run()` returns, the agent stops itself; if it raises, the agent goes to FAILED and `agent.error` holds the exception.
-   `on_stop()`: release resources. It runs once, also after a failure.

`stop()` cancels the task running `on_start()` and `run()`, so whatever it awaits (a model call, a tool) is cancelled too, then runs `on_stop()`. Each phase is bounded by a timeout (10 s by default) and the agent always ends STOPPED. `stop()` is idempotent, and before `start()` it does nothing. STOPPED is final: create a new agent to run again. An illegal transition, such as starting a stopped agent, raises `InvalidTransitionError`.

<!-- test: run -->

```python
import asyncio

from rai.agents import BaseAgent


class Heartbeat(BaseAgent):
    async def on_start(self) -> None:
        self.beats = 0

    async def run(self) -> None:
        for _ in range(3):
            self.beats += 1
            await asyncio.sleep(0.01)


async def main() -> None:
    async with Heartbeat() as agent:  # start() ... stop()
        await agent.wait()  # until it is STOPPED or FAILED
    print(agent.state, agent.beats)


asyncio.run(main())
```

`SyncAgent` is for agents built on blocking code: override `setup()` and `teardown()`, which run in a worker thread.

## Running agents

| Entry point            | Use                                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `async with agent:`    | Inside your own event loop                                                                                                                                                                             |
| `rai.run(*agents)`     | A program's main function: starts the agents in order, blocks until SIGINT, SIGTERM or all of them finish, then stops them in reverse order. Raises an `ExceptionGroup` of the errors of failed agents |
| `ThreadedAgent(agent)` | Sync code that must keep going: runs the agent on its own thread and event loop; `start()`, `stop()`, or `with`                                                                                        |

<!-- test: run -->

```python
import rai
from rai.agents import ThreadedAgent

rai.run(Heartbeat())  # returns when run() returns

with ThreadedAgent(Heartbeat()) as threaded:
    threaded.join(timeout=5.0)
```

All coroutines of one agent must run on the same event loop.

## ReActAgent

`ReActAgent` is a tool-calling agent built on LangChain's `create_agent`.

```python
from rai.agents import ReActAgent

agent = ReActAgent(
    "complex",                    # a role, an alias, "provider:model", or a BaseChatModel
    tools,                        # RaiTools; other LangChain tools are wrapped so hooks still run
    system_prompt="You operate a mobile robot.",
    middleware=[],                # LangChain agent middleware
    recursion_limit=50,           # model and tool steps per turn
    on_busy="interrupt",          # or "queue"
)
```

There are two ways to drive it:

-   **Direct calls.** `await agent.ainvoke(*messages)` runs one turn and returns the final `AIMessage`. No `start()` is needed. Messages are strings, LangChain messages or `HRIMessage`s. Turns of one agent must not overlap. Cancelling the call cancels the turn.
-   **As a service.** Run it with `rai.run` or `ThreadedAgent` and feed it with `agent.submit(*messages)`, which is thread-safe, so it can be a connector callback. With `on_busy="interrupt"` a new message cancels the running turn and the next turn sees all pending messages; with `"queue"` the turn finishes first. `agent.interrupt()` cancels the running turn and its current tool call.

Without a checkpointer the conversation is kept in `agent.history`. Tool errors and hook denials reach the model as error tool messages; the turn goes on. A tool interrupted mid-call gets an error result, so the history stays valid for the next turn. Images that tools return (`{"images": [...]}` artifacts) are shown to the model in each request without being written into the history.

## StateBasedAgent

`StateBasedAgent` is a `ReActAgent` that sees the current robot state on every model call. `state_sources` maps a name to an async callable returning text or content blocks, typically an [aggregator](../guides/aggregators.md):

```python
from rai.agents import StateBasedAgent
from rai.aggregators.ros2 import ImageSummary, RosoutDigest

agent = StateBasedAgent(
    "complex",
    tools,
    state_sources={
        "camera": ImageSummary(connector, "/camera/image_raw", frames=3),
        "logs": RosoutDigest(connector),
    },
)
```

Before each model call the sources are read concurrently, each under `state_timeout_s` (5 s). A source that fails, times out or returns `None` keeps its last value. The state is appended to the request only and never written into the history, so it does not pile up. The agent starts its aggregators in `on_start()` and stops them in `on_stop()`, so run it with `rai.run`, `ThreadedAgent` or `async with`.

## Events

Agents emit typed events from `rai.agents.events`. UIs, HRI output and the [flight recorder](../guides/recorder.md) consume them instead of LangChain callbacks:

| Event          | When                                                                    |
| -------------- | ----------------------------------------------------------------------- |
| `StateChanged` | A lifecycle transition (`old`, `new`)                                   |
| `ModelStart`   | Before a model call (`model`, `messages`)                               |
| `ModelEnd`     | After a model call (`message`)                                          |
| `Observation`  | The robot state a `StateBasedAgent` added to a request                  |
| `ToolStart`    | Before a tool call (`name`, `args`, `effect`)                           |
| `ToolEnd`      | After a tool call (`status`: `success`, `error` or `denied`; `content`) |
| `FinalMessage` | The turn's answer                                                       |
| `Interrupted`  | The turn was cancelled                                                  |
| `AgentError`   | The turn failed                                                         |

Every event carries the agent name and a timestamp; events of a turn carry its `turn` id.

```python
from rai.agents.events import ToolEnd

unsubscribe = agent.subscribe(lambda event: print(type(event).__name__))

async for event in agent.events():  # on the calling event loop; ends when the agent stops
    if isinstance(event, ToolEnd) and event.status == "denied":
        print("denied:", event.name, event.content)
```

`agent.subscribe(output)` calls `output` in the emitting thread; keep it fast.

### Talking back to humans

`HRIOutput` turns the event stream into what a person should hear: text the model says while it uses tools, then the final answer. `hri_connector_sender` publishes it on an HRI topic, one `HRIMessage` per chunk with the turn as its `communication_id`:

```python
from rai.agents import HRIOutput, hri_connector_sender
from rai.communication.ros2 import ROS2HRIConnector

hri = ROS2HRIConnector()
agent.subscribe(HRIOutput(hri_connector_sender(hri, "/to_human"), tool_status=True))
hri.subscribe("/from_human", agent.submit)
```

`tool_status=True` also announces tool starts, errors and denials.

See the [API reference](../api/agents.md) for every parameter.
