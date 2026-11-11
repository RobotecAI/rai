# Robot state (aggregators)

A [`StateBasedAgent`](../concepts/agents.md#statebasedagent) sees the current robot state on every model call. Its state sources are usually aggregators: one topic each, turned into a piece of text or an image for the model.

An aggregator has two halves:

-   **A bounded buffer.** `start()` subscribes to the topic and keeps the last `window` messages with their arrival time; `stop()` closes the subscription. Messages older than `max_age_s` are ignored. Buffering never calls a model.
-   **An on-demand summary.** Awaiting the aggregator runs `summarize()` on the buffered messages, only when the agent is about to call its model. The result is cached for `cache_s` seconds and after that recomputed only if a new message arrived, so a VLM aggregator costs nothing while the agent is idle. With no recent messages the state reads `no messages on <topic> in the last <max_age_s> s`.

```python
import rai
from rai.agents import StateBasedAgent
from rai.aggregators.ros2 import ImageSummary, RosoutDigest
from rai.communication.ros2 import ROS2Connector

connector = ROS2Connector()
agent = StateBasedAgent(
    "complex",
    tools,
    state_sources={
        "camera": ImageSummary(connector, "/camera/image_raw", frames=3),
        "logs": RosoutDigest(connector),
    },
)
rai.run(agent)  # on_start() starts the aggregators, on_stop() stops them
```

## Built-in aggregators

`rai.aggregators.ros2` provides:

| Aggregator     | State                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `RosoutDigest` | Distinct `/rosout` lines of the last 60 s at INFO or above, with repeat counts                 |
| `LastImage`    | The newest camera frame, shown to the model as an image                                        |
| `ImageSummary` | A vision model's description of the newest frame, or with `frames=3` of what changed over them |

`ImageSummary` takes `model=` (a chat model or a [model role](../getting_started/configuration.md#models), default `"simple"`) and `prompt=`. Images are encoded only for the frames sent to the model.

## Writing an aggregator

Subclass `rai.aggregators.Aggregator` and implement `async summarize(msgs)`, which gets the buffered messages, oldest first, and returns text or LangChain content blocks:

<!-- test: run -->

```python
from sensor_msgs.msg import BatteryState

from rai.aggregators import Aggregator


class BatteryLevel(Aggregator[BatteryState]):
    async def summarize(self, msgs: list[BatteryState]) -> str:
        return f"{msgs[-1].percentage:.0%}"
```

```python
battery = BatteryLevel(connector, "/battery_state", msg_type="sensor_msgs/msg/BatteryState", window=1)
```

Any `async def` that returns text, content blocks or `None` (nothing new: keep the last value) can be a state source too. Sources are read concurrently, each under the agent's `state_timeout_s`; one that fails or times out keeps its last good value.

The state is added to each model request and never written to the history. The [flight recorder](recorder.md) stores it as `Observation` events, and [replay](replay.md) compares it.
