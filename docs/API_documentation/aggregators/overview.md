# Aggregators

An aggregator turns one topic into a piece of robot state for [`StateBasedAgent`](../agents/overview.md). It has two halves:

-   **A bounded buffer.** `start()` subscribes to the topic and keeps the last `window` messages with their arrival time; `stop()` closes the subscription. Messages older than `max_age_s` are ignored. Buffering is cheap and never calls a model.
-   **An on-demand summary.** Awaiting the aggregator runs `summarize()` on the buffered messages, only when the agent is about to call its model. The result is cached for `cache_s` seconds and after that recomputed only if a new message arrived. A VLM aggregator therefore costs nothing while the agent is idle. With no recent messages the state reads `no messages on <topic> in the last <max_age_s> s`.

```python
from rai.agents import StateBasedAgent
from rai.aggregators.ros2 import ImageSummary, RosoutDigest

agent = StateBasedAgent(
    tools=tools,
    state_sources={
        "camera": ImageSummary(connector, "/camera/image_raw", frames=3),
        "logs": RosoutDigest(connector),
    },
)
rai.run(agent)  # start() subscribes the aggregators, stop() closes them
```

The state is added to each model request and never written to the history. Sources are read concurrently, each under `state_timeout_s`; a source that fails or times out keeps its last good value. Any `async def` returning text or content blocks can be a state source too.

## Writing an aggregator

```python
class BatteryLevel(Aggregator[BatteryState]):
    async def summarize(self, msgs):
        return f"{msgs[-1].percentage:.0%}"

BatteryLevel(connector, "/battery", msg_type="sensor_msgs/msg/BatteryState", window=1)
```

::: rai.aggregators.Aggregator

## See Also

-   [ROS 2 Aggregators](ROS_2_Aggregators.md)
-   [Agents](../agents/overview.md)
