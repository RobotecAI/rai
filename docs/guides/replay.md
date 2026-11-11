# Record and replay

`rai.replay` runs agents, tools and aggregators on recorded data, without a robot or a simulator. Record a session on the robot with the [flight recorder](recorder.md), then replay it in CI: fast, deterministic, and with a clear report when a prompt, a tool or its output changes.

It needs ROS 2 and the `recorder` extra:

```bash
pip install 'rai_core[recorder]'
```

It reads three kinds of recordings:

-   flight recordings written by `rai.recorder.FlightRecorder` (MCAP),
-   rosbag2 bags in MCAP (`ros2 bag record`, the default storage on Jazzy and later),
-   rosbag2 bags in sqlite3 (`.db3`).

Pass a file, or a rosbag2 directory (its `.mcap` or `.db3` files are merged). Only the ROS topics (CDR) are replayed. The `/rai/...` channels of a flight recording are used by `AgentReplay`.

## Record on the robot, replay in CI

[`examples/replay_in_ci.py`](https://github.com/RobotecAI/rai/blob/main/examples/replay_in_ci.py) shows the whole loop. The agent's tools are built by one function, used in both places:

```python
def tools(connector):
    return [ReceiveROS2MessageTool(connector=connector), GetROS2TransformTool(connector=connector)]
```

On the robot, record a session with the topics the tools read:

```python
connector = ROS2Connector()
agent = ReActAgent(tools=tools(connector), system_prompt=PROMPT, name="robot")
with FlightRecorder("session.mcap", agent=agent, topics=["/tf", "/tf_static", "/odom"], connector=connector):
    asyncio.run(agent.ainvoke("Where is the robot?"))
```

In CI, re-run it offline and check the report:

```python
from rai.replay import AgentReplay

def test_robot_session():
    replay = AgentReplay("tests/data/session.mcap", agent="robot")
    agent = ReActAgent(replay.model, tools(replay.connector), system_prompt=PROMPT, name="robot")
    report = asyncio.run(replay.run(agent))
    assert report.ok, report
```

`AgentReplay` gives the agent two parts:

-   `replay.model`, a `ReplayChatModel`: it answers each model call with the next recorded response, so the agent makes the same tool calls as on the robot.
-   `replay.connector`, a `ReplayConnector`: it serves the recorded topics and TF to the tools on a virtual clock, and records what they publish, call or send instead of sending it.

`replay.run(agent)` replays every recorded turn (`replay.turns()` gives the human messages) and returns a `ReplayReport`. While the agent works, it compares each item below with the recording:

| Check         | Compared                                                                                                                                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`      | Every model request: the type, content, tool calls and tool call ids of each message. Message ids, metadata and token usage are ignored.                                                                          |
| `observation` | What a `StateBasedAgent` added to each request.                                                                                                                                                                   |
| `tool_call`   | The name and arguments of each tool call, by call id.                                                                                                                                                             |
| `tool_result` | The status and content of each tool result.                                                                                                                                                                       |
| `egress`      | The messages each tool published on a recorded topic, compared with the recorded messages on that topic. Topics compared: those the replay published to, recorded topics of a `MOTION` type, and `egress_topics`. |
| `model_calls` | The number of model calls.                                                                                                                                                                                        |

`report.ok` is true when nothing diverged. `str(report)` lists each divergence with where it happened and what was expected:

```
replay DIVERGED: 3/3 model calls, 2/2 tool calls, 1 egress
  - tool_result at tool drive (call_3b1f8d): content: expected "driving 0.5", got "refusing"
```

With `strict=True` (the default), the first divergence makes the next model call fail, so the run stops where the behaviour changed. With `strict=False`, the run goes on and the report collects every divergence. A different tool result usually also changes the next prompt.

### The clock

The connector's clock follows the recorded session. Before each model call and each tool call, it moves to the time that call happened in the recording. A tool reading `/odom` therefore gets the message the original tool could read. A wait moves the clock forward:

-   a `receive` that has no message fresh enough (`max_age`),
-   `Subscription.get`,
-   `transform`,
-   `GoalHandle.result`.

The clock moves to the next recorded message, or by the timeout, which then raises `ROS2TimeoutError` as it would on the robot. Nothing sleeps: a session that took a minute replays in about a second.

## ReplayConnector

`ReplayConnector(source)` is a `ROS2Connector`: pydantic tools that require one accept it, and tools, aggregators and agents run against it unchanged. Use it on its own for unit tests of tools:

```python
from rai.replay import ReplayConnector, ScriptedGoal, expect_published
from rai.tools.ros2.generic.topics import PublishROS2MessageTool, ReceiveROS2MessageTool

replay = ReplayConnector(
    "session.mcap",
    services={"/set_bool": SetBool.Response(success=True)},
    actions={"/navigate_to_pose": ScriptedGoal(result=NavigateToPose.Result(), duration=3.0)},
)
print(ReceiveROS2MessageTool(connector=replay).invoke({"topic": "/odom"}))
PublishROS2MessageTool(connector=replay).invoke(
    {"topic": "/cmd_vel", "message": {"linear": {"x": 0.2}}, "message_type": "geometry_msgs/msg/Twist"}
)
expect_published(replay, "/cmd_vel", {"linear.x": 0.2}, count=1)
```

-   **Topics.** `receive` returns the latest recorded message at the virtual time. `subscribe` delivers messages as the clock advances. Callbacks run inline, in time order, and async callbacks are awaited. In pull mode, `async for msg in sub` ends at the end of the recording. Topics in `latched` (default `/tf_static`) give a new subscription the messages recorded so far.
-   **Egress.** `publish`, `call`, `send_goal` and goal cancels run the egress hooks (`global_egress`, then `replay.egress`), so a test can check safety hooks too. Allowed egress is appended to `replay.sent` as `EgressRecord(kind, name, type, payload, stamp, origin)`, where `origin` is the tool. Denials raise `EgressDenied` and go to `replay.denied`.
-   **Services and actions.** `services` and `actions` script the answers. A value is a response message (or an action Result), an exception to raise, or a function of the request or goal returning either one, sync or async. Use `ScriptedGoal(result, feedback, status, accepted, duration)` for feedback, a goal that is rejected or aborts, or one that takes virtual time. A call or goal with no script times out, like a server that is not running.
-   **Time, TF and graph.** `now()` is the virtual time (ROS time). `transform` and `tf_frames` use the recorded `/tf` and `/tf_static`. The graph queries list the recorded topics, the topics published so far, and the scripted services and actions.
-   **Clock control.** `advance(seconds)`, `advance_to(stamp_ns)` and `seek(stamp_ns)` move the clock. `seek` can also move it backwards. `clock_ns`, `start` and `end` read it. The clock starts at the session start of a flight recording, or at the first message.

## ReplayPlayer

`ReplayPlayer` publishes the recorded topics into a live ROS graph with real publishers. The real `ROS2Connector`, ROS nodes and RViz see the recorded data, as with `ros2 bag play`:

```python
from rai.replay import ReplayPlayer

with ReplayPlayer("session.mcap", clock=True, rate=2.0, loop=True) as player:
    player.wait_for_subscribers(["/camera/image_raw"])
    player.start()
    rai.run(agent)  # the agent's real connector reads the replayed topics
```

-   `topics`, `exclude` and `remap` choose what to play and under which names. `start` gives the seconds into the recording to start at. `seek(seconds)`, `pause()` and `resume()` work while playing.
-   `rate` scales time. `loop` starts over at the end. `wait()` blocks until the end.
-   `clock=True` publishes `/clock` with the recording's time, so nodes and connectors on sim time (`ROS2Connector(use_sim_time=True)`) follow the recording in `now()`.
-   Topics of a `MOTION` type (velocity commands, trajectories, navigation goals) are never played unless `include_motion=True`, so a recording played into a robot's graph cannot drive it. They are listed in `player.skipped`.
-   `latched` topics (default `/tf_static`) are published transient-local.

## Assertions

`rai.replay` has assertions for tests. Each one returns what matched, or raises `AssertionError` listing what was seen:

| Assertion                                                   | Checks                                                                            |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `expect_published(source, topic, predicate, within, count)` | Messages published on a topic.                                                    |
| `expect_called(source, service, predicate, ...)`            | Service requests sent.                                                            |
| `expect_goal(source, action, predicate, ...)`               | Action goals sent.                                                                |
| `expect_no_motion(source, types=MOTION)`                    | Nothing of a motion type was sent (cancels are allowed).                          |
| `expect_nothing_sent(source)`                               | No egress at all.                                                                 |
| `expect_tool_called(source, name, args, status, count)`     | Tool calls of an `AgentReplay`, a `Recording` or path, or a list of agent events. |

`source` is a `ReplayConnector` or an `AgentReplay`. A predicate is a function of the message, or a dict of dotted field paths and values, for example `{"linear.x": 0.5}`. `within` is in virtual seconds from the replay's start. Without `count`, at least one match is required.

## Limitations

-   Replay is exact only for what was recorded. Record the topics your tools read. Service responses and action results are not in a flight recording, so script them.
-   On the robot, a tool reads a topic a few milliseconds after its call starts. The replay reads at the recorded start of the call. A message that arrived in between can make a fast topic read differently.
-   Egress is compared only on recorded topics. A recorded message counts as sent by the latest tool call that started before it, if it arrived at most `egress_grace_s` (1 s) after that call ended. Other egress, such as services and goals, is listed in `report.notes`. Check it with the `expect_*` assertions.
-   Aggregators keep their own wall-clock windows (`max_age_s`). In a replay they see the recorded messages as they are delivered, so their windows may hold more messages than on the robot.
-   `ReplayConnector` does not deliver its own publishes to its subscribers, and cannot serve services or actions (`create_service`, `create_action`).
-   The connector loads the recorded ROS messages into memory. The player streams indexed MCAP and sqlite3 files.

See the [API reference](../api/replay.md) for every parameter.
