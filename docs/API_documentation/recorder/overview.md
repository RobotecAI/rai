# Flight recorder

`rai.recorder.FlightRecorder` writes one [MCAP](https://mcap.dev) file per session: every LLM turn, tool call, hook decision and observation of your agents, next to the ROS topics you choose, on one timeline. Open it in [Foxglove](https://foxglove.dev) to see what the camera showed, what the model was told and what it decided.

It needs the `recorder` extra:

```bash
pip install 'rai_core[recorder]'
```

## Recording

```python
import rai
from rai.recorder import FlightRecorder

with FlightRecorder("run.mcap", agent=agent, topics=["/cmd_vel", "/camera/image_raw"]):
    rai.run(agent)
```

-   `agent`: one agent or several; add more with `recorder.attach(agent)`.
-   `topics`: ROS topics recorded raw (CDR). Give names (the type comes from the ROS graph, and a topic that does not exist yet is picked up when it appears) or `{name: type}`.
-   `audit`: where hook decisions come from. The default (`True`) installs a private `AuditLog` observer on every runtime and RaiTool. Pass the one `rai.safety.install(...)` created (`audit=guard.audit`) to share it, or `False` for none.
-   `connector`: the `ROS2Connector` for topics and the clock. Without one, a connector on the shared runtime is created when topics need it.

`rai.recorder.install(path, *agents, **kwargs)` starts a recorder and returns it, like `rai.safety.install`; call `stop()` to finish the file. In async code, `start()` the recorder and call `stop()` from a thread (`await asyncio.to_thread(recorder.stop)`).

Recording never slows the agent or the ROS executor. Inputs only take a timestamp and put the item on a bounded queue (`maxsize`), and one writer thread encodes and writes. When the queue is full, items are dropped and counted in `recorder.dropped`. The file is flushed every `flush_s` seconds (default 1). If the process dies, the file is still readable up to the last flush. A recorder that is still running at interpreter exit finishes its file.

## Channels

| Topic                                    | Schema                      | Contents                                                                                                                                                                                                                                        |
| ---------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/rai/agent/events`                      | `rai.AgentEvent`            | Every agent event, one compact line each: state changes, model and tool start and end, final message, interruptions, errors.                                                                                                                    |
| `/rai/agent/llm`                         | `rai.LLMCall`               | `request`: model name and the prompt as LangChain message dicts. Only the messages that differ from the agent's previous request are stored (`prefix` counts the shared ones). `response`: the message, text, tool calls, token usage, latency. |
| `/rai/agent/tools`                       | `rai.ToolCall`              | `start`: args and effect. `end`: status (`success`, `error`, `denied`), content, artifact, duration.                                                                                                                                            |
| `/rai/agent/observations`                | `rai.Observation`           | The robot state a `StateBasedAgent` added to each model request, and which sources were fresh.                                                                                                                                                  |
| `/rai/agent/observations/<source>/image` | `foxglove.CompressedImage`  | Images of one state source.                                                                                                                                                                                                                     |
| `/rai/agent/images`                      | `foxglove.CompressedImage`  | Other images: prompts and tool artifacts.                                                                                                                                                                                                       |
| `/rai/agent/log`                         | `foxglove.Log`              | A readable timeline of the above, with hook denials as warnings.                                                                                                                                                                                |
| `/rai/safety/decisions`                  | `rai.HookDecision`          | Every hook decision on both hook points (`AuditRecord`).                                                                                                                                                                                        |
| your topics                              | ROS 2 type (`ros2msg`), CDR | Raw messages, as `ros2 bag record` writes them to MCAP.                                                                                                                                                                                         |

RAI channels are JSON with JSON schemas. Each image is stored once: JSON content refers to it with a `rai-image:<ref>` token in place of its base64 data, and the image's `frame_id` is the ref. WAV audio goes to an attachment named `<ref>.wav`. The `rai.session` metadata record holds the clock, the agents, the topics and the versions. The `rai.recorder.stats` record holds the message and drop counts.

### Time

`log_time` and `publish_time` are equal: the moment the event happened, on one clock. When the connector's node uses sim time (`use_sim_time=True`), that clock is ROS time. Otherwise it is the wall clock, read through the monotonic clock so it never jumps backwards during a session. Agent events are stamped when they are emitted. Hook decisions are stamped when the hook decided, even though they reach the recorder later. ROS messages are stamped when they are received. A message that a tool publishes therefore appears between the tool's `start` and `end`, right after its egress decision.

## Opening in Foxglove

Open the file in Foxglove (**Open local file**). Some useful panels:

-   **Log** on `/rai/agent/log` for the agent's timeline.
-   **Raw Messages** on `/rai/agent/llm` or `/rai/agent/tools` for full prompts and tool results.
-   **Image** on `/rai/agent/observations/<source>/image` and on your camera topic, side by side.
-   **Plot** or **3D** on the ROS topics.

The `mcap` CLI works too: `mcap info run.mcap`, `mcap cat run.mcap --topics /rai/agent/tools --json`.

## Reading in Python

```python
from langchain_core.messages import messages_from_dict
from rai.recorder import Recording

rec = Recording("run.mcap")
for call in rec.values("/rai/agent/tools"):
    print(call["phase"], call["name"], call.get("status"))
prompt = messages_from_dict(rec.prompts("my_agent")[-1]["messages"])  # the full last prompt
twists = rec.values("/cmd_vel")  # ROS messages, decoded with the recorded schema
```

`Recording.messages(topics)` returns `RecordedMessage`s in time order. `restore(value)` puts the base64 data back in place of the media tokens.

::: rai.recorder.FlightRecorder

::: rai.recorder.Recording
