# What's new in 3.0

RAI 3.0 is a rewrite of the core. The goals, in order, were safety, speed and clean code.

## Platforms and packaging

-   ROS 2 Lyrical (Python 3.14) and Jazzy (Python 3.12). Humble is no longer supported.
-   Model providers and optional features are extras of `rai_core`: `openai`, `anthropic`, `aws`, `google`, `ollama`, `langfuse`, `mcp`, `ui`, `policies`, `recorder`, `all`. The base install is much smaller.
-   The developer environment is a uv workspace in a `--system-site-packages` venv (`scripts/dev_env.sh`); no `PYTHONPATH` changes. Docker images for Lyrical and Jazzy.
-   cv_bridge and tf_transformations are no longer needed; numpy 2 everywhere.

## Settings and models

-   `rai.toml` plus `RAI_*` environment variables replace `config.toml`. Models are `"provider:model"` strings; `rai.get_llm` builds them. See [Configuration](getting_started/configuration.md).

## ROS 2 layer

-   `ROS2Connector` is async, with blocking twins under `connector.sync`. Connectors share one runtime (one node, one executor, one spin thread); the default executor is `EventsExecutor`.
-   Native ROS messages in and out, QoS presets, `Subscription` and `GoalHandle` objects, numpy views of images and point clouds, one JPEG encoder for the model.
-   Clean shutdown: `close()` cancels pending calls and destroys entities. No entity leaks over repeated calls.
-   See [Connectors and the runtime](concepts/connectors.md).

## Agents and tools

-   Agents on LangChain 1.x with an explicit lifecycle state machine, `rai.run` and `ThreadedAgent`, interruption that cancels the running tool, and a typed event stream. See [Agents and lifecycle](concepts/agents.md).
-   `StateBasedAgent` adds robot state to each model request without writing it into the history; aggregators summarize on demand.
-   `RaiTool` adds effects, ROS targets and fail-closed hooks to LangChain tools. Tool errors go back to the model. See [Tools and effects](concepts/tools.md).
-   `Text`, `Image` and `Audio` content types replace the multimodal message classes. See [Content and HRI messages](concepts/content.md).

## Safety

-   Two hook points that fail closed: every tool call, and every publish, service call, goal and cancel. Built-in `EStop` (triggered from ROS on `/rai/estop`), `ConfirmActuation`, `RateLimit` and `AuditLog`. The ROS 2 node is no longer exposed, and the CLI write commands are gone. See [Safety](concepts/safety.md).

## New features

-   [ROS 2 diagnostics toolkit](guides/ros2_tools.md#diagnostics): logs, diagnostics, lifecycle states, TF health and topic rates through the connector.
-   [MCP](guides/mcp.md): `rai mcp serve` exposes the ROS 2 tools to any MCP client; agents use MCP servers as tools.
-   [Robot manifest](guides/manifest.md): the robot's spec, rai_whoami description and live ROS graph, for the model and over MCP.
-   [Flight recorder](guides/recorder.md) and [replay](guides/replay.md): record agent runs to MCAP next to ROS topics, view them in Foxglove, replay them in CI.
-   [Learned-policy skills](guides/policies.md): VLA policy servers (openpi, GR00T, HTTP) as cancellable tools.
-   [Streamlit chat](guides/streamlit.md) rebuilt on the event stream.
