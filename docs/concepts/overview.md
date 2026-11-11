# Architecture

RAI sits between an LLM framework (LangChain) and ROS 2. It owns what is specific to robots: connectors to the ROS graph, tools with effects, safety hooks, robot state for the model, and recording.

```text
agents       ReActAgent, StateBasedAgent, your BaseAgent       rai.agents
  |  events: model and tool calls, final messages              rai.agents.events
tools        RaiTool: effect + targets + fail-closed hooks      rai.tools, rai.tools.ros2
aggregators  robot state, summarized on demand                  rai.aggregators
  |
connectors   ROS2Connector, ROS2HRIConnector                    rai.communication.ros2
  |  egress hooks on every publish, service call, goal, cancel
ROS2Runtime  one node, one executor, one spin thread (internal)
  |
rclpy        Jazzy or Lyrical
```

## Design rules

-   **Async first, sync available.** Agents, tools and connectors are async and work from any event loop. `connector.sync`, `ThreadedAgent` and `rai.run` serve blocking code.
-   **Native ROS messages.** Connectors take and return ROS message objects. Images and point clouds become numpy views without copies. Dicts appear only at the LLM boundary, inside tools.
-   **RAI content types.** `Text`, `Image` and `Audio` hold numpy data and are converted to LangChain content blocks only when a model is called.
-   **Fail closed.** A safety hook that denies, crashes or times out blocks the action. There are two hook points, one in every tool call and one in every ROS egress. See [Safety](safety.md).
-   **Errors go to the model.** A tool that fails returns an error tool message; the run continues.
-   **Cheap import.** `import rai` loads nothing heavy; LangChain and rclpy load on first use.

## Pages

| Concept                                     | What it covers                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| [Connectors and the runtime](connectors.md) | Topics, services, actions, TF, QoS presets, threading, lifecycle                 |
| [Agents and lifecycle](agents.md)           | The agent state machine, `ReActAgent`, `StateBasedAgent`, events, running agents |
| [Tools and effects](tools.md)               | `RaiTool`, effects and targets, tool hooks                                       |
| [Safety](safety.md)                         | The two hook points, egress hooks, the MOTION set, EStop, confirmation, audit    |
| [Content and HRI messages](content.md)      | `Text`, `Image`, `Audio`, `HRIMessage` and the LLM boundary                      |

## Packages

| Package                | Contents                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `rai_core`             | The `rai` Python package: everything on this page                                                     |
| `rai_interfaces`       | RAI's ROS 2 messages and services (`HRIMessage`, perception and whoami services); separate repository |
| `rai_bringup`          | Launch files (HRI, voice, open-set perception)                                                        |
| `rai_s2s`              | Speech recognition, text to speech and voice agents                                                   |
| `rai_whoami`           | Robot identity from documentation, images and URDFs                                                   |
| `rai_perception`       | Open-set detection and segmentation (GroundingDINO, Grounded SAM 2)                                   |
| `rai_semap`            | Semantic map of detected objects                                                                      |
| `rai_bench`, `rai_sim` | Benchmarks and simulator bridges                                                                      |
| `rai_finetune`         | Fine-tuning on embodied data (experimental)                                                           |

RAI is a set of Python packages rather than ROS 2 packages: its dependencies come from PyPI and need precise versions, and most of RAI works without ROS 2. The ROS 2 parts are `rai_interfaces` and `rai_bringup`, built with colcon.

## rai_core modules

| Module                        | Purpose                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `rai.settings`, `rai.llm`     | Settings from `rai.toml` and `RAI_*` variables; chat and embeddings models       |
| `rai.agents`                  | Agent lifecycle, `ReActAgent`, `StateBasedAgent`, events, `HRIOutput`, `rai.run` |
| `rai.tools`                   | `RaiTool`, `rai_tool`, effects, tool hooks                                       |
| `rai.tools.ros2`              | Generic ROS 2, Nav2, manipulation and diagnostics tools                          |
| `rai.tools.policies`          | Learned-policy (VLA) skills                                                      |
| `rai.communication`           | `HRIMessage`, `HRIConnector`, connector errors                                   |
| `rai.communication.ros2`      | `ROS2Connector`, `ROS2HRIConnector`, QoS, egress hooks, numpy views              |
| `rai.content`, `rai.messages` | Content types and their conversion to LangChain messages                         |
| `rai.aggregators`             | Robot state sources for `StateBasedAgent`                                        |
| `rai.safety`                  | `EStop`, `ConfirmActuation`, `RateLimit`, `AuditLog`, `install`                  |
| `rai.mcp`                     | MCP server (`rai mcp serve`) and client                                          |
| `rai.manifest`                | What the robot is and what its ROS graph offers, for the model                   |
| `rai.recorder`, `rai.replay`  | MCAP flight recorder; replaying recordings in tests                              |
| `rai.ui`                      | Streamlit chat and configurator                                                  |

The [API reference](../api/index.md) documents each of them.
