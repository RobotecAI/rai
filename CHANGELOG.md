# Changelog

All notable changes to RAI are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the packages follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - 2026-11-11

RAI 3.0 is a rewrite of the core. The goals, in order, were safety, speed and clean code. See [What's new in 3.0](docs/whats_new.md) for an overview.

### Package versions

| Package                     | 2.x    | 3.0.0 release |
| --------------------------- | ------ | ------------- |
| `rai_core`                  | 2.14.1 | 3.0.0         |
| `rai_whoami`                | 0.2.0  | 1.0.0         |
| `rai_s2s`                   | 1.1.0  | 2.0.0         |
| `rai_semap`                 | 0.1.0  | 1.0.0         |
| `rai-sim`                   | 0.0.2  | 1.0.0         |
| `rai-bench`                 | 0.4.0  | 1.0.0         |
| `rai-perception`            | 0.4.0  | 1.0.0         |
| `rai_bringup` (ROS package) | 0.0.1  | 3.0.0         |

Every package's public API changed, so each gets a major bump. Every package that uses `rai_core` requires `rai_core>=3.0.0,<4`, and dependencies between the other packages are pinned the same way (`>=X.0.0,<X+1`).

### Breaking changes

#### Platforms and packaging

-   ROS 2 Lyrical (Python 3.14) and Jazzy (Python 3.12) are supported. Humble is dropped.
-   Model providers and optional features are extras of `rai_core`: `openai`, `anthropic`, `aws`, `google`, `ollama`, `langfuse`, `mcp`, `ui`, `policies`, `recorder` and `all`. The base install no longer pulls any vendor package, langfuse, streamlit or langchain-community. A missing extra raises an `ImportError` that names it.
-   `cv_bridge` and `tf_transformations` are no longer used. `rai.communication.ros2.images` converts images, `rai.tools.ros2.navigation.geometry` converts yaw. numpy 2 is required on both distros.
-   `rai_nomad` is removed (it pins numpy<2, which does not resolve on Python 3.14).
-   Docker: one multi-stage `docker/Dockerfile` with `ARG ROS_DISTRO` (jazzy, lyrical), amd64 only. The `DEPENDENCIES` build argument is gone.

#### Settings and models

-   `config.toml` is no longer read. Settings come from `rai.toml` and `RAI_*` environment variables (pydantic-settings). Models are `"provider:model"` strings built by `rai.get_llm`.
-   `import rai` is lazy and no longer imports ROS 2 or vendor packages.

#### ROS layer

-   `ROS2Connector` is async, with blocking twins under `connector.sync`. The 2.x API is deleted: `send_message`, `receive_message`, `register_callback`, `unregister_callback`, `service_call`, `start_action`, `terminate_action`, `get_transform`, `shutdown` (use `close`), the `executor_type` and `destroy_subscribers` arguments and the `node` property.
-   Deleted with it: `BaseConnector`, `BaseMessage`, `ROS2Message`, `IROS2Message`, `ROS2HRIMessage`, `ros_async.get_future_result`, the waiters polling helpers and `convert_ros_img_to_cv2mat`. `ConnectorException` and `ConnectorClosedError` move to `rai.communication.exceptions`.
-   `ROS2HRIConnector` uses `send`, `subscribe` and `receive`. `HRIMessage` is a plain dataclass of text, images and audio; `HRIConnector` is a Protocol.
-   Native ROS messages are the API. Images and point clouds are numpy views of the message.

#### Agents and tools

-   Agents are rebuilt on LangChain 1.x `create_agent` with an explicit lifecycle state machine. Run them with `rai.run` or `ThreadedAgent`.
-   The multimodal message classes (including `ToolMultimodalMessage`) are replaced by the `Text`, `Image` and `Audio` content types. pydub leaves the API.
-   Aggregators v2: `BaseAggregator` and the `ROS2*Aggregator` classes are replaced by `Aggregator`, `RosoutDigest`, `LastImage` and `ImageSummary`, owned by `StateBasedAgent`.
-   ROS 2 tools are `RaiTool` subclasses on the async connector. Nav2 tools use `send_goal` and `GoalHandle` instead of module globals. `ManipulatorMoveTo` calls through the connector.
-   The ROS 2 CLI toolkit keeps only read commands, and the diagnostics toolkit replaces it.
-   `rai.tools.timeout` is removed.
-   The Streamlit UI is rebuilt on the agent event stream.

#### Extensions

-   `rai_s2s`: speech uses `rai.content.Audio`; `[asr]` and `[tts]` are read through `rai.settings.section`; `SoundDeviceConnector` has its own small API; the speech agents are `SyncAgent`s.
-   `rai_whoami`: the agents are `SyncAgent`s run with `rai.run`.
-   `rai_perception`, `rai_semap`, `rai_sim` and `rai_bench`: ported to the 3.0 connector and tool APIs. The deprecated `GroundedSamAgent` and `GroundingDinoAgent` are removed, and the perception utilities lose their `bridge` argument.

### Added

#### ROS layer

-   `ROS2Runtime`: one node, executor and spin thread shared by the connectors, with an entity registry and clean shutdown. `EventsExecutor` is the default; `executor="single"` selects `SingleThreadedExecutor`. No user code runs on the executor thread.
-   `rai.communication.ros2.compat`: every Jazzy and Lyrical difference, detected by feature probes.
-   QoS presets, `Subscription` and `GoalHandle` objects, `ROS2Connector.declare_parameter` and `ROS2Connector.parameter`.
-   `call(idempotent=True)` resends an unanswered service request.

#### Safety

-   `RaiTool`: LangChain tools with effects, ROS targets and fail-closed hooks. Tool errors go back to the model as tool messages.
-   Egress hooks on every publish, service call, goal and cancel, whichever API issues them. Hooks fail closed (raising, timing out or returning anything else denies) and can grant leases. `MOTION` lists the hardware-moving types.
-   Built-in hooks in `rai.safety`: `EStop` (ROS trigger on `/rai/estop` and `/rai/estop/set`, cancels active MOTION goals on engage), `ConfirmActuation` (opt-in), `RateLimit` and `AuditLog`.
-   A runtime warning once per call site that reaches for `connector.node`.

#### Agents and tools

-   `AgentEvent` stream with the model name, prompt, observations and tool artifacts.
-   `StateBasedAgent` adds robot state to each model request without writing it into the history.
-   ROS 2 diagnostics toolkit: `/rosout`, `/diagnostics`, lifecycle states, TF health and topic rate and QoS through the connector.
-   MCP: `rai mcp serve` exposes the ROS 2 tools to any MCP client with egress hooks enforced; agents use MCP servers as tools through `langchain-mcp-adapters`.
-   Robot capability manifest: the robot spec, the rai_whoami description and the live ROS graph, as prompt middleware and as an MCP resource.
-   Flight recorder: agent runs recorded to MCAP next to ROS topics, viewable in Foxglove.
-   Record and replay harness to run agents from MCAP files in CI without a robot.
-   Learned-policy (VLA) skill adapter: openpi, GR00T and HTTP policy servers as cancellable tools (`policies` extra).

#### Packaging and CI

-   uv workspace in a `--system-site-packages` venv (`scripts/dev_env.sh`), with no `PYTHONPATH` changes.
-   CI runs the suite in the Lyrical and Jazzy images, which are published to ghcr.
-   Coverage gate for `rai_core` at the measured floor (83.94% on Lyrical, fails under 83%), and 85% on patches through Codecov.
-   pyright gate: strict on `rai.communication`, standard on `rai_core` and every other package except `rai_bench`.
-   A/B connector benchmark job (`bench.yml`), non-blocking during its burn-in.
-   `rai_core` ships `py.typed`.

#### Docs

-   Docs restructured and rewritten for 3.0, with a "What's new in 3.0" page. The Python and TOML snippets in the docs are checked by tests.

### Changed

#### ROS layer

-   Measured on Lyrical (median of 3 runs): a 1 kHz stream delivers 2000 of 2000 messages (44 on 2.x, where `MultiThreadedExecutor` busy-loops with rmw_fastrtps), with p50/p99 latency of 0.04/0.08 ms.
-   HRI messages go straight to and from `rai_interfaces/msg/HRIMessage`: a VGA HRI publish takes about 0.1 to 0.5 ms (1.1 to 1.2 ms before), decoding 0.002 ms (0.1 ms before). Audio travels as int16 samples.
-   Tool permission lists (`readable`, `writable`, `forbidden`) compare names resolved by the connector, so `cmd_vel` cannot bypass `forbidden=["/cmd_vel"]`.

#### Packaging

-   Base `rai_core` install: 45 packages and 289 MB (installed RECORD sizes, Python 3.14), down from 122 packages and about 770 MB in 2.14.1.
-   `rai_s2s` declares `torch` directly and drops `torchaudio` and the `onnxruntime-gpu` extra (it overwrites the `onnxruntime` install); the README documents the manual swap.
-   `rai_s2s` and `rai_whoami` declare their dependency on `rai_core`, and `rai-bench` on `rai-perception`.
-   Dependencies bumped: opencv 5, langfuse 5, elevenlabs 2, kokoro-onnx 0.6.
-   `rai_interfaces` is pinned to tag 0.3.0 in `ros_deps.repos`.

### Removed

-   Humble support, `rai_nomad`, `config.toml`, `rai.tools.timeout`, the 2.x connector API and the multimodal message classes (see Breaking changes).
-   Dead code in `rai_core`: the `plan_agent` and `ros_logs` modules and the unused `ROS2TopicAPI` helpers.
-   The CLI toolkit write commands.

### Fixed

#### ROS layer

-   Service and goal responses lost after discovery: rmw_fastrtps drops a response when the server has not yet matched the client's reply reader. With 20 fresh server processes, 6 to 48 of 200 service responses and 7 to 33 of 40 goal responses were lost on both distros. RAI servers now answer only once the requester's reply reader is in their graph (0 lost after the fix), and RAI clients settle before their first request to a new participant or server. A goal whose acceptance timed out is cancelled by its id.
-   A client, publisher or subscription replaced by one of another type is destroyed and gone from the graph before the new one is created; rmw_fastrtps refused the new one.
-   An AUTO QoS subscription created before any publisher lost fragments of large messages (43 of 60 VGA frames, 21 of 40 point clouds at 10 Hz). It is now re-created at its first match.
-   A latched sample that arrived before the first consumer was attached is delivered exactly once.
-   The rclpy context RAI initialized is shut down before exit. `close()` cancels pending calls and destroys entities, with no entity leaks over repeated calls.
-   Hardening branch fixes: NodeDiscovery never exposes entries outside its allowlist, the cancel tool reports the real outcome, action result and feedback tools answer unknown ids instead of raising, and Nav2 tools ignore late callbacks of an older goal.
-   Images honour the message encoding instead of assuming rgb8.

#### Agents, tools and extensions

-   `from_langchain` keeps tool artifacts and error status.
-   Tests no longer write `artifact_database.pkl` into the working directory.
-   `rai mcp serve` serves the diagnostics toolkit and tools whose argument schema is a `pydantic.v1` model, and lists the diagnostics tool group in `--help`.
-   `rai_whoami` answers embodiment info requests when fields are unset.
-   Safety confirmations ask through the HRI connector's `send` and `subscribe`.

[unreleased]: https://github.com/RobotecAI/rai/compare/3.0.0...HEAD
[3.0.0]: https://github.com/RobotecAI/rai/releases/tag/3.0.0
