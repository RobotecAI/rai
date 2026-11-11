# RAI 3.0 plan

Status: final, 2026-11-11. This plan merges five planning reports in this directory:

- `01_rclpy_internals.md`: rclpy, rcl and rmw on Jazzy vs Lyrical, with measurements
- `02_ros_layer.md`: RAI's ROS layer, a review of the hardening branch, and the 3.0 ROS architecture
- `03_agents_tools.md`: the agent, tool, hooks, settings and LLM layer
- `04_packaging_ci.md`: packaging, the dev environment, CI, quality gates, dead code and the docs outline
- `05_competitive_features.md`: 20 features found in similar projects, ranked P0 to P3

Each report has file:line evidence and marks every claim as measured or inferred. This file covers the decisions, the architecture and the order of work.

## 0. Working rules

- **Branch:** all 3.0 work happens on `mm/rai-3.0`. It is pushed to the repo regularly. No PRs are opened. The work items below are commits or commit series on that branch, not PRs.
- Every work item lands with its tests. Lyrical and Jazzy must both stay green in Docker after each item.
- 3.0 ships as one release. Migration guides come later.
- The goals are, in order: safety, speed, clean code. Lines of code should go down overall.

## 1. Decisions taken

| Topic | Decision |
|---|---|
| ROS distros | Lyrical first (Python 3.14), Jazzy second (Python 3.12). Humble dropped. |
| Hardening branch (`t3/identify-core-state-machines`) | Not merged as a whole. Its fault suite and benchmark become the acceptance tests ("oracle"). About 8 small fixes are ported as code (R2 in §3). |
| Packages | Separate rai_* packages stay. rai_finetune stays. rai_nomad is dropped. rai_bench and rai_sim are left alone apart from required API ports. rai_whoami stays. |
| Dev env | `uv venv --system-site-packages` plus an editable uv workspace. No PYTHONPATH tricks. Measured working on both distros. |
| Agent framework | LangChain 1.x: `create_agent` plus middleware, and `langchain-mcp-adapters` for the MCP client. Decided by the D1 bake-off (`d1/REPORT.md`); D2 is not triggered. RAI does not build its own tool or hook system. The fail-closed adapter lives on the RAI tool class, so it also covers the MCP server and direct calls. |
| Agents | Async-first with sync wrappers. ReAct and StateBased stay, rebuilt on the chosen framework. StateBased sensor context is injected per model call and never written into the history. |
| Content types | RAI owns `Text`, `Image` (ndarray plus encoding) and `Audio` (ndarray shaped (n, channels) plus a sample rate). They are converted to framework content blocks only at the LLM boundary. pydub leaves the API. |
| Vendors | Every provider becomes an extra. Tier 1: openai (including compatible servers), anthropic, ollama, google. Tier 2: bedrock. langchain-community is removed. |
| Connectors | Separate classes stay (`ROS2Connector`, `ROS2HRIConnector`), sharing one internal runtime. |
| Tool permissions | Not deny-by-default. `readable`/`writable`/`forbidden` stay, compared on resolved names. |
| Safety | A fail-closed adapter on the RAI tool class on the agent side, and RAI egress hooks on the connector side (see 2.3). Both fail closed. ConfirmActuation is opt-in. EStop can be triggered from ROS. The CLI toolkit loses its write commands. Bypasses log a runtime warning. |
| Messages | Native ROS messages are the API. Images and point clouds become numpy views with no copy after deserialization. Dicts only at the LLM boundary. |
| Aggregators | Kept, split into bounded buffers plus async summarizers that run on demand. |
| MCP | `rai mcp serve` exposes all RAI ROS tools by registering them on the `mcp` SDK's FastMCP, with egress hooks still enforced. Agents use MCP tools as a client through `langchain-mcp-adapters`. |
| Quality gates | pyright strict on `rai.communication` from the start and standard mode elsewhere (ruff doesn't check types). Coverage gate: the measured floor plus 85% on new code, with no hard 85% at the tag. Benchmarks report only for two weeks, then block. No arm64 images. |
| Settings | pydantic-settings with `RAI_*` env vars and an optional `rai.toml` (TOML because `tomllib` is in the standard library). Models are `"provider:model"` strings for `init_chat_model` or the equivalent in PydanticAI. The old `config.toml` is not read. |
| Upstream bugs | Reported to the maintainer only (01 §12.13), not filed upstream. |
| Docker | Official Lyrical and Jazzy images are the dev and CI environment. |
| Docs | Rewritten for 3.0. The demos are left as they are. |

## 2. Architecture

### 2.1 ROS layer (details in 02 §3 and 01 §12)

```
tools / aggregators / agents          (typed calls, async or sync)
ROS2Connector     ROS2HRIConnector    (separate public classes)
         \          /
         ROS2Runtime                  (internal: node, executor, spin thread,
                                       entity registry, lifecycle, egress hooks)
         rai.communication.ros2.compat (all Jazzy/Lyrical differences, detected by feature probes)
         rclpy
```

- **Executor.** Drop `MultiThreadedExecutor`. With rmw_fastrtps it busy-loops: at 1 kHz it delivered 206 of 2000 messages using 2.57 s of CPU, and on a single 30 Hz topic it used 89% CPU (01 §3.3). The candidates are `EventsExecutor` (2000/2000 messages, 0.05 s of CPU, experimental upstream, needs Jazzy rclpy ≥ 7.1.11) and `SingleThreadedExecutor` (2000/2000, 0.19 s). **Gate R3:** EventsExecutor becomes the default only if it passes the full oracle on both distros. Otherwise SingleThreaded is the default and EventsExecutor stays opt-in.
- **No user code runs on the executor thread.** It only takes messages, completes futures and fills queues or buffers. Every callback RAI registers is guarded, because one exception kills `spin()` on every executor.
- **Async API usable from any event loop.** `connector.sync.*` runs the same coroutines on a private loop and is a first-class API for scripts, tests and simple tools. `GoalHandle` and `Subscription` objects replace string handles and the module-level stores.
- **Entity registry.** One publisher, subscription, service client or action client per (name, type), reused and destroyed on close. Today main leaks 1000 clients over 1000 calls.
- **Lifecycle.** A single RUNNING → CLOSED transition. `close()` cancels pending futures (callers get `ConnectorClosedError`), stops and joins the executor, then destroys the entities and the node, in that order. It is idempotent and works as a context manager.
- **Native messages only** in the connector API and in callbacks. LLM-facing tools accept dicts and convert them at the tool boundary. `receive(timeout, max_age=None)` raises if `max_age` is set and no sample is fresh enough. One shared runtime per process by default; a connector can still own a private one.
- **QoS presets** replace auto-matching: `SENSOR`, `COMMAND` (never latched), `LATCHED`, `HRI` (never drops), and `AUTO` for subscriptions only. Subscriptions use `BEST_AVAILABLE`.
- **Messages.**
  - `views.image()` and `views.points()` return numpy views via `np.frombuffer`.
  - `encode_image()` is the single place images are encoded for the LLM (JPEG): about 1.2 ms for VGA, against 88 ms on today's path.
  - The `HRIMessage` publish path drops the dict round trip, which costs about 290 ms per VGA image today.
  - `raw=True` is available only for relays and recorders.
  - cv_bridge and tf_transformations are removed from core, which lets Jazzy use numpy ≥ 2 as well.
- **Services and actions.**
  - Timeouts raise one RAI exception on both distros.
  - Readiness is polled at 10-20 ms intervals instead of rclpy's 250 ms.
  - Cancels and results use `*_async` calls with RAI-side timeouts.
  - `result_timeout` is always passed explicitly (Jazzy defaults to 900 s, Lyrical to 10 s).
- **Not adopted:**
  - `rclpy.experimental.AsyncNode`: it deadlocks under high-rate traffic on Lyrical, has no actions, and doesn't exist on Jazzy.
  - `wait_for_message` on a spinning node.
  - Python-level `set_on_new_message_callback`.

### 2.2 Agent, tool and LLM layer (details in 03 §2, revised)

RAI no longer builds its own tool interface, hook registry or MCP layer. It reuses a framework and owns only what's specific to robots.

- **Framework bake-off (D1): LangChain 1.x won.** The same slice was built in LangChain 1.x and in PydanticAI:
  - two ROS tools;
  - a ReAct loop;
  - StateBased sensor context;
  - an approval gate;
  - the fail-closed wrapper;
  - MCP as server and as client;
  - cancellation.

  Both passed the same 25 tests on Lyrical, against a fake robot and against `ROS2Connector` with a real ROS node. Details are in `d1/REPORT.md`.

  Measured:

  | | LangChain 1.4.9 | PydanticAI 2.67 (`pydantic-ai-slim`) |
  |---|---|---|
  | Lines of code for the slice (excluding tests and shared code) | 144 | 157 (no MCP server support, so RAI writes a 40-line adapter) |
  | Import time, framework / whole slice | 0.35 s / 0.48 s | 0.30 s / 0.54 s |
  | Per ReAct step, fake model, with all middleware (median) | 3.2-3.3 ms | 3.5-4.0 ms |
  | Cancel to run end during a slow tool (median) | 0.2-0.3 ms; the tool's `finally` runs | 0.4-0.5 ms; the tool's `finally` runs |
  | Fail-closed wrapper | Tool-level `ToolException` plus `handle_tool_error`; one wrapper covers every path | `WrapperToolset`; one wrapper covers every path |
  | Approval | `HumanInTheLoopMiddleware`; needs a checkpointer; asks before the hooks run | `ApprovalRequiredToolset`; stateless; the order is up to RAI |
  | StateBased context per call, not stored | `awrap_model_call`, 7 lines | `before_model_request`, 6 lines |
  | MCP | client `langchain-mcp-adapters` 0.3.2; server `to_fastmcp` (drops annotations) | client built in; no server |
  | Migration outside rai_core | none | rai_bench (16 files), rai_whoami (9, including the `langchain_community` RAG stack), rai_perception (3), rai_semap (1), examples (8) |

  PydanticAI's only real edge is that its toolset wrapper composes safety in one place. LangChain gets the same by putting the adapter in the tool. LangChain is equal or better on everything else, and it needs no migration.
- **RAI tool class.** A thin typed layer on the chosen framework's tool type, adding `effect` (READ / WRITE / ACTUATE) and ROS `targets`. Tools are async-native and the sync path is a wrapper. Tool errors go back to the LLM as tool messages and never end the run. `rai.tools.timeout` is removed; timeouts and cancellation come from the framework plus the fail-closed wrapper.
- **Agent lifecycle.**
  - An explicit state machine: CREATED → STARTING → RUNNING → STOPPING → STOPPED, plus FAILED.
  - `stop()` before start does nothing; today it hangs forever.
  - Interrupting cancels the running task, including the current tool.
  - `rai.run()` replaces `AgentRunner`.
- **ReAct and StateBased.** Both are built on the chosen framework's agent loop. The StateBased context is an `awrap_model_call` middleware that appends the observation to the request only. Its sensor summaries run concurrently, each with its own timeout, and keep the last good result. It never writes to the history; today the history piles up as `/rosout: /rosout: ...`.
- **Aggregators v2.** Bounded buffers on the connector side, plus async summarizers run only when the LLM needs context, cached for a maximum age.
- **Content types.** RAI's `Text`, `Image` and `Audio` (see §1) replace the `*MultimodalMessage` types and pydub.
- **Settings and models.** As in §1. `import rai` becomes lazy; today it takes 0.7 s.
- **Moved or deleted.** megamind and the structured-output runnable move to rai_bench. `plan_agent`, the conversational agent and the pickle artifact store are deleted. The Streamlit configurator moves behind the `ui` extra.

### 2.3 Safety: two hook points

- **Agent side.** The fail-closed adapter sits inside the RAI tool class: if a hook denies, times out or crashes, the tool raises `ToolException`, and the LLM gets an error tool message while the run continues. Because it is in the tool, it covers the agent, the MCP server and direct calls. Middleware is used only for agent-only concerns, such as StateBased context. D1 measured that `HumanInTheLoopMiddleware` asks the human before any tool-level hook runs, so `ConfirmActuation` is an awaiting hook in the adapter instead.
- **Connector side (RAI-owned; this is the guarantee).** An egress chain covers every publish, service call, goal and cancel.
  - It matches on message type first and name second, because names get remapped and types don't.
  - Decisions are `Allow(lease_s=…)` or `Deny(reason)`. A lease approves a stream such as `cmd_vel` once.
  - It costs nothing when no hooks are registered.
  - It also covers direct tool calls and the MCP server path, which agent-side middleware never sees.
- **MOTION set.** RAI ships a default set of hardware-moving message types (Twist/TwistStamped, NavigateToPose, NavigateThroughPoses, FollowJointTrajectory, MoveGroup, GripperCommand, ManipulatorMoveTo). Users can extend it.
- **Built-in robot hooks:**
  - `EStop`: guards both hook points, cancels every MOTION goal, and can be triggered from ROS through an `/rai/estop` topic or service.
  - `ConfirmActuation`: opt-in, asks over HRI or the CLI.
  - `RateLimit`.
  - `AuditLog`: also feeds the flight recorder (F2).
- **Closing bypasses.**
  - `connector.node` is removed from the public API.
  - The CLI toolkit's write commands (`pub`, `call`, `send_goal`, `param set`) are removed.
  - Extensions that still reach rclpy directly get a runtime warning.

### 2.4 Packaging and CI (details in 04)

- rai_core install size drops from about 770 MB. About 550 MB of it is optional (streamlit, opencv installed twice, langchain-community, the vendor packages) and becomes extras: `openai`, `anthropic`, `aws`, `google`, `ollama`, `langfuse`, `mcp`, `ui`.
- A real uv workspace, and every dependency bumped to its latest version.
- Docker images built on ros-base for Lyrical and Jazzy, published to ghcr. CI runs on them as a matrix.
- `continue-on-error: true` is removed from the test job; today failing tests still show green.

## 3. Work order

Sizes: S is under 1 day, M is 1-3 days, L is 3-7 days, all assuming agentic implementation plus review. The IDs point back to the reports (P = 04, R = 02, A = 03, F = 05).

### Phase 0: foundation (honest CI, clean base)

| ID | Item | Size | Depends on |
|---|---|---|---|
| P0 | Create `mm/rai-3.0` from main. Land the Lyrical spike cleanly: Python 3.12-3.14, no Humble, the endpoint-info compat import, the ruff known-first-party fix | S | none |
| P1 | Remove `continue-on-error` from CI | XS | P0 |
| P2 | Drop rai_nomad. In the same commit, move its rosdep keys (nav2_msgs, tf_transformations, portaudio19-dev, vision_msgs, …) to `rai_bringup/package.xml` | S | P0 |
| P3 | uv workspace plus the `--system-site-packages` dev env. Shrink `setup_shell.sh` and remove empy and osrf-pycommon from the dependencies | M | P0 |
| R1 | Port the fault suite and benchmark from the hardening branch, rewritten against the API, tolerant of load, and marked xfail where main fails. Add an image and point-cloud path, plus a row comparing hooks on and off | M | P3 |
| P9 | Remove dead code (04 §5): `plan_agent`, `ros_logs`, unused topic API methods, unreachable code | S | P0 |

### Phase 1: cores, in parallel

| ID | Item | Size | Depends on |
|---|---|---|---|
| D1 | Framework bake-off, LangChain 1.x vs PydanticAI (see 2.2). Records the decision in this file | M | P3 |
| D2 | Not triggered: LangChain won D1. (It would have migrated rai_bench, rai_s2s, rai_semap, rai_whoami and rai_perception to PydanticAI.) | none | D1 |
| R2 | Port the small fixes from the hardening branch: `b119762`, `7e47ada`, `048aba0`, `8a625cf`, `1f06344`, the ideas from `29a709d`, and in-flight service-call cancellation | S | R1 |
| R3 | `ROS2Runtime`, the compat layer and the async `ROS2Connector`, including the executor decision gate. Acceptance: 100% delivery at 1 kHz, no entity growth over 1000 calls and 200 goals, shutdown under 1 s even with a stuck callback | L | R1 |
| R6 | Egress hook chain and the MOTION set in `ROS2Runtime` (RAI-owned) | S | R3 |
| A1 | `rai.settings` and `rai.llm`, provider extras, a lazy `rai/__init__` | M | P3 |
| A5 | Agent runtime: the state machine, `ThreadedAgent`, `rai.run` | M | P0 |
| P4 | Cut rai_core dependencies down and split extras; enforce layering with a ruff ban (rai_core must not import rai_s2s or rai_perception) | M | P3 |
| P5 | Drop cv_bridge and tf_transformations, add the numpy image converter, use `numpy>=2` everywhere | M | P4 |
| P6 | Bump every dependency. Bumps that break APIs (elevenlabs, langfuse, kokoro) go in their own commits | M ×3 | P4, P5 |

### Phase 2: tools and adapters

| ID | Item | Size | Depends on |
|---|---|---|---|
| A3 | RAI tool class on LangChain `BaseTool` (effect and targets in `metadata`, async-native, errors returned as tool messages), with the fail-closed hook adapter inside the tool (`ToolException` plus `handle_tool_error`), see `d1/REPORT.md` §5 | S | D1 |
| A10 | `rai mcp serve` and the MCP client through the framework (extra `rai-core[mcp]`), with an end-to-end stdio test | S | A3, R6 |
| R4 | `ROS2HRIConnector` on the runtime, the views and the encoder, and the RAI content types (`Text`, `Image`, `Audio`) | M | R3 |
| A9 | Port about 35 ROS 2 tools: effects and targets, `GoalHandle` instead of globals, cached clients, the CLI read/write split, no use of `connector.node` | L | A3, R3 |

### Phase 3: agents

| ID | Item | Size | Depends on |
|---|---|---|---|
| A6 | ReAct and StateBased on the chosen framework, `HRIOutput`, the `AgentEvent` stream, tracing set up once | M | A1, A3, A5 |
| A7 | Aggregators v2 and `StateBasedAgent` | M | A6, R4 |
| A8 | Built-in hooks: `EStop` (with the ROS trigger), `ConfirmActuation` (opt-in), `RateLimit`, `AuditLog`; the bypass warning | M | A3, A6, R6 |
| A11/A12 | Messages cleanup; Streamlit rebuilt on `AgentEvent` (extra) | S+S | A6 |
| R5 | Delete the old connector classes, mixins, `general_callback` and the polling helpers | S | A9, A7 |
| A13 | Port rai_bench, rai_semap, rai_perception, rai_s2s, rai_whoami and the examples to the new connector and tool APIs (the framework part is already done if D2 ran) | L | A3-A9 |

### Phase 4: P0 features (from 05)

| ID | Feature | Size | Depends on |
|---|---|---|---|
| F1 | Live robot capability manifest: the ROS graph plus a per-robot spec plus rai_whoami, given to the model in the prompt and as an MCP resource | M | R3, A10 |
| F2 | Agent flight recorder (MCAP): LLM turns, tool calls, hook decisions and observations, aligned in time with ROS topics, viewable in Foxglove | M | A8 (AuditLog), A6 |
| F3 | Record and replay connector: run agents from MCAP or rosbag files in CI without a robot | M | R3, F2 |
| F4 | Learned-policy skill adapter: pi0, GR00T, SmolVLA or LeRobot policy servers as long-running tools that can be cancelled | L | A3, A9 |
| F5 | ROS diagnostics toolkit: `/rosout`, `/diagnostics`, lifecycle states, TF health and topic rate/QoS, replacing the CLI subprocess calls | M | A9 |

### Phase 5: gates, docs, release

| ID | Item | Size | Depends on |
|---|---|---|---|
| P7/P8 | Docker images (Lyrical and Jazzy, amd64 only, published to ghcr) and CI running on them | M+M | P2, P3 |
| P10 | Coverage gate: the measured floor (about 79% once dead code is gone) plus 85% on new code | S | P8, P9 |
| P11 | pyright strict on `rai.communication` as it is rewritten, and standard mode at zero errors elsewhere (128 today) | M ×4 | P8, P4 |
| P12 | Benchmark job: compares base and head on the same runner, with exact, ratio and report-only thresholds. Non-blocking for two weeks, then required | M | P8, R1 |
| P13 | Docs rewrite (structure in 04 §6) | L | most items |
| P14 | Release 3.0.0: versions, `rai_core<4` everywhere, changelog, one tag | S | all |

The critical path is P0 → R1 → R3 → A9 → A13 → P14, with D1 → A3 → A6 running in parallel.

### Backlog after 3.0 (05: P1 to P3)

- **P1:** event-driven triggers; spatio-temporal episodic memory (as rai_semap v2); action outcome verification; VLM pointing and spatial grounding tools; recorded-data analysis tools.
- **P2:** Agent Skills (SKILL.md); plan-then-execute with behavior trees; programmatic tool calling; remote transport (Zenoh, rosbridge); adversarial safety benchmark.
- **P3:** semantic safety rules; open-vocabulary 3D scene graph; realtime speech-to-speech backends; multi-robot leases; agent modes and model routing.

## 4. Settled during review

- **rai_interfaces on Lyrical:** the bloom release is deferred. `ros_deps.repos` stays, pinned, for the whole 3.0 effort.
- **D1 result:** LangChain won, so D2 does not run and A3 starts right after D1.

## 5. Upstream bugs found (for the maintainer only, not filed)

From 01 §12.13:

- **rmw_fastrtps busy wait:** it skips `rmw_wait` on a shared DataReader `StatusCondition`, so `MultiThreadedExecutor` busy-loops.
- **`AsyncNode` deadlock on Lyrical:** `take` holds the GIL while it waits on the DDS reader lock.
- **Dead wait in `Executor.shutdown(timeout)`:** it never waits for callbacks that are already running.
- **Missing fix on Lyrical:** the rclpy #1749 fix for a multi-threaded shutdown race is not there.
- **Large-message latency:** Fast DDS shows a bimodal 11 ms latency for large messages at low rates.
