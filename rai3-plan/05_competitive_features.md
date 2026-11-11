# RAI 3.0 plan: features from the competitive landscape

Survey date: 2026-11-11. This report covers features that similar or adjacent projects ship or publish and that RAI does not have yet. Items already planned for 3.0 are excluded: Lyrical/Jazzy only, the hardened zero-copy ROS layer, the async runtime with an explicit lifecycle, RAI-owned typed tools with LangChain and MCP adapters, safety hooks (e-stop, confirm/veto, rate limits, audit log), Pydantic settings, aggregators v2, the benchmark CI gate, the coverage gate and the docs rewrite. Sizes use the scale from `03_agents_tools.md`: **S** is under 1 day, **M** is 1-3 days and **L** is 3-7 days of agentic implementation plus review.

Each feature was judged on two things: whether it fits RAI as a framework (a reusable mechanism, not a product or a demo) and whether it helps robotics engineers who build LLM agents on ROS 2.

## Summary

| # | Pri | Feature | Size | Main 3.0 dependency |
|---|---|---|---|---|
| 1 | P0 | Live robot capability manifest | M | typed tools, settings, MCP server |
| 2 | P0 | Agent flight recorder (MCAP) | M | hooks (observe events), AuditLog, agent event stream |
| 3 | P0 | Record/replay harness (bag-backed connector) | M | connector layer, benchmark CI |
| 4 | P0 | VLA / learned-policy skill adapter | L | async runtime (cancel), ACTUATE effect, safety hooks, zero-copy images |
| 5 | P0 | ROS diagnostics and introspection toolkit | M | typed tools (A9 port) |
| 6 | P1 | Event-driven triggers (reactive wake-ups) | M | async runtime, aggregators v2 |
| 7 | P1 | Spatio-temporal episodic memory | L | zero-copy images, typed tools, rai_semap |
| 8 | P1 | Action outcome verification | M | hooks (post-call), aggregators v2 |
| 9 | P1 | VLM pointing and spatial grounding tools | M | zero-copy images, typed tools, rai_perception |
| 10 | P1 | Recorded-data analysis toolkit | M | MCP server, typed tools, feature 2 |
| 11 | P2 | Agent Skills (SKILL.md playbooks) | S | typed tools, settings |
| 12 | P2 | Explicit plans and behavior-tree execution | L | typed tools, async runtime, hooks |
| 13 | P2 | Programmatic tool calling (code-as-policy sandbox) | M | ToolExecutor, hooks |
| 14 | P2 | Remote transport for off-board agents | L | connector layer |
| 15 | P2 | Adversarial safety benchmark | M | safety hooks, benchmark CI |
| 16 | P3 | Semantic safety rules (preconditions, temporal logic) | L | hooks |
| 17 | P3 | Open-vocabulary 3D scene graph | L | feature 7, rai_perception |
| 18 | P3 | Realtime speech-to-speech model backends | M | async runtime, rai_s2s |
| 19 | P3 | Multi-robot coordination and actuator arbitration | L | async runtime, MCP client, hooks |
| 20 | P3 | Agent modes and model routing | M | settings, async runtime |

---

## P0: must have in 3.0

### 1. Live robot capability manifest

**What it is.** At startup, RAI introspects the live ROS 2 graph (topics, services, actions, types, parameters, TF frames) and merges the result with a small per-robot spec file (limits, preferred high-level interfaces, forbidden interfaces, usage examples). The merged manifest is injected into the system prompt and exposed as an MCP resource.

**Who has it.**
- [ROSClaw](https://arxiv.org/abs/2603.26997) (2026): a discovery node injects a manifest of topics, services, actions and message types plus safety limits (for example 1.0 m/s and 1.5 rad/s on TurtleBot3) into the prompt. Switching robots needed only a config and allowlist change (10-15 minutes).
- [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server) (v3.1.2, Sept 2026, about 1.5k stars): a `robot_specifications/*.yaml` per robot (for example `unitree_go2.yaml`) holds prompts that point the LLM at preferred services, with examples. It also has `robot_config` tools and a `robot_specs` MCP resource.
- [ROSA](https://github.com/nasa-jpl/rosa): `RobotSystemPrompts` and a tool `blacklist` parameter.
- [ros2_lingua](https://github.com/purahan/ros2_lingua): nodes self-advertise typed capabilities through a `LinguaMixin`.

**Why it matters for RAI.** rai_whoami builds an embodiment description offline from documents and URDFs, but nothing tells the agent what the running graph actually offers. Today users hand-write prompts listing topics. A manifest also makes the generic ROS 2 tools much more reliable, because the model stops guessing names and types. ROSClaw attributes part of its gain over ROSA (84.0% vs 75.5% task completion, 9% vs 14% out-of-policy prompts) to this executive layer.

**Size.** M.

**Dependencies.** Typed tools (tool effects and targets map onto the manifest), Pydantic settings (the spec file), and the MCP server (to expose it as a resource). It should reuse rai_whoami's output as the static half, not replace it.

### 2. Agent flight recorder (MCAP)

**What it is.** Every LLM turn, tool call, tool result, hook decision, aggregator observation and HRI message is written as typed channels into an MCAP file, time-aligned with selected ROS topics. The file opens directly in Foxglove or Rerun next to the camera and TF data.

**Who has it.**
- [dimos](https://github.com/dimensionalOS/dimos): `dimos.memory` stores streams in SQLite with an experimental Rust-backed SQLite and MCAP recorder. Replay runs open in Rerun, and the `agentspy` and `lcmspy` CLIs inspect agent and stream activity.
- [ROSClaw](https://arxiv.org/abs/2603.26997): an audit logger records every decision before gating, so blocked attempts are kept too.
- [Foxglove agents](https://docs.foxglove.dev/docs/agents) (Aug 2026): agents inspect loaded recordings and link test failures to recordings. The [Foxglove SDK](https://docs.foxglove.dev/docs/sdk/logging-messages) writes custom JSON-schema channels to MCAP.
- [OM1](https://github.com/OpenMind/OM1): a bundled Prometheus and Grafana stack with LLM and ASR latency dashboards.
- [OpenTelemetry GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): span names such as `invoke_agent` and `execute_tool` (still in development status in 2026).

**Why it matters for RAI.** RAI already sends traces to Langfuse and LangSmith, but those traces are not aligned with robot data. When a robot does something wrong, the engineer needs to see what the camera showed, what the LLM was told and what it decided, all on one timeline. This is the robotics-native version of tracing, and it is also the data source for features 3, 7 and 10 and for rai_finetune.

**Size.** M.

**Dependencies.** The hooks system (observe events are the natural tap), the planned AuditLog hook (the recorder is a second sink for the same events), and the agent event stream (A6). An OTel exporter can be added later as another sink on the same events.

### 3. Record/replay harness (bag-backed connector)

**What it is.** A connector that serves topics, TF and canned service or action responses from an MCAP or rosbag, with a clock, so an agent runs against recorded data without a robot or simulator. Combined with feature 2, a recorded session can be re-run with a new prompt or model and diffed.

**Who has it.**
- [dimos](https://github.com/dimensionalOS/dimos): a global `--replay` flag (for example `dimos --replay run unitree-go2`) runs any blueprint on recorded data, including agentic and memory blueprints.
- [ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr): builds and queries memory offline from ROS 2 bags.
- [Foxglove](https://foxglove.dev/blog/foxglove-goes-agentic): agents work on recordings rather than live robots.

**Why it matters for RAI.** Today RAI needs O3DE, a real robot or hand-written mocks (`rai_bench/tool_calling_agent/mocked_ros2_interfaces.py`) to exercise an agent. A bag-backed connector gives fast, deterministic regression tests for prompts, aggregators and perception, and lets contributors without the simulator work on agent logic. It also makes the planned benchmark CI cheaper.

**Size.** M.

**Dependencies.** The hardened connector layer (the replay connector implements the same interface) and the benchmark-in-CI work. Actuation tools need a mock or record-only mode, which the typed tool effects make easy.

### 4. VLA / learned-policy skill adapter

**What it is.** A standard way to wrap a remote visuomotor policy (pi0/pi0.5, GR00T N1.x, SmolVLA, OpenVLA, ACT) as a single long-running RAI tool such as `run_policy(instruction, max_duration)`. The adapter builds observations from ROS topics, calls the policy server, executes action chunks at control rate on ROS 2 (ros2_control or a topic), reports progress, and supports preemption and timeouts. The LLM stays the orchestrator and the VLA is the motor skill.

**Who has it.**
- [Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) and [Robotics-ER 1.5](https://developers.googleblog.com/en/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/) (Sept 2025): the ER model is an orchestrator that calls a VLA or user functions as tools.
- [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T): a ZMQ `run_gr00t_server.py` and `PolicyClient`. NVIDIA's forum guidance says the server is stateless at the task level, so task memory and sequencing must live in an external manager.
- [LeRobot async inference](https://huggingface.co/docs/lerobot/async): a gRPC policy server and robot client that overlap action chunks.
- [openpi remote inference](https://github.com/Physical-Intelligence/openpi): a websocket `WebsocketClientPolicy.infer(obs)` returning `(chunk, action_dim)` arrays.
- [dimos](https://github.com/dimensionalOS/dimos): VLA and other learned policies alongside classical skills.
- [RoboClaw](https://arxiv.org/abs/2603.11558) (2026): a VLM agent orchestrates learned policy primitives for long-horizon tasks.

**Why it matters for RAI.** This is the main architectural pattern of 2025-2026 ("System 2 LLM/VLM over System 1 VLA"), and none of these projects provides the ROS 2 side. GR00T explicitly leaves the external manager to someone else. RAI is well placed to be that manager because it already owns ROS 2 I/O, safety hooks and the agent loop. Without this, RAI's manipulation story stays tied to GroundingDINO, SAM and MoveIt pick-and-place.

**Size.** L (the generic adapter plus two backends, for example LeRobot and openpi; GR00T can follow).

**Dependencies.** The async runtime (a long-running, cancellable tool with progress), the ACTUATE tool effect plus e-stop and rate-limit hooks (the policy's command stream must also pass through the hooks' halt path), and zero-copy images (observations at policy rate). It should ship as an optional extra, not in core.

### 5. ROS diagnostics and introspection toolkit

**What it is.** Typed, read-only tools for debugging a ROS 2 system: query `/rosout` by node, level and time; read `/diagnostics` and diagnostic_aggregator status; list lifecycle node states; get TF tree health (stale or missing frames); measure topic rate, latency and QoS mismatches; inspect node graphs and parameters. These are native rclpy implementations, not `ros2` CLI subprocesses.

**Who has it.**
- [ROSA](https://github.com/nasa-jpl/rosa): its headline use case is "inspect, diagnose, understand", with ROS 1 and ROS 2 tool packages for logs, nodes, parameters and graph queries.
- [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server): `nodes`, `parameters` and `connection` tool modules, and an industrial end-effector diagnosis demo in which the LLM found a root cause.
- [ROSClaw](https://arxiv.org/abs/2603.26997): `ros2_param_get` and `ros2_param_set` among its eight tools.
- [wise-vision ros2_mcp](https://github.com/wise-vision/ros2_mcp).

**Why it matters for RAI.** RAI's tools are action-oriented (publish, call, navigate, grasp). For introspection it only has `ROS2CLIToolkit`, which shells out to `ros2 node/param/topic` with a 5 s timeout. The fastest way for an engineer to get value from an LLM on a ROS system is "why is my robot not moving", and that is exactly where ROSA and ros-mcp-server win today. Once the planned MCP server ships, this toolkit makes RAI a credible drop-in ROS debugging server for Claude Code, Cursor and similar clients.

**Size.** M.

**Dependencies.** The typed tool port (A9). All of these tools are READ effect, so the safety hooks do not gate them.

---

## P1: should have

### 6. Event-driven triggers (reactive wake-ups)

**What it is.** Declarative triggers that start or interrupt an agent turn when a condition on ROS data holds: a message on a topic, a predicate on a field (battery below 20%), an action result or failure, a perception event, a timer, or a diagnostics level change. Triggers are debounced, have a cooldown and a priority, and enqueue an event with context into the agent.

**Who has it.**
- [OM1](https://docs.openmind.com/): mode transitions with `input_triggered`, `time_based` and `context_aware` rules, priorities and cooldowns, plus a fixed-`hertz` cortex loop.
- [rosaOS](https://aclanthology.org/2026.acl-demo.47/) (ACL 2026 demo): all user input, worker callbacks and errors flow through one event queue that the kernel agent consumes in turn.
- [dimos](https://github.com/dimensionalOS/dimos): agents are modules subscribed to typed streams.

**Why it matters for RAI.** Today a RAI agent either waits for an HRI message or runs a StateBased loop that calls the LLM on every tick. Most real deployments are reactive: an operator wants "tell me if the gripper faults" or "replan when Nav2 aborts". Triggers cut LLM cost and latency and make agents proactive without polling.

**Size.** M.

**Dependencies.** The async runtime (an event queue and interrupting a running turn) and aggregators v2 (a trigger can attach a fresh observation as context).

### 7. Spatio-temporal episodic memory

**What it is.** A store of observations (captions, detections, images and tool outcomes), each with a timestamp, robot pose and embedding, plus tools such as `recall(text, time_range, near_pose)`, `when_last_seen(x)` and `where_was(x, t)`. It answers "where did I leave the cup", "what changed in aisle 3 since yesterday" and "how long ago did someone pass here".

**Who has it.**
- [NVIDIA ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr) ([blog](https://developer.nvidia.com/blog/using-generative-ai-to-enable-robots-to-reason-and-act-with-remembr)): VILA captions with time and pose in Milvus, an LLM agent that calls text, position and time retrieval in a loop, and a ROS 2 demo on Nova Carter.
- [dimos memory](https://github.com/dimensionalOS/dimos): a `Store`/`Stream` API with `.at()`, `.before()`, `.near(pose, radius)`, `.search(embedding, k)` and `.tags()` filters, SQLite and vector backends, and JPEG codecs.
- [RoboOS-NeXT](https://arxiv.org/abs/2505.03673) (Oct 2025): a unified spatio-temporal embodied memory.
- [OM1](https://github.com/OpenMind/OM1): long-term memory of interactions and daily logs.

**Why it matters for RAI.** rai_semap stores object detections against map locations but has no time dimension, no free-text or visual recall and no episodic events. Long-running service robots need "what happened" as much as "where is it". Every serious 2025-2026 competitor has some version of this.

**Size.** L.

**Dependencies.** Zero-copy images (cheap ingestion), typed tools, and the flight recorder (feature 2) as a possible ingestion source. `03_agents_tools.md` rejects persistent memory stores in core, so this should be rai_semap v2 or a `rai_memory` extension, not part of rai_core.

### 8. Action outcome verification

**What it is.** A reusable post-action check: after an ACTUATE tool returns "success", a verifier (VLM on before and after frames, a state predicate, or both) decides whether the intended effect happened and returns a structured verdict with a reason. A failure is fed back to the LLM so it can retry, replan or ask for help.

**Who has it.**
- [Gemini Robotics-ER](https://ai.google.dev/gemini-api/docs/robotics-overview): "success detection", and in ER 2, progress classification on video.
- [AHA](https://aha-vlm.github.io) (NVIDIA, UW, MIT; ICLR 2025): an open VLM for free-form failure reasoning, reporting +21.4% task success when used for sub-task verification.
- [ROS-LLM (Huawei)](https://arxiv.org/abs/2406.19741): reflection from human and environment feedback.
- [rosaOS](https://aclanthology.org/2026.acl-demo.47/): LLM-driven retry or escalation on callback failure.

**Why it matters for RAI.** Nav2 and MoveIt report "succeeded" when the controller finished, not when the task was done (the object may have slipped). LLM agents compound these silent failures over long horizons. A standard verifier hook is cheap and turns RAI's existing perception into closed-loop task reliability.

**Size.** M.

**Dependencies.** The hooks system (a post-call hook on ACTUATE tools, which can rewrite the result) and aggregators v2 (before and after snapshots).

### 9. VLM pointing and spatial grounding tools

**What it is.** Tools in which a VLM returns 2D points, boxes or short trajectories for a natural-language query ("point to the handle", "where can I place this"). RAI lifts them to 3D with depth, camera info and TF, and returns poses in a requested frame. The VLM backend is pluggable (Gemini Robotics-ER, Molmo, Qwen-VL, RoboPoint).

**Who has it.**
- [Gemini Robotics-ER](https://ai.google.dev/gemini-api/docs/robotics-overview): `[{"point": [y, x], "label": ...}]` normalized to 0-1000, plus boxes and trajectories.
- [Molmo](https://molmo.allenai.org) and [RoboPoint](https://robo-point.github.io): open pointing models.
- [GR00T N1.6](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow) and Cosmos Reason: grounding for planning.

**Why it matters for RAI.** rai_perception covers object detection and segmentation (GroundingDINO, SAM), but not affordances or free-form spatial queries such as part, region or placement. Pointing is how the frontier embodied-reasoning models expose spatial understanding, and turning a pixel into a pose in a frame is exactly the ROS-side glue engineers have to write by hand.

**Size.** M.

**Dependencies.** Zero-copy images, typed tools, and the existing rai_perception 3D back-projection (`GetObjectPositionsTool`).

### 10. Recorded-data analysis toolkit

**What it is.** Typed tools, also exposed through the MCP server, that let an agent query MCAP files and rosbags: list topics and time ranges, extract and summarize a time window, compute statistics (rate, gaps, min and max on fields), search `/rosout` in a bag, sample frames for a VLM, and cut a sub-bag around an event.

**Who has it.**
- [Bagel](https://github.com/Extelligence-ai/bagel) (Bagel 2.0, 2026): an MCP server over ROS 1/2 bags, MCAP, PX4 and others, with natural-language data-reduction pipelines that are previewed before they run, and export to Rerun and PlotJuggler.
- [ROSBag MCP Server](https://arxiv.org/abs/2511.03497) ([code](https://github.com/binabik-ai/mcp-rosbags)): a Nov 2025 paper with a UI to benchmark LLM providers on it.
- [Foxglove MCP server](https://docs.foxglove.dev/docs/agents/mcp-server) (Aug 2026): inspect loaded data, control playback, build layouts and write user scripts. Agents can also run [semantic search](https://foxglove.dev/blog/introducing-the-agentic-data-platform-for-physical-ai) with NVIDIA Cosmos.

**Why it matters for RAI.** It pairs with features 2 and 3: an agent can explain its own past run, or a field failure, from the recording. Offline analysis is also the safest first use of LLMs for many robotics teams, because nothing moves.

**Size.** M.

**Dependencies.** The MCP server, typed tools, and the flight recorder format (feature 2). It should share the reader with the replay connector (feature 3).

---

## P2: valuable, after the core lands

### 11. Agent Skills (SKILL.md playbooks)

**What it is.** RAI loads [Agent Skills](https://agentskills.io) folders (a SKILL.md file with name and description frontmatter, instructions and optional scripts) and discloses them progressively: only the descriptions sit in the prompt, and the agent loads the full skill on demand. Typical skills are "docking procedure for robot X", "recover from Nav2 abort" or "calibrate gripper".

**Who has it.**
- [Agent Skills open standard](https://agentskills.io) (published Dec 2025, adopted by roughly 30-40 agent platforms by mid-2026).
- [Foxglove agents](https://foxglove.dev/blog/foxglove-goes-agentic): skills as playbooks, for example diagnosing a broken MCAP file.
- An OpenClaw [ros2-skill](https://openskillindex.com/skills/clawdbot-skills-ros2-skill), and [AgenticROS](https://github.com/agenticros/agenticros) (now redirects to Corebrum/corebrum-synapse) with its skill loader.
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app): per-profile prompts and tools.

**Why it matters for RAI.** Procedural knowledge per robot and site is today buried in long system prompts. Skills keep prompts small, are versioned next to the robot's config, and work with any model. It is cheap and follows a standard RAI's users already know from coding agents.

**Size.** S.

**Dependencies.** Typed tools (scripts inside a skill become tools) and settings (skill search paths).

### 12. Explicit plans and behavior-tree execution

**What it is.** An optional plan-then-execute mode in which the LLM first emits a structured plan artifact (a typed step list, or a BehaviorTree.CPP XML using registered tools as leaves), which is validated and optionally shown for confirmation. The plan is then executed by RAI's executor, or by BehaviorTree.CPP/Nav2's BT engine for BT plans, with progress fed back.

**Who has it.**
- [ROS-LLM (Huawei Noah's Ark)](https://arxiv.org/abs/2406.19741): sequence, behavior-tree and state-machine output modes executed as ROS actions and services.
- [BTGenBot-2](https://arxiv.org/abs/2602.01870) (2026): a 1B model that generates BehaviorTree.CPP XML ([BTGenBot repo](https://github.com/AIRLab-POLIMI/BTGenBot)).
- [MetaTool ROS-Agent](https://arxiv.org/abs/2609.13335) (Sept 2026): a pseudo-code plan before execution, up to about 24% better on complex tasks with open models.
- [ros2_lingua](https://github.com/purahan/ros2_lingua): a backward-chaining planner over declared capabilities.

**Why it matters for RAI.** BehaviorTree.CPP is the de facto ROS 2 standard through Nav2, so a BT plan is something engineers can inspect, edit, reuse and run without the LLM in the loop. An explicit plan also gives the confirm hook something meaningful to approve (a whole plan instead of each call), and it stabilizes smaller local models.

**Size.** L (step-list mode is M; BT export and execution adds the rest).

**Dependencies.** Typed tools (leaf schemas), the async runtime and the hooks (plan-level confirmation).

### 13. Programmatic tool calling (code-as-policy sandbox)

**What it is.** A tool that lets the model write a short Python snippet that calls RAI tools as functions (loops, conditionals, waiting on results), executed in a restricted sandbox. Every inner call still goes through the `ToolExecutor` and hooks.

**Who has it.**
- [Code as Policies](https://code-as-policies.github.io) and [Microsoft PromptCraft-Robotics / "ChatGPT for Robotics"](https://github.com/microsoft/PromptCraft-Robotics): code generation over a high-level robot API.
- [Anthropic programmatic tool calling](https://www.anthropic.com/engineering/advanced-tool-use).
- [Gemini Robotics-ER](https://ai.google.dev/gemini-api/docs/robotics-overview): orchestrates by "calling your robot functions or running generated code".
- [smolagents CodeAgent](https://github.com/huggingface/smolagents).

**Why it matters for RAI.** Tasks like "visit these 6 waypoints and photograph each shelf" cost one LLM round trip per call in ReAct, which is slow and expensive and loses context. A script does it in one turn. Because RAI owns the executor and hooks, it can offer this safely, which ad-hoc code generation cannot.

**Size.** M.

**Dependencies.** The ToolExecutor and hooks (A2, A3). A sandbox choice is needed: a restricted interpreter in-process, or a subprocess with only a tool RPC.

### 14. Remote transport for off-board agents

**What it is.** A connector backend that reaches a robot over rosbridge WebSocket, WebRTC or Zenoh, so the agent host needs no ROS install and can sit on a laptop, a cloud GPU or a different DDS domain or network.

**Who has it.**
- [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server): rosbridge only, and works across ROS 1 and ROS 2 without code changes on the robot.
- [ROSClaw](https://arxiv.org/abs/2603.26997): local DDS, rosbridge (5-10 ms) and WebRTC P2P (20-100 ms), switched by one flag.
- [dimos](https://github.com/dimensionalOS/dimos): Zenoh, LCM, DDS and ROS 2 transports, and dimTELE WebRTC teleop without inbound ports.
- [OM1](https://github.com/OpenMind/OM1): recommends Zenoh for new work.

**Why it matters for RAI.** Many users want to run the LLM agent off-robot (GPU server, developer laptop) against robots they cannot install RAI's dependencies on. rmw_zenoh covers part of this transparently, but a ROS-less client mode widens RAI's reach (Windows or macOS dev machines, cloud) with little core change once the connector interface is clean.

**Size.** L.

**Dependencies.** The hardened connector layer (a second backend behind the same API). Zero-copy guarantees will not hold over this transport, which should be documented.

### 15. Adversarial safety benchmark

**What it is.** A rai_bench suite of adversarial and out-of-policy prompts (jailbreaks, unsafe goals, prompt injection through perception or HRI) against simulated robots. It measures attempted unsafe calls, blocked calls and executed unsafe calls per model and per hook configuration.

**Who has it.**
- [RoboPAIR](https://robopair.org) (jailbreaking LLM-controlled robots).
- [RoboGuard](https://robo-guard.github.io) ([paper](https://arxiv.org/abs/2503.07885), RA-L 2026): cut unsafe plan execution from 92% to under 3%.
- [ROSClaw](https://arxiv.org/abs/2603.26997): 100 adversarial prompts per model, with out-of-policy rates from 9% (GPT-5.2) to 43% (Llama 4) and zero executed after gating.

**Why it matters for RAI.** 3.0 ships safety hooks, and this suite is how to prove they work and stay working. It gives users a model-selection signal that nobody else publishes for ROS 2, and it fits RAI's benchmark identity.

**Size.** M.

**Dependencies.** The safety hooks and the benchmark CI with thresholds.

---

## P3: nice later

### 16. Semantic safety rules (preconditions and temporal logic)

**What it is.** Optional declarative rules on top of hooks: per-tool preconditions and postconditions ("arm stowed before navigation") and scene-grounded rules ("never drive within 1 m of a person", "do not enter zone B while it is occupied"), checked against the plan or the current state before execution. Rules can optionally be compiled to temporal logic and checked against the plan.

**Who has it.**
- [RoboGuard](https://arxiv.org/abs/2503.07885): an LLM grounds rules into LTL, and control synthesis repairs unsafe plans.
- [ros2_lingua](https://github.com/purahan/ros2_lingua): preconditions, postconditions and dependency injection.
- [OM1](https://docs.openmind.com/): governance rules in every fuser prompt (prompt-only, not enforced).
- [ROSClaw](https://arxiv.org/abs/2603.26997): velocity bounds plus an interface allowlist and an optional LiDAR guard.

**Why it matters for RAI.** The planned hooks handle e-stop, confirmation and rate limits. Semantic rules are the next layer, which industrial users will ask for. `03_agents_tools.md` explicitly rejects a permission DSL and policy files in core, so this must be an opt-in hook package, not a core concept.

**Size.** L.

**Dependencies.** The hooks system and feature 1 (rule targets come from the manifest).

### 17. Open-vocabulary 3D scene graph

**What it is.** A hierarchical world model (floors, rooms, places, objects, relations), built from RGB-D and poses, that the LLM can query and expand ("objects in the kitchen", "path from room A to room B"). It also serves as the planning substrate for long-horizon tasks.

**Who has it.** [ConceptGraphs](https://concept-graphs.github.io), [HOV-SG](https://hovsg.github.io), [DovSG](https://arxiv.org/abs/2410.11989) (dynamic updates), [SayPlan](https://sayplan.github.io) (LLM planning over a 3D scene graph with iterative replanning), and [RoboOS](https://github.com/FlagOpen/RoboOS) (shared spatial memory across robots).

**Why it matters for RAI.** It is a natural evolution of rai_semap and feature 7 for large sites, but research code is still fragile and compute-heavy. It is best done after the memory API exists.

**Size.** L.

**Dependencies.** Feature 7, rai_perception and zero-copy point clouds.

### 18. Realtime speech-to-speech model backends

**What it is.** A rai_s2s backend that uses native audio models (OpenAI Realtime, Gemini Live, or open Realtime-compatible servers) end to end, with RAI tools callable from the voice session and barge-in. This complements the current ASR, LLM and TTS pipeline.

**Who has it.** [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) (OpenAI Realtime, Gemini and Hugging Face backends, tools including MCP Spaces), Pollen's [open realtime voice](https://pollen-robotics.com/reachy-mini/blog/open-realtime-voice/) engine, the [Gemini Live with Reachy Mini](https://dev.to/googleai/build-a-talking-robot-with-gemini-live-and-reachy-mini-20e2) tutorial, and [rosaOS](https://aclanthology.org/2026.acl-demo.47/) (Reachy Mini as the agent's front end).

**Why it matters for RAI.** Latency and natural turn-taking dominate the HRI experience. RAI's cascaded pipeline is flexible and local-friendly, but a realtime backend is now the expected option for social and service robots.

**Size.** M.

**Dependencies.** The async runtime (streaming, interruption) and the typed tools (the voice session's tool list).

### 19. Multi-robot coordination and actuator arbitration

**What it is.** Two parts. First, a lease or lock layer so that concurrent agents or tasks cannot command the same actuator group at once. Second, an optional coordinator that delegates tasks to per-robot RAI agents, exposed through MCP or the [A2A protocol](https://a2a-protocol.org), with shared world state.

**Who has it.**
- [RoboOS](https://github.com/FlagOpen/RoboOS) ([paper](https://arxiv.org/abs/2505.03673)): a cloud "brain" plans across robots, per-robot "cerebellum" skill libraries execute, and a Redis shared memory holds state.
- [rosaOS](https://aclanthology.org/2026.acl-demo.47/): kernel and process agents, with the kernel deciding which device each process may access.
- The [OpenClaw-based ROSClaw](https://arxiv.org/abs/2603.26997) work: a parallel ROSClaw by Zhao et al. coordinates heterogeneous robots with a VLM.

**Why it matters for RAI.** Arbitration matters even on one robot as soon as triggers (feature 6), VLA skills (feature 4) and HRI run concurrently. Fleet orchestration is less central for most users. `03_agents_tools.md` rejects multi-agent orchestration in core, so the coordinator belongs in a separate package. Only the lease primitive is a core candidate.

**Size.** L.

**Dependencies.** The async runtime, the MCP client and the hooks (a lease check as a gate hook).

### 20. Agent modes and model routing

**What it is.** Named agent profiles (prompt, tool subset, model and aggregators), for example "idle patrol", "conversation", "manipulation" or "degraded/offline", with rule-based or LLM-requested switching. Routing picks the model per mode or per call (cheap model for chit-chat, ER or VLM model for spatial work, local model when the network drops).

**Who has it.** [OM1](https://docs.openmind.com/) (modes with their own prompt, inputs, actions and LLM, and transition rules with priority and cooldown), [rosaOS](https://aclanthology.org/2026.acl-demo.47/) (self-hosted fallbacks for every external API), and [Gemini Robotics-ER](https://ai.google.dev/gemini-api/docs/robotics-overview) (a `thinking_level` budget per call).

**Why it matters for RAI.** It keeps tool lists small (better tool selection), controls cost, and handles connectivity loss on real robots. It is mostly configuration on top of settings and the runtime, but it is a user-visible concept that needs design.

**Size.** M.

**Dependencies.** Pydantic settings (`provider:model` references), the async runtime, and feature 6 (triggers drive mode switches).

---

## Considered and left out

- **Teleop takeover and web cockpit** (dimos dimTELE, AgenticROS teleop app): this is product surface. RAI's Streamlit integration plus Foxglove covers the framework need.
- **Self-improving skill libraries and autonomous data collection** (RoboClaw's Entangled Action Pairs, Voyager-style macro learning): still research-grade. Revisit after feature 4.
- **Semantic search over recordings** (Foxglove plus NVIDIA Cosmos): a data-platform product. Feature 10 covers the agent-facing part.
- **Ask-for-help under uncertainty** ([KnowNo](https://robot-help.github.io), conformal prediction): the planned confirm/veto hook covers the practical part, and the calibration machinery is model-specific.
- **On-chain governance and identity** (OM1 with FABRIC): hype relative to RAI's users.
- **A generic OpenTelemetry exporter**: deferred by `03_agents_tools.md`. It fits later as another sink of feature 2.

## Surveyed projects

- [NASA JPL ROSA](https://github.com/nasa-jpl/rosa) (about 1.7k stars, active): a LangChain ROS 1/2 agent for inspection and diagnosis. It has `RobotSystemPrompts`, a tool `blacklist`, `astream` events, tool packages and a TurtleSim demo.
- [robotmcp/ros-mcp-server](https://github.com/robotmcp/ros-mcp-server) (about 1.5k stars, v3.1.2 Sept 2026): a FastMCP server over rosbridge, with topics, services, actions, parameters, nodes, image tools and per-robot spec YAMLs.
- [wise-vision/ros2_mcp](https://github.com/wise-vision/ros2_mcp): a native rclpy MCP server with topic, service and action tools and prompts.
- [ajtudela/nav2_mcp_server](https://github.com/ajtudela/nav2_mcp_server): exposes Nav2 navigation, planning, recovery and docking over MCP.
- [Auromix/ROS-LLM](https://github.com/Auromix/ROS-LLM) (last push 2023, stale): a ROS 2 Humble chat-to-robot node with Whisper or AWS ASR. The separate [Huawei ROS-LLM paper](https://arxiv.org/abs/2406.19741) adds BT and state-machine output, imitation-learned actions and reflection.
- [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos) (about 4.7k stars, very active): a ROS-optional agent-native SDK with `@skill` functions over MCP, typed streams over Zenoh, LCM, DDS or ROS 2, a spatio-temporal memory store, replay and simulation modes, and a Rerun viewer.
- [OpenMind OM1](https://github.com/OpenMind/OM1) (about 3k stars, Go runtime with the Python runtime deprecated): a fuser cortex loop at a fixed rate, modes with transition rules, governance prompts, Prometheus and Grafana latency monitoring, and Zenoh, ROS 2 and CycloneDDS plugins.
- [NVIDIA ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr): captioned spatio-temporal memory in Milvus with an LLM retrieval agent and a ROS 2 Nova Carter demo.
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.6, with Cosmos Reason as the planner): a ZMQ policy server and client, stateless at the task level.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) (about 29k stars): [async inference](https://huggingface.co/docs/lerobot/async) policy server and robot client for pi0, pi0.5 and SmolVLA. There is also the community [lerobot-mcp](https://pypi.org/project/lerobot-mcp/) and SO-101 MCP servers.
- [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi): a websocket policy server and a lightweight `openpi-client`.
- [Foxglove agents and MCP server](https://docs.foxglove.dev/docs/agents) (Aug 2026): an agent sidebar, a local MCP server (data inspection, layouts, scripts, playback), skills, and Cosmos-powered semantic search.
- [Microsoft PromptCraft-Robotics](https://github.com/microsoft/PromptCraft-Robotics) (last push 2024, stale): prompt examples and an AirSim sandbox. Its lasting ideas are a high-level API library and code generation.
- [BTGenBot / BTGenBot-2](https://github.com/AIRLab-POLIMI/BTGenBot) ([paper](https://arxiv.org/abs/2602.01870)): small LLMs that generate BehaviorTree.CPP XML.
- [ros2_lingua](https://github.com/purahan/ros2_lingua): an early prototype of a capability contract with preconditions and a backward-chaining grounding planner.
- [ROSClaw (Cardenas et al.)](https://arxiv.org/abs/2603.26997) (Mar 2026): an OpenClaw-to-ROS 2 executive layer with discovery, pre-execution validation, audit logging and three transports. It is benchmarked against ROSA.
- [AgenticROS](https://github.com/agenticros/agenticros) (now Corebrum/corebrum-synapse): an OpenClaw gateway plugin with named robot verbs, a skill loader and a teleop web app.
- [rosaOS](https://aclanthology.org/2026.acl-demo.47/) (ACL 2026 demo, [rosaos.ai](https://rosaos.ai)): an OS-style kernel agent plus process agents over MCP and ROS, with an event queue and Reachy Mini as the user interface.
- [RoboOS](https://github.com/FlagOpen/RoboOS) ([paper](https://arxiv.org/abs/2505.03673)) and RoboOS-NeXT: a brain/cerebellum design with multi-robot shared memory.
- [RoboClaw](https://arxiv.org/abs/2603.11558) ([code](https://github.com/RoboClaw-Robotics/RoboClaw)): a VLM agent unifying data collection, policy learning and deployment.
- [MetaTool ROS-Agent](https://arxiv.org/abs/2609.13335) (Sept 2026): a structured pseudo-code plan before tool execution for open models.
- [Gemini Robotics 1.5 / Robotics-ER](https://ai.google.dev/gemini-api/docs/robotics-overview): an ER orchestrator calling a VLA and user functions, with pointing, success detection and a thinking budget.
- [AHA](https://aha-vlm.github.io): an open failure-reasoning VLM for manipulation.
- [RoboGuard](https://robo-guard.github.io) and [RoboPAIR](https://robopair.org): LLM-robot guardrails via temporal logic, and robot jailbreaking.
- [Bagel](https://github.com/Extelligence-ai/bagel) and [ROSBag MCP Server](https://github.com/binabik-ai/mcp-rosbags) ([paper](https://arxiv.org/abs/2511.03497)): natural-language analysis of bags and MCAP over MCP.
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app): a realtime voice robot agent with profile-scoped tools and MCP Spaces.
- [ConceptGraphs](https://concept-graphs.github.io), [HOV-SG](https://hovsg.github.io), [DovSG](https://arxiv.org/abs/2410.11989) and [SayPlan](https://sayplan.github.io): open-vocabulary 3D scene graphs for LLM planning.
- [Agent Skills standard](https://agentskills.io), [Anthropic advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) and [OpenTelemetry GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/): cross-industry agent standards relevant to RAI's tool and observability layers.
- [ROS 2 Lyrical Luth](https://docs.ros.org/en/lyrical/Releases/Release-Lyrical-Luth.html) (May 2026, LTS to 2031): no agent features. The rosidl::Buffer zero-copy GPU work and the asyncio node tutorial are relevant to the planned ROS layer and async runtime.
