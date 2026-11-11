# RAI 3.0 plan, part 02: the ROS layer

Scope: `src/rai_core/rai/communication/**`, `src/rai_core/rai/tools/ros2/**`, `src/rai_core/rai/aggregators/**`, plus an independent review of the hardening branch `feat/ros2-connector-hardening` (worktree `/home/mmajek/.t3/worktrees/rai/t3-aaf19f73`, commits `17237a4..e8250a4` on `38853fa`).

Evidence markers: **[M]** measured in this session (command and numbers given), **[R]** read in code (file:line), **[I]** inferred, not verified. Line numbers on `main` refer to `git show main:<path>` unless noted; branch line numbers refer to `e8250a4`.

## 0. TL;DR

- **Branch verdict: do not merge wholesale. Rewrite the connector core for 3.0 and use the branch's fault suite and benchmark as the test oracle; port ~8 small, self-contained fixes as code.** The branch is mostly correct and it does work on Lyrical (59/59 fault tests, 5 of 5 runs; 299 passed in `tests/communication` + `tests/tools/ros2`), but it hardens a threading design that 3.0 is replacing (async-first), adds ~545 net source lines, three thread pools per connector, two unmarked private-API uses beyond the three `ponytail:` ones, and several silent semantic changes (latched publishers by default, best-effort subscriptions by default, drop-oldest on every callback including HRI).
- **Biggest single finding [M]:** on Lyrical the stream-delivery problem on `main` is the `MultiThreadedExecutor`, not RAI logic. Raw rclpy 10.0.11, one subscription, 1 kHz for 2 s: MT executor delivers 206/2000 using 2.57 s CPU; SingleThreaded 2000/2000 (0.19 s CPU); `EventsExecutor` 2000/2000 (0.05 s CPU); `AsyncNode` 2000/2000 (0.10 s CPU). A 3.0 core built on a single-threaded or events executor that never runs user code on the executor thread removes the need for most of the branch's machinery.
- **Top abstraction problems:** a generic `BaseConnector[T]` whose message model forces every ROS message through a pydantic wrapper; a leaky connector that 20 files bypass via `connector.node`; duplicated, unsynchronized bookkeeping in the topic/action APIs; goal/handle state spread across module-level globals in tools; implicit lifecycle; per-call client creation with no destruction; and slow conversion paths (HRI publish 335 ms per VGA frame, image tool 88 to 600 ms per frame).
- **Proposed 3.0 architecture:** keep `ROS2Connector` and `ROS2HRIConnector` as separate classes, both composed over one small `ROS2Runtime` (node, single-threaded/events executor, spin thread, entity registry, lifecycle); a loop-agnostic async API with sync wrappers; typed `GoalHandle` and `Subscription` objects instead of string handles; native numpy views for images and point clouds with encoding only at the LLM boundary; an egress hook chain on the connector (publish, service call, goal send) for veto/confirmation of hardware-moving commands; aggregators split into bounded buffers plus async summarizers.

## 1. Current ROS layer on main

### 1.1 Abstractions as they are

| Abstraction | Where | What it is |
|---|---|---|
| `BaseMessage` | `communication/base_connector.py:40-60` | pydantic model: `payload: Any`, `metadata: dict`, `timestamp`. |
| `BaseConnector[T]` | `base_connector.py:73-450` | Generic base. Resolves `T` by reflecting `__orig_bases__` (`:89-96`). Owns a 4-worker `ThreadPoolExecutor` for callbacks (`:85-87`), two dicts of callbacks (`:81-84`), and defines ~10 methods that all take `**kwargs` and raise `NotImplementedError`. |
| `HRIMessage`, `HRIConnector[T]` | `communication/hri_connector.py:37-178` | Text, PIL images, pydub audio, conversions to/from LangChain messages (base64 PNG/WAV). `HRIConnector` adds only `build_message`. |
| `ROS2BaseConnector[T]` | `ros2/connectors/base.py:45-572` | `ROS2ActionMixin + ROS2ServiceMixin + BaseConnector[T]`. Owns a `Node`, a `MultiThreadedExecutor` by default (`:154-157`), a spin thread (`:162-163`), a TF `Buffer` + `TransformListener` created eagerly (`:142-143`), a 1 s performance timer (`:146-148`), a `last_msg` cache (`:167`), and three API objects. |
| `ROS2Connector` | `ros2/connectors/ros2_connector.py:19-20` | `ROS2BaseConnector[ROS2Message]`, empty body. |
| `ROS2HRIConnector` | `ros2/connectors/hri_connector.py:32-182` | `ROS2BaseConnector[ROS2HRIMessage]` + `HRIConnector[ROS2HRIMessage]`: a full ROS 2 connector (node, executor, TF listener, services, actions) that also converts `rai_interfaces/HRIMessage`. |
| `ROS2ARIConnector` | does not exist | `grep -rn ARIConnector` finds nothing on main. |
| `ROS2TopicAPI`, `ROS2ServiceAPI`, `ROS2ActionAPI` | `ros2/api/{topic,service,action}.py` | Thin wrappers over `rclpy.Node`, plus caches. |
| `ROS2Message`, `ROS2HRIMessage` | `ros2/messages.py:41-108` | `ROS2Message` is an empty subclass of `BaseMessage`; `ROS2HRIMessage` converts via `cv_bridge` and PIL. |
| `ROS2Context`, `NodeDiscovery` | `ros2/context.py:23-221` | `rclpy.init/shutdown` context manager; periodic graph discovery with an allowlist. |
| `get_future_result` | `ros2/ros_async.py:25-45` | Blocks on a `threading.Event` set from an rclpy done callback; returns `None` on timeout. |
| Tools | `tools/ros2/base.py`, `generic/*`, `navigation/*`, `manipulation/*`, `cli.py`, `simple.py` | LangChain `BaseTool`s holding a `connector` and optional `readable/writable/forbidden` lists. |
| Aggregators | `aggregators/base.py:23-54`, `aggregators/ros2/aggregators.py` | A deque buffer filled by a raw callback, and `get()` that summarizes (logs dedup, last image, VLM description, VLM diff). Driven by `BaseStateBasedAgent` (`agents/langchain/state_based_agent.py:111-182`). |

### 1.2 Which abstractions are wrong for 3.0

1. **`BaseConnector[T]` as a generic message bus is the wrong center.** [R] Every ROS operation is expressed as "send a `T` with `payload: Any`": `send_message` publishes `message.payload` (`connectors/base.py:269-275`), services wrap the response back into `ROS2Message(payload=msg, metadata={"msg_type": str(type(msg))})` (`service_mixin.py:53-55`), and each received message is wrapped into a pydantic model before dispatch (`base_connector.py:236`, `connectors/base.py:290`) even when every registered callback is `raw=True` (aggregators always are, `state_based_agent.py:119-121`). The generic type is recovered by reflecting `__orig_bases__` (`base_connector.py:89-96`). The wrapper carries no information the ROS message does not already carry, costs a pydantic construction per message, and hides the real types from type checkers. For 3.0 the ROS connector should speak native ROS messages (and dicts only at the LLM boundary, inside tools). `HRIMessage` is a real domain type and should stay.
2. **`**kwargs` everywhere.** [R] Every `BaseConnector` method takes `**kwargs: Optional[Any]`, and the ROS overrides add keyword-only arguments on top (`connectors/base.py:242-251`, `:292-302`). Misspelled options are silently accepted. `create_service(on_done=...)` is accepted and dropped (`connectors/base.py:476-510`). With async-first APIs these signatures have to be redone anyway.
3. **Mixins with dead validation.** [R] `ROS2ActionMixin.__post_init__` and `ROS2ServiceMixin.__post_init__` (`action_mixin.py:22-31`, `service_mixin.py:22-31`) are dataclass hooks on non-dataclasses: they never run. The mixins only forward to the API objects. Composition is simpler.
4. **`ROS2HRIConnector` is a whole robot connector.** [R] It inherits node + executor + spin thread + eager TF listener + service/action machinery (`hri_connector.py:32`, `connectors/base.py:134-163`). An agent that uses one `ROS2Connector` and one `ROS2HRIConnector` runs two nodes, two executors and two TF listeners that each deserialize all `/tf` traffic [I]. Keeping the classes separate (user constraint) is fine, but they should share one runtime.
5. **String handles instead of objects.** [R] Actions return a uuid string; the caller must go back to the connector for feedback/result/cancel (`action.py:202-282`); services return a uuid that maps to nothing reachable (`ros2/api/service.py:118-129`). Tools then add three more global dicts on top (`tools/ros2/generic/actions.py:29-33`). A `GoalHandle` object makes most of that bookkeeping disappear.
6. **`connector.node` as the real API.** [R] 20 files outside `rai/communication` use `connector.node` / `connector._node` (grep: `connector\.node\b|connector\._node`), including core tools: `manipulation/custom.py:88,220,424` (`create_client` per call, never destroyed), `navigation/nav2_blocking.py:112` (new `ActionClient` per call, never destroyed, `send_goal` with no timeout), `navigation/nav2.py:103,114`, `generic/topics.py:280` (`connector._node.get_clock()`). The connector does not offer what tools need (typed service/goal calls with message objects, `now()`), so they bypass it. Anything that bypasses the connector also bypasses any safety hook placed on it.

### 1.3 Leaky layers and duplicated bookkeeping

- **Topic API keeps two of everything.** [R] `ROS2TopicAPI.__init__` has `_publishers` and `publishers`, `_subscriptions` and `subscriptions`, an unused `_last_msg`, and a stored-but-unused `_destroy_subscribers` (`ros2/api/topic.py:63-75` on main). `create_subscriber` writes `subscriptions` while `publish` writes `_publishers` (`topic.py` `create_publisher` vs `_get_or_create_publisher`). No lock; two concurrent first calls create two subscriptions [I, branch test `test_concurrent_first_use_creates_one_subscription_and_publisher` asserts this was broken].
- **The connector keeps a third cache.** [R] `ROS2BaseConnector.last_msg` (`connectors/base.py:167`) is fed by registering `_last_message_callback` as a normal user callback (`:388`), so it runs on the 4-worker callback pool behind slow user callbacks: `receive_message` can starve [I; branch test `test_receive_message_not_starved_by_slow_callbacks` targets this].
- **Action API state lives in a `TypedDict` with `None` fields** (`action.py:62-68`); the record is inserted before the goal is accepted and never removed on rejection or completion (`:213-256`). Feedback fan-out goes to a 10-worker pool with no ordering (`:84-96`), and every feedback is deep-copied per callback (`:108`).
- **Tools keep module-level globals.** [R] `generic/actions.py:29-33` (results, feedbacks, id mapping, two locks) are shared by every toolkit, agent and connector in the process. `navigation/nav2.py` keeps `current_action_id`, `current_feedback`, `current_result` as module globals (`nav2.py:92-170`), so two Nav2 toolkits interfere.
- **Permissions are plain string equality.** [R] `tools/ros2/base.py:51-63` compares names literally, so `cmd_vel` vs `/cmd_vel` (or a remapped name) slips through a `forbidden` list. The CLI toolkit (`tools/ros2/cli.py`) exposes `ros2 topic pub`, `ros2 service call`, `ros2 action send_goal` via subprocess and ignores `readable/writable/forbidden` entirely; when its timeout kills `send_goal`, the goal keeps running on the robot [I].
- **Bug: cancel tool never reports success.** [R] `CancelROS2ActionTool._run` compares a `CancelGoal.Response` to `CancelResponse.ACCEPT` (`generic/actions.py:255-259`), which is never equal. It also has no `is_writable` check.

### 1.4 Implicit state machines

- **Connector lifecycle** has no state at all on main: after `shutdown()` (`connectors/base.py:556-572`) calls fail in arbitrary ways depending on which entity was destroyed first. Shutdown also calls `self._tf_listener.unregister()` then `destroy_node()` while the MT executor may still be inside a callback [I].
- **Goal state** is inferred from which `TypedDict` fields are `None` (`action.py:258-282`): "accepted" means `client_goal_handle is not None`, "done" means `result_future.done()`, and a rejected goal leaves an entry with `result_future=None` forever.
- **Subscription state** for a topic is the union of `_topic_api.subscriptions`, `last_msg`, and whether `_last_message_callback` was registered (`connectors/base.py:375-388`); `register_callback` and `receive_message` each run their own variant of this logic.
- **LLM-visible action ids** are a two-way mapping kept in three global dicts (`generic/actions.py:165-187, 209-242`).

### 1.5 Slow paths (measured on Lyrical, rclpy 10.0.11, Python 3.14)

Benchmark: the branch's `tests/communication/ros2/bench/ros2_connector_bench.py`, run against main code (this worktree, with only the Lyrical import fallback) and against the branch (with the same fallback), same machine (32 cores), separate `ROS_DOMAIN_ID`s. Raw outputs: `.../scratchpad/bench_main.json`, `.../scratchpad/bench_hardening.json`, logs `bench_main.log`, `bench_hardening.log` in the scratchpad.

| Metric [M] | main | branch |
|---|---|---|
| Registered raw callback, 100 Hz String, 2 s: delivered | **39 / 200** | 199 / 200 |
| Same at 1 kHz: delivered | **45 / 2000** | 2000 / 2000 |
| HRI publish, VGA image: p50 / throughput | **335 ms / 3 msg/s** | 0.23 ms / 2698 msg/s |
| HRI 30 Hz dispatch latency p50 (60/60 delivered both) | 0.97 ms | 11.2 ms (single run, cause not investigated) |
| `receive_message` cold p50 / p99 | 120 ms / 1492 ms | 50 ms / 126 ms |
| service call, first call | 259 ms | 115 ms |
| service call, `reuse_client=False` p99 | 81.6 ms | 1.0 ms |
| service call, 8 threads x 2 services | 1084 calls/s | 4301 calls/s |
| start_action accepted p99 | 762 ms | 114 ms |
| 1000 service calls with `reuse_client=False`: leaked clients | **1000** (+105 MB RSS) | 0 |
| 200 goals: action entries / waitables / threads after | **200 / 206 / 38** | 32 / 3 / 6 |
| shutdown duration | 4 ms | 45 ms |
| fresh-client service calls that timed out (of 1000) | 2 | 5 |

Isolating the delivery problem on main [M] (`.../scratchpad/exp/qos_exp.py`, main code, Lyrical):

| main `register_callback`, raw String | delivered |
|---|---|
| MT executor, auto QoS (depth 1), 100 Hz | 41 / 200 |
| MT executor, explicit depth 10, 100 Hz | 190 / 200 |
| MT executor, depth 10, 1 kHz | 418 / 2000 |
| MT executor, depth 100, 1 kHz | 400 / 2000 |
| **SingleThreaded executor**, auto QoS (depth 1), 100 Hz | **200 / 200** |
| **SingleThreaded executor**, depth 10, 1 kHz | **2000 / 2000** |

Raw rclpy, no RAI code [M] (`.../scratchpad/exp/exec_exp.py`): one `std_msgs/String` subscription, depth 10, 2 s stream.

| Executor | 1 kHz delivered / subscriber CPU | 5 kHz delivered / CPU |
|---|---|---|
| `MultiThreadedExecutor` | 206 / 2000, 2.57 s | not run |
| `SingleThreadedExecutor` | 2000 / 2000, 0.19 s | 10000 / 10000, 0.85 s |
| `rclpy.experimental.EventsExecutor` | 2000 / 2000, 0.05 s | 10000 / 10000, 0.19 s |
| `rclpy.experimental.AsyncNode` (asyncio) | 2000 / 2000, 0.10 s | 3 kHz: 6000 / 6000, 0.23 s (the 5 kHz run hung in my harness; not investigated) |

Conclusion [M]: the MT executor on rclpy 10 caps one subscription at roughly 200 msg/s while burning CPU; depth 1 makes it worse. QoS depth matters, but the executor choice dominates.

Image conversion cost [M] (`.../scratchpad/exp/img_exp.py`, smooth synthetic RGB frames with light noise, in-process):

| Path | 640x480 | 1920x1080 |
|---|---|---|
| deserialize `sensor_msgs/Image` | 0.05 ms | 0.36 ms |
| `np.frombuffer(msg.data).reshape` (view, no copy) | 0.0004 ms | 0.0004 ms |
| `cv_bridge.imgmsg_to_cv2` | 0.04 ms | 0.22 ms |
| **current image tool path** (`GetROS2ImageTool`: cv_bridge + `preprocess_image`, PIL PNG + base64) | **88 ms** | **602 ms** |
| `convert_ros_img_to_base64` (aggregators; cv2 PNG) | 5.6 ms | 37 ms |
| numpy view + (downscale if > 1024 px) + cv2 JPEG q85 + base64 | 1.2 ms | 1.9 ms |

With a high-entropy frame the current tool path took 343 ms (VGA) and 2.36 s (1080p) [M]. `preprocess_image` always re-encodes to PNG through PIL (`rai/messages/conversion.py`, end of `preprocess_image`). Other slow paths [R]: HRI publish built the ROS message, converted it to an `OrderedDict`, and rebuilt it with `set_message_fields` (`messages.py:78-104` + `topic.py publish`); `ROS2HRIMessage.from_ros2` creates a `CvBridge` and PIL images for every received message on the executor thread (`messages.py:46-76`); `ROS2ImgVLMDescriptionAggregator.get` base64-encodes every buffered image and then uses only the last (`aggregators/ros2/aggregators.py:90-102`); `receive_message` polls with `sleep(0.1)` (`connectors/base.py:392-395`) and `wait_for_transform` does the same (`:426-431`); every connector deserializes all TF traffic because the listener is eager (`:142-143`).

### 1.6 Aggregators

[R] `BaseAggregator.__call__` appends to an unbounded deque by default (`max_size=None`, `aggregators/base.py:29-40`), so an image aggregator on a 30 Hz camera buffers every raw frame between intervals. `get_buffer()` then `clear_buffer()` are two separate operations (`aggregators/ros2/aggregators.py:35-52`, `:90-91`, `:140-147`), so frames arriving in between are lost [I]. Summarization (VLM calls) is synchronous inside the state loop's thread pool (`state_based_agent.py:149-182`). The concept (buffer per source, periodic summary into agent state) is right; the implementation needs atomic drain, mandatory bounds, lazy encoding of only the frames used, and async summarizers.

## 2. Review of the hardening branch

### 2.1 What it changes (size)

[M] `git diff --shortstat 38853fa e8250a4`: `src`: 17 files, +1160 / -615. `tests`: 9 files, +2360 / -26 (fault suite 1474 lines, 54 test functions, 59 cases; benchmark 615 lines). Per file (main to branch): `base_connector.py` 450 to 544, `ros2/connectors/base.py` 572 to 770, `ros2/api/action.py` 291 to 413, `ros2/api/service.py` 129 to 222, `ros2/api/topic.py` 249 to 195 (shrunk; dedup of the double dicts). 27 commits, mostly scoped and well described. Merge against current main has one conflict (`ros2/api/action.py`, from `71f255c`) [M via `git merge-tree`].

### 2.2 Does it work on Lyrical?

[M] Scratch worktree at `e8250a4` + Lyrical compat (pyproject/uv.lock from the spike worktree, `TopicEndpointInfo` import fallback in `ros2/api/base.py`; `topic.py` on the branch no longer imports it), venv via `uv sync --python /usr/bin/python3.14`, colcon install reused from the spike worktree. Worktree removed afterwards.

- Fault suite: **59 passed**, 5 of 5 runs, about 24 s each.
- `tests/communication` + `tests/tools/ros2` (sound device tests excluded, `sounddevice` missing in this venv): **299 passed, 4 skipped, 2 xfailed**.
- Under full CPU saturation (32 busy processes on 32 cores): `test_action_feedback_order` fails 7 of 7 runs with 36 to 45 of 100 feedback messages delivered (threshold > 50); ordering still holds (0 inversions). No "too slow, dropping" warning is logged, so the loss is upstream of the dispatcher (executor starvation or the action client's depth-10 feedback subscription) [I]. Everything else passed under load.
- All private hooks still exist on rclpy 10.0.11 / Python 3.14 [M]: `threading._register_atexit`, `ThreadPoolExecutor._threads`, `ActionClient._feedback_callbacks` (`rclpy/action/client.py:280`), `MultiThreadedExecutor._executor` (`rclpy/executors.py:1090`), `MultiThreadedExecutor._spin_once_impl` (`:1093`).

So "never tested on Lyrical" is now answered: it runs and its claims hold on Lyrical, except feedback completeness under heavy load.

### 2.3 Design and concurrency review

What is good:

- `SerialDispatcher` (`base_connector.py:90-157`): per-key FIFO on a shared pool, re-submits itself after each call, so one key never occupies more than one worker and keys run in parallel. The lock discipline is correct: `submit` schedules `_run` while holding the lock, `_run` takes the lock before popping, so the entry always exists [R]. Exceptions in user callbacks are caught (`Exception` only; a `BaseException` from a callback would leave the key's entry in `_pending` forever and silently stall that callback [R/I]).
- Explicit lifecycle: `ConnectorState` + `_running_only` wrapper + `ConnectorClosedError` (`connectors/base.py:72-99`), shutdown idempotent and callable from a callback (`:687-770`). Waiting `receive_message` / `service_call` callers are woken at shutdown.
- Services: client cache keyed by resolved name and type, in-flight futures tracked so `release_client`/type change/shutdown cancel callers instead of hanging them (`api/service.py:44-222`); concurrent calls through one client no longer serialized (measured 4x throughput).
- Actions: one cached `ActionClient` per name with its own mutually exclusive group (`api/action.py:164-192`), late-accepted goals cancelled (`:153-162`), bounded record retention (`MAX_FINISHED_GOALS = 32`, `:72`), feedback and done callback on one serial key so `on_done` runs after the goal's feedback (`:116-140`).
- Lots of real bugs fixed with tests: caller payload mutation (`b119762`), name-resolution bypass in tool permissions (`29a709d`), cancel tool result (`29a709d`), NodeDiscovery allowlist race (`1f06344`), HRI publish round trip (`7e47ada`), lazy TF (`048aba0`).

Problems:

1. **It optimizes the wrong executor.** [R] `_MultiThreadedExecutor` overrides the private `_spin_once_impl` to add `time.sleep(0)` and reaches into the private `_executor` pool for shutdown (`connectors/base.py:102-127, 739`); neither is marked `ponytail:`. Lyrical's rclpy already yields (`os.sched_yield()`, `executors.py:1119`) and has a public `shutdown(wait_for_threads=False)` (`:1149-1166`). The measured data in 1.5 says the right fix is not a better MT executor but no MT executor.
2. **Thread count.** [R] Per connector: spin thread + MT pool (defaults to `len(os.sched_getaffinity(0))`, 32 here) + 4 callback workers + 10 action-callback workers, all lazily created. Every `ROS2HRIConnector` has the same. Fine on a workstation, wasteful on a Jetson [I].
3. **Drop-oldest for everything.** [R] Every registered callback has a 10-deep queue that drops the oldest pending message (`base_connector.py:130`, default `maxlen=10`). Right for camera frames, wrong for HRI text from a human (a burst of 11 messages while the agent callback is busy loses the first) [I]. The policy must be per subscription.
4. **Silent semantic changes.** [R]
   - Publishers default to RELIABLE + **TRANSIENT_LOCAL**, depth 10, and `auto_qos_matching` is ignored for publishers (`api/topic.py:34-39, 129-138`). For command topics this means a late-joining TRANSIENT_LOCAL subscriber receives up to 10 stale commands (for example `cmd_vel` replayed to a restarted controller or a recorder that republishes) [I]. Latching must be opt-in, never the default for commands.
   - Subscriptions without publishers default to BEST_EFFORT + VOLATILE depth 10 (`api/base.py:81-90`); the first caller fixes QoS and message type for the topic forever, later differing requests only log a warning (`api/topic.py:80-94`).
   - `receive_message` returns a latched sample of any age if the subscription happens to be TRANSIENT_LOCAL (`connectors/base.py:511-518`), otherwise only samples newer than `timeout_sec`. The meaning of `timeout_sec` (how long to wait vs maximum sample age) is still conflated, as on main.
   - `destroy_subscribers` was removed; `create_service(on_done=...)` was removed from the signature; `create_service` still returns a uuid that refers to nothing (`api/service.py:202-214`).
5. **Lifecycle is bolted on.** [R] `_running_only` wraps methods one by one, and the mixin methods are re-bound in the class body to get the guard (`service_call = _running_only(ROS2ServiceMixin.service_call)`, `connectors/base.py:672-674`). The error conversion "any exception while not RUNNING becomes `ConnectorClosedError`" (`:88-97`) is pragmatic but can mask real errors raised during shutdown races. A process-wide `WeakSet` of live connectors is shut down from `threading._register_atexit` at import time (`:50-69`): importing `rai.communication.ros2` installs an exit hook.
6. **Polling remains.** [R] `wait_until_ready` polls every 5 ms (`ros_async.py:44-58`) in place of `wait_for_service`. This is a latency fix, not a design: it burns a thread per waiter. On Lyrical, `AsyncClient.wait_for_service` also polls (0.1 s), so there is no event to wait on in rclpy either [R, `rclpy/experimental/async_client.py`].
7. **Action client replacement can orphan in-flight goals.** [R] `_get_action_client` destroys an existing client cached for another type (`api/action.py:186-190`) without failing the goals that client is tracking, whose `result_future` then never completes [I]. The service API does cancel in-flight futures in the same situation; the action API is asymmetric.
8. **Thread-centric by construction.** Everything is `threading.Condition`, pools and blocking waits. 3.0 agents are asyncio-first; this code would need a second, async surface on top, doubling the API.

Test quality: the fault suite is mostly black-box (12 private-attribute uses in 1474 lines: `._thread` x5, `._state` x4, `._executor`, `._topic_api`, `._actions_api`) [M by grep], uses real rclpy nodes for servers and publishers, and is organized by area (dispatch, topics, services, actions, lifecycle, performance). Weak spots: fixed thresholds that fail under load (2.2), 24 `time.sleep` calls, and several tests that encode the branch's chosen semantics (`test_receive_message_returns_latched_sample_of_any_age`, `test_hri_subscription_is_reliable_without_publishers`, `test_publisher_latches_only_for_transient_local_subscribers`, `test_every_call_after_shutdown_raises_closed` with `ConnectorClosedError`). The benchmark is good: isolated subprocesses, real transport, JSON output with `--compare`, leak counts. Both are worth keeping.

### 2.4 Verdict

**Rewrite the core, keep the branch as the oracle, port a short list of fixes.**

Why not wholesale: 3.0 changes the concurrency model (asyncio-first) and the executor (measured reason to drop MT), and drops Humble, which removes the reason for several workarounds. Merging the branch first and then replacing its dispatcher, lifecycle wrapper, executor subclass and blocking waits means writing the 3.0 core anyway, plus migrating users across two semantic changes (TRANSIENT_LOCAL publishers, drop-oldest callbacks) that 3.0 would then revert or make explicit.

Why not ignore it: it is the best specification RAI has of what the connector must survive. 54 fault tests and a reproducible benchmark that already pass on Lyrical are worth more than the code.

Port as code (small, independent, apply to 3.0 regardless of the new core; also candidates for a 2.x patch release if one is planned):
- `b119762` stop mutating caller payloads, cache message classes;
- `7e47ada` HRI publish without the dict round trip (`to_ros2_msg`);
- `048aba0` lazy TF listener;
- `8a625cf` per-instance HRI node names, forward `use_sim_time`, steady performance timer;
- `1f06344` NodeDiscovery allowlist race;
- `29a709d` tool permission name resolution and cancel-tool result (the permission part depends on `connector.node`; keep the idea, call a connector `resolve_name` instead);
- the service in-flight-future tracking idea from `7d4afb3`/`fe28737` (cancel callers when a client is destroyed);
- the `SerialDispatcher` algorithm only if the 3.0 design keeps a thread pool for sync callbacks (section 3.4 makes it an asyncio consumer task instead).

Adopt as the 3.0 acceptance oracle: the fault suite (rewritten against the 3.0 API, thresholds made load-tolerant, semantics tests updated to the decisions in section 3) and the benchmark (add an async variant, an image/point-cloud path, a hooks-on/hooks-off overhead row), run on Lyrical and Jazzy in CI.

## 3. Proposed RAI 3.0 ROS layer

Design goals: one execution model, no user code on the executor thread, native message types inside, explicit ownership (objects, not string handles), and a single egress choke point for safety hooks. Lines of code should go down, not up.

### 3.1 Layers

```
tools / aggregators / agents
        |            (typed calls, async or sync)
ROS2Connector    ROS2HRIConnector          <- separate public classes (constraint)
        \            /
         ROS2Runtime                         <- shared, internal: node, executor, spin thread,
        (one per process by default)            entity registry, lifecycle, egress hooks
        |
      rclpy (SingleThreadedExecutor default, EventsExecutor opt-in)
```

- **`ROS2Runtime` (internal, ~300 lines target)**: owns one `Node`, one executor, one spin thread, the entity registry, the hook chain, and the state (`RUNNING`, `CLOSED`). Connectors compose it (`ROS2Connector(runtime=None)` creates or reuses the default runtime; passing one shares it). This replaces `ROS2BaseConnector`, both mixins, and the generic `BaseConnector[T]` machinery for ROS.
- **Executor:** `SingleThreadedExecutor` by default on both distros, `EventsExecutor` behind a flag (`executor="events"`; experimental upstream, 4x less CPU in 1.5) [M]. No `MultiThreadedExecutor`. The executor thread only takes messages, completes futures and enqueues; it never runs user code, preprocessing or hooks. This is what makes the single thread sufficient [M: 10 kHz delivered with ST].
- **`rclpy.experimental.AsyncNode`** exists on Lyrical (Python >= 3.12) but not on Jazzy 7.1.11 (only `EventsExecutor` there) [M], is experimental, and has no action client/server [R]. Do not build on it in 3.0; keep the runtime boundary narrow enough that an `AsyncNode` backend can replace the spin thread later.
- **Entity registry:** one dict per kind keyed by resolved name, holding `(type, entity, qos)`. Publishers, subscriptions, service clients and action clients are created once, reused, and destroyed at close. Type conflicts raise. Replaces the double dicts in the topic API, `last_msg`, `_persistent_clients`, `actions`, and the tools' global stores.
- **Lifecycle:** `RUNNING -> CLOSED` (one transition, guarded by a lock), checked at the top of every public method by one helper. `close()` cancels every pending future in the registry (callers get `ConnectorClosedError`), stops the executor, joins the spin thread with a timeout, destroys entities, then the node. Context managers: `with ROS2Connector() as c:` and `async with`. No import-time atexit hook: the spin thread is a daemon and the docs require `close()` or a context manager [I trade-off; the branch's `test_process_exits_without_shutdown` stays as an oracle].

### 3.2 Public API shape (async first, sync wrappers)

```python
class ROS2Connector:
    # topics
    async def publish(self, topic: str, msg: RosMsg, *, qos: QoS = QoS.COMMAND) -> None
    async def receive(self, topic: str, *, msg_type: type | str | None = None,
                      timeout: float = 1.0, max_age: float | None = None) -> RosMsg
    def subscribe(self, topic: str, *, msg_type=None, qos: QoS = QoS.AUTO,
                  maxsize: int = 10, overflow: Literal["drop_oldest", "block"] = "drop_oldest"
                  ) -> Subscription          # async iterator; .on(callback) for push style; .close()
    def latest(self, topic: str, ...) -> Latest   # inline ring buffer of size 1 (receive cache, aggregators)
    # services and actions, native request/goal objects (dicts accepted and converted)
    async def call(self, service: str, request: RosSrvRequest, *, timeout: float = 5.0) -> RosSrvResponse
    async def send_goal(self, action: str, goal: RosGoal, *, accept_timeout: float = 5.0) -> GoalHandle
    # TF, time, graph
    async def transform(self, target: str, source: str, *, timeout: float = 1.0) -> TransformStamped
    def now(self) -> Time
    def resolve(self, name: str, kind: Literal["topic", "service", "action"]) -> str
    def graph(self) -> Graph                  # topics/services/actions with types
    # lifecycle, hooks
    def close(self) -> None
    hooks: HookChain
    sync: SyncROS2Connector                   # same methods, blocking, with timeouts

class GoalHandle:                             # replaces string handles and global stores
    id: str
    async def result(self, timeout: float | None = None) -> RosResult
    def feedback(self) -> AsyncIterator[RosFeedback]   # bounded, latest-wins on overflow
    async def cancel(self, timeout: float = 2.0) -> bool
    status: GoalStatus                        # explicit enum, not inferred from None fields
```

Bridging rule: every rclpy future gets one done callback that moves the result into the caller's world. Async path: `loop.call_soon_threadsafe(fut.set_result, ...)` on the caller's running loop, so the connector works from any event loop and needs no loop of its own. Sync path (`connector.sync.*`): a private event loop thread, started lazily on the first sync call, runs the same coroutines (`run_coroutine_threadsafe(...).result(timeout)`). One implementation, two surfaces; no `asyncio.run` inside library code (it breaks under Jupyter, Streamlit and running agents) [I]. Calling a `sync` method from inside the private loop raises instead of deadlocking.

Callbacks: a subscription has one bounded queue per consumer. The executor thread does `queue.put_nowait` via `call_soon_threadsafe` (or applies `drop_oldest`). Each registered callback gets one consumer task: `async def` callbacks are awaited, sync callbacks run with `asyncio.to_thread`. One consumer per callback gives the branch's ordering guarantee without a lock-and-resubmit dispatcher. `Latest`/ring-buffer consumers are "inline": they run on the executor thread, are O(1), and are the only code that does (receive cache, aggregator buffers).

QoS presets replace auto-matching guesswork: `SENSOR` (best effort, volatile, depth 5), `COMMAND` (reliable, volatile, depth 10; never latched), `STATE`/`LATCHED` (reliable, transient local, depth 1; opt-in), `HRI` (reliable, volatile, depth 100, overflow `block` so human messages are never dropped), `AUTO` for subscriptions only (match publisher offers if any exist, else `SENSOR`). `receive(max_age=...)` separates "how long to wait" from "how old a sample may be"; latched topics simply pass `max_age=None`.

`ROS2HRIConnector` keeps its current public role (send/receive `HRIMessage`), shares the runtime, and gets no TF listener, no services/actions surface. The generic `BaseConnector`/`HRIConnector` interfaces become small `Protocol`s (`HRIConnector` protocol: `send`, `subscribe`, `close`) so non-ROS HRI transports (sound device, Streamlit) still plug into agents.

### 3.3 Native and zero-copy message paths (default for images and point clouds)

- rclpy deserializes `uint8[]` into `array.array` [M: `type(msg.data).__name__ == "array"`], which supports the buffer protocol, so a numpy view is free [M: 0.0004 ms]. RMW-level zero-copy (loaned messages) is not available to Python subscribers [I, not checked in rclpy 10 source]; "zero-copy" in RAI therefore means no copy after deserialization and no intermediate formats.
- `rai.communication.ros2.views` (new, small): `image(msg) -> np.ndarray` (handles `step`, `rgb8/bgr8/mono8/16UC1/32FC1`, `CompressedImage` decoded once with cv2), `points(msg: PointCloud2) -> np.ndarray` structured view built from `fields` with `np.frombuffer` (no `sensor_msgs_py.read_points` Python loop), `depth(msg)`.
- One LLM-boundary encoder, `encode_image(array, max_side=1024, format="jpeg", quality=85) -> str`, used by tools, aggregators and `HRIMessage.to_langchain`. Measured 1.2 ms (VGA) / 1.9 ms (1080p) versus 88 / 602 ms for the current PNG path [M]. PNG stays available as an option for exact pixels.
- `HRIMessage` stores images as numpy arrays (or raw ROS `Image`s) and converts to PIL/base64 lazily in `to_langchain`; `from_ros2` stops building a `CvBridge` and PIL images per message. The executor thread never decodes images.
- Raw serialized pass-through (`create_subscription(..., raw=True)`) is exposed as `subscribe(..., raw=True)` for forwarding/recording paths that never need the Python object.
- `cv_bridge` becomes optional in core (it pins numpy 1.x on Jazzy per the spike's pyproject comment) [R: `src/rai_core/pyproject.toml` diff in the spike worktree].

### 3.4 Aggregators, rethought

Keep the concept, split the two jobs:

1. **Buffers** live on the connector side and are inline consumers: `Latest` (size 1), `Window(n)` (ring buffer, required bound), `TimeWindow(seconds)`. `drain()` swaps the deque under a lock, so nothing is lost between read and clear. Messages stay native (no wrapping, no encoding at arrival).
2. **Summarizers** are plain async callables `async def summarize(items: Sequence[RosMsg]) -> BaseMessage | None`. The existing four become: logs dedup (pure), last image (encode only the last frame), VLM description (encode only the last frame), VLM diff (encode only the 3 key frames). VLM summarizers get a per-call timeout.

The state loop in the agent becomes an asyncio task that, every interval, drains each buffer and runs summarizers concurrently (`asyncio.gather` with timeouts), keeping the last good result per source. Registration API: `connector.aggregate(topic, Window(30), summarizer)` returns a handle with `.close()`.

### 3.5 Safety hooks (connector and tool side)

Goal: a confirmation or veto point before anything that can move hardware, without making tools deny-by-default and without per-message overhead when no hooks are registered.

**Egress hook chain on the runtime** (the backstop that every path goes through):

```python
@dataclass(frozen=True)
class Egress:
    kind: Literal["publish", "service", "goal", "cancel"]
    name: str              # resolved
    type: str              # e.g. "geometry_msgs/msg/Twist", "nav2_msgs/action/NavigateToPose"
    payload: Any           # the ROS message / request / goal (read-only by convention)
    origin: str | None     # tool name or caller tag, from a contextvar set by the tool wrapper

class Decision: Allow(lease_s: float = 0) | Deny(reason: str)

connector.hooks.add(hook, match=Match(types={"geometry_msgs/msg/Twist", "nav2_msgs/action/NavigateToPose", ...},
                                       names={"/cmd_vel"}, kinds={"publish", "goal"}))
# hook: (Egress) -> Decision | Awaitable[Decision]
connector.hooks.after(audit_fn)      # observe-only, for logs/telemetry
```

- **Matching by type first, name second.** Names get remapped; types do not. RAI ships a `MOTION` set (Twist/TwistStamped, `NavigateToPose`, `NavigateThroughPoses`, `FollowJointTrajectory`, `MoveGroup`, `GripperCommand`, `ManipulatorMoveTo`, ...) users can extend.
- **Default: no hooks, everything allowed** (constraint). Lookup is a dict hit on `(kind, type)` and `(kind, name)` computed once per entity and cached on the registry entry, so a 50 Hz teleop publish without matching hooks pays one attribute check [I].
- **Confirmation without spamming:** a hook returns `Allow(lease_s=10)` to approve that `(kind, name, origin)` for a period, so a confirmed `cmd_vel` stream is not re-asked per message. `Deny` raises `EgressDenied(reason)` to the caller; tools turn it into a normal tool result the LLM can read ("Operator denied: ...").
- **Async-native:** hooks may await (for example ask the operator through a `ROS2HRIConnector` and wait for "yes" with a timeout; the timeout policy belongs to the hook). Sync callers run hooks on the private sync loop. Hooks never run on the executor thread.
- **Stop path:** because goals are tracked as `GoalHandle`s in the registry, `connector.cancel_all_goals(match=MOTION)` is cheap to provide and is what an E-stop hook or a "stop" HRI command calls. It is not a substitute for a hardware E-stop [I].

**Tool side:**
- Tools declare `effects: frozenset[Literal["read", "write", "motion"]]` (class attribute). Generic publish/call/goal tools are `write` and become `motion` dynamically when the target type is in `MOTION`. The agent layer (another planner's section) can use this for tool-level confirmation UX before the call is even made; the connector hook remains the guarantee.
- Tools set the `origin` contextvar so hooks and audit logs know which tool (and agent) issued the egress.
- Close the bypasses: core tools stop using `connector.node` (manipulation and Nav2 tools call `connector.call` / `connector.send_goal`); the CLI toolkit loses `topic pub`, `service call` and `action send_goal` (or routes them through the connector). Otherwise hooks are advisory. Extensions that still use `connector.node` are documented as outside the hook guarantee.
- `readable/writable/forbidden` stay (not deny-by-default) but compare resolved names via `connector.resolve`.

### 3.6 Tools after the change

- Generic action tools keep a per-toolkit `GoalRegistry` (bounded dict of LLM-visible id to `GoalHandle`) instead of module globals; feedback is read from the handle (latest N), results from `await handle.result(0)`; cancel uses `handle.cancel()` and reports the real outcome.
- Nav2 tools (`nav2.py`, `nav2_blocking.py`) become ~half their size on `send_goal` + `GoalHandle`; the blocking variant gets a timeout and cancels on timeout.
- Image tools use `views.image` + `encode_image`; `GetROS2TransformTool` uses `connector.now()`.

### 3.7 What gets deleted

`ROS2BaseConnector`, `ROS2ActionMixin`, `ROS2ServiceMixin`, `ROS2Message` wrapping on hot paths, `BaseConnector.general_callback*`, `ParametrizedCallback`, `callback_id_mapping`, `ros_async.get_future_result`, `waiters` polling helpers (replace with `graph()` + async wait), the executor performance timer (replace with a queue-lag metric on subscriptions), `destroy_subscribers`, Humble branches. Expected net reduction in `communication/ros2` despite new features [I].

### 3.8 Sequencing

1. Land the benchmark and the fault suite on main first (marked xfail where main fails), adapted to be API-level and load tolerant. CI on Lyrical and Jazzy.
2. Port the small fixes listed in 2.4.
3. Build `ROS2Runtime` + async `ROS2Connector` behind the new API; rewrite the oracle tests against it; benchmark gates: 1 kHz 100% delivery, 10 kHz with `SENSOR` QoS no executor stall, no entity growth over 1000 calls/200 goals, shutdown under 1 s with a stuck callback.
4. Move `ROS2HRIConnector`, tools and aggregators; add views/encoder; add hooks with the `MOTION` set and the tool `effects` metadata.
5. Delete the old classes.

### 3.9 Open questions and risks

- `EventsExecutor` is experimental upstream on both distros; default stays `SingleThreadedExecutor` until it passes the oracle on both [I].
- One shared runtime per process lowers threads and nodes but couples connectors' lifecycles; closing a connector must not close a shared runtime (reference count) [I].
- Dropping MT executors means a long-running user action server `execute_callback` must not block the executor thread; 3.0 action servers should run `execute` as an asyncio task or in a worker, which needs one more design pass [I].
- The HRI 30 Hz latency difference (main 1 ms vs branch 11 ms p50) was a single run and is unexplained [M, single sample].
- Fresh service clients occasionally lose the first response on Lyrical (2 to 5 per 1000, both main and branch) [M]; this is the known DDS discovery race and argues for cached clients by default [I].
