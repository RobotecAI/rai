# 01: rclpy and the layers below it, as they matter to RAI 3.0

Scope: rclpy (Python + pybind11 `_rclpy`), rcl, rmw_fastrtps, rosidl_python, rosidl_runtime_py, compared on Jazzy and Lyrical, judged against what RAI's ROS layer (`src/rai_core/rai/communication/ros2/`, `base_connector.py`) does today.

## 0. Key findings

1. **RAI's default executor is broken for subscriptions on both distros.** `MultiThreadedExecutor` plus a subscription that has QoS event handlers (rclpy adds two by default to every subscription) plus rmw_fastrtps make the executor busy-loop whenever a message is pending. Measured on Lyrical with one 30 Hz topic: 89% CPU, 59 ms median latency, 314 ms p99. Single-threaded executor on the same load: 1% CPU and 0.15 ms. At 1 kHz the multi-threaded executor delivers 78 msg/s. RAI's `TransformListener` on the connector node is enough to trigger this whenever `/tf` is published. Root cause is in section 3.3.
2. **The dict round trip is roughly 5000x the cost of native messages for images and point clouds.** Measured: `message_to_ordereddict` + `set_message_fields` on a VGA rgb8 Image costs 286 ms on Lyrical (254 ms on Jazzy). The native path (`Image(data=bytes)` + `np.frombuffer`) costs 0.02 ms. For a 4.9 MB PointCloud2 the dict path costs 1.6 s against 0.08 ms native. RAI's HRI connector does exactly this for every image it publishes.
3. **No executor survives a callback exception.** Single-threaded, multi-threaded and Events executors all re-raise out of `spin()`, which kills RAI's spin thread silently.
4. **`EventsExecutor` (experimental on both distros, same code) is the fastest and lightest.** Measured: 139k to 157k msg/s against 26k for single-threaded, and a 0.03 ms service round trip against 0.16 ms (single) and 0.35 ms (multi). It also has no wait-set rebuild cost. Limits: it is single-threaded, it ignores callback groups, and it does not support guard conditions.
5. **Lyrical's experimental `AsyncNode` deadlocks under high-rate traffic with rmw_fastrtps.** I captured a GIL-versus-DataReader-mutex ABBA deadlock with gdb (section 11.1). Do not build RAI 3.0 on it yet.
6. **Real Jazzy-vs-Lyrical differences RAI must abstract:**
   - `Client.call` timeout: returns `None` on Jazzy, raises `TimeoutError` on Lyrical.
   - `set_message_fields` mutates its input on Jazzy (a second call with the same dict fails); on Lyrical it deep-copies instead, which is slow.
   - `topic_endpoint_info` moved to `endpoint_info`.
   - `rclpy.shutdown()` destroys every node on Lyrical but not on Jazzy.
   - `ActionServer` `result_timeout` default dropped from 900 s to 10 s.
   - `rclpy.init()` returns a context manager on Lyrical only.
   - Full list in section 10.

## 1. Method and versions

| | Lyrical | Jazzy |
|---|---|---|
| rclpy installed | 10.0.11, identical to the `lyrical` branch head `4a3f3d8` | 7.1.11 in `rai-dev:jazzy` and `osrf/ros:jazzy-desktop-full`. The branch head is 7.1.13 (`521012e`). |
| Python | 3.14.4, GIL build (`sys._is_gil_enabled()` is True) | 3.12.3 |
| RMW | rmw_fastrtps_cpp (the only RMW installed) | rmw_fastrtps_cpp (the only RMW installed) |
| numpy | 2.3.5 | 1.26.4 |

Upstream sources were shallow-cloned at the `jazzy`, `lyrical` and `rolling` branches into `scratchpad/upstream/`: rclpy, rcl, rmw, rosidl_python, rosidl_runtime_py, and rmw_fastrtps (jazzy and lyrical only). Every `file:line` citation below is relative to that directory.

Benchmarks are in `scratchpad/bench/`, and raw outputs are kept next to them as `*_lyrical.txt` and `*_jazzy.txt`. Lyrical ran on the host with `source /opt/ros/lyrical/setup.bash` and `/usr/bin/python3`. Jazzy ran in `rai-dev:jazzy`. All runs used `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and a unique `ROS_DOMAIN_ID` per run. The host has 32 cores.

For about 10 minutes another workload on the host ran 30 CPU burners. I re-ran every timing-sensitive benchmark after the load dropped (load average about 2), and the numbers below come from those re-runs. Absolute numbers are single-host and indicative. The ratios are large enough to be robust.

Each claim is tagged **[measured]** or **[code]** (read in source).

## 2. What RAI's ROS layer relies on today

- `connectors/base.py:155-163`: one `Node` and a `MultiThreadedExecutor()` (default thread count, 32 on this host) spinning in a non-daemon `threading.Thread`. Every entity sits in the node's default `MutuallyExclusiveCallbackGroup`.
- `connectors/base.py:143`: a `TransformListener` on the same node. It creates `/tf` and `/tf_static` subscriptions that carry rclpy's default event handlers.
- `base_connector.py:235-241`: `general_callback` runs on the executor thread and calls `general_callback_preprocessor` there. User callbacks are offloaded to a `ThreadPoolExecutor(4)`.
- `api/base.py:115,124`, `api/topic.py:168`, `types/ros2/convert.py:32,47`, `messages.py:78-94`: payloads go through `set_message_fields` and `message_to_ordereddict` dicts, and HRI images go through them too.
- `api/service.py:92-108`: an optional per-call `Client`, `wait_for_service(timeout)`, then a blocking `Client.call(timeout)` from the caller's thread.
- `api/action.py:228-263`:
  - It creates a new `ActionClient` per goal and never destroys it.
  - It deep-copies every feedback message (line 108).
  - It cancels with a blocking `cancel_goal()` that has no timeout.
- `connectors/base.py:392-395`, `api/topic.py:183-191`, `context.py:181`: graph and message waits poll with `time.sleep(0.1)` or `0.5`.
- `connectors/base.py:556-572`: shutdown order is `destroy_node`, then actions and topics, then `executor.shutdown`, then `join`.

## 3. Executors (Q1)

### 3.1 How the Python executors work [code]

`Executor._wait_for_ready_callbacks` (`rclpy-lyrical/rclpy/rclpy/executors.py:716-961`) rebuilds a brand new `_rclpy.WaitSet` on every iteration (line 861). It first walks every node's subscriptions, timers, clients, services, guards and waitables through `can_execute`, then calls `wait()`, then walks the entities again to find the ready ones. The cost is O(number of entities) per dispatched callback, all in Python with the GIL held.

- `SingleThreadedExecutor._spin_once_impl` (`:1013-1027`) runs the handler inline.
- `MultiThreadedExecutor._spin_once_impl` (`:1093-1119`):
  - It submits the handler to a `ThreadPoolExecutor`.
  - It marks the entity `_executor_event = True` (`:662`), which keeps the entity out of the next wait set until a worker has taken the data.
  - It then calls `os.sched_yield()`. That yield is the Lyrical "perf fix (hopefully)" #1547 (`:1119`), and Jazzy 7.1.11 does not have it.
- The C++ `EventsExecutor` (`rclpy-*/rclpy/src/rclpy/events_executor/`) uses rcl `set_on_new_*_callback` hooks. The rmw callback only enqueues work into a C++ queue without taking the GIL (`events_executor/rcl_support.cpp:59-70`), and dispatch then runs on the thread that called `spin()`. It is the same code on Jazzy (backport #1435, async callbacks in 7.1.11) and Lyrical. On both distros it lives in `rclpy.experimental`.

### 3.2 Throughput and latency [measured]

The publisher is a separate process. The subscriber uses KEEP_LAST 10, RELIABLE, `std_msgs/Header`. Files: `bench_pubsub.py`, `run_pubsub.sh`, `pubsub_*.txt`, `mtfix_*.txt`, `cpu_*.txt`.

| Load | Single (Lyr / Jazzy) | Multi, RAI default (Lyr / Jazzy) | Events (Lyr / Jazzy) |
|---|---|---|---|
| Flood: max msg/s delivered | 26.4k / 23.0k | **18 / 18** | 139k to 157k / 171k |
| 1 kHz: delivered, p50 latency | 1000/s, 0.06 ms | **78 to 128/s, 4.7 to 5.7 ms** (Jazzy 86/s, 5.4 ms) | 1000/s, 0.03 ms |
| 30 Hz: CPU, p50/p99 latency (Lyrical) | 1%, 0.15/0.40 ms | **89%, 59/314 ms** | 0%, 0.10/0.30 ms |
| 100 Hz: delivered, CPU, p50 (Lyrical) | 100/s, 2%, 0.14 ms | **70/s, 108%, 33 ms** | 100/s, 1%, 0.09 ms |
| `Client.call` round trip from a foreign thread (Lyr / Jazzy) | 0.16 / 0.19 ms | 0.35 / 0.41 ms | 0.03 / 0.03 ms |

Idle `spin_once(0)` cost, which every dispatched callback pays in the Python executors (`iter_cost.py`), on Lyrical:

| Entities on the node | Cost per iteration |
|---|---|
| Node only | 73 µs |
| + 50 `ActionClient`s | 206 µs |
| + 200 `ActionClient`s | 840 µs |
| + 200 subscriptions | 1333 µs |

Jazzy is about 10-25% slower. RAI's per-goal `ActionClient` leak therefore slows every later callback linearly, at about 4 µs per leaked client per callback.

### 3.3 Root cause of the multi-threaded collapse [code + measured]

1. rclpy attaches default `incompatible_qos` and `incompatible_type` event handlers to every subscription (`rclpy-lyrical/rclpy/rclpy/event_handler.py` `SubscriptionEventCallbacks`, `use_default_callbacks=True`). These handlers are waitables in the subscription's callback group.
2. In rmw_fastrtps, an event's status condition is the DataReader's own `StatusCondition` (`rmw_fastrtps-lyrical/rmw_fastrtps_shared_cpp/src/custom_subscriber_info.cpp:120-122`). The subscription enables `data_available` on that condition (`rmw_subscription.cpp:186-188`).
3. `__rmw_wait` first calls `has_triggered_condition`, which returns true when any event's `get_statuscondition().get_trigger_value()` is set (`rmw_wait.cpp:90`; Jazzy `rmw_wait.cpp:70`). The wait is then skipped and `RMW_RET_OK` is returned (`:166`, `:359`) with nothing marked ready.
4. The multi-threaded executor excludes a subscription whose handler has been submitted but not yet run (`_executor_event`), while that subscription's event handlers stay in the wait set. Unread data therefore turns `rcl_wait` into a no-op. The spin thread busy-loops rebuilding wait sets while holding the GIL, which starves the worker that would take the message.

Instrumented proof (`mt_trace.py`, `mt_ready.py`):
- Every wait set was `(subs=0, guards=2, …, services=7, events=4)`.
- `wait()` returned in about 2 µs with nothing ready.
- There were about 100k wait sets per 200 messages.
- Only 2 Python guard triggers happened per message (`mt_trig.py`), so the wakes do not come from guard conditions.

Confirmation (`mt_fix.py`, both distros): creating the subscription with `event_callbacks=SubscriptionEventCallbacks(use_default_callbacks=False)` restores the multi-threaded executor to 1000/s at 1 kHz (0.07 ms) and to 13k/s under flood.

You cannot pass that option to `tf2_ros.TransformListener`. Its `/tf` subscriptions keep the default handlers.

### 3.4 Exceptions [measured + code]

A timer callback that raises kills `spin()` in all three executors on both distros (`sem.py exception`): the bad callback ran once, the good timer ran 0 times, and the spin thread died with `ValueError`.

- Single-threaded: `executors.py:1025-1027`.
- Multi-threaded: `future.result()` at `:1116`, raised one iteration later.
- Events: logs `[FATAL] Exception in 'timers' callback` and then re-raises.

In RAI, the spin thread is a plain `threading.Thread`, so the connector silently stops receiving everything. RAI's own executor-side code is exposed: `general_callback_preprocessor`, service handlers, action goal and cancel callbacks, and `_executor_performance_callback`.

### 3.5 Shutdown semantics [measured + code]

| Action while `spin()` runs in a thread | Single | Multi | Events |
|---|---|---|---|
| `executor.shutdown()` | `spin` returns, join under 1 ms | returns | returns |
| `rclpy.shutdown()` | **raises `ExternalShutdownException`** | returns normally (swallowed at `:1101`) | **raises `ExternalShutdownException`** |
| `node.destroy_node()` | spin keeps running (no error); the executor must still be shut down | same | same |

Same behaviour on both distros.

Code bugs in shutdown:
- `Executor.shutdown(timeout)` never waits for in-flight callbacks. It sets `_is_shutdown = True` and then tests `if not self._is_shutdown:` before `_work_tracker.wait` (`rclpy-lyrical/rclpy/rclpy/executors.py:321-328`, Jazzy `:288-294`), so the wait is dead code.
- Lyrical 10.0.11's `MultiThreadedExecutor.shutdown` closes the pool without waiting for the spin loop to exit (`:1149-1167`). That race is fixed only on the Jazzy branch, 7.1.13 #1749 (`rclpy-jazzy/.../executors.py:1050-1083`, `_spin_done`), and the installed Jazzy 7.1.11 does not have the fix either.
- I saw "exception was never retrieved: cannot use Destroyable because destruction was requested" on multi-threaded teardown.

### 3.6 Recommendation (Q1)

- **Both distros: run `EventsExecutor` as the RAI default.** It needs `rclpy>=7.1.11` on Jazzy for async service and subscription callbacks.
  - It is about 5x the single-threaded throughput.
  - It has the lowest latency and CPU use.
  - It has no O(entities) rebuild.
  - It keeps RAI's existing model: one spin thread, with user work offloaded to a thread pool.
- **Requirements for EventsExecutor:**
  - No guard conditions. `EventsExecutor` throws "Guard conditions not supported" (`events_executor.cpp:232,557`). Neither RAI nor tf2_ros uses them today.
  - Nothing may block inside an executor callback. That includes action `execute_callback`s, which run inline as executor tasks.
- **Keep `SingleThreadedExecutor` as a fallback flag**, because `EventsExecutor` is experimental. Pass `executor_type="events" | "single"`.
- **Remove `MultiThreadedExecutor` as a default.** Do not offer it unless every subscription is created without event handlers, which is impossible for tf2.

## 4. Callback groups and reentrancy (Q2) [measured]

Test: a service callback calls another service synchronously with `Client.call(timeout=2)`. The inner server sits in its own group, which makes it behave like a remote node (`sem.py nested`).

| Executor | Both in default group | Client in its own group (`split`) | Reentrant group |
|---|---|---|---|
| Single | timeout | timeout | timeout |
| Multi | timeout | **ok (0.001 s)** | **ok** |
| Events | timeout | timeout | timeout |

On timeout, Jazzy's inner `call` returns `None` and Lyrical's raises `TimeoutError`.

Rule: a blocking call from inside a callback works only on a multi-threaded executor, and only when the client sits in a group other than the one that is running. Upstream says the same: "Do not call this method in a callback" (`client.py:183`, `action/client.py:515`).

ActionServer `execute_callback`s run as executor tasks with no callback group (`act.py`). On the multi-threaded executor the server node's timers kept ticking at 100 Hz during a 1 s goal. On single-threaded and Events executors, a blocking `execute_callback` stalls everything.

RAI 3.0 should never block in executor context: use `call_async` and futures, or hop to a worker thread. RAI already offloads user callbacks, so the remaining work is making RAI-owned executor-side callbacks non-blocking.

## 5. Message conversion (Q3)

### 5.1 Data path and copies [code]

- **Native take:** `rcl_take` deserializes CDR into a C struct (copy 1) (`rclpy-lyrical/rclpy/src/rclpy/subscription.cpp:177`). `convert_to_py` then creates a `PyBytes` (copy 2) and `array.frombytes` (copy 3) for `uint8[]` (`rosidl_python-lyrical/rosidl_generator_py/resource/_msg_support.c.em:702-707`).
- **Raw take (`raw=True`):** `rcl_take_serialized_message` (copy 1) and then `py::bytes` (copy 2) (`subscription.cpp:171`).
- **Publish:** `Publisher.publish(msg)` copies into a C struct and serializes. `Publisher.publish(bytes)` publishes serialized bytes directly (`publisher.py:57-71`, both distros).
- **No loaned messages and no type adaptation in rclpy** on either distro. `grep loan`, `TypeAdapter` and `masquerad` over `rclpy/src` and `rclpy/rclpy` return nothing. There is no intra-process path either: same-process pub/sub goes through DDS, which uses SHM on the same host.
- **Lyrical adds `rosidl_buffer.Buffer`** (native buffer backends, `acceptable_buffer_backends=` on `create_subscription`, `node.py:1972`). CPU data is still delivered to Python as `array.array('B')`. Only vendor (for example GPU) backends yield a `Buffer` (`/opt/ros/lyrical/.../rosidl_buffer/__init__.py`). So this is not a CPU zero-copy path for Python.
- **Setter difference:**
  - On Lyrical the generated `uint8[]` setter keeps an `array.array` without copying.
  - Jazzy's setter always re-wraps it with `array.array('B', value)`.
  - With `ROS_PYTHON_CHECK_FIELDS=1`, both distros iterate every byte for non-`array` inputs.

### 5.2 Cost [measured, `bench_conv.py`, median]

| Operation | Image 640x480 rgb8, Lyrical | Image, Jazzy | PointCloud2 307k pts x 16 B (4.9 MB), Lyrical | PointCloud2, Jazzy |
|---|---|---|---|---|
| `Cls(data=bytes)` | 0.017 ms | 0.56 ms | 0.076 ms | 0.15 ms |
| `np.frombuffer(msg.data)` view | about 0 | about 0 | about 0 | about 0 |
| `serialize_message` / `deserialize_message` | 0.045 / 0.050 ms | 0.043 / 0.052 ms | 0.24 / 0.26 ms | 0.24 / 0.26 ms |
| `copy.deepcopy(msg)` | 0.024 ms | 0.027 ms | 0.096 ms | 0.104 ms |
| `message_to_ordereddict` | **242 ms** | **244 ms** | **1321 ms** | **1346 ms** |
| `set_message_fields(Cls(), dict)` | **44 ms** (32 ms of it is the new deep copy) | 10 ms | **258 ms** | fails on reuse (see 10) |
| `copy.deepcopy(dict)` | 32 ms | 102 ms | 198 ms | n/a |

**End to end [measured]:** VGA image flood across processes, single-threaded executor.

| Distro | Native | `raw=True` + `np.frombuffer` over CDR |
|---|---|---|
| Lyrical | 1098/s, p50 1.86 ms | 1124/s, p50 1.86 ms |
| Jazzy | 1006/s, 2.2 ms | 1052/s, 2.2 ms |

The native path's extra two copies cost only about 2.5%. Separately, at 10-30 Hz both images and point clouds show a bimodal latency of about 1 ms or about 11-22 ms. That happens for native and raw alike, on every executor, and with either `RMW_FASTRTPS_PUBLICATION_MODE`. So it sits in Fast DDS large-data transport, not in rclpy, and is worth a separate look (SHM or data-sharing settings, or rmw_zenoh).

### 5.3 Recommendation (Q3)

Native messages should be the default and the only typed path. Expose images and point clouds as numpy views:
- `np.frombuffer(msg.data, dtype).reshape(h, w, c)` for images.
- A structured dtype built from `PointField`s for point clouds.

Both are zero-copy and writable, so the view aliases the message. Document that.

Never dict-convert `uint8[]` payloads. Keep `message_to_ordereddict` only for LLM-facing summaries, and pass `no_arr=True` or `truncate_length` so arrays are not expanded.

Use `raw=True` plus `bytes` publishing only for relays and recorders. It adds CDR-parsing complexity for a 2.5% gain.

## 6. Clients, discovery and graph (Q4)

- **`wait_for_service` polls every 0.25 s** (`client.py:281`, Jazzy `:184`). `ActionClient.wait_for_server` does the same (`action/client.py:751`, Jazzy `:596`). Measured: `wait_for_service(0.1)` on a missing service returns after **0.250 s**, overshooting its timeout.
- **When the server exists, a brand-new client is ready immediately** [measured, `xproc_client.py`, cross-process, both distros]:
  - New client, `service_is_ready()` on the first check: 10 of 10 runs.
  - Per-call client: `create` + `wait_for_service` 0.2-0.3 ms, then `call` 0.3-0.5 ms, and `destroy` 0.01 ms.
  - Reused client: 0.17-0.19 ms per call.
  - Per-goal `ActionClient`: `create` + `wait_for_server` 0.8 ms.

  Creating clients is cheap. The cost comes from never destroying them (section 3.2).
- **No graph guard condition is exposed in rclpy.** There is no `rcl_node_get_graph_guard_condition` binding (grep over `rclpy-*/rclpy/src`). `AsyncClient.wait_for_service` and `AsyncNode.wait_for_node` still poll (`experimental/async_client.py`, `async_node.py`).
- **Event-driven alternatives that do exist:**
  - Publisher and subscription `matched` event callbacks (`event_handler.py`, `QoSSubscriptionMatchedInfo`, both distros) can replace "is anyone publishing?" polling.
  - On Lyrical, `Subscription`, `Client`, `Service` and `Timer` handles expose `set_on_new_message_callback`, `set_on_new_response_callback` and `set_on_new_request_callback` (#1496, `src/rclpy/client.cpp:207-254`). Jazzy has them only inside the C++ `EventsExecutor`.
- **Graph queries are cheap** [measured]: `get_topic_names_and_types` 1-2 µs, `get_publishers_info_by_topic` 11-14 µs, `get_action_names_and_types` 2 µs. Polling them at 10 Hz is not a cost problem. Only the latency of `sleep(0.1)` and `sleep(0.5)` matters.
- **`Client.call(timeout)` semantics:**
  - **Jazzy:** returns `None` on timeout (`rclpy-jazzy/rclpy/rclpy/client.py:104-110`).
  - **Lyrical:** raises `TimeoutError` (`rclpy-lyrical/rclpy/rclpy/client.py:207-211`, #1271).
  - Both remove the pending request (measured: 0 pending afterwards).
  - Without a timeout, both block forever.
- **`rclpy.wait_for_message`** exists on both distros (`wait_for_message.py:25`). It creates a subscription on the given node and waits on a private wait set. If the same node is also being spun, the executor can take the message first, so it is unsuitable for RAI's spinning node.

## 7. Actions (Q5)

- **ActionClient lifecycle:**
  - The constructor calls `node.add_waitable(self)` (`action/client.py:287`).
  - `destroy()` removes it (`:778`). Only explicit `destroy()` cleans up.
  - Each live client adds 3 clients, 2 subscriptions and 1 waitable to every wait set (the `iter_cost.py` numbers above).
  - Measured create cost is 0.9-1.2 ms. Destroy cost is 0.5 ms on Lyrical and 3.1 ms on Jazzy.
- **Goal handle cleanup:**
  - `_goal_handles` holds weak references and drops goals on terminal status (`:470-488`).
  - `_feedback_callbacks` entries are removed only once a result response arrives (`_remove_pending_result_request`, `:328-336`). If you never call `get_result_async`, the feedback callbacks stay registered.
  - RAI does call it.
- **Cancel:**
  - `ClientGoalHandle.cancel_goal()`, `get_result()` and `ActionClient.send_goal()` all block with `event.wait()` and no timeout (Lyrical `action/client.py:627,693`; Jazzy `:412,487,540`).
  - Measured on both distros: with the server gone, `cancel_goal()` was still blocked after 3 s, and one pending cancel request was left behind.
  - RAI's `terminate_goal` (`api/action.py:263`) can therefore hang forever.
- **Server defaults:**
  - `ActionServer(result_timeout)` defaults to 900 s on Jazzy (`server.py:219`) and **10 s** on Lyrical (`server.py:309`). RAI passes 900 explicitly.
  - Lyrical's `publish_feedback` only takes effect in the executing state (#1639).
  - Lyrical adds `enable_feedback_msg_optimization` on `ActionClient`, which content-filters feedback by goal ID (#1633, `action/client.py:194`; only rmw_fastrtps and Connext support content filtering).
- **No async action client or server in `rclpy.experimental`.** `AsyncNode` has no actions.

## 8. QoS (Q6)

- **`BEST_AVAILABLE` exists on both distros** for reliability, durability, liveliness, deadline and lease, plus `qos_profile_best_available`.
  - The RMW resolves it once, at creation, against the endpoints discovered so far (`rmw-lyrical/rmw/include/rmw/types.h:403-421`).
  - Measured on both distros, a `BEST_AVAILABLE` subscription:
    - received a TRANSIENT_LOCAL latched message, which a default subscription did not;
    - matched a BEST_EFFORT sensor publisher (5 of 5 messages);
    - resolved to RELIABLE + TRANSIENT_LOCAL in the graph.

  This is exactly what RAI's hand-written `adapt_requests_to_offers` (`api/base.py:58`) computes. The RMW version also covers deadline and liveliness. It has the same discovery race as RAI's version.
- **`qos_check_compatible(pub, sub)` exists on both distros** and returns `(QoSCompatibility.ERROR, 'ERROR: Best effort publisher and reliable subscription;')`.
- **Content-filtered topics** work on both distros (`content_filter_options=ContentFilterOptions(expr, params)`; backported to Jazzy 7.1.8). Lyrical adds `Subscription.is_cft_supported` (#1618).
- **Type adaptation or type masquerading: none in rclpy** on either distro. Lyrical's buffer backends (section 5.1) are the only new transport hook.

## 9. Context and lifecycle (Q7)

Measured with `sem.py signals` and `ctx.py`, both distros.

- **`rclpy.init()` ownership:**
  - A second `rclpy.init()` on the same context raises `RuntimeError: Context.init() must only be called once`.
  - Multiple contexts work: a second `Context()` with its own `domain_id` keeps working after the default context shuts down.
  - On Lyrical, `rclpy.init()` returns `InitContextManager` (`__init__.py:71`), so `with rclpy.init(): ...` works. On Jazzy it returns `None`.
- **Signals:** `rclpy.init()` installs C-level SIGINT and SIGTERM handlers. They are invisible to `signal.getsignal`, but `get_current_signal_handlers_options()` reports `ALL`. Delivery shuts down every context.
  - Measured: `kill` (SIGTERM) on a running publisher produced `RCLError: publisher's context is invalid`.
  - `rclpy.init(signal_handler_options=SignalHandlerOptions.NO)` opts out. A library like RAI should use that and leave signals to the application.
- **Node destruction on shutdown:**
  - **Lyrical:** `rclpy.shutdown()` destroys every tracked node (`Context._cleanup`, `context.py:150-163`, plus the fix #1752 that stopped it skipping every other node).
    - `node.get_name()` afterwards raises `InvalidHandle`.
    - `destroy_node()` afterwards is a no-op.
  - **Jazzy:** nodes survive. `get_name()` still works, and creating entities raises `RCLError` instead of `InvalidHandle`.
- **Destroying a node or subscription while spinning:**
  - 200 create/destroy cycles under traffic (rclpy#1142, the scenario behind RAI's `destroy_subscribers=False` workaround) gave 0 errors and kept spin alive on all three executors and both distros.
  - On the multi-threaded executor each destroy took 50-90 ms, because the busy loop keeps the handle in use.
  - Lyrical's "streamlined destroy" (#1629) makes `entity.destroy()` remove itself from the node (`node.py:2584-2600`). On Jazzy you must go through `node.destroy_*()`. `publisher.destroy()` in RAI's `api/topic.py:252` leaves stale entries on Jazzy.

## 10. Jazzy vs Lyrical differences RAI must abstract (Q8)

Taken from the full public-API diff (`api_dump.py` → `api_diff.txt`) plus behaviour probes.

| Area | Jazzy 7.1.x | Lyrical 10.0.x | RAI impact |
|---|---|---|---|
| Endpoint info module | `rclpy.topic_endpoint_info.TopicEndpointInfo`, `TopicEndpointTypeEnum` | `rclpy.endpoint_info.TopicEndpointInfo`, `EndpointTypeEnum`, plus `ServiceEndpointInfo` | Import shim (already in the working tree) |
| `rclpy.qos_event`, `rclpy.destroyable` | present (deprecated) | removed | Use `rclpy.event_handler` |
| `Client.call` on timeout | returns `None` | raises `TimeoutError` | Normalize to one exception |
| `set_message_fields(msg, d)` | **mutates `d`**: nested message lists are replaced in place (`set_message.py:72,99`); a second call with the same dict raises `TypeError` [measured] | deep-copies `d` first (`set_message.py:61`), safe but +32 ms on a VGA dict [measured] | Do not reuse dicts; avoid on hot paths |
| `rclpy.shutdown()` | nodes left alive | destroys all nodes | Make connector shutdown idempotent |
| `rclpy.init()` return value | `None` | `InitContextManager` | Do not rely on it |
| `try_shutdown` | `(context)` | `(context, uninstall_handlers)` | n/a |
| `ActionServer` `result_timeout` default | 900 | 10 | Always pass it explicitly |
| `default_cancel_callback` parameter | `cancel_request` | `goal_handle` | Call by position |
| `Node`/`create_node` kwarg | n/a | `rosout_qos_profile=` | n/a |
| `create_subscription` kwarg | n/a | `acceptable_buffer_backends=` | n/a |
| Node graph helpers | `rclpy.action.graph.*` only | also `Node.get_action_names_and_types()`, `get_clients_info_by_service()`, `get_servers_info_by_service()` | Use the module functions for portability |
| `declare_parameter(name)` with no value | warning | `TypeError` | RAI call sites all pass defaults |
| QoS enum aliases `RMW_QOS_POLICY_*` | deprecated | removed | n/a |
| `Clock` | n/a | `Clock(*, clock_type=)` keyword-only | n/a |
| Timer callback | no arguments | may take `TimerInfo` (inspected via `inspect.signature` on every fire, `executors.py:511-513`) | n/a |
| `MessageInfo` | no `publisher_gid` | `publisher_gid` | n/a |
| Message base classes | plain `object` | `rosidl_pycommon.interface_base_classes.BaseMessage` (ABC); services and actions get `BaseService` and `BaseAction` | `isinstance` checks on Lyrical only |
| `uint8[]` setter | always copies into a new `array` | keeps a passed `array.array`; accepts `rosidl_buffer.Buffer` | n/a |
| `Subscription`/`Client`/`Service`/`Timer` handle `set_on_new_*_callback` | no | yes | n/a |
| `rclpy.experimental` | `EventsExecutor` | `EventsExecutor` plus `AsyncNode`, `AsyncClient` and friends (Python 3.12+) | n/a |
| `Future.add_done_callback` | `(cb)` | `(cb, *, unique=False)` | n/a |
| `MultiThreadedExecutor.shutdown` | `(timeout)`; 7.1.13 waits for spin exit | `(timeout, *, wait_for_threads=True)`; no spin-exit wait | n/a |
| Entity constructors (`Subscription`, `Client`, `Service`, `Publisher`, `Timer`) | positional `callback_group` | `callback_group` keyword-only, plus `on_destroy` | Only matters if subclassing |
| `Subscription.is_cft_supported` | no | yes | n/a |

Notes on rows marked n/a, in table order:
- `acceptable_buffer_backends=`: Lyrical-only.
- `Clock`: pass `clock_type` by keyword on both distros.
- Timer `TimerInfo`: this inspection adds per-fire overhead on Lyrical.
- `publisher_gid`: useful for de-duplication.
- `set_on_new_*_callback`: see section 11.
- `MultiThreadedExecutor.shutdown`: see section 3.5.

Everything else RAI touches has the same signature on both distros, including `Client.call/call_async/wait_for_service`, `ActionClient`, `serialize_message`/`deserialize_message`, `message_to_ordereddict`, `wait_for_message`, and `qos_check_compatible`.

Python versions differ (3.12 vs 3.14), but no Python-level incompatibility appeared in the probes.

## 11. New in Lyrical worth building on, or avoiding (Q9)

### 11.1 `rclpy.experimental.AsyncNode`: avoid for now

`rclpy.experimental.AsyncNode` (#1620) drives subscriptions, services, clients and timers on an asyncio loop, with no executor. It uses Python `set_on_new_*_callback` and `loop.call_soon_threadsafe` (`experimental/async_subscription.py:56,68`).

Measured on Lyrical:
- In-process service call: 0.031 ms.
- 1 kHz subscription: p50 0.04 ms.
- `asyncio.timeout` cancellation works.

**Deadlock under flood** [measured with gdb, `async_starve.py`]:
- The DDS SHM receive thread holds the DataReader mutex in `DataReaderImpl::set_read_communication_status` → `RMWSubscriptionEvent::update_data_available`. It then calls rclpy's Python on-new-message callback, which waits for the GIL.
- Meanwhile the main thread holds the GIL inside `take_message` → `rcl_take` → `DataReaderImpl::read_or_take` and waits for that same mutex (`scratchpad/bench/gdb.out`).
- The whole process froze. Even a watchdog thread never ran.

Other gaps:
- No action support.
- A busy subscription can monopolize the event loop, because the non-concurrent read loop yields without awaiting while data keeps arriving.

**Verdict:** do not adopt `AsyncNode` in 3.0. Revisit when it leaves experimental and `take` releases the GIL.

### 11.2 Other Lyrical additions

- **EventsExecutor:** the same as on Jazzy, so it is the portable fast path (section 3).
- **Type description service:** `~/get_type_description` on every node, on both distros. RAI can fetch full message definitions from remote nodes for LLM tool schemas without installing the package. rclpy has no dynamic typesupport, so it cannot deserialize such types.
- **Action graph and service endpoint info:** `Node.get_action_names_and_types()` and service endpoint info including QoS (`get_clients_info_by_service`).
- **Action feedback content filter:** `enable_feedback_msg_optimization`.
- **`TimerInfo`, `MessageInfo.publisher_gid`, and context-manager init.**
- **Native buffer backends (`rosidl_buffer`):** only relevant if RAI ever consumes GPU-resident images published by a vendor backend.

## 12. Recommendations for RAI 3.0 ROS layer

1. **Executor**
   - Make `EventsExecutor` the default on both distros.
   - Keep `SingleThreadedExecutor` as a supported fallback (`executor="events" | "single"`).
   - Drop `MultiThreadedExecutor`. If kept at all, document that it is unusable with rmw_fastrtps and any subscription that has event handlers, including tf2.
   - Keep one spin thread per connector (or one shared executor per context), daemonized, and owned by the connector.
   - Require `rclpy>=7.1.11` on Jazzy.
2. **Never block, never raise in executor context**
   - Wrap every RAI-registered callback (subscription preprocessors, service handlers, goal, cancel and accept callbacks, timers) in a guard that logs and swallows the exception. A single exception otherwise kills `spin()` on every executor.
   - Run action `execute_callback`s and user handlers on RAI's worker pool. Bridge results back through rclpy futures (`executor.create_future()`, both distros).
   - Supervise the spin thread, and surface a health flag if it dies.
3. **Shutdown order**
   - Run `executor.shutdown()`, then `join` the spin thread, then destroy RAI-owned entities (action clients first), then `destroy_node()`.
   - Make every step idempotent, because Lyrical's `rclpy.shutdown()` already destroyed the node.
   - Do not trust `Executor.shutdown(timeout)` to wait for callbacks (dead code, section 3.5). Track in-flight RAI work yourself.
   - Initialize with `SignalHandlerOptions.NO` when RAI owns `rclpy.init`. Otherwise let the application own signals.
4. **Messages**
   - Make native messages the API: `send_message(msg: RosMsg)` and callbacks that receive the message object.
   - Provide `image_view(msg) -> np.ndarray` and `pointcloud_view(msg) -> structured ndarray`, both as zero-copy `np.frombuffer` views.
   - Build outgoing images with `Image(data=ndarray.tobytes())` or `array.array`.
   - Keep dict conversion only as an explicit adapter for LLM-facing tools. Reject or summarize large arrays (`no_arr`, `truncate_length`).
   - Fix `ROS2HRIMessage.to_ros2_dict()` (`messages.py:78`) to publish native messages. It currently costs about 290 ms per VGA image.
   - Remove `copy.deepcopy` of feedback in `api/action.py:108` and `:225`.
5. **Raw path:** offer an opt-in `raw=True` subscription and `bytes` publish for relays and recorders only. The measured gain for images is about 2.5%.
6. **Services**
   - Cache clients per `(name, type)` and destroy them on release.
   - Normalize timeouts: catch Lyrical's `TimeoutError`, map Jazzy's `None` to the same exception, and raise one RAI exception type.
   - Replace `wait_for_service` and `wait_for_server` (0.25 s quantization) with RAI's own `service_is_ready()` and `server_is_ready()` loop at 10-20 ms granularity, bounded by the caller's timeout.
7. **Actions**
   - Cache one `ActionClient` per `(name, type)` instead of one per goal, and `destroy()` it on release and shutdown. Measured cost per leaked client: about 4 µs per callback, forever.
   - Use `cancel_goal_async()` and `get_result_async()` with RAI-side timeouts. Never use the blocking variants, which have no timeout.
   - Always pass `result_timeout` explicitly.
8. **QoS:** replace `adapt_requests_to_offers` with `QoSProfile(... reliability=BEST_AVAILABLE, durability=BEST_AVAILABLE ...)` (or `qos_profile_best_available` with an explicit depth) on both distros. Keep `qos_check_compatible` for diagnostics. Note the shared discovery race: re-create the subscription on a `matched` or incompatible-QoS event if needed.
9. **Graph waits:** replace `sleep(0.1)` and `sleep(0.5)` polling with subscription and publisher `matched` event callbacks where an endpoint exists. Elsewhere, poll the cheap graph APIs at 10-20 ms. Delete the fixed `time.sleep(0.5)` in `NodeDiscovery.__init__`.
10. **Compat layer:** add a tiny `rai.communication.ros2.compat` module that holds every item in the section 10 table:
    - endpoint-info import;
    - `call` timeout normalization;
    - `set_message_fields` without reuse;
    - shutdown idempotence;
    - action `result_timeout`;
    - `destroy` via `node.destroy_*`;
    - `BaseMessage` detection with a duck-typed fallback on Jazzy.

    Branch on feature probes such as `hasattr` or `ImportError`, never on `ROS_DISTRO`.
11. **Subscriber destruction:** re-enable real `destroy_subscription` (drop the rclpy#1142 workaround). There were 0 failures in 200 cycles under traffic on both distros with the recommended executors. Keep a regression test.
12. **Do not adopt yet:**
    - `AsyncNode` (deadlock, no actions).
    - `rclpy.wait_for_message` on a spinning node.
    - Python-level `set_on_new_message_callback`, which hits the same GIL-versus-DDS-lock hazard as `AsyncNode`.
13. **Follow-ups worth filing upstream:**
    - rmw_fastrtps skips `rmw_wait` on a shared DataReader `StatusCondition` (section 3.3).
    - `AsyncNode` deadlock: `take` should release the GIL.
    - Dead wait in `Executor.shutdown`.
    - Lyrical is missing the #1749 multi-threaded shutdown race fix.
    - Investigate separately the Fast DDS bimodal 11 ms large-message latency at low rates.
