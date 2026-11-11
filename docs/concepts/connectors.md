# Connectors and the runtime

A connector is how RAI code talks to the robot. `ROS2Connector` covers topics, services, actions, TF, parameters and the ROS graph. `ROS2HRIConnector` carries human-robot messages (see [Content and HRI messages](content.md)). Both speak native ROS 2 messages, have an async API that works from any event loop, and have blocking twins under `connector.sync`.

## The runtime

Connectors run on a `ROS2Runtime`: one node, one executor and one spin thread. Connectors created without arguments share one runtime per process, so ten tools with ten connectors still make one node. rclpy is initialized on first use; no `rclpy.init()` or `ROS2Context` is needed.

```python
from rai.communication.ros2 import ROS2Connector

shared = ROS2Connector()                         # the process-wide runtime
private = ROS2Connector("my_node")               # its own node
sim = ROS2Connector("sim_agent", use_sim_time=True)
single = ROS2Connector("worker", executor="single")
```

-   **Executor.** `EventsExecutor` by default, `executor="single"` for `SingleThreadedExecutor`. The executor thread only takes messages and completes futures.
-   **Callbacks.** User code (subscription callbacks, service handlers, action callbacks) never runs on the executor thread. Each callback runs on a worker, one call at a time, in arrival order. An exception in a callback is logged and never stops delivery. Async callbacks are awaited.
-   **Entities.** One publisher, subscription, service client or action client per name and type, created on first use and reused.
-   **Lifecycle.** `close()`, or leaving a `with` / `async with` block, stops the connector's subscriptions and fails its pending calls with `ConnectorClosedError`. The runtime closes with its last connector: it stops the executor, then destroys entities and the node. `close()` is idempotent.

## Topics

<!-- test: run -->

```python
from std_msgs.msg import String

from rai.communication.ros2 import QoS, ROS2Connector

with ROS2Connector() as connector:
    connector.sync.publish("/status", String(data="ready"), qos=QoS.LATCHED)
    msg = connector.sync.receive("/status", timeout=5.0)
    print(msg.data)
```

-   `publish(topic, msg, qos=QoS.COMMAND)` publishes a message.
-   `receive(topic, timeout=1.0, max_age=None)` returns the latest message. Without `max_age` a cached sample of any age is returned (useful for latched topics). With `max_age` only a sample at most that many seconds old is returned, waiting up to `timeout` for one; otherwise it raises `ROS2TimeoutError`.
-   `subscribe(topic, callback=None, maxsize=10)` returns a `Subscription`. With a callback, messages are delivered to it. Without one, iterate it (`async for msg in sub`) or `await sub.get(timeout)`. A full queue drops its oldest message; `maxsize=None` never drops. `sub.close()` ends it.
-   `msg_type` (a class or `"pkg/msg/Name"`) defaults to the type in the ROS graph.

### QoS presets

| Preset        | Profile                                   | Use                                                                        |
| ------------- | ----------------------------------------- | -------------------------------------------------------------------------- |
| `QoS.COMMAND` | reliable, volatile, depth 10              | Publisher default. Never replays old commands to late subscribers          |
| `QoS.SENSOR`  | best effort, volatile, depth 5            | High-rate sensor streams                                                   |
| `QoS.LATCHED` | reliable, transient local, depth 1        | State such as a map or a status                                            |
| `QoS.HRI`     | reliable, volatile, depth 100             | Human-robot messages; never drops                                          |
| `QoS.AUTO`    | best available reliability and durability | Subscription default: matches the publishers discovered when it is created |

## Services

<!-- test: run -->

```python
from std_srvs.srv import Trigger

from rai.communication.ros2 import ROS2Connector


def reset(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
    response.success = True
    return response


with ROS2Connector() as connector:
    connector.create_service("/reset", reset, service_type=Trigger)
    print(connector.sync.call("/reset", Trigger.Request(), timeout=5.0))
```

`call(service, request, timeout=5.0)` returns the response or raises `ROS2TimeoutError` if the service is not available or does not answer in time. `idempotent=True` resends an unanswered request with backoff; use it only for requests that are safe to run twice. `create_service(name, handler, service_type=...)` serves a service; the handler runs off the executor thread.

## Actions

```python
from nav2_msgs.action import NavigateToPose

async def go(connector):
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x = 1.0
    handle = await connector.send_goal("/navigate_to_pose", goal, on_feedback=print)
    try:
        return await handle.result(timeout=120.0)
    except TimeoutError:
        await handle.cancel()
        raise
```

`send_goal` returns a `GoalHandle` once the goal is accepted, or raises `ROS2GoalRejectedError` (rejected) or `ROS2TimeoutError` (no server, or no answer within `accept_timeout`). The handle has `id`, `status`, `done`, `feedback()` (feedback received since the last call), `await result(timeout)` and `await cancel()`; `handle.sync` has blocking `result` and `cancel`. A goal accepted after its timeout is cancelled. `create_action(name, execute, action_type=...)` serves an action.

## TF, time, parameters and the graph

| Method                                                                                                                                                                           | Returns                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `await transform(target, source, timeout=1.0)`                                                                                                                                   | The latest `TransformStamped`; raises `tf2_ros.LookupException`            |
| `tf_frames()`                                                                                                                                                                    | The TF frames and their parents                                            |
| `now()`                                                                                                                                                                          | The node's clock (`use_sim_time` follows `/clock`)                         |
| `declare_parameter(name, default)`                                                                                                                                               | The parameter's value; `--ros-args -p` and launch files override `default` |
| `resolve(name, kind="topic")`                                                                                                                                                    | The fully qualified name, remaps applied (`"cmd_vel"` to `"/cmd_vel"`)     |
| `get_topics_names_and_types()`, `get_services_names_and_types()`, `get_actions_names_and_types()`, `get_node_names()`, `get_node_interfaces(node)`, `get_topic_endpoints(topic)` | Graph queries from the local discovery cache                               |

The node itself is not exposed, so every publish, service call and goal passes the connector's [egress hooks](safety.md#connector-side-egress-hooks).

## Images and point clouds

`rai.communication.ros2.views` turns sensor messages into numpy arrays without copying the message data. `encode_image` is the one place images are encoded for a model (JPEG, downscaled to 1024 px on the long side by default).

```python
from rai.communication.ros2 import encode_image, views

pixels = views.image(image_msg)     # (H, W, C) view in the message's encoding
cloud = views.points(cloud_msg)     # structured array: cloud["x"], cloud["y"], ...
b64 = encode_image(image_msg)       # base64 JPEG for the LLM
```

A view aliases the message: copy it to change it or to keep it after the message is gone.

## Errors

| Exception                                | Raised when                                                    |
| ---------------------------------------- | -------------------------------------------------------------- |
| `ROS2TimeoutError` (a `TimeoutError`)    | A receive, service call or goal does not complete in time      |
| `ROS2GoalRejectedError`                  | An action server rejects a goal                                |
| `rai.communication.ConnectorClosedError` | A call is made on, or was pending in, a closed connector       |
| `EgressDenied` (a `PermissionError`)     | An egress hook refuses a publish, service call, goal or cancel |

## Use from async code

<!-- test: run -->

```python
import asyncio

from std_msgs.msg import String

from rai.communication.ros2 import QoS, ROS2Connector


async def main() -> None:
    async with ROS2Connector() as connector:
        await connector.publish("/status", String(data="ready"), qos=QoS.LATCHED)
        msg = await connector.receive("/status", timeout=5.0)
        print(msg.data)


asyncio.run(main())
```

Do not call `connector.sync` from an async callback or a coroutine: it blocks the event loop. Use the async methods there.

See the [API reference](../api/communication.md) for every parameter.
