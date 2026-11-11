# ROS2 Connectors

RAI provides robust connectors for interacting with ROS 2 middleware, supporting both standard and human-robot interaction (HRI) message flows.

| Connector          | Description                                                             | Example Usage                        |
| ------------------ | ----------------------------------------------------------------------- | ------------------------------------ |
| `ROS2Connector`    | Standard connector for generic ROS 2 topics, services, and actions.     | `connector = ROS2Connector()`        |
| `ROS2HRIConnector` | Connector for multimodal HRI messages (text, images, audio) over ROS 2. | `hri_connector = ROS2HRIConnector()` |

## `ROS2Connector`

`ROS2Connector` publishes, subscribes, and calls services and actions. Its API is async and works from any event loop; `connector.sync` runs the same calls from blocking code. Messages are native ROS 2 messages.

### Class definition

::: rai.communication.ros2.connectors.ros2_connector.ROS2Connector

### Example Usage

```python
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

from rai.communication.ros2 import QoS, ROS2Connector

with ROS2Connector() as connector:
    connector.sync.publish("/cmd_vel", Twist(), qos=QoS.COMMAND)
    scan = connector.sync.receive("/scan", timeout=1.0, max_age=0.5)
    response = connector.sync.call("/reset", Trigger.Request(), timeout=5.0)

    sub = connector.subscribe("/odom", callback=print)  # returns a Subscription
    sub.close()
```

In async code, use `await connector.publish(...)`, `await connector.receive(...)`, `await connector.call(...)` and `await connector.send_goal(...)`, which returns a `GoalHandle` with `result()`, `feedback()` and `cancel()`.

### Node Lifecycle and Threading

Connectors run on a `ROS2Runtime`: one node, one executor (`EventsExecutor` where available, otherwise `SingleThreadedExecutor`) and one spin thread. Connectors created with default arguments share one runtime per process. User callbacks never run on the executor thread. `close()`, or leaving the `with` block, cancels pending calls with `ConnectorClosedError` and destroys the node's entities.

## `ROS2HRIConnector`

`ROS2HRIConnector` sends and receives `rai.communication.HRIMessage`s (text, `rai.content.Image`s and `rai.content.Audio`, a communication id and sequence numbers) as `rai_interfaces/msg/HRIMessage`, on the same runtime as `ROS2Connector`. HRI QoS is reliable and its subscriptions never drop, so every message from a human arrives, in order. Received images and audio are numpy views of the ROS message.

### Class definition

::: rai.communication.ros2.connectors.hri_connector.ROS2HRIConnector

### Example Usage

```python
import numpy as np

from rai.communication import HRIMessage
from rai.communication.ros2 import ROS2HRIConnector
from rai.content import Image

with ROS2HRIConnector() as hri_connector:
    hri_connector.sync.send("/to_human", "Hello")  # text only
    image = Image(np.zeros((480, 640, 3), np.uint8), "rgb8")
    hri_connector.sync.send("/to_human", HRIMessage("Look", images=[image]))

    sub = hri_connector.subscribe("/from_human", callback=print)
    last = hri_connector.sync.receive("/from_human", timeout=5.0)
```

## Usage in Agents

```python
from rai.agents import HRIOutput, ReActAgent, hri_connector_sender
from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector
from rai.tools.ros2 import ROS2Toolkit

ros2_connector = ROS2Connector()
hri_connector = ROS2HRIConnector()

agent = ReActAgent(tools=ROS2Toolkit(connector=ros2_connector).get_tools())
agent.subscribe(HRIOutput(hri_connector_sender(hri_connector, "/to_human")))
hri_connector.subscribe("/from_human", agent.submit)
```

## ROS2 Utilities

RAI provides utilities for error handling and parameter access when working with ROS2 connectors:

-   `ROS2ServiceError`: Exception raised for service-related errors (e.g., service unavailable, timeout). Includes service name, timeout duration, and diagnostic suggestions.

-   `ROS2ParameterError`: Exception raised for parameter-related errors (e.g., missing parameter, type mismatch). Includes parameter name, expected type/value, and suggestions for resolution.

-   `get_param_value()`: Utility function for simplified ROS2 parameter access with automatic type conversion and default value support. Reduces boilerplate compared to direct ROS2 parameter API usage.

`ROS2Connector` calls raise `ROS2TimeoutError` when a service or action server does not answer in time, and `ROS2GoalRejectedError` when a goal is rejected.

```python
from std_srvs.srv import Trigger

from rai.communication.ros2 import ROS2Connector, ROS2TimeoutError

with ROS2Connector() as connector:
    rate = connector.declare_parameter("rate", 1.0)  # --ros-args overrides win
    try:
        response = connector.sync.call("/reset", Trigger.Request(), timeout=5.0)
    except ROS2TimeoutError as e:
        print(f"No answer: {e}")
```

## See Also

-   [Connectors Overview](./overview.md)
-   [ROS 2 Aggregators](../aggregators/ROS_2_Aggregators.md)
-   [ROS 2 Tools](../langchain_integration/ROS_2_tools.md)
