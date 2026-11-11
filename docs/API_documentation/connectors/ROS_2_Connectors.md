# ROS2 Connectors

RAI provides robust connectors for interacting with ROS 2 middleware, supporting both standard and human-robot interaction (HRI) message flows.

| Connector          | Description                                                                                     | Example Usage                        |
| ------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------ |
| `ROS2Connector`    | Standard connector for generic ROS 2 topics, services, and actions.                             | `connector = ROS2Connector()`        |
| `ROS2HRIConnector` | Connector for multimodal HRI messages over ROS 2, combining ROS2BaseConnector and HRIConnector. | `hri_connector = ROS2HRIConnector()` |

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

The `ROS2HRIConnector` extends `ROS2BaseConnector` and implements the `HRIConnector` interface for multimodal human-robot interaction messages. It is specialized for exchanging `ROS2HRIMessage` objects, which can contain text, images, and audio.

### Key Features

-   Publishes and subscribes to `rai_interfaces/msg/HRIMessage` topics
-   Converts between ROS 2 HRI messages and the internal multimodal format
-   Supports all standard connector operations (topics, services, actions)
-   Suitable for integrating AI agents with human-facing ROS 2 interfaces

### Example Usage

```python
from rai.communication.ros2.connectors import ROS2HRIConnector

hri_connector = ROS2HRIConnector()

# Send a multimodal HRI message to a topic
hri_connector.send_message(
    message=my_hri_msg,  # ROS2HRIMessage
    target="/to_human"
)

# Register a callback for incoming HRI messages
hri_connector.register_callback(
    source="/from_human",
    callback=on_human_message
)
```

### Message Conversion

The `ROS2HRIConnector` automatically converts between ROS 2 `rai_interfaces/msg/HRIMessage` and the internal `ROS2HRIMessage` format for seamless multimodal communication.

## Usage in Agents

Both connectors are commonly used in RAI agents to interface with ROS 2 environments. Example:

```python
from rai.communication.ros2.connectors import ROS2Connector, ROS2HRIConnector

ros2_connector = ROS2Connector()
hri_connector = ROS2HRIConnector()

# Use with tools or agents
agent = ReActAgent(
    target_connectors={"/to_human": hri_connector},
    tools=ROS2Toolkit(connector=ros2_connector).get_tools()
)

# Subscribe to human input
agent.subscribe_source("/from_human", hri_connector)
```

## ROS2 Utilities

RAI provides utilities for error handling and parameter access when working with ROS2 connectors:

-   `ROS2ServiceError`: Exception raised for service-related errors (e.g., service unavailable, timeout). Includes service name, timeout duration, and diagnostic suggestions.

-   `ROS2ParameterError`: Exception raised for parameter-related errors (e.g., missing parameter, type mismatch). Includes parameter name, expected type/value, and suggestions for resolution.

-   `get_param_value()`: Utility function for simplified ROS2 parameter access with automatic type conversion and default value support. Reduces boilerplate compared to direct ROS2 parameter API usage.

```python
from rai.communication.ros2 import ROS2ServiceError, ROS2ParameterError, get_param_value

# Access parameters with defaults
service_name = get_param_value(node, "/my_service/name", default="/default_service")

# Handle service errors
try:
    response = connector.service_call(...)
except ROS2ServiceError as e:
    print(f"Service {e.service_name} unavailable: {e.suggestion}")
```

## See Also

-   [Connectors Overview](./overview.md)
-   [ROS 2 Aggregators](../aggregators/ROS_2_Aggregators.md)
-   [ROS 2 Tools](../langchain_integration/ROS_2_tools.md)
