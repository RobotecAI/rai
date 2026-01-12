# ROS2 Connectors

RAI provides robust connectors for interacting with ROS 2 middleware, supporting both standard and human-robot interaction (HRI) message flows.

| Connector          | Description                                                                                     | Example Usage                        |
| ------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------ |
| `ROS2Connector`    | Standard connector for generic ROS 2 topics, services, and actions.                             | `connector = ROS2Connector()`        |
| `ROS2HRIConnector` | Connector for multimodal HRI messages over ROS 2, combining ROS2BaseConnector and HRIConnector. | `hri_connector = ROS2HRIConnector()` |

## `ROS2Connector`

The `ROS2Connector` is the main interface for publishing, subscribing, and calling services/actions in a ROS 2 system. It is a concrete implementation of `ROS2BaseConnector` for standard ROS 2 messages.

### Class definition

??? info "ROS2BaseConnector class definition"

    !!! info "ROS2Connector vs ROS2BaseConnector"

        `ROS2Connector` is a simple alias for `ROS2BaseConnector`. It exists mainly to provide a more intuitive and consistent class name for users, but does not add any new functionality.

    ::: rai.communication.ros2.connectors.base.ROS2BaseConnector

### Key Features

-   Manages ROS 2 node lifecycle and threading (via `MultiThreadedExecutor`, see [ROS 2 Executors](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Executors.html))
-   Supports topic-based message passing (publish/subscribe, see [ROS 2 Topics](https://docs.ros.org/en/jazzy/Concepts/Basic/About-Topics.html))
-   Service calls (request/response, see [ROS 2 Services](https://docs.ros.org/en/jazzy/Concepts/Basic/About-Services.html))
-   Actions (long-running operations with feedback, see [ROS 2 Actions](https://docs.ros.org/en/jazzy/Concepts/Basic/About-Actions.html))
-   TF (Transform) operations, see [ROS 2 TF](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html)
-   Callback registration for asynchronous notifications

### Example Usage

```python
from rai.communication.ros2.connectors import ROS2Connector

connector = ROS2Connector()

# Send a message to a topic
connector.send_message(
    message=my_msg,  # ROS2Message
    target="/my_topic",
    msg_type="std_msgs/msg/String"
)

# Register a callback for a topic
connector.register_callback(
    source="/my_topic",
    callback=my_callback,
    msg_type="std_msgs/msg/String"
)

# Call a service
response = connector.service_call(
    message=my_request_msg,
    target="/my_service",
    msg_type="my_package/srv/MyService"
)

# Start an action
handle = connector.start_action(
    action_data=my_goal_msg,
    target="/my_action",
    msg_type="my_package/action/MyAction",
    on_feedback=feedback_cb,
    on_done=done_cb
)

# Get available topics
topics = connector.get_topics_names_and_types()
```

### Node Lifecycle and Threading

The connector creates a dedicated ROS 2 node and runs it in a background thread, using a `MultiThreadedExecutor` for asynchronous operations. This allows for concurrent message handling and callback execution.

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
