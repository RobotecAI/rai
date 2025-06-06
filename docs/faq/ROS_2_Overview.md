# Overview

RAI provides an abstraction layer for ROS 2 which streamlines the process of integrating ROS 2 with LLMs and other AI components. This integration bridge is essential for modern robotics systems that need to leverage artificial intelligence capabilities alongside traditional robotics frameworks.

At the heart of this integration is the `rai.communication.ros2.ROS2Connector`, which provides a unified interface to ROS 2's subscription, service, and action APIs. This abstraction layer makes it straightforward to build ROS 2 agents and LangChain tools that can seamlessly communicate with both robotics and AI components.

??? info "Why is RAI not a ROS 2 package?"

    RAI was initially developed as a ROS 2 package, but this approach proved problematic for several key reasons:

    1. **Dependency Management**
         - ROS 2's dependency system (rosdep) is too vague and inflexible for RAI's needs
         - RAI heavily relies on the Python ecosystem, particularly for AI/LLM integration
         - Poetry provides more precise version control and dependency resolution, crucial for AI/ML components

    2. **Architectural Separation**
         - The initial ROS 2 package approach made it difficult to maintain clear separation between:
             - Core AI/LLM functionality
             - Communication layer
             - Robot-specific implementations
         - This separation is crucial for maintainability and extensibility

    3. **Development Workflow**
         - ROS 2's build system adds unnecessary complexity for Python-based AI development
         - The mixed C++/Python ecosystem of ROS 2 doesn't align well with RAI's Python-first approach
         - Faster development cycles are possible with a Python-centric architecture

    4. **Flexibility and Portability**
         - RAI needs to work both with and without ROS 2 which allows vast amount of use cases which contribute to the RAI ecosystem
         - The framework should be deployable in various environments
         - Different communication protocols should be easily supported

    The current architecture, with RAI as a Python framework that works *with* ROS 2 rather than *as* ROS 2, provides the best of both worlds:
     - Full ROS 2 compatibility through the `ROS2Connector` abstraction
     - Clean separation of concerns
     - Flexible dependency management
     - Better maintainability and extensibility

## Key Features

-   **Unified Communication Interface**: Simple, generic interface for ROS 2 topics, services, and actions
-   **Automatic QoS Matching**: Handles Quality of Service profiles automatically
-   **Thread-Safe Operations**: Built-in thread safety for concurrent operations

## Supported ROS 2 Features

-   Topics (publish/subscribe)
-   Services (request/response)
-   Actions (long-running operations with feedback)
-   TF2 transforms

## Integration Examples

```python
from rai.communication.ros2 import ROS2Connector, ROS2Context, ROS2Message


@ROS2Context()
def main():
    connector = ROS2Connector()

    # Subscribe to a topic
    def my_custom_callback(message: ROS2Message):
        message.payload  # actual ROS 2 message

    connector.register_callback("/topic", my_custom_callback)

    # Receive a message
    message = connector.receive_message("/topic")
    message.payload  # actual ROS 2 message

    # Publish a message
    message = ROS2Message(payload={"data": "Hello, ROS 2!"})
    connector.send_message(message, "/topic", msg_type="std_msgs/msg/String")

    # Call a service
    request = ROS2Message(payload={"data": True})
    response = connector.service_call(
        message=request, target="/service", msg_type="std_srvs/msg/SetBool"
    )

    # Start an action
    def my_custom_feedback_callback(feedback: ROS2Message):
        feedback.payload  # actual ROS 2 feedback

    message = ROS2Message(
        payload={"pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}}}
    )
    action_id = connector.start_action(
        action_data=message,
        target="/navigate_to_pose",
        msg_type="nav2_msgs/action/NavigateToPose",
        feedback_callback=my_custom_feedback_callback,
    )

    # Cancel an action
    connector.terminate_action(action_id)
```

## Best Practices

1. **Context Management**

    - Always use `ROS2Context` as a decorator or context manager
    - Ensures proper initialization and cleanup

2. **Resource Management**

    - Use the connector's shutdown method when done
    - Handle exceptions appropriately

3. **Performance Considerations**

    - Use appropriate QoS profiles
    - Consider deregistering callbacks when not needed

## Common Use Cases

1. **Robot Control**

    - Navigation commands
    - Manipulation tasks
    - Sensor data processing

2. **System Integration**
    - Connecting AI components to ROS 2
    - Multi-agent coordination

## Troubleshooting

Common issues and their solutions:

1. **Connection Problems**

    - Check ROS 2 network configuration
    - Verify topic/service names
    - Ensure proper QoS matching

2. **Performance Issues**

    - Monitor thread usage
    - Check QoS settings
    - Verify message sizes

3. **Resource Management**
    - Proper cleanup of resources
    - Handling of node lifecycle
    - Memory management

## Future Developments

-   Performance optimizations
