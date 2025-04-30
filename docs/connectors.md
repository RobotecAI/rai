# Connectors

![connectors](./imgs/connectors.png)

Connectors are a set of abstract interfaces and implementations designed to provide a unified way to
interact with various communication systems, including robot middleware like ROS2, sound devices,
and other I/O systems.

## Connector Architecture

The connector architecture is built on a hierarchy of abstract base classes and concrete
implementations:

### Base Classes

- **BaseConnector\<T>**: The foundation interface that defines common communication patterns:

  - Message passing (publish/subscribe)
  - Service calls (request/response)
  - Actions (long-running operations with feedback)
  - Callback registration for asynchronous notifications

- **HRIConnector\<T>**: Extends BaseConnector with Human-Robot Interaction capabilities:
  - Supports multimodal messages (text, images, audio)
  - Provides conversion to/from Langchain message formats
  - Handles message sequencing and conversation IDs

### Concrete Implementations

#### ROS 2 Connectors

ROS 2 connectors provide integration with the ROS 2 middleware:

- **ROS2BaseConnector\<T>**: Core implementation for ROS 2 communication

  - Manages ROS 2 node lifecycle and threading
  - Implements topic-based message passing
  - Provides TF (Transform) functionality
  - Uses a MultiThreadedExecutor for async operations

- **ROS2Connector**: A concrete implementation of ROS2BaseConnector using standard ROS 2 messages

- **ROS2HRIConnector**: Combines ROS2BaseConnector and HRIConnector to provide human-robot
  interaction capabilities over ROS2

ROS 2 connectors use a mixin-based design with specialized components:

- **ROS2ActionMixin**: Implements action client/server functionality
- **ROS2ServiceMixin**: Implements service client/server functionality

#### Sound Device Connector

- **SoundDeviceConnector**: Provides audio streaming capabilities using the sounddevice library
  - Supports audio playback and recording
  - Implements the HRIConnector interface for human-robot audio interaction
  - Supports both synchronous and asynchronous audio operations

## Key Features

### Message Types

Connectors are generic over message types derived from BaseMessage:

- **BaseMessage**: Foundation message type with payload and metadata
- **ROS2Message**: Message type for ROS 2 communication
- **HRIMessage**: Multimodal message type with text, images, and audio
- **ROS2HRIMessage**: HRIMessage specialized for ROS 2 transport
- **SoundDeviceMessage**: Specialized message for audio operations

### Communication Patterns

Connectors support multiple communication patterns:

1. **Publish/Subscribe**

   - `send_message(message, target, **kwargs)`: Send a message to a target
   - `receive_message(source, timeout_sec, **kwargs)`: Receive a message from a source
   - `register_callback(source, callback, **kwargs)`: Register for asynchronous notifications

2. **Request/Response**

   - `service_call(message, target, timeout_sec, **kwargs)`: Make a synchronous service call

3. **Actions**
   - `start_action(action_data, target, on_feedback, on_done, timeout_sec, **kwargs)`: Start a
     long-running action
   - `terminate_action(action_handle, **kwargs)`: Cancel an ongoing action
   - `create_action(action_name, generate_feedback_callback, **kwargs)`: Create an action server

## Threading Model

Connectors implement thread-safe operations:

- ROS 2 connectors use a dedicated thread with MultiThreadedExecutor
- Callbacks are executed in a ThreadPoolExecutor for concurrent processing
- Proper synchronization for shared resources
- Clean shutdown handling for all resources

## Usage Examples

### Basic ROS 2 Message Publishing

```python
from rai.communication.ros2 import ROS2Connector, ROS2Message

# Create connector
connector = ROS2Connector()

# Create and send a message
message = ROS2Message(
    payload={"data": "Hello, Robot!"},
    metadata={"msg_type": "std_msgs/msg/String"}
)
connector.send_message(
    message=message,
    target="/chatter",
    msg_type="std_msgs/msg/String"
)

# Clean up
connector.shutdown()
```

### Asynchronous ROS 2 Action

```python
from rai.communication.ros2 import ROS2Connector, ROS2Message

# Create connector
connector = ROS2Connector()

# Define callbacks
def on_feedback(feedback):
    print(f"Action progress: {feedback}")

def on_done(result):
    print(f"Action completed: {result}")

# Start navigation action
message = ROS2Message(
    payload={"pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}}}
)
action_handle = connector.start_action(
    action_data=message,
    target="/navigate_to_pose",
    on_feedback=on_feedback,
    on_done=on_done,
    msg_type="nav2_msgs/action/NavigateToPose"
)

# Later, cancel the action if needed
connector.terminate_action(action_handle)

# Clean up
connector.shutdown()
```

## Error Handling

Connectors implement robust error handling:

- All operations have appropriate timeout parameters
- Exceptions are properly propagated and documented
- Callbacks are executed in a protected manner to prevent crashes
- Resources are properly cleaned up during shutdown
