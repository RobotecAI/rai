# Aggregators

## Overview

Aggregators in RAI are components that collect and process messages from various sources, transforming them into summarized or analyzed information. They are particularly useful in state-based agents where they help maintain and update the agent's state through periodic aggregation.

## BaseAggregator

`BaseAggregator` is the abstract base class for all aggregator implementations in the RAI framework. It provides a generic interface for collecting and processing messages of a specific type.

### Class Definition

```python
class BaseAggregator(ABC, Generic[T]):
    def __init__(self, max_size: int | None = None) -> None:
        self._buffer: Deque[T] = deque()
        self.max_size = max_size

    def __call__(self, msg: T) -> None:
        if self.max_size is not None and len(self._buffer) >= self.max_size:
            self._buffer.popleft()
        self._buffer.append(msg)

    @abstractmethod
    def get(self) -> BaseMessage | None:
        """Returns the aggregated message"""
        pass

    def clear_buffer(self) -> None:
        self._buffer.clear()

    def get_buffer(self) -> List[T]:
        return list(self._buffer)
```

### Purpose

The `BaseAggregator` class serves as the foundation for message aggregation in RAI, providing:

-   A buffer for collecting messages
-   Size management to prevent memory overflow
-   A consistent interface for processing and returning aggregated results
-   Type safety through generics

### Core Methods

#### `__init__(max_size: int | None = None)`

Initializes a new aggregator with an optional maximum buffer size.

#### `__call__(msg: T)`

Adds a new message to the buffer. If max_size is set and exceeded, removes the oldest message.

#### `get() -> BaseMessage | None`

Abstract method that must be implemented by subclasses. Processes the buffered messages and returns a summary or analysis.

#### `clear_buffer()`

Clears all messages from the buffer.

#### `get_buffer() -> List[T]`

Returns a copy of the current buffer contents.

## Usage in State-Based Agents

Aggregators are typically used in state-based agents to maintain and update the agent's state:

```python
config = StateBasedConfig(
    aggregators={
        ("/camera/camera/color/image_raw", "sensor_msgs/msg/Image"): [
            ROS2ImgVLMDiffAggregator()
        ],
        "/rosout": [
            ROS2LogsAggregator()
        ]
    }
)

agent = ROS2StateBasedAgent(
    config=config,
    target_connectors={"to_human": hri_connector},
    tools=tools
)
```

## Direct Registration via Connector

Aggregators can also be registered directly with a connector using the `register_callback` method. This allows for more flexible message processing outside of state-based agents:

```python
# Create a connector
connector = ROS2Connector()

# Create an aggregator
image_aggregator = ROS2GetLastImageAggregator()

# Register the aggregator as a callback for a specific topic
connector.register_callback(
    topic="/camera/camera/color/image_raw",
    msg_type="sensor_msgs/msg/Image",
    callback=image_aggregator
)

# The aggregator will now process all messages received on the topic
# You can retrieve the aggregated result at any time
aggregated_message = image_aggregator.get()
```

This approach is useful when you need to:

-   Process messages from specific topics independently
-   Combine multiple aggregators for the same topic
-   Use aggregators in non-state-based agents
-   Have more control over when aggregation occurs

## Best Practices

1. **Buffer Management**: Set appropriate max_size to prevent memory issues
2. **Resource Cleanup**: Clear buffers when no longer needed
3. **Error Handling**: Handle empty buffers and processing errors gracefully
4. **Type Safety**: Use appropriate generic types for message types
5. **Performance**: Consider the computational cost of aggregation operations

## Implementation Example

```python
class CustomAggregator(BaseAggregator[CustomMessage]):
    def get(self) -> HumanMessage | None:
        msgs = self.get_buffer()
        if not msgs:
            return None

        # Process messages
        summary = process_messages(msgs)

        # Clear buffer after processing
        self.clear_buffer()

        return HumanMessage(content=summary)
```

## See Also

-   [ROS 2 Aggregators](ROS_2_Aggregators.md): For more information on the different types of ROS 2 aggregators in RAI
-   [Agents](../agents/overview.md): For more information on the different types of agents in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the different types of connectors in RAI
-   [Multimodal messages](../langchain_integration/multimodal_messages.md): For more information on the different types of connectors in RAI
-   [Runners](../runners/overview.md): For more information on the different types of runners in RAI
