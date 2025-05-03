# ROS2 Aggregators

RAI provides several specialized aggregators for ROS2 messages:

| Aggregator                        | Description                                                                      | Example Usage                                                  |
| --------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `ROS2LogsAggregator`              | Processes ROS2 log messages, removing duplicates while maintaining order         | `aggregator = ROS2LogsAggregator()`                            |
| `ROS2GetLastImageAggregator`      | Returns the most recent image from the buffer as a base64-encoded string         | `aggregator = ROS2GetLastImageAggregator()`                    |
| `ROS2ImgVLMDescriptionAggregator` | Uses a Vision Language Model to analyze and describe the most recent image       | `aggregator = ROS2ImgVLMDescriptionAggregator(llm=chat_model)` |
| `ROS2ImgVLMDiffAggregator`        | Compares multiple images (first, middle, and last) to identify changes over time | `aggregator = ROS2ImgVLMDiffAggregator(llm=chat_model)`        |

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

## Direct Registration via ROS2Connector

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

## See Also

-   [Aggregators Overview](../overview.md): For more information on the base aggregator class
-   [Agents](../agents/overview.md): For more information on the different types of agents in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the different types of connectors in RAI
-   [Multimodal messages](../langchain_integration/multimodal_messages.md): For more information on the different types of connectors in RAI
-   [Runners](../runners/overview.md): For more information on the different types of runners in RAI
