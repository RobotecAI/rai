# Agents

## Overview

Agents in RAI are modular components that encapsulate specific functionalities and behaviors. They follow a consistent interface defined by the `BaseAgent` class and can be combined to create complex robotic systems.

## BaseAgent

`BaseAgent` is the abstract base class for all agent implementations in the RAI framework. It defines the minimal interface that all agents must implement while providing common functionality like logging.

### Class Definition

??? info "BaseAgent class definition"

    ::: rai.agents.base.BaseAgent

### Purpose

The `BaseAgent` class serves as the cornerstone of RAI's agent architecture, establishing a uniform interface for various agent implementations. This enables:

-   Consistent lifecycle management (starting/stopping)
-   Standardized logging mechanisms
-   Interoperability between different agent types
-   Integration with management utilities like `AgentRunner`

## Best Practices

1. **Resource Management**: Always clean up resources in the `stop()` method
2. **Thread Safety**: Use locks for shared resources when implementing multi-threaded agents
3. **Error Handling**: Implement proper exception handling in long-running agent threads
4. **Logging**: Use the provided `self.logger` for consistent logging
5. **Graceful Shutdown**: Handle interruptions and cleanup properly

## Architecture

In the RAI framework, agents typically interact with:

-   **Connectors**: For communication (ROS2, audio devices, etc.)
-   **Aggregators**: For processing and summarizing input data
-   **Models**: For AI capabilities (LLMs, vision models, speech recognition)
-   **Tools**: For implementing specific actions an agent can take

## See Also

-   [Aggregators](../aggregators/overview.md): For more information on the different types of aggregators in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the different types of connectors in RAI
-   [Multimodal messages](../langchain_integration/multimodal_messages.md): For more information on the different types of connectors in RAI
-   [Runners](../runners/overview.md): For more information on the different types of runners in RAI
