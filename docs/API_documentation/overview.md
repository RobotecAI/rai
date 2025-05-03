# RAI API Documentation

## Introduction

rai core provides a comprehensive set of tools and components for developing intelligent robotic systems powered by multimodal LLMs. This package bridges the gap between advanced AI capabilities and robotic platforms, enabling natural language understanding, reasoning, and multimodal interactions in robotic applications.

## Core Components

<div style="text-align: center;"><img src="../../imgs/rai_api.png" alt="rai-api"></div>

RAI consists of several key components that work together to create intelligent robotic systems:

### [Agents](agents/overview.md)

Agents are the central components that encapsulate specific functionalities and behaviors.

### [Connectors](connectors/overview.md)

Connectors provide a unified way to interact with various communication systems e.g., ROS 2.

### [Aggregators](aggregators/overview.md)

Aggregators collect and process messages from various sources, transforming them into summarized or analyzed information.

### [LangChain Integration](langchain_integration/overview.md)

RAI leverages LangChain to bridge the gap between large language models and robotic systems:

-   Standardized interfaces across different LLM providers
-   Rich tool ecosystem for complex tasks
-   Enhanced agent capabilities tailored for robotics

### [Multimodal Messages](langchain_integration/multimodal_messages.md)

Enables image support in LangChain messages.

### [Runners](runners/overview.md)

Manages the lifecycle of agents.

## Getting Started

For practical examples and tutorials on using RAI, refer to the tutorials section. The API documentation provides detailed information about each component, its purpose, and usage patterns.

## Best Practices

When working with RAI:

1. Design agents with clear responsibilities and interfaces
2. Use appropriate connectors for your target platforms
3. Leverage aggregators to process complex sensor data
4. Follow established patterns for tool development
5. Consider performance implications for real-time robotic applications
