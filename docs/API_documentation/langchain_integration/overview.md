# Langchain Integration

## Overview

RAI integrates with Langchain to enable natural language, reasoning, and multimodal capabilities in robotic applications. This API documentation describes how Langchain is used and extended within RAI, and how to leverage it for agent and tool development.

## Key Concepts

-   **Standardized Interfaces**: Consistent APIs for LLMs and tools.
-   **Tool Ecosystem**: Use and extend tools for text and multimodal operations.
-   **Agent-Based Reasoning**: Build agents that combine LLMs, tools, and context/memory.
-   **Multimodal Communication**: Support for images, audio, and sensor data within messages.
-   **Robotic Integration**: ROS 2 communication and robotic toolsets.

## Getting Started

```python
from rai import get_llm_model

# Initialize LLM configured in config.toml
llm = get_llm_model(model_type='complex_model')
```

## Agent and Tool Patterns

### Agent-Based Systems

```python
from rai.agents.langchain.runnables import create_react_runnable

agent = create_react_runnable(
    llm=llm,
    tools=[ros2_topic, get_image]
)
agent.invoke({"messages": [HumanMessage(content="Analyze this image")]})
```

### Multimodal Communication

```python
from rai.messages import HumanMultimodalMessage, preprocess_image

message = HumanMultimodalMessage(
    content="Analyze this image",
    images=[preprocess_image(image_uri)]
)
```

### Tool Integration

```python
from langchain_core.tools import tool

@tool
def custom_operation(input: str) -> str:
    # Tool implementation
    return result
```

#### Multimodal Tool Example

```python
from langchain_core.tools import tool
from rai.messages import MultimodalArtifact

@tool(response_format="content_and_artifact")
def custom_operation(input: str) -> str:
    # Tool implementation
    return result, MultimodalArtifact(images=[base64_encoded_png_image])
```

## Best Practices

-   Use appropriate message types for text and media
-   Follow Langchain tool patterns and document capabilities
-   Keep agents focused and specialized

## See Also

-   [Tool tutorial](../../tutorials/tools.md): For more information on how to create custom LangChain tools
-   [Agents](../agents/overview.md)
-   [Aggregators](../aggregators/overview.md)
-   [Connectors](../connectors/overview.md)
-   [Multimodal messages](./multimodal_messages.md)
-   [Runners](../runners/overview.md)
