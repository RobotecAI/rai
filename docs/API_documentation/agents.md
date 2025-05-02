# Agents

## Overview

Agents in RAI are modular components that encapsulate specific functionalities and behaviors. They follow a consistent interface defined by the `BaseAgent` class and can be combined to create complex robotic systems.

## BaseAgent

`BaseAgent` is the abstract base class for all agent implementations in the RAI framework. It defines the minimal interface that all agents must implement while providing common functionality like logging.

### Class Definition

```python
class BaseAgent(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass
```

### Purpose

The `BaseAgent` class serves as the cornerstone of RAI's agent architecture, establishing a uniform interface for various agent implementations. This enables:

-   Consistent lifecycle management (starting/stopping)
-   Standardized logging mechanisms
-   Interoperability between different agent types
-   Integration with management utilities like `AgentRunner`

### Core Methods

#### `__init__()`

Initializes a new agent instance and sets up logging with the class name.

#### `run()`

Abstract method that must be implemented by all subclasses. Starts the agent's main execution loop. In some cases, concrete run implementation may not be needed. In that case use `pass` as a placeholder.

#### `stop()`

Abstract method that must be implemented by all subclasses. Gracefully terminates the agent's execution and cleans up resources.

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

-   [AgentRunner](runner.md): For managing multiple agents
-   [LangChainAgent](langchain.md): For LLM-powered agents
-   [StateBasedAgent](state_based.md): For agents that maintain state through periodic aggregation
