# Runners

There are two ways to manage the agents in RAI:

1. `AgentRunner` - a class for starting and stopping the agents
2. `wait_for_shutdown` - a function for waiting for interruption signals

!!! info "Usage of `Runners` is optional"

    You can start and stop the agents manually for more control over the agents lifecycle.

## Usage

### AgentRunner

```python
from rai.agents import AgentRunner
from rai.agents.ros2 import ROS2StateBasedAgent
from rai.agents import ReActAgent

state_based_agent = ROS2StateBasedAgent()
react_agent = ReActAgent()

runner = AgentRunner([state_based_agent, react_agent])

runner.run_and_wait_for_shutdown() # starts the agents and blocks until the shutdown signal (Ctrl+C or SIGTERM)
```

### wait_for_shutdown

```python
from rai.agents import wait_for_shutdown
from rai.agents.ros2 import ROS2StateBasedAgent
from rai.agents import ReActAgent

state_based_agent = ROS2StateBasedAgent()
react_agent = ReActAgent()

# start the agents manually
state_based_agent.run()
react_agent.run()

# blocks until the shutdown signal (Ctrl+C or SIGTERM)
wait_for_shutdown([state_based_agent, react_agent])
```

## See Also

-   [Agents](../agents/overview.md): For more information on the different types of agents in RAI
-   [Aggregators](../aggregators/overview.md): For more information on the different types of aggregators in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the different types of connectors in RAI
-   [Multimodal messages](../langchain_integration/multimodal_messages.md): For more information on the different types of connectors in RAI
