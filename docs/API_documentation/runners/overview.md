# Runners

`rai.run(*agents)` starts the agents, waits for SIGINT, SIGTERM or every agent finishing, then stops them in reverse order. `ThreadedAgent` runs one agent on its own thread and event loop, for code that is not async.

## run

::: rai.agents.runner.run

## ThreadedAgent

::: rai.agents.runner.ThreadedAgent

## Usage

```python
import rai
from rai.agents import ReActAgent

rai.run(ReActAgent())  # blocks until Ctrl+C or SIGTERM
```

## See Also

-   [Agents](../agents/overview.md): For more information on the different types of agents in RAI
-   [Aggregators](../aggregators/overview.md): For more information on the different types of aggregators in RAI
-   [Connectors](../connectors/overview.md): For more information on the different types of connectors in RAI
-   [Langchain Integration](../langchain_integration/overview.md): For more information on the LangChain integration within RAI
-   [Multimodal messages](../langchain_integration/multimodal_messages.md): For more information on the multimodal LangChain messages in RAI
