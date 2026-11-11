# rai_core

The core of [RAI](https://github.com/RobotecAI/rai), an agent framework for robots: LLM agents with a lifecycle and an event stream, tools with effects and fail-closed safety hooks, async ROS 2 connectors, an MCP server and client, a robot manifest, and an MCAP flight recorder with replay.

Documentation: [robotecai.github.io/rai](https://robotecai.github.io/rai/)

## Install

ROS 2 Lyrical (Python 3.14) or Jazzy (Python 3.12) is optional; the ROS 2 modules need it sourced and a venv that sees the system site-packages:

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install "rai-core[openai]"
```

Extras: `openai`, `anthropic`, `aws`, `google`, `ollama` (model providers), `langfuse` (tracing), `mcp`, `ui` (Streamlit), `policies` (VLA policy clients), `recorder` (MCAP), or `all`.

## Configure

Models are `"provider:model"` strings in an optional `rai.toml` (`./rai.toml` or `$RAI_CONFIG`), overridden by `RAI_*` environment variables such as `RAI_MODELS__COMPLEX=ollama:llama3.1:70b`. API keys stay in each provider's own variable (`OPENAI_API_KEY`, ...).

```toml
[models]
simple = "openai:gpt-4o-mini"
complex = "openai:gpt-4o"
embeddings = "openai:text-embedding-3-small"
```

## Use

```python
import asyncio

from rai.agents import ReActAgent
from rai.tools import Effect, rai_tool


@rai_tool(Effect.READ)
async def battery_level() -> str:
    """The robot's battery charge."""
    return "87%"


agent = ReActAgent("complex", [battery_level])
print(asyncio.run(agent.ainvoke("How much battery is left?")).text)
```

See [Your first agent](https://robotecai.github.io/rai/getting_started/first_agent/) for ROS 2 tools, HRI topics and the e-stop.

## License

Apache 2.0. Contributions: see the [contribution guide](https://github.com/RobotecAI/rai/blob/main/CONTRIBUTING.md).
