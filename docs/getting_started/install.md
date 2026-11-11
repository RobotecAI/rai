# Install

This page installs RAI as a library from PyPI. To work on RAI itself, set up the [developer environment](dev_env.md) instead. To skip the setup, use the [Docker images](docker.md).

## Requirements

| ROS 2 distribution | Ubuntu | Python |
| ------------------ | ------ | ------ |
| Lyrical (primary)  | 26.04  | 3.14   |
| Jazzy              | 24.04  | 3.12   |

RAI is tested on amd64. ROS 2 is optional: agents, tools, settings and content types work without it, and only the ROS 2 modules (`rai.communication.ros2`, `rai.tools.ros2`, ...) need a sourced ROS 2.

## 1. Create a virtual environment

rclpy and the other ROS 2 Python packages are built for the system interpreter and live in its site-packages. Create the venv with that interpreter and let it see them:

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
python3 -m venv --system-site-packages .venv   # or: uv venv --system-site-packages --python /usr/bin/python3
source .venv/bin/activate
```

No `PYTHONPATH` changes are needed, and `ros2 run` / `ros2 launch` work from the activated venv.

## 2. Install rai_core with extras

```bash
pip install "rai-core[openai]"
```

The base install has no model provider. Pick the extras you need:

| Extra       | Adds                                                                              |
| ----------- | --------------------------------------------------------------------------------- |
| `openai`    | OpenAI and OpenAI-compatible servers (vLLM, Ollama's OpenAI endpoint, ...)        |
| `anthropic` | Anthropic models                                                                  |
| `aws`       | AWS Bedrock                                                                       |
| `google`    | Google Gemini                                                                     |
| `ollama`    | Ollama                                                                            |
| `langfuse`  | Langfuse tracing (LangSmith tracing needs no extra)                               |
| `mcp`       | [MCP](../guides/mcp.md) server (`rai mcp serve`) and client                       |
| `ui`        | [Streamlit chat and configurator](../guides/streamlit.md)                         |
| `policies`  | Clients for [learned-policy servers](../guides/policies.md) (openpi, GR00T, HTTP) |
| `recorder`  | [Flight recorder and replay](../guides/recorder.md) (MCAP)                        |
| `all`       | Everything above                                                                  |

A missing provider extra fails with a clear message, for example `Provider 'anthropic' needs langchain_anthropic. Install it with pip install rai_core[anthropic].`

## 3. Install rai_interfaces

The HRI connector, rai_whoami, rai_perception and the manipulation tools use RAI's ROS 2 messages from [rai_interfaces](https://github.com/RobotecAI/rai_interfaces).

On Jazzy, install the Debian package:

```bash
sudo apt install ros-jazzy-rai-interfaces
```

On Lyrical, or if the package is older than you need, build it from source in a colcon workspace:

```bash
mkdir -p ~/rai_ws/src && cd ~/rai_ws
git clone --branch 0.3.0 https://github.com/RobotecAI/rai_interfaces.git src/rai_interfaces
colcon build --symlink-install
source install/setup.bash
```

## 4. Configure a model and check the install

```bash
export OPENAI_API_KEY="sk-..."
python -c "from rai import get_llm; print(get_llm('complex').invoke('Say hi').text)"
```

The default models are `openai:gpt-4o` (complex) and `openai:gpt-4o-mini` (simple). To use another provider or a local model, see [Configuration](configuration.md).

## Other RAI packages

The other packages of the repository build on rai_core:

| Package          | Purpose                                                      | Docs                                                               |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `rai_s2s`        | Speech recognition, text to speech and voice agents          | [Speech to speech](../speech_to_speech/overview.md)                |
| `rai_whoami`     | Robot identity and documentation from docs, images and URDFs | [Robot identity](../guides/whoami.md)                              |
| `rai-perception` | Open-set detection and segmentation services and tools       | [Perception](../guides/perception.md)                              |
| `rai_semap`      | Semantic map of detected objects                             | [README](https://github.com/RobotecAI/rai/tree/main/src/rai_semap) |
| `rai-sim`        | Connects RAI to simulators                                   | [RAI Sim](../simulation_and_benchmarking/rai_sim.md)               |
| `rai-bench`      | Benchmarks for models, agents and tools                      | [RAI Bench](../simulation_and_benchmarking/rai_bench.md)           |

If a package you need is not on PyPI, install it from the repository with the [developer environment](dev_env.md).
