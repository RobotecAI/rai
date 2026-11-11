# FAQ

## General

??? question "What do I need to run RAI?"

    Ubuntu 26.04 with ROS 2 Lyrical, or Ubuntu 24.04 with ROS 2 Jazzy, on amd64. A model: a cloud provider's API key, or a local model through Ollama or an OpenAI-compatible server. See [Install](../getting_started/install.md).

??? question "Do I need a robot?"

    No. The [demos](../demos/overview.md) run in simulation, [replay](../guides/replay.md) runs agents on recordings, and agents, tools and settings work without ROS 2 at all. With a robot, RAI builds on its existing ROS 2 stack: it does not include drivers, navigation or control.

??? question "Can I use RAI without ROS 2?"

    Yes. `rai.agents`, `rai.tools`, `rai.settings`, `rai.llm` and `rai.content` do not need ROS 2. Importing `rai.communication.ros2` or `rai.tools.ros2` without a sourced ROS 2 raises `ImportError: This is a ROS2 feature`.

??? question "Is RAI limited to LLMs?"

    No. `BaseAgent` is a lifecycle, not an LLM loop: implement `run()` with any model or algorithm (speech recognition in rai_s2s is one). See [Agents and lifecycle](../concepts/agents.md).

??? question "Why is RAI not a ROS 2 package?"

    RAI depends on fast-moving Python packages (LangChain, model SDKs) that need exact versions, which rosdep cannot express, and most of it works without ROS 2. So RAI is a set of Python packages in a uv workspace that work with ROS 2, and only `rai_interfaces` and `rai_bringup` are built with colcon.

??? question "Does RAI cost money?"

    RAI is free and open source under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0), commercial use included. Cloud models and services bill you for what you use; local models do not.

??? question "Is my data sent anywhere?"

    Only to the model providers and tracing services you configure. For full privacy, use local models and leave tracing off, or use the offline [flight recorder](../guides/recorder.md).

## Troubleshooting

??? question "`import rclpy` fails inside the venv"

    The venv was created without access to the system site-packages. Recreate it with `python3 -m venv --system-site-packages .venv` (or `scripts/dev_env.sh` in the repository), using the system Python of your ROS 2 distribution, and source ROS 2 before activating it.

??? question "`Provider 'x' needs langchain_x`"

    Install the provider's extra, for example `pip install "rai-core[anthropic]"`. See the extras in [Install](../getting_started/install.md#2-install-rai_core-with-extras).

??? question "`No module named rai_interfaces`"

    Install `ros-jazzy-rai-interfaces`, or on Lyrical build `rai_interfaces` from source, and source its workspace. See [Install](../getting_started/install.md#3-install-rai_interfaces).

??? question "The first service call or goal to a node that just started is slow or times out"

    DDS discovery takes up to a few seconds. RAI waits for the server to be ready within the call's timeout; give the first call a longer one. For requests that are safe to repeat, `call(..., idempotent=True)` resends unanswered requests.

??? question "The agent does not see my topics"

    Check that both sides use the same `ROS_DOMAIN_ID` and RMW (`RMW_IMPLEMENTATION`), and that the topic is not in the tools' `forbidden` list or missing from `readable`. `list_ros2_nodes` and `get_ros2_topic_info` from the [diagnostics toolkit](../guides/ros2_tools.md#diagnostics) show what the agent's node sees, with QoS incompatibilities.

??? question "Speech recognition or text to speech fails with a PortAudio error"

    Install PortAudio (`sudo apt install portaudio19-dev`) and check the device name with `python -c "import sounddevice; print(sounddevice.query_devices())"`. See [Speech to speech](../speech_to_speech/overview.md).

??? question "A tool call was denied and I do not know why"

    The error tool message names the hook and its reason, and the `ToolEnd` event has status `denied`. Install an `AuditLog` (`rai.safety.install(audit_path="audit.jsonl")`) to log every decision of both hook points. See [Safety](../concepts/safety.md).

## Contributing and support

??? question "How do I contribute?"

    Set up the [developer environment](../getting_started/dev_env.md) and read the [contribution guide](contributing/CONTRIBUTING.md). If you are looking for something to work on, browse the [issues](https://github.com/RobotecAI/rai/issues) or ask on [Discord](https://discord.gg/3PGHgTaJSB).

??? question "Where do I get help?"

    The [Q&A forum](https://github.com/RobotecAI/rai/discussions/categories/q-a), [issues](https://github.com/RobotecAI/rai/issues), [Discord](https://discord.gg/3PGHgTaJSB) and the [Embodied AI Community Group](https://github.com/ros-wg-embodied-ai).
