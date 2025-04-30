# RAI

> [!IMPORTANT]  
> **Development Status**: RAI is currently undergoing significant development on the development
> branch, focusing on version 2.0. This major version update will introduce substantial improvements
> and is not backward compatible with version 1.0. We are targeting an early May 2025 release for
> RAI 2.0. For the latest stable release, please refer to the
> [main branch](https://github.com/RobotecAI/rai/tree/main).

RAI is a flexible AI agent framework to develop and deploy Embodied AI features for your robots.

---

<div align="center">

![rai-image](./docs/imgs/RAI_simple_diagram_medium.png)

---

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Release](https://img.shields.io/github/v/release/RobotecAI/rai)
![Contributors](https://img.shields.io/github/contributors/robotecai/rai)

![Static Badge](https://img.shields.io/badge/Ubuntu-24.04-orange)
![Static Badge](https://img.shields.io/badge/Ubuntu-22.04-orange)
![Static Badge](https://img.shields.io/badge/Python-3.12-blue)
![Static Badge](https://img.shields.io/badge/Python-3.10-blue)
![Static Badge](https://img.shields.io/badge/ROS2-jazzy-blue)
![Static Badge](https://img.shields.io/badge/ROS2-humble-blue)

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/3PGHgTaJSB)](https://discord.gg/3PGHgTaJSB)

</div>

---

## 🎯 Overview

The RAI framework is designed to revolutionize robotics by:

🤖 **Empowering Multi-Agent Systems**

- Seamlessly integrate Gen AI capabilities into your robots
- Enable sophisticated agent-based architectures

🔄 **Enhancing Robot Intelligence**

- Add natural human-robot interaction capabilities
- Bring flexible problem-solving to your existing stack
- Provide ready-to-use AI features out of the box

🌟 **Supporting Multi-Modal Interaction**

- Handle diverse data types natively
- Enable rich sensory integration
- Process multiple input/output modalities simultaneously

## Table of Contents

- [Framework](#rai-framework)
- [Setup](#setup)
- [Usage examples (demos)](#simulation-demos)
- [Communication Protocols](#communication-protocols)
- [Developer resources](#developer-resources)

## RAI framework

- [x] rai core: Core functionality for multi-agent system, human-robot interaction and
      multi-modalities.
- [x] rai whoami: Tool to extract and synthesize robot embodiment information from a structured
      directory of documentation, images, and URDFs.
- [x] rai_asr: Speech-to-text models and tools.
- [x] rai_tts: Text-to-speech models and tools.
- [x] rai_sim: Package for connecting RAI to simulation environments.
- [x] rai_bench: Benchmarking suite for RAI. Test agents, models, tools, simulators, etc.
- [x] rai_openset: Openset detection models and tools.
- [x] rai_nomad: Integration with NoMaD for navigation.
- [ ] rai_finetune: Finetune LLMs on your embodied data.

## Setup

Before going further, make sure you have ROS 2 (Jazzy or Humble) installed and sourced on your
system. If you don't have ROS 2 follow the installation documentation for
[Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) or
[Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html). Make sure that
`ros-dev-tools` are installed.

> [!TIP]  
> RAI has experimental docker images. See the [docker](docs/setup_docker.md) for instructions.

### 1. Setting up the workspace:

#### 1.1 Install poetry

RAI uses [Poetry](https://python-poetry.org/) for python packaging and dependency management.
Install poetry with the following line:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Alternatively, you can opt to do so by following the
[official docs](https://python-poetry.org/docs/#installation).

#### 1.2 Clone the repository:

```bash
git clone https://github.com/RobotecAI/rai.git
cd rai
```

#### 1.3 Create poetry virtual environment and install dependencies:

```bash
poetry install
rosdep install --from-paths src --ignore-src -r -y
```

> [!TIP]  
> RAI is modular. If you want to use features such as speech-to-speech, simulation and benchmarking
> suite, openset detection, or NoMaD integration, install additional dependencies:
>
> ```bash
> poetry install --with openset,nomad,s2s,simbench
> ```

#### 1.4 Configure RAI

Run the configuration tool to set up your LLM vendor and other settings:

```bash
poetry run streamlit run src/rai_core/rai/frontend/configurator.py
```

> [!TIP]  
> If the web browser does not open automatically, open the URL displayed in the terminal manually.

### 2. Build the project:

#### 2.1 Build RAI workspace

```bash
colcon build --symlink-install
```

#### 2.2 Activate the virtual environment:

```bash
source ./setup_shell.sh
```

### 3. Setting up vendors

RAI is vendor-agnostic. Use the configuration in [config.toml](./config.toml) to set up your vendor
of choice for RAI modules. Vendor choices for RAI and our recommendations are summarized in
[Vendors Overview](docs/vendors_overview.md).

> [!TIP]  
> We strongly recommend you to use of best-performing AI models to get the most out of RAI!

Pick your local solution or service provider and follow one of these guides:

- **[Ollama](https://ollama.com/download)**
- **[OpenAI](https://platform.openai.com/docs/quickstart)**
- **[AWS Bedrock](https://console.aws.amazon.com/bedrock/home?#/overview)**

## What's next?

RAI provides a comprehensive set of tools and capabilities for your robotic applications. With the
help of our [developer guide](docs/developer_guide.md), you can:

- Deploy RAI on your robot and engage in natural conversations about its documentation and
  capabilities
- Extend the framework with custom tools and integrate them into your interactions
- Implement complex, multi-step tasks that leverage RAI's reasoning and planning capabilities

## Communication Protocols

RAI provides first-class support for ROS 2 Humble and Jazzy distributions. While ROS 2 serves as our
Tier 1 communication protocol, RAI's architecture includes a powerful abstraction layer that:

- Simplifies communication across different networks and protocols
- Enables seamless integration with various communication backends
- Allows for future protocol extensions while maintaining a consistent interface

This design philosophy means that while RAI is fully compatible with ROS 2, most of its features can
be utilized independently of the ROS 2 environment. The framework's modular architecture makes it
suitable not only for different robotic platforms but also for non-robotic applications, offering
flexibility in deployment across various domains.

### Simulation demos

| Try RAI yourself with these demos: | Application | Robot | Description | Docs Link |     |
| ---------------------------------- | ----------- | ----- | ----------- | --------- | --- |

---

| ------------------------------------------------------------- | | Mission and obstacle reasoning
in orchards | Autonomous tractor | In a beautiful scene of a virtual orchard, RAI goes beyond
obstacle detection to analyze best course of action for a given unexpected situation. |
[link](docs/demos/agriculture.md) | | Manipulation tasks with natural language | Robot Arm (Franka
Panda) | Complete flexible manipulation tasks thanks to RAI and Grounded SAM 2 |
[link](docs/demos/manipulation.md) | | Autonomous mobile robot demo | Husarion ROSbot XL |
Demonstrate RAI's interaction with an autonomous mobile robot platform for navigation and control |
[link](docs/demos/rosbot_xl.md) | | Turtlebot demo | Turtlebot | Showcase RAI's capabilities with
the popular Turtlebot platform | [link](docs/demos/turtlebot.md) | | Speech-to-speech interaction
with autonomous taxi | Simulated car | Demonstrate RAI's speech-to-speech interaction capabilities
for specifying destinations to an autonomous taxi in awsim with autoware environment |
[link](docs/demos/taxi.md) |

## Community

### Embodied AI Community Group

RAI is one of the main projects in focus of the
[Embodied AI Community Group](https://github.com/ros-wg-embodied-ai). If you would like to join the
next meeting, look for it in the
[ROS Community Calendar](https://calendar.google.com/calendar/u/0/embed?src=c_3fc5c4d6ece9d80d49f136c1dcd54d7f44e1acefdbe87228c92ff268e85e2ea0@group.calendar.google.com&ctz=Etc/UTC).

### Publicity

- A talk about [RAI at ROSCon 2024](https://vimeo.com/1026029511).

### RAI Q&A

Please take a look at [Q&A](https://github.com/RobotecAI/rai/discussions/categories/q-a).

### Developer Resources

See our [Developer Guide](docs/developer_guide.md) for a deeper dive into RAI, including
instructions on creating a configuration specifically for your robot.

### Contributing

You are welcome to contribute to RAI! Please see our [Contribution Guide](CONTRIBUTING.md).
