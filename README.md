# RAI

> [!IMPORTANT]  
> **RAI is meant for R&D. Make sure to understand its limitations.**

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

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/GZGfejUSjt)](https://discord.gg/GZGfejUSjt)

</div>

---

## Overview

The RAI framework aims to:

- Supply a general multi-agent system, bringing Gen AI features to your robots.
- Add human interactivity, flexibility in problem-solving, and out-of-box AI features to existing robot stacks.
- Provide first-class support for multi-modalities, enabling interaction with various data types.

## Limitations

- Limitations of LLMs and VLMs in use apply: poor spatial reasoning, hallucinations, jailbreaks, latencies, costs, ...
- Resource use (memory, CPU) is not addressed yet.​
- Requires connectivity and / or an edge platform.​

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage examples (demos)](#simulation-demos)
- [Developer resources](#developer-resources)

## Features

- [x] Voice interaction (both ways).
- [x] Customizable robot identity, including constitution (ethical code) and documentation (understanding own capabilities).
- [x] Accessing camera ("What do you see?"), utilizing VLMs.
- [x] Summarizing own state through ROS logs.
- [x] ROS 2 action calling and other interfaces. The Agent can dynamically list interfaces, check their message type, and publish.
- [x] Integration with LangChain to abstract vendors and access convenient AI tools.
- [x] Tasks in natural language to nav2 goals.
- [x] [NoMaD](https://general-navigation-models.github.io/nomad/) integration.
- [x] Tracing.
- [x] Grounded SAM 2 integration.
- [x] Improved Human-Robot Interaction with voice and text.
- [x] Additional tooling such as GroundingDino.
- [x] Support for at least 3 different AI vendors.
- [ ] SDK for RAI developers.
- [ ] UI for configuration to select features and tools relevant for your deployment.

## Setup

Before going further, make sure you have ROS 2 (Jazzy or Humble) installed and sourced on your system.

### 1. Setting up the workspace:

#### 1.1 Install poetry

RAI uses [Poetry](https://python-poetry.org/) for python packaging and dependency management. Install poetry (1.8+) with the following line:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Alternatively, you can opt to do so by following the [official docs](https://python-poetry.org/docs/#installation).

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
> If you want to use features such as Grounded SAM 2 or NoMaD install additional dependencies:
>
> ```bash
> poetry install --with openset,nomad
> ```

#### 1.4 Configure RAI

Run the configuration tool to set up your vendor and other settings:

```bash
poetry shell
streamlit run src/rai/rai/utils/configurator.py
```

> [!TIP]  
> If the web browser does not open automatically, open the URL displayed in the terminal manually.

### 2. Build the project:

#### 2.1 Build RAI workspace

```bash
colcon build --symlink-install
```

#### 2.2 Activate a virtual environment:

```bash
source ./setup_shell.sh
```

### 3. Setting up vendors

RAI is vendor-agnostic. Use the configuration in [config.toml](./config.toml) to set up your vendor of choice for RAI modules.
Vendor choices for RAI and our recommendations are summarized in [Vendors Overview](docs/vendors_overview.md).

> We strongly recommend you to use of best-performing AI models to get the most out of RAI!

Pick your local solution or service provider and follow one of these guides:

- **[Ollama](https://ollama.com/download)**
- **[OpenAI](https://platform.openai.com/docs/quickstart)**
- **[AWS Bedrock](https://console.aws.amazon.com/bedrock/home?#/overview)**

## What's next?

Once you know your way around RAI, try the following challenges, with the aid the [developer guide](docs/developer_guide.md):

- Run RAI on your own robot and talk to it, asking questions about what is in its documentation (and others!).
- Implement additional tools and use them in your interaction.
- Try a complex, multi-step task for your robot, such as going to several points to perform observations!

### Simulation demos

Try RAI yourself with these demos:
| Application | Robot | Description | Demo Link | Docs Link |
| ------------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- | -------------------------------- |
| Mission and obstacle reasoning in orchards | Autonomous tractor | In a beautiful scene of a virtual orchard, RAI goes beyond obstacle detection to analyze best course of action for a given unexpected situation. | [🌾 demo](https://github.com/RobotecAI/rai-rosbot-xl-demo) | [📚](docs/demos/agriculture.md) |
| Manipulation tasks with natural language | Robot Arm (Franka Panda) | Complete flexible manipulation tasks thanks to RAI and Grounded SAM 2 | [🦾 demo](https://github.com/RobotecAI/rai-manipulation-demo) | [📚](docs/demos/manipulation.md) |
| Autonomous mobile robot demo | Husarion ROSbot XL | Demonstrate RAI's interaction with an autonomous mobile robot platform for navigation and control | [🤖 demo](https://github.com/RobotecAI/rai-rosbot-xl-demo) | [📚](docs/demos/rosbot_xl.md) |
| Turtlebot demo | Turtlebot | Showcase RAI's capabilities with the popular Turtlebot platform | [🐢 demo](docs/demos/turtlebot.md) | [📚](docs/demos/turtlebot.md) |
| Speech-to-speech interaction with autonomous taxi | Simulated car | Demonstrate RAI's speech-to-speech interaction capabilities for specifying destinations to an autonomous taxi in awsim with autoware environment | [🚕 demo](docs/demos/taxi.md) | [📚](docs/demos/taxi.md) |

## Community

### Embodied AI Community Group

RAI is one of the main projects in focus of the [Embodied AI Community Group](https://github.com/ros-wg-embodied-ai). If you would like to join the next meeting, look for it in the [ROS Community Calendar](https://calendar.google.com/calendar/u/0/embed?src=c_3fc5c4d6ece9d80d49f136c1dcd54d7f44e1acefdbe87228c92ff268e85e2ea0@group.calendar.google.com&ctz=Etc/UTC).

### Publicity

- A talk about [RAI at ROSCon 2024](https://vimeo.com/1026029511).

### RAI Q&A

Please take a look at [Q&A](https://github.com/RobotecAI/rai/discussions/categories/q-a).

### Developer Resources

See our [Developer Guide](docs/developer_guide.md) for a deeper dive into RAI, including instructions on creating a configuration specifically for your robot.

### Contributing

You are welcome to contribute to RAI! Please see our [Contribution Guide](CONTRIBUTING.md).
