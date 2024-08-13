# RAI

> [!IMPORTANT]  
> **RAI is currently a work in progress. We are consistently developing the framework, aiming for stabilization in time for ROSCon 2024.**

Welcome to the RAI Framework repository! We are dedicated to advancing robotics by integrating Generative AI to enable intelligent task fulfillment and enhance conventional algorithms.

## Overview

The RAI framework aims to:

- Advance robotics through the integration of GenAI.
- Enable intelligent task fulfillment.
- Enhance conventional algorithms.
- Develop a sophisticated multiagent system.
- Incorporate an advanced database for persistent agent memory.
- Create sophisticated ROS 2-oriented tooling for agents.
- Build a comprehensive task/mission orchestrator.

# Table of Contents

- [Quick Start](#installation)
- [Usage examples (demos)](#planned-demos)
- [Available vendors](#available-llm-vendors)
- [Documentation](#scenario-definition)
- [Integration with Robotic Systems](#integration-with-robotic-systems)
- [Further documentation](#further-documentation)

# Quick Start

## Prerequisites

- python3.10 or python3.12
- poetry `>=1.8.0`
- ROS 2 humble or ROS 2 jazzy

### 0. Packages installation:

- Install `poetry >= 1.8.0` by following the official [docs](https://python-poetry.org/docs/#installation)

- Remember to add `poetry` to your `PATH`.

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 1. Clone the repository:

```bash
git clone git@github.com:RobotecAI/rai-private.git
cd rai-private
```

### 2. Create poetry virtual environment and install dependencies:

```bash
poetry install
rosdep install --from-paths src --ignore-src -r -y
```

### 2. Build project:

#### 2.1 Download demos (Optional)

See [docs/demos.md](docs/demos.md)

#### 2.2 Build ros project

```bash
. /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install
```

> [!NOTE]
> symlink install allows the IDEs to properly resolve python definitions

#### 2.3 Activate a virtual environment:

```bash
. /opt/ros/${ROS_DISTRO}/setup.bash
. ./install/setup.bash
poetry shell
source /opt/ros/${ROS_DISTRO}/setup.bash
```

### 3. Setting up vendors

While RAI strives to be fully vendor-agnostic, most of the development work currently utilizes OpenAI models. Setting the `OPENAI_API_KEY` environment variable will yield the best results.

#### OpenAI

If you do not have a key, see how to generate one [here](https://platform.openai.com/docs/quickstart).

```
export OPENAI_API_KEY=""
```

## Installation verification (optional)

### 1. Set up vendor keys

### 2. Run pytest

```bash
pytest -m billable
```

> [!WARNING]
> Running the tests will trigger paid api calls.

# Planned demos

- [agriculture demo 🌾](https://github.com/RobotecAI/rai-agriculture-demo)
- [husarion demo 🤖](https://github.com/RobotecAI/rai-husarion-demo)
- [manipulation demo 🦾](https://github.com/RobotecAI/rai-manipulation-demo)

# Further documentation

For examples see [examples](./examples/README.md)\
For Message definition: [messages.md](docs/messages.md)\
For Scenario definition: [scenarios.md](docs/scenarios.md)\
For available ROS2 packages: [ros-packages.md](docs/ros-packages.md)\

For more information see readmes in respective folders.
