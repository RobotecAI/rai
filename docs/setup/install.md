# Quick setup guide

Before going further, make sure you have ROS 2 (jazzy or humble) installed and sourced on your system.

!!! tip "Docker images"

    RAI has experimental docker images. See the [docker](setup_docker.md) for
    instructions.

There are two ways to start using RAI:

1. Installing RAI using pip (recommended for end users)

2. Setting up a developer environment using poetry (recommended for developers)

## Installing RAI

??? tip "Virtual environment"

    We recommend installing RAI in a virtual environment (e.g., [virtualenv](https://docs.python.org/3/library/venv.html), [uv](https://docs.astral.sh/uv/), or [poetry](https://python-poetry.org/docs/)) to keep your dependencies organized. Make sure to use the same version of python as the one used for ROS 2 (typically `python3.10` for Humble and `python3.12` for Jazzy).

    If you plan to use ROS 2 commands (`ros2 run` or `ros2 launch`), you'll need to add your virtual environment's Python packages to your `$PYTHONPATH`. This step is only necessary for ROS 2 integration - if you're just running RAI directly with Python, you can skip this step.

    For reference, here's how to set this up when installing RAI from source: [setup_shell.sh](https://github.com/RobotecAI/rai/blob/245efa95cdb83a81294bc28da814962bff84be20/setup_shell.sh#L32)

1.  Install core functionality:

    ```bash
    pip install rai-core
    ```

2.  Initialize the global configuration file:

    ```bash
    rai-config-init
    ```

3.  Optionally install ROS 2 dependencies:

    ```bash
    sudo apt install ros-${ROS_DISTRO}-rai-interfaces
    ```

!!! important "Package availability"

    `rai_perception` and `rai_nomad` are not yet available through pip. If your workflow relies on openset detection or NoMaD integration, please refer to the
    [developer environment instructions](#setting-up-developer-environment) setup.

    `rai_interfaces` is available as `apt` package. However, due to package distribution delays, the latest version may not be immediately available. If you encounter missing imports, please build `rai_interfaces` from [source](https://github.com/RobotecAI/rai_interfaces).

??? tip "RAI modules"

    RAI is a modular framework. You can install only the modules you need.

    | Module | Description | Documentation |
    |--------|-------------|-------------|
    | rai-core | Core functionality | [link](../API_documentation/overview.md) |
    | rai-whoami | Embodiment module | [link](https://github.com/RobotecAI/rai/tree/{{branch}}/src/rai_whoami) |
    | rai-s2s | Speech-to-Speech module | [link](../speech_to_speech/overview.md) |
    | rai-sim | Simulation module | [link](../simulation_and_benchmarking/overview.md) |
    | rai-bench | Benchmarking module | [link](../simulation_and_benchmarking/overview.md) |

??? tip "RAI outside of ROS 2"

    RAI can be used outside of ROS 2. This means that no ROS 2 related features will be available.

    You can still use RAI's core agent framework, tool system, message passing, and integrations such as LangChain, even if ROS 2 is not installed or sourced on your machine. This is useful for:

    - Developing and testing AI logic, tools, and workflows independently of any robotics middleware
    - Running RAI agents in simulation or cloud environments where ROS 2 is not present
    - Using RAI as a generic multimodal agent framework for non-robotic applications

    If you later decide to integrate with ROS 2, you can simply install and source ROS 2, and all ROS 2-specific RAI features (such as connectors, aggregators, and tools) will become available automatically.

## Setting up developer environment

### 1.1 Install poetry

RAI uses [Poetry](https://python-poetry.org/)(2.1+) for python packaging and dependency management.
Install poetry with the following line:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Alternatively, you can opt to do so by following the
[official docs](https://python-poetry.org/docs/#installation).

### 1.2 Clone the repository:

```bash
git clone https://github.com/RobotecAI/rai.git
cd rai
```

### 1.3 Download [rai_interfaces](https://github.com/RobotecAI/rai_interfaces)

```bash
vcs import < ros_deps.repos
```

### 1.4 Create poetry virtual environment and install dependencies:

```bash
poetry install
rosdep install --from-paths src --ignore-src -r -y
```

!!! tip "Additional dependencies"

    RAI is modular. If you want to use features such as
    speech-to-speech, simulation and benchmarking suite, openset detection, or NoMaD integration,
    install additional dependencies:

    ```bash
    poetry install --with perception,nomad,s2s,simbench # or `--all-groups` for full setup
    ```

    | Group Name | Description | Dependencies |
    |------------|-------------|--------------|
    | [s2s][s2s] | Speech-to-Speech functionality | rai_asr, rai_tts |
    | [simbench][simbench] | Simulation and benchmarking tools | rai_sim, rai_bench |
    | [perception][perception] | Open-set detection capabilities | groundingdino, groundedsam |
    | [nomad][nomad] | Visual Navigation - NoMaD integration | visualnav_transformer |
    | docs | Documentation-related dependencies | mkdocs, mkdocs-material, pymdown-extensions |

### 1.5 Configure RAI

Run the configuration tool to set up your LLM vendor and other settings:

```bash
poetry run streamlit run src/rai_core/rai/frontend/configurator.py
```

!!! tip "Web browser"

    If the web browser does not open automatically, open the URL displayed in the terminal manually.

## 2. Build the project:

### 2.1 Build RAI workspace

```bash
colcon build --symlink-install
```

### 2.2 Activate the virtual environment:

```bash
source ./setup_shell.sh
```

## 3. Setting up vendors

RAI is vendor-agnostic. Use the configuration in `config.toml` to set up your vendor
of choice for RAI modules. Vendor choices for RAI and our recommendations are summarized in
[Vendors Overview](vendors.md).

!!! tip "Best-performing AI models"

    We strongly recommend you to use of best-performing AI models to get the most out of RAI!

Pick your local solution or service provider and follow one of these guides:

-   **[Ollama](https://ollama.com/download)**
-   **[OpenAI](https://platform.openai.com/docs/quickstart)**
-   **[AWS Bedrock](https://console.aws.amazon.com/bedrock/home?#/overview)**

[s2s]: ../tutorials/voice_interface.md
[simbench]: ../simulation_and_benchmarking/overview.md
[perception]: ../extensions/perception.md
[nomad]: ../extensions/nomad.md
