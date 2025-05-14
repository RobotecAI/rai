# Husarion Robot XL demo

This demo utilizes Open 3D Engine simulation and allows you to work with RAI on a small mobile
platform in a nice apartment.

![Screenshot1](../imgs/o3deSimulation.png)

## Quick start

> [!TIP] LLM model
>
> The demo uses the `complex_model` LLM configured in `config.toml`. The model should be a
> multimodal, tool-calling model. See [Vendors](../setup/vendors.md#llm-model-configuration-in-rai).

!!! tip "ROS 2 Sourced"

    Make sure ROS 2 is sourced.

1. Download the newest binary release:

    ```bash
    ./scripts/download_demo.sh rosbot
    ```

2. Install and download required packages

    ```bash
    vcs import < demos.repos
    rosdep install --from-paths src --ignore-src -r -y
    poetry install --with openset
    ```

## Alternative: Demo source build

If you would like more freedom to adapt the simulation to your needs, you can make changes using [O3DE Editor](https://www.docs.o3de.org/docs/welcome-guide/) and build the project yourself. Please refer to [rai husarion rosbot xl demo](https://github.com/RobotecAI/rai-rosbot-xl-demo) for more details.

# Running RAI

1. Running rai nodes and agents, navigation stack and O3DE simulation.

    ```bash
    ros2 launch ./examples/rosbot-xl.launch.py game_launcher:=demo_assets/rosbot/RAIROSBotXLDemoGamePackage/RAIROSBotXLDemo.GameLauncher
    ```

2. Run streamlit gui:

    ```bash
    streamlit run examples/rosbot-xl-demo.py
    ```

3. Play with the demo, prompting the agent to perform tasks. Here are some examples:

    - Where are you now?
    - What do you see?
    - What is the position of bed?
    - Navigate to the kitchen.

!!! tip "Building from source"

    If you are having trouble running the binary, you can build it from source
    [here](https://github.com/RobotecAI/rai-rosbot-xl-demo).
