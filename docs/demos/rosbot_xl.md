# Husarion Robot XL demo

This demo utilizes Open 3D Engine simulation and allows you to work with RAI on a small mobile
platform in a nice apartment.

![Screenshot1](../imgs/o3deSimulation.png)

## Quick start

> [!TIP] LLM model
>
> The demo uses the `complex_model` LLM configured in `config.toml`. The model should be a multimodal, tool-calling model. See [Vendors](../setup/vendors.md#llm-model-configuration-in-rai).

!!! tip "ROS 2 Sourced"

    Make sure ROS 2 is sourced. (e.g. `source /opt/ros/humble/setup.bash`)

1. Download the newest binary release:

    ```bash
    ./scripts/download_demo.sh rosbot
    ```

2. Install and download required packages

    ```bash
    sudo apt install ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-nav2-bringup
    vcs import < demos.repos
    rosdep install --from-paths src --ignore-src -r -y
    poetry install --with openset
    ```

!!! tip "Alternative: Demo source build"

    If you would like more freedom to adapt the simulation to your needs, you can make changes using [O3DE Editor](https://www.docs.o3de.org/docs/welcome-guide/) and build the binary yourself. Please refer to [rai husarion rosbot xl demo](https://github.com/RobotecAI/rai-rosbot-xl-demo) for more details.

## Running RAI

1. Running rai nodes and agents, navigation stack and O3DE simulation.

    ```bash
    ros2 launch ./examples/rosbot-xl.launch.py game_launcher:=demo_assets/rosbot/RAIROSBotXLDemo/RAIROSBotXLDemo.GameLauncher
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
    - Please bring me coffee something from the kitchen (_this one should be rejected thanks to robot embodiment module_)

!!! tip "Changing camera view"

    To change camera in the simulation use 1,2,3 keys on your keyboard.

## How it works

The rosbot demo utilizes several components:

1. Vision processing using Grounded SAM 2 and Grounding DINO for object detection and segmentation. See [RAI OpenSet Vision](../extensions/openset.md).
2. RAI agent to process the request and interact with environment via [tool-calling](https://python.langchain.com/docs/concepts/tool_calling/) mechanism.
3. Navigation is enabled via [nav2 toolkit](../API_documentation/langchain_integration/ROS_2_tools.md#nav2), which interacts with [ROS 2 nav2](https://docs.nav2.org/) asynchronously by calling [ros2 actions](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Actions/Understanding-ROS2-Actions.html).
4. Embodiment of the Rosbot is achieved using [RAI Whoami](../tutorials/create_robots_whoami.md) module. This makes RAI agent aware of the hardware platform and its capabilities.

The details of the demo can be checked in `examples/rosbot-xl-demo.py`.
