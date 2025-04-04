# Husarion Robot XL demo

This demo utilizes Open 3D Engine simulation and allows you to work with RAI on a small mobile platform in a nice apartment.

![Screenshot1](../imgs/o3deSimulation.png)

## Quick start

1. Download the newest binary release:

- Ubuntu 22.04 & ros2 humble: [link](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIROSBotDemo_1.0.0_jammyhumble.zip)
- Ubuntu 24.04 & ros2 jazzy: [link](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIROSBotDemo_1.0.0_noblejazzy.zip)

2. Install required packages

   ```bash
   sudo apt install ros-${ROS_DISTRO}-ackermann-msgs ros-${ROS_DISTRO}-gazebo-msgs ros-${ROS_DISTRO}-control-toolbox ros-${ROS_DISTRO}-nav2-bringup
   poetry install --with openset
   ```

3. Unpack the binary with the simulation:

   ```bash
   # u24
   unzip RAIROSBotDemo_1.0.0_noblejazzy.zip
   # u22
   unzip RAIROSBotDemo_1.0.0_jammyhumble.zip
   ```

## Alternative: Demo source build

If you would like more freedom to adapt the simulation to your needs, you can make changes using
[O3DE Editor](https://www.docs.o3de.org/docs/welcome-guide/) and build the project
yourself.
Please refer to [rai husarion rosbot xl demo](https://github.com/RobotecAI/rai-rosbot-xl-demo) for more details.

# Running RAI

1. Running rai nodes and agents, navigation stack and O3DE simulation.

   ```bash
   ros2 launch ./examples/rosbot-xl.launch.py game_launcher:=path/to/RAIROSBotXLDemo.GameLauncher
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

> [!TIP]
> If you are having trouble running the binary, you can build it from source [here](https://github.com/RobotecAI/rai-rosbot-xl-demo).
