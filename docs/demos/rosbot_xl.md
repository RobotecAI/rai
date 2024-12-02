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
Please refer to [rai husarion rosbot xl demo][rai rosbot demo] for more details.

# Running RAI

1. Robot identity

   Process of setting up the robot identity is described in [create_robots_whoami](../create_robots_whoami.md).
   We provide ready whoami for RosBotXL in the package.

   ```bash
   cd rai
   vcs import < demos.repos
   colcon build --symlink-install --packages-select rosbot_xl_whoami
   ```

2. Running rai nodes and agents, navigation stack and O3DE simulation.

   ```bash
   ros2 launch ./examples/rosbot-xl.launch.py game_launcher:=path/to/RAIROSBotXLDemo.GameLauncher
   ```

3. Play with the demo, adding tasks to the RAI agent. Here are some examples:

   ```bash
   # Ask robot where it is. RAI will use camera to describe the environment
   ros2 action send_goal -f /perform_task rai_interfaces/action/Task "{priority: 10, description: '', task: 'Where are you?'}"

   # See integration with the navigation stack
   ros2 action send_goal -f /perform_task rai_interfaces/action/Task "{priority: 10, description: '', task: 'Drive 1 meter forward'}"
   ros2 action send_goal -f /perform_task rai_interfaces/action/Task "{priority: 10, description: '', task: 'Spin 90 degrees'}"

   # Try out more complicated tasks
   ros2 action send_goal -f /perform_task rai_interfaces/action/Task "{priority: 10, description: '', task: ' Drive forward if the path is clear, otherwise backward'}"
   ```

> **NOTE**: For now agent is capable of performing only 1 task at once.
> Human-Robot Interaction module is not yet included in the demo (coming soon!).

### What is happening?

By looking at the example code in [rai/examples/rosbot-xl-demo.py](../../examples/rosbot-xl-demo.py) `examples` you can see that:

- This node has no information about the robot besides what it can get from `rai_whoami_node`.
- Topics can be whitelisted to only receive information about the robot.
- Before every LLM decision, `rai_node` sends its state to the LLM Agent. By default, it contains ros interfaces (topics, services, actions) and logs summary, but the state can be extended.
- In the example we are also adding description of the camera image to the state.

If you wish, you can learn more about [configuring RAI for a specific robot](../create_robots_whoami.md).

[rai rosbot demo]: https://github.com/RobotecAI/rai-rosbot-xl-demo
