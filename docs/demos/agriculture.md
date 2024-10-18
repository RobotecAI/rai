# Autonomous Tractor Demo

This demo showcases autonomous tractors operating in an agricultural field using ROS 2. The tractor is controlled using a conventional navigation stack. Sometimes, due to the ever-changing environment, the tractor may encounter unexpected situations. The conventional stack is not designed to handle these situations and usually ends up replanning the path. The tractor can handle this optimally by calling the RAI agent to decide what to do.

![Tractor in field meets an obstacle](../imgs/agriculture_demo.gif)

## Quick Start

1. **Download the Latest Release**

   Download the latest binary release (`release.zip`) from [rai-agriculture-demo -> releases](https://github.com/RobotecAI/rai-agriculture-demo/releases)

2. **Install Required Packages**

   ```bash
   sudo apt install ros-${ROS_DISTRO}-ackermann-msgs ros-${ROS_DISTRO}-gazebo-msgs ros-${ROS_DISTRO}-control-toolbox ros-${ROS_DISTRO}-nav2-bringup
   ```

3. **Unpack the Binary and Run the Simulation**

   ```bash
   unzip release.zip
   . /opt/ros/${ROS_DISTRO}/setup.bash
   ./release/RAIAgricultureDemo.GameLauncher -bg_ConnectToAssetProcessor=0
   ```

4. **Start the Tractor Node**

   ```bash
   python examples/agriculture-demo.py --tractor_number 1
   ```

You are now ready to run the demo and see the tractor in action!

## Running the Demo

The demo simulates a scenario where the tractor stops due to an unexpected situation. The RAI Agent decides the next action based on the current state.

### RAI Agent decisions

RAI Agent's mission is to decide the next action based on the current state of the anomaly. There are three exposed services to control the tractor:

- continue

Used when the anomaly is flagged as a false positive.

- flash

Used to flash the lights on the tractor to e.g. get the attention of the animals

- replan

Used to replan the path/skip the alley.

### What Happens in the Demo?

- The node listens for the tractor's state and calls the RaiNode using ROS 2 action when an anomaly is detected.
- The RaiNode decides the next action based on the current state.
- The tractor performs the action and the demo continues.

For more details on configuring RAI for specific robots, refer to the [RAI documentation](../create_robots_whoami.md).
