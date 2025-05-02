# Autonomous Tractor Demo

This demo showcases autonomous tractors operating in an agricultural field using ROS 2. The tractor
is controlled using a conventional navigation stack. Sometimes, due to the ever-changing
environment, the tractor may encounter unexpected situations. The conventional stack is not designed
to handle these situations and usually ends up replanning the path. The tractor can handle this
optimally by calling the RAI agent to decide what to do.

![Tractor in field meets an obstacle](../imgs/agriculture_demo.gif)

## Quick Start

1. **Download the Latest Release**

    Download the latest binary release for your ROS 2 distribution.

    - [ros2-humble-agriculture-demo](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIAgricultureDemo_1.0.0_jammyhumble.zip)
    - [ros2-jazzy-agriculture-demo](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIAgricultureDemo_1.0.0_noblejazzy.zip)

2. **Unpack the Binary and Run the Simulation** Unpack the binary

    - For Jazzy:

    ```bash
    unzip RAIAgricultureDemo_1.0.0_noblejazzy.zip
    ```

    - For Humble:

    ```bash
    unzip RAIAgricultureDemo_1.0.0_jammyhumble.zip
    ```

    ```bash
    . /opt/ros/${ROS_DISTRO}/setup.bash
    ./RAIAgricultureDemoGamePackage/RAIAgricultureDemo.GameLauncher -bg_ConnectToAssetProcessor=0
    ```

3. **Start the Tractor Node**

    ```bash
    python examples/agriculture-demo.py --tractor_number 1
    ```

You are now ready to run the demo and see the tractor in action!

## Running the Demo

The demo simulates a scenario where the tractor stops due to an unexpected situation. The RAI Agent
decides the next action based on the current state.

### RAI Agent decisions

RAI Agent's mission is to decide the next action based on the current state of the anomaly. There
are three exposed services to control the tractor:

-   continue

Used when the anomaly is flagged as a false positive.

-   flash

Used to flash the lights on the tractor to e.g. get the attention of the animals

-   replan

Used to replan the path/skip the alley.

### What Happens in the Demo?

-   The node listens for the tractor's state and calls the RaiNode using ROS 2 action when an anomaly
    is detected.
-   The RaiNode decides the next action based on the current state.
-   The tractor performs the action and the demo continues.

For more details on configuring RAI for specific robots, refer to the
[RAI documentation](../create_robots_whoami.md).

!!! tip "Building from source"

    If you are having trouble running the binary, you can build it from source
    [here](https://github.com/RobotecAI/rai-agriculture-demo).
