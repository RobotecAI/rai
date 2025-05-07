# Autonomous Tractor Demo

This demo showcases autonomous tractors operating in an agricultural field using ROS 2. The tractor
is controlled using a conventional navigation stack. Sometimes, due to the ever-changing
environment, the tractor may encounter unexpected situations. The conventional stack is not designed
to handle these situations and usually ends up replanning the path. The tractor can handle this
optimally by calling the RAI agent to decide what to do.

<div style="text-align: center;"><img src="../../imgs/agriculture_demo.gif" alt="agriculture-demo"></div>

## Quick Start

!!! tip "Remain in sourced shell"

    Ensure that every command is run in a sourced shell using `source setup_shell.sh`
    Ensure ROS 2 is sourced.

1. **Download the Latest Release**

    ```bash
    ./scripts/download_demo.sh agriculture
    ```

2. **Run the Simulation**

    ```bash
    ./demo_assets/agriculture/RAIAgricultureDemoGamePackage/RAIAgricultureDemo.GameLauncher -bg_ConnectToAssetProcessor=0
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
[API documentation](../API_documentation/overview.md).

!!! tip "Building from source"

    If you are having trouble running the binary, you can build it from source
    [here](https://github.com/RobotecAI/rai-agriculture-demo).
