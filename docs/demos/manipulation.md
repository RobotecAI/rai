# Manipulation tasks with natural language

This demo showcases the capabilities of RAI (Robotec AI) in performing manipulation tasks using natural language commands. The demo utilizes a robot arm (Franka Emika Panda) in a simulated environment, demonstrating how RAI can interpret complex instructions and execute them using advanced vision and manipulation techniques.

![Manipulation Demo](../imgs/manipulation_demo.gif)

> [!NOTE]  
> This readme is a work in progress.

## Setup

1. Follow the RAI setup instructions in the [main README](../../README.md#setup).
2. Download additional dependencies:

```shell
poetry install --with openset
```

3. Clone the manipulation demo repository:

   ```bash
   git clone https://github.com/RobotecAI/rai-manipulation-demo.git src/examples/rai-manipulation-demo
   ```

4. Download the latest binary release for your ROS 2 distribution:

   - [ros2-humble-manipulation-demo](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIManipulationDemo_1.0.0_jammyhumble.zip)
   - [ros2-jazzy-manipulation-demo](https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIManipulationDemo_1.0.0_noblejazzy.zip)

5. Unpack the binary:

   For Humble:

   ```bash
   unzip RAIManipulationDemo_1.0.0_jammyhumble.zip
   ```

   For Jazzy:

   ```bash
   unzip RAIManipulationDemo_1.0.0_noblejazzy.zip
   ```

6. Build the ROS 2 workspace:

   ```bash
   colcon build --symlink-install
   ```

## Running the Demo

> **Note**: Ensure that every command is run in a sourced shell using `source setup_shell.sh`

1. Start the demo
   ```shell
   ros2 launch examples/manipulation-demo.launch.py
   ```
2. Interact with the robot arm using natural language commands. For example:
   ```
   Enter a prompt: Pick up the red cube and drop it on other cube
   ```

## How it works

The manipulation demo utilizes several components:

1. Vision processing using Grounded SAM 2 and Grounding DINO for object detection and segmentation.
2. RAI agent to process the request and plan the manipulation sequence.
3. Robot arm control for executing the planned movements.

The main logic of the demo is implemented in the `ManipulationDemo` class, which can be found in:

```python
examples/manipulation-demo.py
```
