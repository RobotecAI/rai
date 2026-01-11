<!--- --8<-- [start:sec1] -->

# RAI Perception

RAI Perception brings powerful computer vision capabilities to your ROS2 applications. It integrates [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [Grounded-SAM-2](https://github.com/RobotecAI/Grounded-SAM-2) to detect objects, create segmentation masks, and calculate gripping points.

The package includes two ready-to-use ROS2 service nodes (`GroundedSamAgent` and `GroundingDinoAgent`) that you can easily add to your applications. It also provides tools that work seamlessly with [RAI LLM agents](../tutorials/walkthrough.md) to build conversational robot scenarios.

## Prerequisites

Before installing `rai-perception`, ensure you have:

1. **ROS2 installed** (Jazzy recommended, or Humble). If you don't have ROS2 yet, follow the official ROS2 installation guide for [jazzy](https://docs.ros.org/en/jazzy/Installation.html) or [humble](https://docs.ros.org/en/humble/Installation.html).
2. **Python 3.8+** and `pip` installed (usually pre-installed on Ubuntu).
3. **NVIDIA GPU** with CUDA support (required for optimal performance).
4. **wget** installed (required for downloading model weights):
    ```bash
    sudo apt install wget
    ```

## Installation

**Step 1:** Source ROS2 in your terminal:

```bash
# For ROS2 Jazzy (recommended)
source /opt/ros/jazzy/setup.bash

# For ROS2 Humble
source /opt/ros/humble/setup.bash
```

**Step 2:** Install ROS2 dependencies. `rai-perception` requires its ROS2 packages that needs to be installed separately:

```bash
# Update package lists first
sudo apt update

# Install rai_interfaces as a debian package
sudo apt install ros-jazzy-rai-interfaces  # or ros-humble-rai-interfaces for Humble
```

**Step 3:** Install `rai-perception` via pip:

```bash
pip install rai-perception
```

> [!TIP]
> It's recommended to install `rai-perception` in a virtual environment to avoid conflicts with other Python packages.

> [!TIP]
> To avoid sourcing ROS2 in every new terminal, add the source command to your `~/.bashrc` file:
>
> ```bash
> echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc  # or humble
> ```

<!--- --8<-- [end:sec1] -->

<!--- --8<-- [start:sec4] -->

## Getting Started

This section provides a step-by-step guide to get you up and running with RAI Perception.

### Quick Start

After installing `rai-perception`, launch the perception agents:

**Step 1:** Open a terminal and source ROS2:

```bash
source /opt/ros/jazzy/setup.bash  # or humble
```

**Step 2:** Launch the perception agents:

```bash
python -m rai_perception.scripts.run_perception_agents
```

> [!NOTE]
> The weights will be downloaded to `~/.cache/rai` directory on first use.

The agents create two ROS 2 nodes: `grounding_dino` and `grounded_sam` using [ROS2Connector](../API_documentation/connectors/ROS_2_Connectors.md).

### Testing with Example Client

The `rai_perception/talker.py` example demonstrates how to use the perception services for object detection and segmentation. It shows the complete pipeline: GroundingDINO for object detection followed by GroundedSAM for instance segmentation, with visualization output.

**Step 1:** Open a terminal and source ROS2:

```bash
source /opt/ros/jazzy/setup.bash  # or humble
```

**Step 2:** Launch the perception agents:

```bash
python -m rai_perception.scripts.run_perception_agents
```

**Step 3:** In a different terminal (remember to source ROS2 first), run the example client:

```bash
source /opt/ros/jazzy/setup.bash  # or humble
python -m rai_perception.examples.talker --ros-args -p image_path:="<path-to-image>"
```

You can use any image containing objects like dragons, lizards, or dinosaurs. For example, use the `sample.jpg` from the package's `images` folder. The client will detect these objects and save a visualization with bounding boxes and masks to `masks.png` in the current directory.

> [!TIP]
>
> If you wish to integrate open-set vision into your ros2 launch file, a premade launch
> file can be found in `rai/src/rai_bringup/launch/openset.launch.py`

### ROS2 Service Interface

The agents can be triggered by ROS2 services:

-   `grounding_dino_classify`: `rai_interfaces/srv/RAIGroundingDino`
-   `grounded_sam_segment`: `rai_interfaces/srv/RAIGroundedSam`

<!--- --8<-- [end:sec4] -->

<!--- --8<-- [start:sec5] -->

## Dive Deeper: Tools and Integration

This section provides information for developers looking to integrate RAI Perception tools into their applications.

### RAI Tools

`rai_perception` package contains tools that can be used by [RAI LLM agents](../tutorials/walkthrough.md)
to enhance their perception capabilities. For more information on RAI Tools see
[Tool use and development](../tutorials/tools.md) tutorial.

<!--- --8<-- [start:sec2] -->

### `GetDetectionTool`

This tool calls the GroundingDINO service to detect objects from a comma-separated prompt in the provided camera topic.

<!--- --8<-- [end:sec2] -->

> [!TIP]
>
> you can try example below with [rosbotxl demo](../demos/rosbot_xl.md) binary.
> The binary exposes `/camera/camera/color/image_raw` and `/camera/camera/depth/image_rect_raw` topics.

<!--- --8<-- [start:sec3] -->

**Example call**

```python
import time
from rai_perception.tools import GetDetectionTool
from rai.communication.ros2 import ROS2Connector, ROS2Context

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")

    # Wait for topic discovery to complete
    print("Waiting for topic discovery...")
    time.sleep(3)

    x = GetDetectionTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        object_names=["bed", "bed pillow", "table lamp", "plant", "desk"],
    )
    print(x)
```

**Example output**

```
I have detected the following items in the picture plant, table lamp, table lamp, bed, desk
```

### `GetDistanceToObjectsTool`

This tool calls the GroundingDINO service to detect objects from a comma-separated prompt in the provided camera topic. Then it utilizes messages from the depth camera to estimate the distance to detected objects.

**Example call**

```python
from rai_perception.tools import GetDistanceToObjectsTool
from rai.communication.ros2 import ROS2Connector, ROS2Context
import time

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")
    connector.node.declare_parameter("conversion_ratio", 1.0)  # scale parameter for the depth map

    # Wait for topic discovery to complete
    print("Waiting for topic discovery...")
    time.sleep(3)

    x = GetDistanceToObjectsTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/depth/image_rect_raw",
        object_names=["desk"],
    )

    print(x)
```

**Example output**

```
I have detected the following items in the picture desk: 2.43m away
```

### Debug Mode

Tools like `GetObjectGrippingPointsTool` support an optional `debug` parameter that enables progressive evaluation and debugging. When `debug=True`, the tool publishes intermediate pipeline results to ROS2 topics for visualization in RVIZ and logs detailed stage information including point counts and timing. This allows you to inspect individual pipeline stages (point cloud extraction, filtering, estimation) without running the full pipeline. Topics published include `/debug/gripping_points/raw_point_clouds` and `/debug/gripping_points/filtered_point_clouds`. Note that debug mode adds computational overhead and is not suitable for production use.

<!--- --8<-- [end:sec3] -->

<!--- --8<-- [end:sec5] -->
