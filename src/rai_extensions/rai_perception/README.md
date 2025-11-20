<!--- --8<-- [start:sec1] -->

# RAI Perception

This package provides ROS2 integration with [Idea-Research GroundingDINO Model](https://github.com/IDEA-Research/GroundingDINO) and [Grounded-SAM-2, RobotecAI fork](https://github.com/RobotecAI/Grounded-SAM-2) for object detection, segmentation, and gripping point calculation. The `GroundedSamAgent` and `GroundingDinoAgent` are ROS2 service nodes that can be readily added to ROS2 applications. It also provides tools that can be used with [RAI LLM agents](../tutorials/walkthrough.md) to construct conversational scenarios.

In addition to these building blocks, this package includes utilities to facilitate development, such as a ROS2 client that demonstrates interactions with agent nodes.

## Installation

While installing `rai_perception` via Pip is being actively worked on, to incorporate it into your application, you will need to set up a ROS2 workspace.

### ROS2 Workspace Setup

Create a ROS2 workspace and copy this package:

```bash
mkdir -p ~/rai_perception_ws/src
cd ~/rai_perception_ws/src

# only checkout rai_perception package
git clone --depth 1 --branch main https://github.com/RobotecAI/rai.git temp
cd temp
git archive --format=tar --prefix=rai_perception/ HEAD:src/rai_extensions/rai_perception | tar -xf -
mv rai_perception ../rai_perception
cd ..
rm -rf temp
```

### ROS2 Dependencies

Add required ROS dependencies. From the workspace root, run

```bash
rosdep install --from-paths src --ignore-src -r
```

### Build and Run

Source ROS2 and build:

```bash
# Source ROS2 (humble or jazzy)
source /opt/ros/${ROS_DISTRO}/setup.bash

# Build workspace
cd ~/rai_perception_ws
colcon build --symlink-install

# Source ROS2 packages
source install/setup.bash
```

### Python Dependencies

`rai_perception` depends on `rai-core` and `sam2`. There are many ways to set up a virtual environment and install these dependencies. Below, we provide an example using Poetry.

**Step 1:** Copy the following template to `pyproject.toml` in your workspace root, updating it according to your directory setup:

```toml
# rai_perception_project pyproject template
[tool.poetry]
name = "rai_perception_ws"
version = "0.1.0"
description = "ROS2 workspace for RAI perception"
package-mode = false

[tool.poetry.dependencies]
python = "^3.10, <3.13"
rai-core = ">=2.5.4"
rai-perception = {path = "src/rai_perception", develop = true}

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

**Step 2:** Install dependencies:

First, we create Virtual Environment with Poetry:

```bash
cd ~/rai_perception_ws
poetry lock
poetry install
```

Now, we are ready to launch perception agents:

```bash
# Activate virtual environment
source "$(poetry env info --path)"/bin/activate
export PYTHONPATH
PYTHONPATH="$(dirname "$(dirname "$(poetry run which python)")")/lib/python$(poetry run python --version | awk '{print $2}' | cut -d. -f1,2)/site-packages:$PYTHONPATH"

# run agents
python src/rai_perception/scripts/run_perception_agents.py
```

> [!TIP]
> To manage ROS 2 + Poetry environment with less friction: Keep build tools (colcon) at system level, use Poetry only for runtime dependencies of your packages.

<!--- --8<-- [end:sec1] -->

`rai-perception` agents create two ROS 2 nodes: `grounding_dino` and `grounded_sam` using [ROS2Connector](../../../docs/API_documentation/connectors/ROS_2_Connectors.md).
These agents can be triggered by ROS2 services:

-   `grounding_dino_classify`: `rai_interfaces/srv/RAIGroundingDino`
-   `grounded_sam_segment`: `rai_interfaces/srv/RAIGroundedSam`

> [!TIP]
>
> If you wish to integrate open-set vision into your ros2 launch file, a premade launch
> file can be found in `rai/src/rai_bringup/launch/openset.launch.py`

> [!NOTE]
> The weights will be downloaded to `~/.cache/rai` directory.

## RAI Tools

`rai_perception` package contains tools that can be used by [RAI LLM agents](../../../docs/tutorials/walkthrough.md)
to enhance their perception capabilities. For more information on RAI Tools see
[Tool use and development](../../../docs/tutorials/tools.md) tutorial.

<!--- --8<-- [start:sec2] -->

### `GetDetectionTool`

This tool calls the GroundingDINO service to detect objects from a comma-separated prompt in the provided camera topic.

<!--- --8<-- [end:sec2] -->

> [!TIP]
>
> you can try example below with [rosbotxl demo](../../../docs/demos/rosbot_xl.md) binary.
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

## Simple ROS2 Client Node Example

The `rai_perception/talker.py` example demonstrates how to use the perception services for object detection and segmentation. It shows the complete pipeline: GroundingDINO for object detection followed by GroundedSAM for instance segmentation, with visualization output.

This example is useful for:

-   Testing perception services integration
-   Understanding the ROS2 service call patterns
-   Seeing detection and segmentation results with bounding boxes and masks

Run the example:

```bash
cd ~/rai_perception_ws
python src/rai_perception/scripts/run_perception_agents.py
```

In a different window, run

```bash
cd ~/rai_perception_ws
ros2 run rai_perception talker --ros-args -p image_path:=src/rai_perception/images/sample.jpg
```

The example will detect objects (dragon, lizard, dinosaur) and save a visualization with bounding boxes and masks to `masks.png`.

<!--- --8<-- [end:sec3] -->
