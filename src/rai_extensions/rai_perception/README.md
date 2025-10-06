<!--- --8<-- [start:sec1] -->

# RAI Open Set Vision

This package provides a ROS2 Node which is an interface to the [Idea-Research GroundingDINO Model](https://github.com/IDEA-Research/GroundingDINO).
It allows for open-set detection.

## Installation

In your workspace you need to have an `src` folder containing this package `rai_perception` and the `rai_interfaces` package.

### Preparing the GroundingDINO

Add required ROS dependencies:

```
rosdep install --from-paths src --ignore-src -r
```

## Build and run

In the base directory of the `RAI` package install dependencies:

```
poetry install --with openset
```

Source the ros installation

```
source /opt/ros/${ROS_DISTRO}/setup.bash
```

Run the build process:

```
colcon build --symlink-install
```

Source the environment

```
source setup_shell.sh
```

Run the `GroundedSamAgent` and `GroundingDinoAgent` agents.

```
python run_vision_agents.py
```

<!--- --8<-- [end:sec1] -->

Agents create two ROS 2 Nodes: `grounding_dino` and `grounded_sam` using [ROS2Connector](../../../docs/API_documentation/connectors/ROS_2_Connectors.md).
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
enhance their perception capabilities. For more information on RAI Tools see
[Tool use and development](../../../docs/tutorials/tools.md) tutorial.

<!--- --8<-- [start:sec3] -->

### `GetDetectionTool`

This tool calls the grounding dino service to use the model to see if the message from the provided camera topic contains objects from a comma separated prompt.

<!--- --8<-- [end:sec3] -->

> [!TIP]
>
> you can try example below with [rosbotxl demo](../../../docs/demos/rosbot_xl.md) binary.
> The binary exposes `/camera/camera/color/image_raw` and `/camera/camera/depth/image_raw` topics.

<!--- --8<-- [start:sec4] -->

**Example call**

```python
from rai_perception.tools import GetDetectionTool
from rai.communication.ros2 import ROS2Connector, ROS2Context

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")
    x = GetDetectionTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        object_names=["chair", "human", "plushie", "box", "ball"],
    )
```

**Example output**

```
I have detected the following items in the picture - chair, human
```

### `GetDistanceToObjectsTool`

This tool calls the grounding dino service to use the model to see if the message from the provided camera topic contains objects from a comma separated prompt. Then it utilises messages from depth camera to create an estimation of distance to a detected object.

**Example call**

```python
from rai_perception.tools import GetDetectionTool
from rai.communication.ros2 import ROS2Connector, ROS2Context

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")
    connector.node.declare_parameter("conversion_ratio", 1.0) # scale parameter for the depth map
    x = GetDistanceToObjectsTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/depth/image_rect_raw",
        object_names=["chair", "human", "plushie", "box", "ball"],
    )

```

**Example output**

```
I have detected the following items in the picture human: 3.77m away
```

## Simple ROS2 Client Node Example

An example client is provided with the package as `rai_perception/talker.py`

You can see it working by running:

```
python run_vision_agents.py
cd rai # rai repo BASE directory
ros2 run rai_perception talker --ros-args -p image_path:=src/rai_extensions/rai_perception/images/sample.jpg
```

If everything was set up properly you should see a couple of detections with classes `dinosaur`, `dragon`, and `lizard`.

<!--- --8<-- [end:sec4] -->
