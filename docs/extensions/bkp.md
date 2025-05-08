--8<-- "src/rai_extensions/rai_open_set_vision/README.md:1:50"
Agents create two ROS 2 Nodes: `grounding_dino` and `grounded_sam` using [ROS2Connector](../API_documentation/connectors/ROS_2_Connectors.md).
--8<-- "src/rai_extensions/rai_open_set_vision/readme.md:52:56"

> [!TIP]
>
> If you wish to integrate open-set vision into your ros2 launch file, a premade launch
> file can be found in `rai/src/rai_bringup/launch/openset.launch.py`

> [!NOTE]
> The weights will be downloaded to `~/.cache/rai` directory.

## RAI Tools

`rai_open_set_vision` package contains tools that can be used by [RAI LLM agents](../../../docs/tutorials/walkthrough.md)
enhance their perception capabilities. For more information on RAI Tools see
[Tool use and development](../tutorials/tools.md) tutorial.

--8<-- "src/rai_extensions/rai_open_set_vision/README.md:71:74"

> [!TIP]
>
> you can try example below with [rosbotxl demo](../demos/rosbot_xl.md) binary.
> The binary exposes `/camera/camera/color/image_raw` and `/camera/camera/depth/image_raw` topics.
> --8<-- "src/rai_extensions/rai_open_set_vision/README.md:79:139"
