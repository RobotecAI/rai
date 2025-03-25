# RAI Open Set Vision

This package provides a ROS2 Node which is an interface to the [Idea-Research GroundingDINO Model](https://github.com/IDEA-Research/GroundingDINO).
It allows for open-set detection.

## Installation

In your workspace you need to have an `src` folder containing this package `rai_open_set_vision` and the `rai_interfaces` package.

### Preparing the GroundingDINO

Add required ROS dependencies:

```
rosdep install --from-paths src --ignore-src -r
```

### Build and run

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

Run the ROS node using `ros2 launch`:

```
ros2 launch rai_open_set_vision gdino_launch.xml [weights_path:=PATH/TO/WEIGHTS]
```

> [!NOTE]
> By default the weights will be downloaded to `$(ros2 pkg prefix rai_open_set_vision)/share/weights/`.
> You can change this path if you downloaded the weights manually or moved them.

### RAI Tools

This package provides the following tools:

- `GetDetectionTool`
  This tool calls the grounding dino service to use the model to see if the message from the provided camera topic contains objects from a comma separated prompt.

  **Example call**

  ```
  x = GetDetectionTool(node=RaiBaseNode(node_name="test_node"))._run(
      camera_topic="/camera/camera/color/image_raw",
      object_names=["chair", "human", "plushie", "box", "ball"],
  )

  ```

  **Example output**

  ```
  I have detected the following items in the picture - chair, human
  ```

- `GetDistanceToObjectsTool`
  This tool calls the grounding dino service to use the model to see if the message from the provided camera topic contains objects from a comma separated prompt. Then it utilises messages from depth camera to create an estimation of distance to a detected object.

  **Example call**

  ```
  x = GetDistanceToObjectsTool(node=RaiBaseNode(node_name="test_node"))._run(
      camera_topic="/camera/camera/color/image_raw",
      depth_topic="/camera/camera/depth/image_rect_raw",
      object_names=["chair", "human", "plushie", "box", "ball"],
  )

  ```

  **Example output**

  ```
  I have detected the following items in the picture human: 1.68 m away, chair: 2.20 m away
  ```

### Example

An example client is provided with the package as `rai_open_set_vision/talker.py`

You can see it working by running:

```
ros2 launch rai_open_set_vision example_communication_launch.xml image_path:=src/rai_extensions/rai_open_set_vision/images/sample.jpg [dino_weights_path:=PATH/TO/DINO/WEIGHTS] [sam_weights_path:=PATH/TO/SAM/WEIGHTS]
```

If everything was set up properly you should see a couple of detections with classes `dinosaur`, `dragon`, and `lizard`.
