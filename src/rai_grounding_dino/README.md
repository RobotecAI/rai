# RAI GroundingDINO

This package provides a ROS2 Node which is an interface to the [Idea-Research GroundingDINO Model](https://github.com/IDEA-Research/GroundingDINO).
It allows for open-set detection.

## Installation

In your workspace you need to have an `src` folder containing this package `rai_grounding_dino` and the `rai_interfaces` package.

### Preparing the GroundingDINO

Download the weights of the model:

```
mkdir PATH/TO/WEIGHTS
cd PATH/TO/WEIGHTS
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Add required ROS dependencies:

```
rosdep install --from-paths src --ignore-src -r
```

### Build and run

In the base directory of the `RAI` package install dependancies:

```
poetry install --with gdino
```
Source the ros installation

```
source /opt/ros/${ROS_DISTRO}/setup.bash
```
Run the build process:

```
colcon build
```
Source the local installation:

```
source ./install/setup.bash
```
Activate the poetry environment:

```
poetry shell
```
Run the ROS node using `ros2 launch`:

```
ros2 launch rai_grounding_dino gdino_launch.xml weights_path:=PATH/TO/WEIGHTS
```
### Example

An example client is provided with the package as `rai_grounding_dino/talker.py`

You can see it working by running:

```
ros2 launch rai_grounding_dino example_communnication_launch.xml weights_path:=PATH/TO/WEIGHTS image_path:=src/rai_grounding_dino/images/sample.jpg
```
If everything was set up properly you should see a couple of detections with classes `dinosaur`, `dragon`, and `lizard`.
