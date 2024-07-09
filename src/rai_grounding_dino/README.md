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
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```

Then prepare the virtual environment for this package

```
cd src/rai_grounding_dino
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### Build and run

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

Activate the virtucal environment:

```
. src/rai_grounding_dino/.venv/bin/activate
```

Run the ROS node using `ros2 run`:

```
ros2 run rai_grounding_dino grounding_dino

```
