# Demos

## Installation

Download repositories

```bash
vcs import < demos.repos
```

## Husarion ROSBot XL demo

![Screenshot1](imgs/o3deSimulation.png)

Please refer to [rai husarion rosbot xl demo][rai husarion demo] to install and run the simulation.

### Running RAI

You can set the task for the agent in the `examples/nav2_example_ros_actions.py` file.

```bash
. /opt/ros/${ROS_DISTRO}/setup.bash # for e.g. ROS_DISTRO=jazzy
. ./install/setup.bash
poetry shell
python examples/nav2_ros_actions.py
```

[rai husarion demo]: https://github.com/RobotecAI/rai-husarion-demo-private

## Agriculture demo

TBD

## Manipulation demo

TBD
