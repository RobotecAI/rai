# Demos

## Installation

Download repositories

```bash
vcs import < demos.repos
```

## Husarion ROSBot XL demo

![Screenshot1](imgs/o3deSimulation.png)

Please refer to [rai husarion rosbot xl demo][rai rosbot demo] to install and run the simulation.

### Running RAI

You can set the task for the agent in the `examples/nav2_example_ros_actions.py` file.

1. Prepare the robot description.

```bash
colcon build --symlink-install --packages-select rosbot_xl_whoami

. install/setup.bash
```

2. Start `rai_whoami_node`
   who_am_i node. It loads files from robot [description](https://github.com/RobotecAI/rai-rosbot-xl-demo/tree/development/src/rosbot_xl_whoami/description) folder to server robot identity.

```bash
source setup_shell.sh

ros2 run rai_whoami rai_whoami_node --ros-args -p robot_description_package:="rosbot_xl_whoami"
```

3. Start `rai_node`.

By looking at the example code in `src/examples/rosbot-xl-generic-node-demo.py` you can see that:

- This node has no information about the robot besides what it can get from `rai_whoami_node`
- Topics can be whitelisted to only receive information about the robot
- Before every LLM decision, `rai_node` sends it's state to the LLM Agent. By default it contains ros interfaces (topics, services, actions) and rosout summary, but state can be extended. In the example we also adding the summary of the camera image.

```bash
source setup_shell.sh

python examples/rosbot-xl-generic-node-demo.py
```

4. Send the task to the node:

> **NOTE**: For now agent is capable of performing only 1 task at once.

```bash
# Using robots camera to describe environment
ros2 topic pub --once /task_addition_requests std_msgs/msg/String "data: 'Where are you now?'"

# Automatic integration with navigation stack
ros2 topic pub --once /task_addition_requests std_msgs/msg/String "data: 'Drive 1 meter forward'"
ros2 topic pub --once /task_addition_requests std_msgs/msg/String "data: 'Spin 90 degrees'"

# Knowledge composition to achieve more complicated tasks
ros2 topic pub --once /task_addition_requests std_msgs/msg/String "data: 'Drive forward if the path is clear, otherwise backward'"
```

[rai rosbot demo]: https://github.com/RobotecAI/rai-rosbot-xl-demo
