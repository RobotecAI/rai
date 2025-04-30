# Tools

RAI provides various ROS 2 tools, both generic (mimics ros2cli) and specific (e.g., nav2, moveit2, etc.)

## Interface

`BaseROS2Tool` is the base class for all ROS 2 tools. It provides a common interface for all ROS 2 tools including name allowlist/blocklist for all the communication protocols (messages, services, actions)

```python
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2.base import BaseROS2Tool

connector = ROS2Connector()

BaseROS2Tool(
    connector=connector,
    readable=["/odom", "/scan"], # readable topics, services and actions
    writable=["/robot_position"], # writable topics, services and actions
    forbidden=["/cmd_vel"], # forbidden topics, services and actions
)
```

## Generic ROS 2 Tools

RAI provides a generic ROS 2 toolkit, which allows the Agent to interact with any ROS 2 topics, services and actions.

```python
from rai.tools.ros2 import ROS2Toolkit
from rai.communication.ros2 import ROS2Connector

connector = ROS2Connector()
tools = ROS2Toolkit(connector=connector).get_tools()
```

Below is the list of tools provided by the generic ROS 2 toolkit that can also be used as standalone tools (except for the ROS 2 action tools, which should be used via the `ROS2ActionToolkit` as they share a state):

### Topics

| Tool Name                        | Description                                           |
| -------------------------------- | ----------------------------------------------------- |
| `PublishROS2MessageTool`         | Tool for publishing messages to ROS 2 topics          |
| `ReceiveROS2MessageTool`         | Tool for receiving messages from ROS 2 topics         |
| `GetROS2ImageTool`               | Tool for retrieving image data from topics            |
| `GetROS2TopicsNamesAndTypesTool` | Tool for listing all available topics and their types |
| `GetROS2MessageInterfaceTool`    | Tool for getting message interface information        |
| `GetROS2TransformTool`           | Tool for retrieving transform data                    |

### Services

| Tool Name                          | Description                                             |
| ---------------------------------- | ------------------------------------------------------- |
| `GetROS2ServicesNamesAndTypesTool` | Tool for listing all available services and their types |
| `CallROS2ServiceTool`              | Tool for calling ROS 2 services                         |

### Actions

| Tool Name                         | Description                                            |
| --------------------------------- | ------------------------------------------------------ |
| `GetROS2ActionsNamesAndTypesTool` | Tool for listing all available actions and their types |
| `StartROS2ActionTool`             | Tool for starting ROS 2 actions                        |
| `GetROS2ActionFeedbackTool`       | Tool for retrieving action feedback                    |
| `GetROS2ActionResultTool`         | Tool for retrieving action results                     |
| `CancelROS2ActionTool`            | Tool for canceling running actions                     |
| `GetROS2ActionIDsTool`            | Tool for getting action IDs                            |

## Specific ROS 2 Tools

RAI provides specific ROS 2 tools for certain ROS 2 packages.

### Nav2

| Tool Name                       | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| `NavigateToPoseTool`            | Tool for navigating to a pose                                 |
| `GetNavigateToPoseFeedbackTool` | Tool for retrieving the feedback of a navigate to pose action |
| `GetNavigateToPoseResultTool`   | Tool for retrieving the result of a navigate to pose action   |
| `CancelNavigateToPoseTool`      | Tool for canceling a navigate to pose action                  |
| `GetOccupancyGridTool`          | Tool for retrieving the occupancy grid                        |

### Custom Tools

| Tool Name                | Description                                  |
| ------------------------ | -------------------------------------------- |
| `MoveToPointTool`        | Tool for moving to a point                   |
| `GetObjectPositionsTool` | Tool for retrieving the positions of objects |
