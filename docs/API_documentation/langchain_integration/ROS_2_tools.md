# Tools

RAI provides various ROS 2 tools, both generic (mimics ros2cli) and specific (e.g., nav2, moveit2, etc.)

## Class Definition

`BaseROS2Tool` is the base class for all ROS 2 tools. It provides a common interface for all ROS 2 tools including name allowlist/blocklist for all the communication protocols (messages, services, actions)

### Class Definition

::: rai.tools.ros2.base.BaseROS2Tool

### Security Model

ROS2 tools implement a security model using three access control parameters:

-   **`readable`** (Allowlist for Reading): Whitelist of topics the agent can read/subscribe to

    -   If `None`: all topics are readable (permissive)
    -   If set: only topics in the list are readable (restrictive)
    -   Note: Only applies to topics; services and actions are not checked against this parameter

-   **`writable`** (Allowlist for Writing): Whitelist of topics/actions/services the agent can write/publish to

    -   If `None`: all topics/actions/services are writable (permissive)
    -   If set: only topics/actions/services in the list are writable (restrictive)

-   **`forbidden`** (Denylist): Blacklist of topics/actions/services that are always denied
    -   Highest priority - checked first and overrides both `readable` and `writable`
    -   If a resource is forbidden, it cannot be accessed regardless of allowlists

**Priority Order:** `forbidden` > `readable`/`writable` > default (all allowed)

### Usage example

```python
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2.base import BaseROS2Tool

connector = ROS2Connector()

BaseROS2Tool( # BaseROS2Tool cannot be used directly, this is just an example
    connector=connector,
    readable=["/odom", "/scan"], # readable topics
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

#### GetROS2TopicsNamesAndTypesTool Behavior

`GetROS2TopicsNamesAndTypesTool` lists ROS2 topics and their types with filtering and categorization based on `readable`, `writable`, and `forbidden` parameters:

**Filtering Logic:**

-   **Forbidden topics**: Always excluded regardless of allowlists
-   **When only `readable` is set**: Only topics in the readable list are included
-   **When only `writable` is set**: Only topics in the writable list are included
-   **When both `readable` and `writable` are set**: Topics in readable **OR** writable lists are included (OR logic)
-   **When neither is set**: All topics are included (except forbidden)

**Output Format:**

-   **No restrictions**: Simple list of all topics
-   **With restrictions**: Topics are categorized and displayed in sections:
    1. Topics in both readable and writable lists (no section header)
    2. "Readable topics:" section (topics only in readable list)
    3. "Writable topics:" section (topics only in writable list)

**Example:**

```python
tool = GetROS2TopicsNamesAndTypesTool(
    connector=connector,
    readable=["/odom", "/scan", "/camera/image"],
    writable=["/goal_pose", "/scan"],
    forbidden=["/cmd_vel"]
)
# /scan appears in first section (both readable and writable)
# /odom and /camera/image appear in "Readable topics:" section
# /goal_pose appears in "Writable topics:" section
# /cmd_vel is excluded (forbidden)
```

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

| Tool Name                | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `MoveToPointTool`        | Tool for moving to a point                          |
| `MoveObjectFromToTool`   | Tool for moving an object from one point to another |
| `GetObjectPositionsTool` | Tool for retrieving the positions of objects        |
