# ROS 2 tools

`rai.tools.ros2` has ready-made tools on a [`ROS2Connector`](../concepts/connectors.md): generic tools for any topic, service or action, Nav2 navigation, manipulation, and read-only diagnostics. Each is a [`RaiTool`](../concepts/tools.md) with an effect and resolved ROS targets, so safety hooks see what it touches.

```python
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import Nav2Toolkit, ROS2DiagnosticsToolkit, ROS2Toolkit

connector = ROS2Connector()
tools = [
    *ROS2Toolkit(connector=connector, forbidden=["/cmd_vel"]).get_tools(),
    *Nav2Toolkit(connector=connector, workspace_bounds_min=(-5, -5, -1), workspace_bounds_max=(5, 5, 1)).get_tools(),
    *ROS2DiagnosticsToolkit(connector=connector).get_tools(),
]
```

## Access lists

Every ROS 2 tool and toolkit takes three lists, compared on resolved names, so `"cmd_vel"`, `"/cmd_vel"` and remapped forms match:

| Parameter   | Meaning                                                                      | `None` (default) |
| ----------- | ---------------------------------------------------------------------------- | ---------------- |
| `readable`  | Topics the tools may read                                                    | all              |
| `writable`  | Topics, services and actions the tools may publish to, call or send goals to | all              |
| `forbidden` | Names that are always denied, read or write; wins over both lists            | none             |

A denied name fails the call with an error tool message. The listing tools filter by the same lists: `GetROS2TopicsNamesAndTypesTool` shows only topics the agent may use, grouped into readable and writable when lists are set, and never forbidden ones.

The access lists shape what the model is offered. To enforce a rule no matter which tool or code path is used, add an [egress hook](../concepts/safety.md#connector-side-egress-hooks).

## Generic tools

`ROS2Toolkit` combines the three toolkits below. Each tool can also be used on its own, except the action tools, which share the toolkit's goals.

| Toolkit               | Tool (LLM name)                                                          | Effect  |
| --------------------- | ------------------------------------------------------------------------ | ------- |
| `ROS2TopicsToolkit`   | `PublishROS2MessageTool` (`publish_ros2_message`)                        | ACTUATE |
|                       | `ReceiveROS2MessageTool` (`receive_ros2_message`)                        | READ    |
|                       | `GetROS2ImageTool` (`get_ros2_image`)                                    | READ    |
|                       | `GetROS2TransformTool` (`get_ros2_transform`)                            | READ    |
|                       | `GetROS2TopicsNamesAndTypesTool` (`get_ros2_topics_names_and_types`)     | READ    |
|                       | `GetROS2MessageInterfaceTool` (`get_ros2_message_interface`)             | READ    |
| `ROS2ServicesToolkit` | `CallROS2ServiceTool` (`call_ros2_service`)                              | WRITE   |
|                       | `GetROS2ServicesNamesAndTypesTool` (`get_ros2_services_names_and_types`) | READ    |
| `ROS2ActionToolkit`   | `StartROS2ActionTool` (`start_ros2_action`)                              | ACTUATE |
|                       | `CancelROS2ActionTool` (`cancel_ros2_action`)                            | WRITE   |
|                       | `GetROS2ActionFeedbackTool` (`get_ros2_action_feedback`)                 | READ    |
|                       | `GetROS2ActionResultTool` (`get_ros2_action_result`)                     | READ    |
|                       | `GetROS2ActionIDsTool` (`get_ros2_action_ids`)                           | READ    |
|                       | `GetROS2ActionsNamesAndTypesTool` (`get_ros2_actions_names_and_types`)   | READ    |

The model passes messages as dicts with the type name (`"geometry_msgs/msg/Twist"`); the tools convert them to native messages. Images come back as image artifacts the model sees. A started action returns an id the model uses for feedback, result and cancel.

### Configured tools

When the model should not choose the topic, configure it in the tool:

| Tool                             | LLM name                  | Effect | Configured with                               |
| -------------------------------- | ------------------------- | ------ | --------------------------------------------- |
| `GetROS2ImageConfiguredTool`     | `get_ros2_camera_image`   | READ   | `topic`, `timeout_sec`                        |
| `GetROS2TransformConfiguredTool` | `get_ros2_robot_position` | READ   | `source_frame`, `target_frame`, `timeout_sec` |

## Navigation (Nav2)

`Nav2Toolkit` starts `NavigateToPose` goals without blocking, so the agent can watch feedback, answer the user or cancel:

| Tool                            | LLM name                        | Effect  |
| ------------------------------- | ------------------------------- | ------- |
| `NavigateToPoseTool`            | `navigate_to_pose`              | ACTUATE |
| `GetNavigateToPoseFeedbackTool` | `get_navigate_to_pose_feedback` | READ    |
| `GetNavigateToPoseResultTool`   | `get_navigate_to_pose_result`   | READ    |
| `CancelNavigateToPoseTool`      | `cancel_navigate_to_pose`       | WRITE   |

`frame_id` (default `map`) and `action_name` (default `navigate_to_pose`) configure the Nav2 stack. `workspace_bounds_min` and `workspace_bounds_max` reject goals outside an (x, y, z) box before they are sent. `rai.tools.ros2.navigation.nav2.GetOccupancyGridTool` returns the map as an image with the robot's pose drawn on it.

## Manipulation

These tools drive an arm through the `rai_interfaces/srv/ManipulatorMoveTo` service (see the [manipulation demo](../demos/manipulation.md)):

| Tool                     | LLM name               | Effect  |
| ------------------------ | ---------------------- | ------- |
| `MoveToPointTool`        | `move_to_point`        | ACTUATE |
| `MoveObjectFromToTool`   | `move_object_from_to`  | ACTUATE |
| `GetObjectPositionsTool` | `get_object_positions` | READ    |

`manipulator_frame` is required. `calibration_x`, `calibration_y`, `calibration_z` and `min_z` correct and bound the targets. `GetObjectPositionsTool` is deprecated: use `GetObjectGrippingPointsTool` from [rai_perception](perception.md).

## Diagnostics

`ROS2DiagnosticsToolkit` answers "why is my robot not moving" questions. Every tool is READ and works through the connector's graph queries, subscriptions and service calls; nothing shells out to the `ros2` CLI. Outputs are short text with units, bounded in length for the model.

| LLM name                    | Reports                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `list_ros2_nodes`           | Nodes in the graph; flags duplicate names and the agent's own node                                               |
| `get_ros2_node_info`        | What a node publishes, subscribes to, serves and calls, with types                                               |
| `get_ros2_topic_info`       | A topic's type, publishers and subscribers with their QoS, and incompatible QoS pairs                            |
| `measure_ros2_topic_rate`   | Rate (Hz), period jitter, bandwidth and header delay over a short window                                         |
| `get_ros2_parameters`       | A node's parameter values, optionally with descriptions                                                          |
| `get_ros2_logs`             | `/rosout` entries filtered by level, node, text and age                                                          |
| `get_ros2_diagnostics`      | `/diagnostics` (or `/diagnostics_agg`): worst level overall and per hardware id, problems, stale entries         |
| `get_ros2_lifecycle_states` | Lifecycle (managed) nodes and their state                                                                        |
| `get_ros2_tf_tree`          | The TF trees with each transform's rate and age, stale transforms, and the chain or missing link of a frame pair |
| `list_ros2_interfaces`      | Installed interface packages, or the types of one package                                                        |

`get_ros2_logs`, `get_ros2_diagnostics` and `get_ros2_tf_tree` start listening at their first call and keep a bounded buffer, so the first call waits briefly for data. `measure_ros2_topic_rate` subscribes for its window only.

The toolkit does not repeat the listing tools of `ROS2Toolkit`. A debugging agent adds `GetROS2TopicsNamesAndTypesTool`, `GetROS2ServicesNamesAndTypesTool`, `GetROS2ActionsNamesAndTypesTool` and `GetROS2MessageInterfaceTool`, as in `examples/debugging_assistant.py` ([demo](../demos/debugging_assistant.md)).

## Serving them over MCP

`rai mcp serve` exposes the same toolkits to any MCP client, with the same access lists and hooks. See [MCP](mcp.md).

See the [API reference](../api/ros2_tools.md) for every parameter, and [Writing tools](writing_tools.md) to add tools for your robot.
