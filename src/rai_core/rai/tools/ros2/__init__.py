# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util

if importlib.util.find_spec("rclpy") is None:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

from .cli import (
    ROS2CLIToolkit,
    ros2_action,
    ros2_interface,
    ros2_node,
    ros2_param,
    ros2_service,
    ros2_topic,
)
from .generic import (
    CallROS2ServiceTool,
    CancelROS2ActionTool,
    GetROS2ActionFeedbackTool,
    GetROS2ActionIDsTool,
    GetROS2ActionResultTool,
    GetROS2ActionsNamesAndTypesTool,
    GetROS2ImageTool,
    GetROS2MessageInterfaceTool,
    GetROS2ServicesNamesAndTypesTool,
    GetROS2TopicsNamesAndTypesTool,
    GetROS2TransformTool,
    PublishROS2MessageTool,
    ReceiveROS2MessageTool,
    ROS2ActionToolkit,
    ROS2ServicesToolkit,
    ROS2Toolkit,
    ROS2TopicsToolkit,
    StartROS2ActionTool,
)
from .manipulation.custom import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
    MoveObjectFromToToolInput,
    MoveToPointTool,
    MoveToPointToolInput,
)
from .navigation.nav2 import (
    CancelNavigateToPoseTool,
    GetNavigateToPoseFeedbackTool,
    GetNavigateToPoseResultTool,
    Nav2Toolkit,
    NavigateToPoseTool,
)
from .simple import (
    GetROS2ImageConfiguredTool,
    GetROS2TransformConfiguredTool,
)

__all__ = [
    "CallROS2ServiceTool",
    "CancelNavigateToPoseTool",
    "CancelROS2ActionTool",
    "GetNavigateToPoseFeedbackTool",
    "GetNavigateToPoseResultTool",
    "GetObjectPositionsTool",
    "GetROS2ActionFeedbackTool",
    "GetROS2ActionIDsTool",
    "GetROS2ActionResultTool",
    "GetROS2ActionsNamesAndTypesTool",
    "GetROS2ImageConfiguredTool",
    "GetROS2ImageTool",
    "GetROS2MessageInterfaceTool",
    "GetROS2ServicesNamesAndTypesTool",
    "GetROS2TopicsNamesAndTypesTool",
    "GetROS2TransformConfiguredTool",
    "GetROS2TransformTool",
    "MoveObjectFromToTool",
    "MoveObjectFromToToolInput",
    "MoveToPointTool",
    "MoveToPointToolInput",
    "Nav2Toolkit",
    "NavigateToPoseTool",
    "PublishROS2MessageTool",
    "ROS2ActionToolkit",
    "ROS2CLIToolkit",
    "ROS2ServicesToolkit",
    "ROS2Toolkit",
    "ROS2TopicsToolkit",
    "ReceiveROS2MessageTool",
    "StartROS2ActionTool",
    "ros2_action",
    "ros2_interface",
    "ros2_node",
    "ros2_param",
    "ros2_service",
    "ros2_topic",
]
