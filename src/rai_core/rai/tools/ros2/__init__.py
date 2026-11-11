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

from .diagnostics import (
    GetROS2DiagnosticsTool,
    GetROS2LifecycleStatesTool,
    GetROS2LogsTool,
    GetROS2NodeInfoTool,
    GetROS2ParametersTool,
    GetROS2TFTreeTool,
    GetROS2TopicInfoTool,
    ListROS2InterfacesTool,
    ListROS2NodesTool,
    MeasureROS2TopicRateTool,
    ROS2DiagnosticsToolkit,
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
    "GetROS2DiagnosticsTool",
    "GetROS2ImageConfiguredTool",
    "GetROS2ImageTool",
    "GetROS2LifecycleStatesTool",
    "GetROS2LogsTool",
    "GetROS2MessageInterfaceTool",
    "GetROS2NodeInfoTool",
    "GetROS2ParametersTool",
    "GetROS2ServicesNamesAndTypesTool",
    "GetROS2TFTreeTool",
    "GetROS2TopicInfoTool",
    "GetROS2TopicsNamesAndTypesTool",
    "GetROS2TransformConfiguredTool",
    "GetROS2TransformTool",
    "ListROS2InterfacesTool",
    "ListROS2NodesTool",
    "MeasureROS2TopicRateTool",
    "MoveObjectFromToTool",
    "MoveObjectFromToToolInput",
    "MoveToPointTool",
    "MoveToPointToolInput",
    "Nav2Toolkit",
    "NavigateToPoseTool",
    "PublishROS2MessageTool",
    "ROS2ActionToolkit",
    "ROS2DiagnosticsToolkit",
    "ROS2ServicesToolkit",
    "ROS2Toolkit",
    "ROS2TopicsToolkit",
    "ReceiveROS2MessageTool",
    "StartROS2ActionTool",
]
