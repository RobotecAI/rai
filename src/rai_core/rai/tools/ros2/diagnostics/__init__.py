# Copyright (C) 2026 Robotec.AI
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

"""Read-only ROS 2 diagnostics and introspection on the connector (no ros2 CLI)."""

from typing import List

from langchain_core.tools import BaseTool

from rai.tools.ros2.base import BaseROS2Toolkit

from .graph import (
    GetROS2NodeInfoTool,
    GetROS2ParametersTool,
    ListROS2InterfacesTool,
    ListROS2NodesTool,
)
from .lifecycle import GetROS2LifecycleStatesTool
from .logs import GetROS2DiagnosticsTool, GetROS2LogsTool
from .tf import GetROS2TFTreeTool
from .topics import GetROS2TopicInfoTool, MeasureROS2TopicRateTool


class ROS2DiagnosticsToolkit(BaseROS2Toolkit):
    """Every tool is READ. Pair it with the listing tools of ROS2Toolkit
    (topics, services, actions, message interfaces) for a debugging agent."""

    name: str = "ROS2DiagnosticsToolkit"
    description: str = "Read-only ROS 2 diagnostics and introspection"

    def get_tools(self) -> List[BaseTool]:
        return self._build(
            ListROS2NodesTool,
            GetROS2NodeInfoTool,
            GetROS2TopicInfoTool,
            MeasureROS2TopicRateTool,
            GetROS2ParametersTool,
            GetROS2LogsTool,
            GetROS2DiagnosticsTool,
            GetROS2LifecycleStatesTool,
            GetROS2TFTreeTool,
            ListROS2InterfacesTool,
        )


__all__ = [
    "GetROS2DiagnosticsTool",
    "GetROS2LifecycleStatesTool",
    "GetROS2LogsTool",
    "GetROS2NodeInfoTool",
    "GetROS2ParametersTool",
    "GetROS2TFTreeTool",
    "GetROS2TopicInfoTool",
    "ListROS2InterfacesTool",
    "ListROS2NodesTool",
    "MeasureROS2TopicRateTool",
    "ROS2DiagnosticsToolkit",
]
