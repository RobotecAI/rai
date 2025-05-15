# Copyright (C) 2025 Robotec.AI
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

from typing import Annotated, List, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import ConfigDict

from rai.communication.ros2.connectors import ROS2Connector


class BaseROS2Tool(BaseTool):
    """
    Base class for all ROS2 tools.

    Attributes
    ----------
    connector : ROS2Connector
        The connector to the ROS 2 system.
    readable : Optional[List[str]]
        The topics that can be read. If the list is not provided, all topics can be read.
    writable : Optional[List[str]]
        The names (topics/actions/services) that can be written. If the list is not provided, all topics can be written.
    forbidden : Optional[List[str]]
        The names (topics/actions/services) that are forbidden to read and write.
    name : str
        The name of the tool.
    description : str
        The description of the tool.
    """

    connector: ROS2Connector
    readable: Optional[List[str]] = None
    writable: Optional[List[str]] = None
    forbidden: Optional[List[str]] = None

    name: str = ""
    description: str = ""

    def is_readable(self, topic: str) -> bool:
        if self.forbidden is not None and topic in self.forbidden:
            return False
        if self.readable is None:
            return True
        return topic in self.readable

    def is_writable(self, topic: str) -> bool:
        if self.forbidden is not None and topic in self.forbidden:
            return False
        if self.writable is None:
            return True
        return topic in self.writable


class BaseROS2Toolkit(BaseToolkit):
    """
    Base class for all ROS2 toolkits.

    Parameters
    ----------
    connector : ROS2Connector
        The connector to the ROS2 system.
    readable : Optional[List[str]]
        The topics that can be read. If the list is not provided, all topics can be read.
    writable : Optional[List[str]]
        The names (topics/actions/services) that can be written. If the list is not provided, all topics can be written.
    forbidden : Optional[List[str]]
        The names (topics/actions/services) that are forbidden to read and write.
    """

    connector: ROS2Connector
    readable: Optional[
        Annotated[
            List[str],
            """The topics that can be read.
            If the list is not provided, all topics can be read.""",
        ]
    ] = None
    writable: Optional[
        Annotated[
            List[str],
            """The names (topics/actions/services) that can be written.
            If the list is not provided, all topics can be written.""",
        ]
    ] = None
    forbidden: Optional[
        Annotated[
            List[str],
            """The names (topics/actions/services) that are forbidden to read and write.""",
        ]
    ] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
