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

from typing import Annotated, List, Optional, Tuple

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
        The names (topics/actions/services) that can be written. If the list is not provided, all topics/actions/services can be written.
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

    def _check_permission_and_include(
        self, name: str, check_readable: bool = True
    ) -> Tuple[bool, bool, bool]:
        """
        Check permissions and determine if resource should be included.

        Args:
            name: Resource name (topic/service/action)
            check_readable: If False, only checks writable (for services/actions).
                            If True, checks both readable and writable (for topics).

        Returns:
            (should_include, is_readable, is_writable)
        """
        # Skip forbidden resources
        if self.forbidden is not None and name in self.forbidden:
            return (False, False, False)

        is_readable_resource = self.is_readable(name) if check_readable else False
        is_writable_resource = self.is_writable(name)

        # Determine if resource should be included based on whitelist state
        # If only readable is set: include only if resource is in readable list
        # If only writable is set: include only if resource is in writable list
        # If both are set: include if resource is in readable OR writable list
        # If neither is set: include all resources (except forbidden)
        should_include = True
        if check_readable and self.readable is not None and self.writable is not None:
            # Both whitelists are set, resource must be in at least one
            should_include = is_readable_resource or is_writable_resource
        elif check_readable and self.readable is not None:
            # Only readable whitelist is set, resource must be readable
            should_include = is_readable_resource
        elif self.writable is not None:
            # Only writable whitelist is set, resource must be writable
            should_include = is_writable_resource

        return (should_include, is_readable_resource, is_writable_resource)

    def _categorize(self, is_readable: bool, is_writable: bool) -> Optional[str]:
        """
        Categorize resource into readable, writable, or both.

        Returns:
            Category name: "readable_and_writable", "readable", "writable", or None
        """
        if is_readable and is_writable:
            return "readable_and_writable"
        elif is_readable:
            return "readable"
        elif is_writable:
            return "writable"
        return None


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
        The names (topics/actions/services) that can be written. If the list is not provided, all topics/actions/services can be written.
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
            If the list is not provided, all topics/actions/services can be written.""",
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
