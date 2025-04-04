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

from typing import List

from langchain_core.tools import BaseTool

from rai.tools.ros2.base import BaseROS2Toolkit


class ROS2Toolkit(BaseROS2Toolkit):
    def get_tools(self) -> List[BaseTool]:
        # lazy import to avoid circular import
        from rai.tools.ros2.generic import (
            ROS2ActionToolkit,
            ROS2ServicesToolkit,
            ROS2TopicsToolkit,
        )

        return [
            *ROS2TopicsToolkit(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ).get_tools(),
            *ROS2ServicesToolkit(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ).get_tools(),
            *ROS2ActionToolkit(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ).get_tools(),
        ]
