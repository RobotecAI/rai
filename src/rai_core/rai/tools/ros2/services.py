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

try:
    pass
except ImportError:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from rai.tools.utils import wrap_tool_input  # type: ignore


class CallROS2ServiceToolInput(BaseModel):
    service_name: str = Field(..., description="The service to call")
    service_type: str = Field(..., description="The type of the service")
    args: Dict[str, Any] = Field(
        ..., description="The arguments to pass to the service"
    )


class CallROS2ServiceTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "call_ros2_service"
    description: str = "Call a ROS2 service"
    args_schema: Type[CallROS2ServiceToolInput] = CallROS2ServiceToolInput

    @wrap_tool_input
    def _run(self, tool_input: CallROS2ServiceToolInput) -> str:
        message = ROS2ARIMessage(
            payload=tool_input.args, metadata={"msg_type": tool_input.service_type}
        )
        response = self.connector.service_call(message, tool_input.service_name)
        return str(
            {
                "payload": response.payload,
                "metadata": response.metadata,
            }
        )
