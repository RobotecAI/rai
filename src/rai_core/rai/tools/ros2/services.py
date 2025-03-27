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

from pydantic import BaseModel, Field

from rai.communication.ros2.connectors import ROS2ARIMessage
from rai.tools.ros2.base import BaseROS2Tool


class CallROS2ServiceToolInput(BaseModel):
    service_name: str = Field(..., description="The service to call")
    service_type: str = Field(..., description="The type of the service")
    service_args: Dict[str, Any] = Field(
        ..., description="The arguments to pass to the service"
    )


class CallROS2ServiceTool(BaseROS2Tool):
    name: str = "call_ros2_service"
    description: str = "Call a ROS2 service"
    args_schema: Type[CallROS2ServiceToolInput] = CallROS2ServiceToolInput

    def _run(
        self, service_name: str, service_type: str, service_args: Dict[str, Any]
    ) -> str:
        if not self.is_writable(service_name):
            raise ValueError(f"Service {service_name} is not writable")
        message = ROS2ARIMessage(payload=service_args)
        response = self.connector.service_call(
            message, service_name, msg_type=service_type
        )
        return str(
            {
                "payload": response.payload,
                "metadata": response.metadata,
            }
        )
