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

from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field

from rai.communication.ros2.runtime import message_class
from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit, from_dict, to_dict


class ROS2ServicesToolkit(BaseROS2Toolkit):
    name: str = "ROS2ServicesToolkit"
    description: str = "A toolkit for ROS2 services"

    def get_tools(self) -> List[BaseTool]:
        return self._build(CallROS2ServiceTool, GetROS2ServicesNamesAndTypesTool)


class GetROS2ServicesNamesAndTypesTool(BaseROS2Tool):
    """Lists the services the tool may call (all but forbidden, or the writable ones)."""

    default_effect = Effect.READ
    name: str = "get_ros2_services_names_and_types"
    description: str = "Get the names and types of all ROS2 services"

    async def _acall(self) -> str:
        return "\n".join(
            stringify_dict({"service": service, "type": types})
            for service, types in self.connector.get_services_names_and_types()
            if self.is_writable(service, service=True)
        )


class CallROS2ServiceToolInput(BaseModel):
    service_name: str = Field(min_length=1, description="The service to call")
    service_type: str = Field(min_length=1, description="The type of the service")
    service_args: Optional[Dict[str, Any]] = Field(
        default={},
        description="A dictionary mapping each field name of the service request message to its value. For example, for std_srvs/srv/SetBool use {'data': True}.",
    )
    timeout_sec: float = Field(
        default=5.0,
        gt=0,
        description="The timeout for the service call in seconds",
    )


class CallROS2ServiceTool(BaseROS2Tool):
    default_effect = Effect.WRITE
    target_args = {"service_name": "service"}
    name: str = "call_ros2_service"
    description: str = "Call a ROS2 service"
    args_schema: Type[CallROS2ServiceToolInput] = CallROS2ServiceToolInput

    async def _acall(
        self,
        service_name: str,
        service_type: str,
        service_args: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 5.0,
    ) -> str:
        self.check_writable(service_name, "service")
        srv_cls = message_class(service_type)
        response = await self.connector.call(
            service_name,
            from_dict(srv_cls.Request, service_args),
            srv_type=srv_cls,
            timeout=timeout_sec,
        )
        return str({"payload": to_dict(response)})
