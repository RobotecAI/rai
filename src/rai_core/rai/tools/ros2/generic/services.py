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

from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field

from rai.communication.ros2 import ROS2Message
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit


class ROS2ServicesToolkit(BaseROS2Toolkit):
    name: str = "ROS2ServicesToolkit"
    description: str = "A toolkit for ROS2 services"

    def get_tools(self) -> List[BaseTool]:
        return [
            CallROS2ServiceTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2ServicesNamesAndTypesTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
        ]


class GetROS2ServicesNamesAndTypesToolInput(BaseModel):
    pass


class GetROS2ServicesNamesAndTypesTool(BaseROS2Tool):
    name: str = "get_ros2_services_names_and_types"
    description: str = "Get the names and types of all ROS2 services"
    args_schema: Type[GetROS2ServicesNamesAndTypesToolInput] = (
        GetROS2ServicesNamesAndTypesToolInput
    )

    def _run(self) -> str:
        services_and_types = self.connector.get_services_names_and_types()
        if all([self.readable is None, self.writable is None, self.forbidden is None]):
            response = [
                {"service": service, "type": type}
                for service, type in services_and_types
            ]
            return "\n".join([stringify_dict(service) for service in response])
        else:
            writable_services: List[Dict[str, Any]] = []

            for service, type in services_and_types:
                if self.is_writable(service):
                    writable_services.append({"service": service, "type": type})
                    continue

            text_response = "\n".join(
                [
                    stringify_dict(service_description)
                    for service_description in writable_services
                ]
            )
            return text_response


class CallROS2ServiceToolInput(BaseModel):
    service_name: str = Field(description="The service to call")
    service_type: str = Field(description="The type of the service")
    service_args: Optional[Dict[str, Any]] = Field(
        default={},
        description="A dictionary mapping each field name of the service request message to its value. For example, for std_srvs/srv/SetBool use {'data': True}.",
    )
    timeout_sec: float = Field(
        default=5.0,
        description="The timeout for the service call in seconds",
    )


class CallROS2ServiceTool(BaseROS2Tool):
    name: str = "call_ros2_service"
    description: str = "Call a ROS2 service"
    args_schema: Type[CallROS2ServiceToolInput] = CallROS2ServiceToolInput

    def _run(
        self,
        service_name: str,
        service_type: str,
        service_args: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 5.0,
    ) -> str:
        if not self.is_writable(service_name):
            raise ValueError(f"Service {service_name} is not writable")
        if service_args is None:
            service_args = {}
        message = ROS2Message(payload=service_args)
        response = self.connector.service_call(
            message, service_name, msg_type=service_type, timeout_sec=timeout_sec
        )
        return str(
            {
                "payload": response.payload,
                "metadata": response.metadata,
            }
        )
