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

from typing import Any

from rai.communication.ros2.api import ROS2ServiceAPI
from rai.communication.ros2.messages import ROS2Message


class ROS2ServiceMixin:
    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        if not hasattr(self, "_service_api"):
            raise AttributeError(
                f"{self.__class__.__name__} instance must have an attribute '_service_api' of type ROS2ServiceAPI"
            )
        self._service_api: ROS2ServiceAPI
        if not isinstance(self._service_api, ROS2ServiceAPI):
            raise AttributeError(
                f"{self.__class__.__name__} instance must have an attribute '_service_api' of type ROS2ServiceAPI"
            )

    def service_call(
        self,
        message: ROS2Message,
        target: str,
        timeout_sec: float = 5.0,
        *,
        msg_type: str,
        **kwargs: Any,
    ) -> ROS2Message:
        msg = self._service_api.call_service(
            service_name=target,
            service_type=msg_type,
            request=message.payload,
            timeout_sec=timeout_sec,
        )
        return ROS2Message(
            payload=msg, metadata={"msg_type": str(type(msg)), "service": target}
        )
