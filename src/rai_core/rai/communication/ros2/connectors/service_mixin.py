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

from typing import Any, Union

from rai.communication.ros2.api import IROS2Message, ROS2ServiceAPI
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

    def release_client(self, service_name: str) -> bool:
        return self._service_api.release_client(service_name)

    def service_call(
        self,
        message: Union[ROS2Message, IROS2Message],
        target: str,
        timeout_sec: float = 5.0,
        *,
        msg_type: str | None = None,
        reuse_client: bool = True,
        **kwargs: Any,
    ) -> Union[ROS2Message, IROS2Message]:
        """Call a ROS2 service.

        Provides dual support:
        - LLM support: ROS2Message with dict payload + msg_type string
        - Typed (human-friendly): Direct service Request class instance (msg_type inferred)

        Parameters
        ----------
        message : Union[ROS2Message, IROS2Message]
            The service request. Can be:
            - ROS2Message with dict payload (requires msg_type)
            - Service Request class instance (e.g., SetBool.Request(), msg_type optional)
        target : str
            The target service name.
        timeout_sec : float, optional
            Timeout in seconds, by default 5.0.
        msg_type : str | None, optional
            The ROS2 service type string (e.g., 'std_srvs/srv/SetBool').
            Required if message is ROS2Message (dict), optional if message is Request instance.
        reuse_client : bool, optional
            Whether to reuse cached client, by default True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Union[ROS2Message, IROS2Message]
            Service response. Returns ROS2Message if input was ROS2Message,
            otherwise returns Response class instance.
        """
        # Check if message is ROS2Message (dict) or Request class instance
        if isinstance(message, ROS2Message):
            # LLM support path: dict-based message
            if msg_type is None:
                raise ValueError(
                    "msg_type must be provided when message is ROS2Message (dict-based). "
                    "Either pass msg_type or use a Request class instance directly."
                )
            request = message.payload
        else:
            # Typed (human-friendly) path: Request class instance
            if msg_type is not None:
                # Allow explicit msg_type but warn it's redundant
                pass  # Will be ignored in call_service if introspection succeeds
            request = message

        response = self._service_api.call_service(
            service_name=target,
            service_type=msg_type,
            request=request,
            timeout_sec=timeout_sec,
            reuse_client=reuse_client,
        )

        # Return type matches input: ROS2Message for dicts, Response instance for classes
        if isinstance(message, ROS2Message):
            return ROS2Message(
                payload=response,
                metadata={"msg_type": str(type(response)), "service": target},
            )
        else:
            return response
