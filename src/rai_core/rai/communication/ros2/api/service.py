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

import os
import uuid
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
)

import rclpy
import rclpy.action
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
from rclpy.client import Client
from rclpy.service import Service

from rai.communication.ros2.api.base import (
    BaseROS2API,
)
from rai.communication.ros2.api.conversion import import_message_from_str


class ROS2ServiceAPI(BaseROS2API):
    """Handles ROS 2 service operations including calling services."""

    def __init__(self, node: rclpy.node.Node) -> None:
        self.node = node
        self._logger = node.get_logger()
        self._services: Dict[str, Service] = {}
        self._persistent_clients: Dict[str, Client] = {}
        self._persistent_clients_lock = Lock()

    def release_client(self, service_name: str) -> bool:
        with self._persistent_clients_lock:
            return self._persistent_clients.pop(service_name, None) is not None

    def call_service(
        self,
        service_name: str,
        service_type: str,
        request: Any,
        timeout_sec: float = 5.0,
        *,
        reuse_client: bool = True,
    ) -> Any:
        """
        Call a ROS 2 service.

        Args:
            service_name: Fully-qualified service name.
            service_type: ROS 2 service type string (e.g., 'std_srvs/srv/SetBool').
            request: Request payload dict.
            timeout_sec: Seconds to wait for availability/response.
            reuse_client: Reuse a cached client. Client creation is synchronized; set
                False to create a new client per call.

        Returns:
            Response message instance.

        Raises:
            ValueError: Service not available within the timeout.
            AttributeError: Service type or request cannot be constructed.

        Note:
            With reuse_client=True, access to the cached client (including the
            service call) is serialized by a lock, preventing concurrent calls
            through the same client. Use reuse_client=False for per-call clients
            when concurrent service calls are required.
        """
        srv_msg, srv_cls = self.build_ros2_service_request(service_type, request)

        def _call_service(client: Client, timeout_sec: float) -> Any:
            is_service_available = client.wait_for_service(timeout_sec=timeout_sec)
            if not is_service_available:
                raise ValueError(
                    f"Service {service_name} not ready within {timeout_sec} seconds. "
                    "Try increasing the timeout or check if the service is running."
                )
            if os.getenv("ROS_DISTRO") == "humble":
                return client.call(srv_msg)
            else:
                return client.call(srv_msg, timeout_sec=timeout_sec)

        if reuse_client:
            with self._persistent_clients_lock:
                client = self._persistent_clients.get(service_name, None)
                if client is None:
                    client = self.node.create_client(srv_cls, service_name)  # type: ignore
                    self._persistent_clients[service_name] = client
                return _call_service(client, timeout_sec)
        else:
            client = self.node.create_client(srv_cls, service_name)  # type: ignore
            return _call_service(client, timeout_sec)

    def get_service_names_and_types(self) -> List[Tuple[str, List[str]]]:
        return self.node.get_service_names_and_types()

    def create_service(
        self,
        service_name: str,
        service_type: str,
        callback: Callable[[Any, Any], Any],
        **kwargs,
    ) -> str:
        srv_cls = import_message_from_str(service_type)
        service = self.node.create_service(srv_cls, service_name, callback, **kwargs)
        handle = str(uuid.uuid4())
        self._services[handle] = service
        return handle
