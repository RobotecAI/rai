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

import importlib.util

if importlib.util.find_spec("rclpy") is None:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

import logging

from rai.agents import BaseAgent
from rai.communication.ros2 import ROS2Connector

from rai_interfaces.srv import EmbodimentInfo as EmbodimentInfoSrv
from rai_whoami.models import EmbodimentInfo


class ROS2EmbodimentInfoAgent(BaseAgent):
    def __init__(self, root_dir: str, service_name: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.root_dir = root_dir
        self.service_name = service_name
        self.connector = ROS2Connector()
        self.embodiment_info = EmbodimentInfo.from_directory(self.root_dir)

        self.connector.create_service(
            service_name=self.service_name,
            service_type="rai_interfaces/srv/EmbodimentInfo",
            on_request=self.service_callback,
        )
        self.logger.info(
            f"Embodiment info agent initialized with service name {service_name}"
        )

    def service_callback(
        self, request: EmbodimentInfoSrv.Request, response: EmbodimentInfoSrv.Response
    ):
        self.logger.info(f"Received request: {request}")
        response.rules = self.embodiment_info.rules
        response.capabilities = self.embodiment_info.capabilities
        response.behaviors = self.embodiment_info.behaviors
        response.description = self.embodiment_info.description
        response.images = self.embodiment_info.images
        self.logger.info("Sending response.")
        return response

    def run(self):
        pass

    def stop(self):
        self.connector.shutdown()
