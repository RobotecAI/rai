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

"""The base of the perception tools: READ tools on the detection service."""

from typing import Any, ClassVar, Mapping

from pydantic import Field
from rai.tools import Effect
from rai.tools.ros2.base import BaseROS2Tool
from sensor_msgs.msg import Image

from rai_perception.components.service_utils import (
    DETECTION_SERVICE,
    SERVICE_TIMEOUT_S,
    detect,
    service_available,
)


class PerceptionTool(BaseROS2Tool):
    """A READ tool that reads camera topics and calls the detection service.

    Service names are tool fields; the default is the generic "/detection".
    """

    default_effect = Effect.READ
    target_fields: ClassVar[Mapping[str, str]] = {"detection_service_name": "service"}

    detection_service_name: str = Field(default=DETECTION_SERVICE, exclude=True)
    box_threshold: float = Field(default=0.35, description="Box threshold for GDINO")
    text_threshold: float = Field(default=0.45, description="Text threshold for GDINO")
    service_timeout: float = Field(default=SERVICE_TIMEOUT_S, exclude=True)
    receive_timeout: float = Field(default=5.0, exclude=True)

    @property
    def service_names(self) -> list[str]:
        return [getattr(self, field) for field in self.target_fields]

    def check_service_dependencies(self) -> dict[str, bool]:
        """Each required service and whether it is in the ROS graph now."""
        return {s: service_available(self.connector, s) for s in self.service_names}

    async def receive(self, topic: str, msg_type: Any = Image) -> Any:
        self.check_readable(topic)
        return await self.connector.receive(
            topic, msg_type=msg_type, timeout=self.receive_timeout
        )

    async def detect(self, image: Image, classes: str) -> Any:
        return await detect(
            self.connector,
            image,
            classes,
            service=self.detection_service_name,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            timeout=self.service_timeout,
        )
