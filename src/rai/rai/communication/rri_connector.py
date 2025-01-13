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

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class RRIMessage(BaseModel):
    payload: Any = Field(description="The payload of the message")


class ROS2RRIMessage(RRIMessage):
    ros_message_type: str = Field(
        description="The string representation of the ROS message type (e.g. 'std_msgs/String')"
    )
    python_message_class: Optional[type] = Field(
        description="The Python class of the ROS message type", default=None
    )


class RRIConnector(ABC):
    """
    Base class for Robot-Robot Interaction (RRI) connectors.
    """

    @abstractmethod
    def send_message(self, message: Any, target: str):
        pass

    @abstractmethod
    def receive_message(self, source: str, timeout_sec: float = 1.0) -> RRIMessage:
        pass

    @abstractmethod
    def service_call(
        self, message: RRIMessage, target: str, timeout_sec: float = 1.0
    ) -> RRIMessage:
        pass

    @abstractmethod
    def start_action(
        self,
        action: RRIMessage,
        target: str,
        on_feedback: Callable[[Any], None],
        on_done: Callable[[Any], None],
        timeout_sec: float = 1.0,
    ) -> str:
        pass
