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

from typing import Optional

from pydantic import Field

from rai.communication import BaseConnector, BaseMessage


class ARIMessage(BaseMessage):
    pass


# TODO: Move this to ros2 module
class ROS2RRIMessage(ARIMessage):
    ros_message_type: str = Field(
        description="The string representation of the ROS message type (e.g. 'std_msgs/String')"
    )
    python_message_class: Optional[type] = Field(
        description="The Python class of the ROS message type", default=None
    )


class ARIConnector(BaseConnector[ARIMessage]):
    """
    Base class for Robot-Robot Interaction (RRI) connectors.
    """
