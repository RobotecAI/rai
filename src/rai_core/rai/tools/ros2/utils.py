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

from typing import Any, OrderedDict

import rosidl_runtime_py.convert
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities


def ros2_message_to_dict(message: Any) -> OrderedDict[str, Any]:
    """Convert any ROS2 message into a dictionary.

    Args:
        message: A ROS2 message instance

    Returns:
        A dictionary representation of the message

    Raises:
        TypeError: If the input is not a valid ROS2 message
    """
    msg_dict: OrderedDict[str, Any] = rosidl_runtime_py.convert.message_to_ordereddict(
        message
    )  # type: ignore
    return msg_dict
