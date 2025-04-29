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

import importlib
from typing import Any

import rosidl_runtime_py

from rai.communication.ros2.api.conversion import import_message_from_str

from ..base import ROS2BaseModel


def to_ros2_msg(base_model: ROS2BaseModel) -> Any:
    if not isinstance(base_model, ROS2BaseModel):
        raise TypeError(f"Expected Ros2BaseModel, got {type(base_model)}")
    msg_name = base_model.get_msg_name()
    ros2_msg_cls = import_message_from_str(msg_name)
    msg_args = base_model.model_dump()
    ros2_msg = ros2_msg_cls()
    rosidl_runtime_py.set_message.set_message_fields(ros2_msg, msg_args)
    return ros2_msg


def from_ros2_msg(msg: Any) -> ROS2BaseModel:
    if not hasattr(msg, "__class__"):
        raise TypeError(f"Expected ROS2 message, got {type(msg)}")
    msg_name = msg.__class__.__name__
    types_module = importlib.import_module("rai.types")
    try:
        base_model_cls: ROS2BaseModel = getattr(types_module, msg_name)
    except AttributeError:
        raise ImportError(
            f"Could not find ROS2BaseModel class for {msg_name} in rai.types module"
        )
    msg_dict = rosidl_runtime_py.message_to_ordereddict(msg)
    return base_model_cls.model_validate(msg_dict)
