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

import base64
import functools
from typing import Any, OrderedDict, Type

import cv2
import numpy as np
import rosidl_runtime_py.convert
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
import sensor_msgs.msg
from rosidl_parser.definition import NamespacedType
from rosidl_runtime_py.import_message import import_message_from_namespaced_type
from rosidl_runtime_py.utilities import get_namespaced_type

from rai.communication.ros2.images import compressed_imgmsg_to_numpy, imgmsg_to_numpy


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


@functools.cache
def import_message_from_str(msg_type: str) -> Type[object]:
    msg_namespaced_type: NamespacedType = get_namespaced_type(msg_type)
    return import_message_from_namespaced_type(msg_namespaced_type)


def convert_ros_img_to_ndarray(
    msg: sensor_msgs.msg.Image, encoding: str = ""
) -> np.ndarray:
    if encoding == "":
        encoding = msg.encoding.lower()

    if encoding == "rgb8":
        image_data = np.frombuffer(msg.data, np.uint8)
        image = image_data.reshape((msg.height, msg.width, 3))
    elif encoding == "bgr8":
        image_data = np.frombuffer(msg.data, np.uint8)
        image = image_data.reshape((msg.height, msg.width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding == "rgba8":
        image_data = np.frombuffer(msg.data, np.uint8)
        image = image_data.reshape((msg.height, msg.width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif encoding == "mono8":
        image_data = np.frombuffer(msg.data, np.uint8)
        image = image_data.reshape((msg.height, msg.width))
    elif encoding == "16uc1":
        image_data = np.frombuffer(msg.data, np.uint16)
        image = image_data.reshape((msg.height, msg.width))
    elif encoding == "32fc1":
        image_data = np.frombuffer(msg.data, np.float32)  # Handle 32-bit float
        image = image_data.reshape((msg.height, msg.width))
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return image


def convert_ros_img_to_base64(
    msg: sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage,
) -> str:
    """Encode the image as a base64 PNG."""
    if isinstance(msg, sensor_msgs.msg.Image):
        bgr = imgmsg_to_numpy(msg, "bgr8")
    elif isinstance(msg, sensor_msgs.msg.CompressedImage):
        bgr = compressed_imgmsg_to_numpy(msg, "bgr8")
    else:
        raise ValueError(f"Unsupported message type: {type(msg)}")
    return base64.b64encode(cv2.imencode(".png", bgr)[1].tobytes()).decode("utf-8")
