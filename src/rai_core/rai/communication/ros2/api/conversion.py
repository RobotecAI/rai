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
from typing import Any, OrderedDict, Type, cast

import cv2
import numpy as np
import rosidl_runtime_py.convert
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
import sensor_msgs.msg
from cv_bridge import CvBridge
from rosidl_parser.definition import NamespacedType
from rosidl_runtime_py.import_message import import_message_from_namespaced_type
from rosidl_runtime_py.utilities import get_namespaced_type


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


def convert_ros_img_to_cv2mat(msg: sensor_msgs.msg.Image) -> cv2.typing.MatLike:
    bridge = CvBridge()
    cv_image = cast(cv2.Mat, bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))  # type: ignore
    if cv_image.shape[-1] == 4:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
    else:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image


def convert_ros_img_to_base64(
    msg: sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage,
) -> str:
    bridge = CvBridge()
    msg_type = type(msg)
    if msg_type == sensor_msgs.msg.Image:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    elif msg_type == sensor_msgs.msg.CompressedImage:
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
    else:
        raise ValueError(f"Unsupported message type: {msg_type}")

    if cv_image.shape[-1] == 4:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        return base64.b64encode(bytes(cv2.imencode(".png", cv_image)[1])).decode(
            "utf-8"
        )
    elif cv_image.shape[-1] == 1:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        return base64.b64encode(bytes(cv2.imencode(".png", cv_image)[1])).decode(
            "utf-8"
        )
    else:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_data = cv2.imencode(".png", cv_image)[1].tostring()  # type: ignore
        return base64.b64encode(image_data).decode("utf-8")  # type: ignore
