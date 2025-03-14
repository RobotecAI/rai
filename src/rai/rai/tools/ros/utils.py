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
from typing import Type, Union, cast

import cv2
import numpy as np
import rclpy
import rclpy.node
import rclpy.time
import sensor_msgs.msg
from cv_bridge import CvBridge
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from rosidl_parser.definition import NamespacedType
from rosidl_runtime_py.import_message import import_message_from_namespaced_type
from rosidl_runtime_py.utilities import get_namespaced_type
from tf2_ros import Buffer, TransformListener, TransformStamped


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


def convert_ros_img_to_base64(msg: sensor_msgs.msg.Image) -> str:
    bridge = CvBridge()
    cv_image = cast(cv2.Mat, bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))  # type: ignore
    if cv_image.shape[-1] == 4:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        return base64.b64encode(bytes(cv2.imencode(".png", cv_image)[1])).decode(
            "utf-8"
        )
    elif cv_image.shape[-1] == 1:
        cv_image = cv2.cvtColor(cv_image, cv2.GRAY2RGB)
        return base64.b64encode(bytes(cv2.imencode(".png", cv_image)[1])).decode(
            "utf-8"
        )

    else:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_data = cv2.imencode(".png", cv_image)[1].tostring()  # type: ignore
        return base64.b64encode(image_data).decode("utf-8")  # type: ignore


# Copied from https://github.com/ros2/rclpy/blob/jazzy/rclpy/rclpy/wait_for_message.py, to support humble
def wait_for_message(
    msg_type,
    node: "Node",
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1,
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(
        msg_type, topic, lambda _: None, qos_profile=qos_profile
    )
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities("subscription")
        guards_ready = wait_set.get_ready_entities("guard_condition")

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None


def get_transform(
    node: rclpy.node.Node, target_frame: str, source_frame: str
) -> TransformStamped:
    tf_buffer = Buffer(node=node)
    tf_listener = TransformListener(tf_buffer, node)

    transform = None
    while transform is None:
        rclpy.spin_once(node, timeout=0.5)
        if tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
            transform = tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
    tf_listener.unregister()
    return transform
