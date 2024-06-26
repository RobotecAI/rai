import base64
import logging
import subprocess
from typing import Any, Callable, Union, cast

import cv2
import rclpy
import rclpy.qos
from cv_bridge import CvBridge
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from sensor_msgs.msg import Image


def wait_for_message(
    msg_type: type,
    node: Node,
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait: int = -1,
) -> tuple[bool, Any]:
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


class SingleMessageGrabber:
    def __init__(
        self,
        topic: str,
        message_type: type,
        timeout_sec: int,
        logging_level: int = logging.INFO,
        postprocess: Callable[[Any], Any] = lambda x: x,
    ):
        self.topic = topic
        self.message_type = message_type
        self.timeout_sec = timeout_sec
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)
        self.initialized = False
        self.postprocess = getattr(self, "postprocess", postprocess)

    def grab_message(self) -> Any:
        if not rclpy.ok():
            rclpy.init()
            self.initialized = True

        node = rclpy.create_node(self.__class__.__name__ + "_node")  # type: ignore
        qos_profile = rclpy.qos.qos_profile_sensor_data

        success, msg = wait_for_message(
            self.message_type,
            node,
            self.topic,
            qos_profile=qos_profile,
            time_to_wait=self.timeout_sec,
        )

        if success:
            self.logger.info(
                f"Received message of type {self.message_type.__class__.__name__} from topic {self.topic}"  # type: ignore
            )
        else:
            self.logger.error(
                f"Failed to receive message of type {self.message_type.__class__.__name__} from topic {self.topic}"  # type: ignore
            )

        node.destroy_node()
        if self.initialized:
            rclpy.shutdown()
        return msg

    def get_data(self) -> Any:
        msg = self.grab_message()
        return self.postprocess(msg)


class SingleImageGrabber(SingleMessageGrabber):
    def __init__(
        self, topic: str, timeout_sec: int = 10, logging_level: int = logging.INFO
    ):
        self.topic = topic
        super().__init__(
            topic=topic,
            message_type=Image,
            timeout_sec=timeout_sec,
            logging_level=logging_level,
        )

    def postprocess(self, msg: Image) -> str:
        bridge = CvBridge()
        cv_image = cast(cv2.Mat, bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))  # type: ignore
        if cv_image.shape[-1] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
            base64_image = base64.b64encode(
                bytes(cv2.imencode(".png", cv_image)[1])
            ).decode("utf-8")
            return base64_image

        image_data = cv2.imencode(".png", cv_image)[1].tostring()  # type: ignore
        base64_image = base64.b64encode(image_data).decode("utf-8")  # type: ignore
        return base64_image


class ReadAvailableTopics:
    def __init__(self, logging_level: int = logging.INFO):
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)

    def get_data(self):
        command = "ros2 topic list"
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        return output


class ReadAvailableNodes:
    def __init__(self, logging_level: int = logging.INFO):
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)

    def get_data(self):
        command = "ros2 node list"
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        return output


class ReadAvailableServices:
    def __init__(self, logging_level: int = logging.INFO):
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)

    def get_data(self):
        command = (
            "for service in $(ros2 service list); do\n"
            'echo -n "service $service "\n'
            'echo "type: $(ros2 service type $service)"\n'
            "done"
        )
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        return output


class ReadAvailableActions:
    def __init__(self, logging_level: int = logging.INFO):
        self.logging_level = logging_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.logging_level)

    def get_data(self):
        command = "ros2 action list"
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        return output
