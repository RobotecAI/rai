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
#

import base64
import logging
import subprocess
from typing import Any, Callable, Dict, List, Literal, Sequence, Union, cast

import cv2
import rclpy
import rclpy.qos
from cv_bridge import CvBridge
from deprecated import deprecated
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from rclpy.duration import Duration
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSLivelinessPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from rai.messages import ToolMultimodalMessage


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


@deprecated(reason="Multimodal images are handled using rai.messages.multimodal")
def images_to_vendor_format(images: List[str], vendor: str) -> List[Dict[str, Any]]:
    if vendor == "openai":
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                },
            }
            for image in images
        ]
    else:
        raise ValueError(f"Vendor {vendor} not supported")


@deprecated(reason="Running tool is langchain.agent based now")
def run_tool_call(
    tool_call: ToolCall,
    tools: Sequence[BaseTool],
) -> Dict[str, Any] | Any:
    logger = logging.getLogger(__name__)
    selected_tool = {k.name: k for k in tools}[tool_call["name"]]

    try:
        if selected_tool.args_schema is not None:
            args = selected_tool.args_schema(**tool_call["args"]).dict()
        else:
            args = dict()
    except Exception as e:
        err_msg = f"Error in preparing arguments for {selected_tool.name}: {e}"
        logger.error(err_msg)
        return err_msg

    logger.info(f"Running tool: {selected_tool.name} with args: {args}")

    try:
        tool_output = selected_tool.run(args)
    except Exception as e:
        err_msg = f"Error in running tool {selected_tool.name}: {e}"
        logger.warning(err_msg)
        return err_msg

    logger.info(f"Successfully ran tool: {selected_tool.name}. Output: {tool_output}")
    return tool_output


@deprecated(reason="Running tool is langchain.agent based now")
def run_requested_tools(
    ai_msg: AIMessage,
    tools: Sequence[BaseTool],
    messages: List[BaseMessage],
    llm_type: Literal["openai", "bedrock"],
):
    internal_messages: List[BaseMessage] = []
    for tool_call in ai_msg.tool_calls:
        tool_output = run_tool_call(tool_call, tools)
        assert isinstance(tool_call["id"], str), "Tool output must have an id."
        if isinstance(tool_output, dict):
            tool_message = ToolMultimodalMessage(
                content=tool_output.get("content", "No response from the tool."),
                images=tool_output.get("images"),
                tool_call_id=tool_call["id"],
            )
            tool_message = tool_message.postprocess(format=llm_type)
        else:
            tool_message = [
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            ]
        if isinstance(tool_message, list):
            internal_messages.extend(tool_message)
        else:
            internal_messages.append(tool_message)

    # because we can't answer an aiMessage with an alternating sequence of tool and human messages
    # we sort the messages by type so that the tool messages are sent first
    # for more information see implementation of ToolMultimodalMessage.postprocess

    internal_messages.sort(key=lambda x: x.__class__.__name__, reverse=True)
    messages.extend(internal_messages)
    return messages


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
        self.postprocess = getattr(self, "postprocess", postprocess)

    def grab_message(self) -> Any:
        node = rclpy.create_node(self.__class__.__name__ + "_node")  # type: ignore
        qos_profile = rclpy.qos.qos_profile_sensor_data
        if (
            self.topic == "/map"
        ):  # overfitting to husarion TODO(maciejmajek): find a better way
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_ALL,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                lifespan=Duration(seconds=0),
                deadline=Duration(seconds=0),
                liveliness=QoSLivelinessPolicy.AUTOMATIC,
                liveliness_lease_duration=Duration(seconds=0),
            )
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
        else:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

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


class TF2Listener(Node):
    def __init__(self):
        super().__init__("tf2_listener")

        # Create a buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # This will store the transform when received
        self.transform = None

    def get_transform(self):
        try:
            # Lookup transform between base_link and map
            now = rclpy.time.Time()
            self.transform = self.tf_buffer.lookup_transform("map", "base_link", now)
        except Exception as ex:
            self.get_logger().debug(f"Could not transform: {ex}")


class TF2TransformFetcher:
    def get_data(self):
        node = TF2Listener()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        try:
            while rclpy.ok() and node.transform is None:
                node.get_transform()
                rclpy.spin_once(node, timeout_sec=1.0)
        except KeyboardInterrupt:
            pass

        transform = node.transform
        node.destroy_node()
        return transform
