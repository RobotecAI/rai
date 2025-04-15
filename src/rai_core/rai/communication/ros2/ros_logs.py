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

import logging
from collections import deque
from typing import Deque, Literal, Optional

import rcl_interfaces.msg
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

try:
    import rai_interfaces.srv
except ImportError:
    logging.warning(
        "rai_interfaces is not installed, RaiStateLogsParser will not work."
    )
from rai.communication.ros2.ros_async import get_future_result


class BaseLogsParser:
    def summarize(self) -> str:
        raise NotImplementedError


def create_logs_parser(
    parser_type: Literal["rai_state_logs", "llm"],
    node: rclpy.node.Node,
    llm: Optional[BaseChatModel] = None,
    callback_group: Optional[rclpy.callback_groups.ReentrantCallbackGroup] = None,
    bufsize: Optional[int] = 100,
) -> BaseLogsParser:
    if parser_type == "rai_state_logs":
        return RaiStateLogsParser(node, callback_group)
    elif parser_type == "llm":
        if any([v is None for v in [llm, callback_group, bufsize]]):
            raise ValueError("Must provide llm, callback_group, and bufsize")
        return LlmRosoutParser(llm, node, callback_group, bufsize)
    else:
        raise ValueError(f"Unknown summarizer type: {parser_type}")


class RaiStateLogsParser(BaseLogsParser):
    """Use rai_state_logs node to get logs"""

    SERVICE_NAME = "/get_log_digest"

    def __init__(
        self, node: rclpy.node.Node, callback_group: rclpy.callback_groups.CallbackGroup
    ) -> None:
        self.node = node

        self.rai_state_logs_client = node.create_client(
            rai_interfaces.srv.StringList,
            self.SERVICE_NAME,
            callback_group=callback_group,
        )
        while not self.rai_state_logs_client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info(
                f"'{self.SERVICE_NAME}' service is not available, waiting again..."
            )

    def summarize(self) -> str:
        request = rai_interfaces.srv.StringList.Request()
        future = self.rai_state_logs_client.call_async(request)

        response: Optional[rai_interfaces.srv.StringList.Response] = get_future_result(
            future
        )

        if response is None or not response.success:
            self.node.get_logger().error(f"'{self.SERVICE_NAME}' service call failed")
            return ""
        self.node.get_logger().info(
            f"'{self.SERVICE_NAME}' service call done. Response: {response.success=}, {response.string_list=}"
        )
        return "\n".join(response.string_list)


class LlmRosoutParser(BaseLogsParser):
    """Bufferize `/rosout` and summarize is with LLM"""

    def __init__(
        self,
        llm: BaseChatModel,
        node: rclpy.node.Node,
        callback_group: rclpy.callback_groups.CallbackGroup,
        bufsize: int = 100,
    ):
        self.bufsize = bufsize
        self._buffer: Deque[str] = deque()
        self.template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Shorten the following log keeping its format - for example merge similar or repeating lines",
                ),
                ("human", "{rosout}"),
            ]
        )
        self.llm = self.template | llm

        self.node = node

        rosout_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.rosout_subscription = self.init_rosout_subscription(
            self.node, callback_group, rosout_qos
        )

    def init_rosout_subscription(
        self,
        node: rclpy.node.Node,
        callback_group: rclpy.callback_groups.CallbackGroup,
        qos_profile: rclpy.qos.QoSProfile,
    ) -> rclpy.subscription.Subscription:
        return node.create_subscription(
            rcl_interfaces.msg.Log,
            "/rosout",
            callback=self.rosout_callback,
            callback_group=callback_group,
            qos_profile=qos_profile,
        )

    def rosout_callback(self, msg: rcl_interfaces.msg.Log):
        self.node.get_logger().debug(f"Received rosout: {msg}")

        if "rai_node" in msg.name:
            return

        self.append(f"[{msg.stamp.sec}][{msg.name}]:{msg.msg}")

    def clear(self):
        self._buffer.clear()

    def append(self, line: str):
        self._buffer.append(line)
        if len(self._buffer) > self.bufsize:
            self._buffer.popleft()

    def get_raw_logs(self, last_n: int = 30) -> str:
        return "\n".join(list(self._buffer)[-last_n:])

    def summarize(self):
        if len(self._buffer) == 0:
            return "No logs"
        buffer = self.get_raw_logs()
        self.clear()
        response = self.llm.invoke({"rosout": buffer})
        return str(response.content)
