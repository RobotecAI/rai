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

import functools
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from action_msgs.msg import GoalStatus
from rai.tools.ros.utils import import_message_from_str
from rai.utils.ros import NodeDiscovery
from rai.utils.ros_async import get_future_result
from rclpy.action.client import ActionClient
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.topic_endpoint_info import TopicEndpointInfo


def ros2_build_msg(msg_type: str, msg_args: Dict[str, Any]) -> Tuple[object, Type]:
    """
    Import message and create it. Return both ready message and message class.

    msgs args can have two formats:
    { "goal" : {arg 1 : xyz, ... } or {arg 1 : xyz, ... }
    """

    msg_cls: Type = rosidl_runtime_py.utilities.get_interface(msg_type)
    msg = msg_cls.Goal()

    if "goal" in msg_args:
        msg_args = msg_args["goal"]
    rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
    return msg, msg_cls


class Ros2TopicsAPI:
    def __init__(
        self,
        node: rclpy.node.Node,
        callback_group: rclpy.callback_groups.CallbackGroup,
        ros_discovery_info: NodeDiscovery,
    ) -> None:
        self.node = node
        self.callback_group = callback_group
        self.last_subscription_msgs_buffer = dict()
        self.qos_profile_cache: Dict[str, QoSProfile] = dict()

        self.ros_discovery_info = ros_discovery_info

    def get_logger(self):
        return self.node.get_logger()

    def adapt_requests_to_offers(
        self, publisher_info: List[TopicEndpointInfo]
    ) -> QoSProfile:
        if not publisher_info:
            return QoSProfile(depth=1)

        num_endpoints = len(publisher_info)
        reliability_reliable_count = 0
        durability_transient_local_count = 0

        for endpoint in publisher_info:
            profile = endpoint.qos_profile
            if profile.reliability == ReliabilityPolicy.RELIABLE:
                reliability_reliable_count += 1
            if profile.durability == DurabilityPolicy.TRANSIENT_LOCAL:
                durability_transient_local_count += 1

        request_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            liveliness=LivelinessPolicy.AUTOMATIC,
        )

        # Set reliability based on publisher offers
        if reliability_reliable_count == num_endpoints:
            request_qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            if reliability_reliable_count > 0:
                self.get_logger().warning(
                    "Some, but not all, publishers are offering RELIABLE reliability. "
                    "Falling back to BEST_EFFORT as it will connect to all publishers. "
                    "Some messages from Reliable publishers could be dropped."
                )
            request_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Set durability based on publisher offers
        if durability_transient_local_count == num_endpoints:
            request_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        else:
            if durability_transient_local_count > 0:
                self.get_logger().warning(
                    "Some, but not all, publishers are offering TRANSIENT_LOCAL durability. "
                    "Falling back to VOLATILE as it will connect to all publishers. "
                    "Previously-published latched messages will not be retrieved."
                )
            request_qos.durability = DurabilityPolicy.VOLATILE

        return request_qos

    def create_subscription_by_topic_name(self, topic):
        if self.has_subscription(topic):
            self.get_logger().warning(
                f"Subscription to {topic} already exists. To override use destroy_subscription_by_topic_name first"
            )
            return

        msg_type = self.get_msg_type(topic)

        if topic not in self.qos_profile_cache:
            self.get_logger().debug(f"Getting qos profile for topic: {topic}")
            qos_profile = self.adapt_requests_to_offers(
                self.node.get_publishers_info_by_topic(topic)
            )
            self.qos_profile_cache[topic] = qos_profile
        else:
            self.get_logger().debug(f"Using cached qos profile for topic: {topic}")
            qos_profile = self.qos_profile_cache[topic]

        topic_callback = functools.partial(
            self.generic_state_subscriber_callback, topic
        )

        self.node.create_subscription(
            msg_type,
            topic,
            callback=topic_callback,
            callback_group=self.callback_group,
            qos_profile=qos_profile,
        )

    def get_msg_type(self, topic: str, n_tries: int = 5) -> Any:
        """Sometimes node fails to do full discovery, therefore we need to retry"""
        for _ in range(n_tries):
            if topic in self.ros_discovery_info.topics_and_types:
                msg_type = self.ros_discovery_info.topics_and_types[topic]
                return import_message_from_str(msg_type)
            else:
                # Wait for next discovery cycle
                self.get_logger().info(f"Waiting for topic: {topic}")
                if self.ros_discovery_info:
                    time.sleep(self.ros_discovery_info.period_sec)
                else:
                    time.sleep(1.0)
        raise KeyError(f"Topic {topic} not found")

    def set_ros_discovery_info(self, new_ros_discovery_info: NodeDiscovery):
        self.ros_discovery_info = new_ros_discovery_info

    def get_raw_message_from_topic(
        self, topic: str, timeout_sec: int = 5, topic_wait_sec: int = 2
    ) -> Any:
        self.get_logger().debug(f"Getting msg from topic: {topic}")

        ts = time.perf_counter()

        for _ in range(topic_wait_sec * 10):
            if topic not in self.ros_discovery_info.topics_and_types:
                time.sleep(0.1)
                continue
            else:
                break

        if topic not in self.ros_discovery_info.topics_and_types:
            raise KeyError(
                f"Topic {topic} not found. Available topics: {self.ros_discovery_info.topics_and_types.keys()}"
            )

        if topic in self.last_subscription_msgs_buffer:
            self.get_logger().info("Returning cached message")
            return self.last_subscription_msgs_buffer[topic]
        else:
            self.create_subscription_by_topic_name(topic)
            try:
                msg = self.last_subscription_msgs_buffer.get(topic, None)
                while msg is None and time.perf_counter() - ts < timeout_sec:
                    msg = self.last_subscription_msgs_buffer.get(topic, None)
                    self.get_logger().info("Waiting for message...")
                    time.sleep(0.1)

                success = msg is not None

                if success:
                    self.get_logger().debug(
                        f"Received message of type {type(msg)} from topic {topic}"
                    )
                    return msg
                else:
                    error = f"No message received in {timeout_sec} seconds from topic {topic}"
                    self.get_logger().error(error)
                    return error
            finally:
                self.destroy_subscription_by_topic_name(topic)

    def generic_state_subscriber_callback(self, topic_name: str, msg: Any):
        self.get_logger().debug(
            f"Received message of type {type(msg)} from topic {topic_name}"
        )
        self.last_subscription_msgs_buffer[topic_name] = msg

    def has_subscription(self, topic: str) -> bool:
        for sub in self.node._subscriptions:
            if sub.topic == topic:
                return True
        return False

    def destroy_subscription_by_topic_name(self, topic: str):
        self.last_subscription_msgs_buffer.clear()
        for sub in self.node._subscriptions:
            if sub.topic == topic:
                self.node.destroy_subscription(sub)


class Ros2ActionsAPI:
    def __init__(self, node: rclpy.node.Node):
        self.node = node

        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status: Optional[int] = None
        self.client: Optional[ActionClient] = None
        self.action_feedback: Optional[Any] = None

    def get_logger(self):
        return self.node.get_logger()

    def run_action(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        if not self.is_task_complete():
            raise AssertionError(
                "Another ros2 action is currently running and parallel actions are not supported. Please wait until the previous action is complete before starting a new one. You can also cancel the current one."
            )

        if action_name[0] != "/":
            action_name = "/" + action_name
            self.get_logger().info(f"Action name corrected to: {action_name}")

        try:
            goal_msg, msg_cls = ros2_build_msg(action_type, action_goal_args)
        except Exception as e:
            return f"Failed to build message: {e}"

        self.client = ActionClient(self.node, msg_cls, action_name)
        self.msg_cls = msg_cls

        retries = 0
        while not self.client.wait_for_server(timeout_sec=1.0):
            retries += 1
            if retries > 5:
                raise Exception(
                    f"Action server '{action_name}' is not available. Make sure `action_name` is correct..."
                )
            self.get_logger().info(
                f"'{action_name}' action server not available, waiting..."
            )

        self.get_logger().info(f"Sending action message: {goal_msg}")

        send_goal_future = self.client.send_goal_async(
            goal_msg, self._feedback_callback
        )
        self.get_logger().info("Action goal sent!")

        self.goal_handle = get_future_result(send_goal_future)

        if not self.goal_handle:
            raise Exception(f"Action '{action_name}' not sent to server")

        if not self.goal_handle.accepted:
            raise Exception(f"Action '{action_name}' not accepted by server")

        self.result_future = self.goal_handle.get_result_async()
        self.get_logger().info("Action sent!")
        return f"{action_name} started successfully with args: {action_goal_args}"

    def get_task_result(self) -> str:
        if not self.is_task_complete():
            return "Task is not complete yet"

        def parse_status(status: int) -> str:
            return {
                GoalStatus.STATUS_SUCCEEDED: "succeeded",
                GoalStatus.STATUS_ABORTED: "aborted",
                GoalStatus.STATUS_CANCELED: "canceled",
                GoalStatus.STATUS_ACCEPTED: "accepted",
                GoalStatus.STATUS_CANCELING: "canceling",
                GoalStatus.STATUS_EXECUTING: "executing",
                GoalStatus.STATUS_UNKNOWN: "unknown",
            }.get(status, "unknown")

        result = self.result_future.result()

        self.destroy_client()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            msg = f"Result succeeded: {result.result}"
            self.get_logger().info(msg)
            return msg
        else:
            str_status = parse_status(result.status)
            error_code_str = self.parse_error_code(result.result.error_code)
            msg = f"Result {str_status}, because of: error_code={result.result.error_code}({error_code_str}), error_msg={result.result.error_msg}"
            self.get_logger().info(msg)
            return msg

    def parse_error_code(self, code: int) -> str:
        code_to_name = {
            v: k for k, v in vars(self.msg_cls.Result).items() if isinstance(v, int)
        }
        return code_to_name.get(code, "UNKNOWN")

    def _feedback_callback(self, msg):
        self.get_logger().info(f"Received ros2 action feedback: {msg}")
        self.action_feedback = msg

    def is_task_complete(self):
        if not self.result_future:
            # task was cancelled or completed
            return True

        result = get_future_result(self.result_future, timeout_sec=0.10)
        if result is not None:
            self.status = result.status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().debug(
                    f"Task with failed with status code: {self.status}"
                )
                return True
        else:
            self.get_logger().info("There is no result")
            # Timed out, still processing, not complete yet
            return False

        self.get_logger().info("Task succeeded!")
        return True

    def cancel_task(self) -> Union[str, bool]:
        self.get_logger().info("Canceling current task.")
        try:
            if self.result_future and self.goal_handle:
                future = self.goal_handle.cancel_goal_async()
                result = get_future_result(future, timeout_sec=1.0)
                return "Failed to cancel result" if result is None else True
            return True
        finally:
            self.destroy_client()

    def destroy_client(self):
        if self.client:
            self.client.destroy()
