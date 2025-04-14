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

import logging
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

import rclpy
import rclpy.action
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.topic_endpoint_info import TopicEndpointInfo

from rai.communication.ros2.api.conversion import import_message_from_str


@runtime_checkable
class IROS2Message(Protocol):
    __slots__: list

    def get_fields_and_field_types(self) -> dict: ...


class BaseROS2API:
    @staticmethod
    def adapt_requests_to_offers(publisher_info: List[TopicEndpointInfo]) -> QoSProfile:
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
                logging.warning(
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
                logging.warning(
                    "Some, but not all, publishers are offering TRANSIENT_LOCAL durability. "
                    "Falling back to VOLATILE as it will connect to all publishers. "
                    "Previously-published latched messages will not be retrieved."
                )
            request_qos.durability = DurabilityPolicy.VOLATILE

        return request_qos

    @staticmethod
    def build_ros2_msg(
        msg_type: str | type[rclpy.node.MsgType], msg_args: Dict[str, Any]
    ) -> object:
        """Build a ROS2 message instance from string or MsgType and content dictionary."""
        if isinstance(msg_type, str):
            msg_cls = import_message_from_str(msg_type)
        else:
            msg_cls = msg_type
        msg = msg_cls()
        rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
        return msg

    @staticmethod
    def build_ros2_service_request(
        service_type: str, service_request_args: Dict[str, Any]
    ) -> Tuple[object, Type[Any]]:
        msg_cls = import_message_from_str(service_type)
        msg = msg_cls.Request()
        rosidl_runtime_py.set_message.set_message_fields(msg, service_request_args)
        return msg, msg_cls  # type: ignore
