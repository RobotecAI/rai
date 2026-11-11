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

"""Every Jazzy (rclpy 7.1) vs Lyrical (rclpy 10) difference RAI relies on.

Branches are feature probes, never ROS_DISTRO checks (rai3-plan/01 section 10).
Differences RAI avoids instead of wrapping: ``Client.call`` timeouts (RAI only
uses ``call_async``), ``rclpy.init`` return values, and the blocking
``cancel_goal``/``get_result``/``send_goal`` action calls.
"""

import copy
from typing import Any

import rosidl_runtime_py.set_message
from geometry_msgs.msg import PoseArray
from rclpy.client import Client
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.service import Service
from rclpy.subscription import Subscription

try:
    from rclpy.endpoint_info import TopicEndpointInfo  # Lyrical
except ImportError:
    from rclpy.topic_endpoint_info import TopicEndpointInfo  # Jazzy

try:
    from rclpy.experimental import EventsExecutor  # Jazzy >= 7.1.11, Lyrical
except ImportError:
    EventsExecutor = None

# ActionServer's result_timeout default is 900 s on Jazzy and 10 s on Lyrical.
ACTION_RESULT_TIMEOUT_S = 900


def _set_message_fields_mutates() -> bool:
    values: dict[str, Any] = {"poses": [{}]}
    rosidl_runtime_py.set_message.set_message_fields(PoseArray(), values)
    return not isinstance(values["poses"][0], dict)


# Jazzy's set_message_fields replaces nested dicts of the input in place;
# Lyrical deep-copies the input itself.
_SET_FIELDS_MUTATES = _set_message_fields_mutates()


def set_message_fields(msg: Any, values: dict[str, Any]) -> Any:
    """Fill msg from a dict without touching the caller's dict."""
    if _SET_FIELDS_MUTATES:
        values = copy.deepcopy(values)
    rosidl_runtime_py.set_message.set_message_fields(msg, values)
    return msg


def destroy(node: Node, entity: Any) -> None:
    """Destroy an entity and remove it from the node.

    On Jazzy ``entity.destroy()`` leaves the node's lists stale, so the
    node.destroy_* methods are used; they also work on Lyrical. Action clients
    and servers remove themselves. Lyrical's rclpy.shutdown() already destroyed
    every node, so destroying twice must be harmless.
    """
    destroyers = {
        Publisher: node.destroy_publisher,
        Subscription: node.destroy_subscription,
        Client: node.destroy_client,
        Service: node.destroy_service,
    }
    try:
        for kind, destroyer in destroyers.items():
            if isinstance(entity, kind):
                destroyer(entity)
                return
        entity.destroy()
    except Exception:  # an invalid handle after rclpy.shutdown()
        pass


__all__ = [
    "ACTION_RESULT_TIMEOUT_S",
    "EventsExecutor",
    "TopicEndpointInfo",
    "destroy",
    "set_message_fields",
]
