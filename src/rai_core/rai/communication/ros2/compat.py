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
from typing import Any, Optional

import rosidl_runtime_py.set_message
from geometry_msgs.msg import PoseArray
from rclpy.action.client import ClientGoalHandle
from rclpy.client import Client
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
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

RCLError = _rclpy.RCLError  # a RuntimeError; not re-exported by rclpy

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


def requesters_without_reply_reader(node: Node, service: str) -> set[bytes]:
    """Participants that write requests to service but whose reply readers this
    node has not discovered yet (empty when the RMW has no such topics).

    rmw_fastrtps drops a response whose reply reader its writer has not matched
    within the writer's max_blocking_time (100 ms; rmw_response.cpp), while the
    client's ``service_is_ready()`` only reflects the client's own matches.
    Discovery is asymmetric, so a server may see a request seconds before it
    sees the reader the response must go to. The request and reply DDS topics
    are ``rq<name>Request`` and ``rr<name>Reply``; an endpoint gid starts with
    its participant's 12-byte GUID prefix, and each client has one of each.
    """

    def per_participant(endpoints: list[Any]) -> dict[bytes, int]:
        counts: dict[bytes, int] = {}
        for endpoint in endpoints:
            prefix = bytes(endpoint.endpoint_gid)[:12]
            counts[prefix] = counts.get(prefix, 0) + 1
        return counts

    writers = per_participant(
        node.get_publishers_info_by_topic(f"rq{service}Request", no_mangle=True)
    )
    if not writers:
        return set()
    readers = per_participant(
        node.get_subscriptions_info_by_topic(f"rr{service}Reply", no_mangle=True)
    )
    return {p for p, n in writers.items() if readers.get(p, 0) < n}


def service_server_ids(node: Node, service: str) -> frozenset[bytes]:
    """The gids of the request readers of service, one per server: a changed
    set means a server (re)appeared. Empty when the RMW has no such topic."""
    return frozenset(
        bytes(endpoint.endpoint_gid)
        for endpoint in node.get_subscriptions_info_by_topic(
            f"rq{service}Request", no_mangle=True
        )
    )


def own_participant(node: Node) -> Optional[bytes]:
    """The GUID prefix of node's participant, from its /parameter_events
    publisher (every node has one unless parameter services are disabled)."""
    for endpoint in node.get_publishers_info_by_topic("/parameter_events"):
        if (endpoint.node_name, endpoint.node_namespace) == (
            node.get_name(),
            node.get_namespace(),
        ):
            return bytes(endpoint.endpoint_gid)[:12]
    return None


def dds_type_name(msg_class: Any) -> str:
    """The DDS type name the graph reports on a no_mangle query, e.g.
    ``std_msgs::msg::dds_::String_`` (rosidl_typesupport_fastrtps)."""
    package, kind = msg_class.__module__.split(".")[:2]
    return f"{package}::{kind}::dds_::{msg_class.__name__}_"


def entity_topics(name: str, kind: str) -> list[tuple[str, bool]]:
    """The DDS topics (name, is_publisher) behind an entity of kind
    publisher, subscription, client or action_client on resolved name."""
    if kind in ("publisher", "subscription"):
        return [(f"rt{name}", kind == "publisher")]
    if kind == "client":
        return [(f"rq{name}Request", True), (f"rr{name}Reply", False)]
    topics = []
    for service in ("send_goal", "cancel_goal", "get_result"):
        base = f"{name}/_action/{service}"
        topics += [(f"rq{base}Request", True), (f"rr{base}Reply", False)]
    return topics + [(f"rt{name}/_action/{t}", False) for t in ("feedback", "status")]


def own_endpoints(node: Node, topics: list[tuple[str, bool]]) -> list[Any]:
    """node's own endpoints on DDS topics (name, is_publisher). The graph drops
    an endpoint when the RMW really destroys it, not when rclpy defers that."""
    me = (node.get_name(), node.get_namespace())
    found = []
    for topic, publisher in topics:
        get = (
            node.get_publishers_info_by_topic
            if publisher
            else node.get_subscriptions_info_by_topic
        )
        found += [
            e
            for e in get(topic, no_mangle=True)
            if (e.node_name, e.node_namespace) == me
        ]
    return found


def cancel_goal_id(action_client: Any, goal_id: Any) -> None:
    """Send a cancel request for goal_id without waiting for the answer, also
    when its goal response never arrived (rclpy's public cancel needs the
    handle from that response). Same private API on Jazzy and Lyrical."""
    action_client._cancel_goal_async(ClientGoalHandle(action_client, goal_id, None))


__all__ = [
    "ACTION_RESULT_TIMEOUT_S",
    "EventsExecutor",
    "RCLError",
    "TopicEndpointInfo",
    "cancel_goal_id",
    "dds_type_name",
    "destroy",
    "entity_topics",
    "own_endpoints",
    "own_participant",
    "requesters_without_reply_reader",
    "service_server_ids",
    "set_message_fields",
]
