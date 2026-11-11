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

"""The ROS 2 connector: an async API usable from any event loop, its blocking
twin ``connector.sync``, and the typed Subscription and GoalHandle objects.

Native ROS messages go in and come out; dicts belong at the LLM boundary.
"""

import asyncio
import enum
import inspect
import sys
import threading
import time
import uuid
import warnings
from collections import deque
from typing import Any, Callable, Optional, cast

import rclpy.action
from action_msgs.srv import CancelGoal
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import LookupException, TransformStamped

from rai.communication.base_connector import (
    BaseConnector,
    ConnectorClosedError,
    ConnectorException,
)
from rai.communication.ros2 import compat
from rai.communication.ros2.egress import EgressHooks, type_name
from rai.communication.ros2.exceptions import ROS2GoalRejectedError
from rai.communication.ros2.messages import ROS2Message
from rai.communication.ros2.runtime import (
    POLL_S,
    Client,
    Dispatcher,
    ExecutorKind,
    ROS2Runtime,
    Topic,
    Waiter,
    acquire,
    fail,
    message_class,
    release,
)


def _profile(depth: int, reliability: Any, durability: Any) -> QoSProfile:
    return QoSProfile(depth=depth, reliability=reliability, durability=durability)


class QoS:
    """QoS presets. Publishers default to COMMAND, subscriptions to AUTO."""

    SENSOR = _profile(5, ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE)
    COMMAND = _profile(10, ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE)
    LATCHED = _profile(1, ReliabilityPolicy.RELIABLE, DurabilityPolicy.TRANSIENT_LOCAL)
    HRI = _profile(100, ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE)
    # Subscriptions only: the RMW matches the publishers discovered at creation
    # (latched ones included) and falls back to best effort + volatile.
    AUTO = _profile(
        10, ReliabilityPolicy.BEST_AVAILABLE, DurabilityPolicy.BEST_AVAILABLE
    )


def _companion(obj: Any, suffix: str) -> Any:
    """The service or action class of a Foo_Request or Foo_Goal instance."""
    cls = type(obj)
    package = sys.modules[cls.__module__.rsplit(".", 1)[0]]
    return getattr(package, cls.__name__.removesuffix(suffix))


def _invoke(runtime: ROS2Runtime, callback: Callable[[Any], Any], arg: Any) -> None:
    result = callback(arg)
    if inspect.isawaitable(result):
        runtime.run(result)


class Subscription:
    """A connector's subscription to a topic.

    With a callback, messages are delivered off the executor thread, one call
    at a time, in arrival order (async callbacks are awaited on the connector's
    loop). Without one, iterate it: ``async for msg in sub`` or ``await
    sub.get()``. A full queue drops its oldest message; maxsize=None never drops.
    """

    def __init__(
        self,
        runtime: ROS2Runtime,
        topic: Topic,
        name: str,
        callback: Optional[Callable[[Any], Any]],
        maxsize: Optional[int],
        on_close: Callable[[], None] = lambda: None,
        replay: bool = False,
        decode: Callable[[Any], Any] = lambda msg: msg,
    ) -> None:
        self.topic = name
        self._decode = decode
        self.closed = False
        self._on_close = on_close
        self._runtime = runtime
        self._topic = topic
        self._waiters: deque[Waiter] = deque()
        self._buffer: deque[Any] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._dispatcher: Optional[Dispatcher] = None
        if callback is not None:
            self._dispatcher = Dispatcher(
                runtime, lambda msg: _invoke(runtime, callback, decode(msg)), maxsize
            )
            self._consume: Callable[[Any], None] = self._dispatcher.push
        else:
            self._consume = self._put
        topic.add(self._consume, replay)

    def _put(self, msg: Any) -> None:  # executor thread
        with self._lock:
            if self._waiters:
                self._waiters.popleft().resolve(msg)
            else:
                self._buffer.append(msg)

    async def get(self, timeout: Optional[float] = None) -> Any:
        """The next message (pull mode)."""
        with self._lock:
            if self.closed:
                raise ConnectorClosedError(f"Subscription to {self.topic} is closed")
            if self._buffer:
                return self._decode(self._buffer.popleft())
            waiter = Waiter()
            self._waiters.append(waiter)
        try:
            msg = await self._runtime.wait(
                waiter, timeout, f"receive from {self.topic}"
            )
            return self._decode(msg)
        finally:
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Any:
        try:
            return await self.get()
        except ConnectorClosedError:
            raise StopAsyncIteration from None

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            waiters, self._waiters = list(self._waiters), deque()
        self._topic.remove(self._consume)
        self._on_close()
        if self._dispatcher is not None:
            self._dispatcher.close()
        fail(set(waiters), ConnectorClosedError(f"Subscription to {self.topic} closed"))


class GoalStatus(enum.IntEnum):
    UNKNOWN = 0
    ACCEPTED = 1
    EXECUTING = 2
    CANCELING = 3
    SUCCEEDED = 4
    CANCELED = 5
    ABORTED = 6


class GoalHandle:
    """An accepted action goal: its status, feedback, result and cancel.

    on_feedback and on_done callbacks run off the executor thread, in order,
    on_done after the goal's last feedback callback.
    """

    def __init__(
        self,
        runtime: ROS2Runtime,
        client: Client,
        action: str,
        groups: tuple[set[Waiter], ...],
        on_feedback: Optional[Callable[[Any], Any]],
        on_done: Optional[Callable[["GoalHandle"], Any]],
        feedback_maxlen: int = 100,
    ) -> None:
        self.id = ""
        self.action = action
        self.type = type_name(client.msg_type)  # "pkg/action/Name"
        self._runtime = runtime
        self._client = client
        self._groups = (client.pending, *groups)
        self._goal: Any = None
        self._result_future: Any = None
        self._response: Any = None
        self._feedback: deque[Any] = deque(maxlen=feedback_maxlen)
        self._waiters: set[Waiter] = set()
        self._lock = threading.Lock()
        self._on_feedback, self._on_done = on_feedback, on_done
        self._dispatcher = (
            Dispatcher(runtime, self._dispatch, None)
            if on_feedback or on_done
            else None
        )
        self.sync = SyncGoalHandle(self)

    def _feedback_callback(self, msg: Any) -> None:  # executor thread
        self._feedback.append(msg.feedback)
        if self._dispatcher is not None and self._on_feedback is not None:
            self._dispatcher.push(msg.feedback)

    def _accepted(self, goal: Any) -> None:
        self._goal = goal
        self.id = uuid.UUID(bytes=bytes(goal.goal_id.uuid)).hex
        self._result_future = goal.get_result_async()
        self._result_future.add_done_callback(self._done_callback)
        self._runtime.track_goal(self)

    def _done_callback(self, future: Any) -> None:  # executor thread
        try:
            response = future.result()
        except Exception as e:
            response, error = None, e
        else:
            error = None
        with self._lock:
            self._response = response
            waiters = set(self._waiters)
        self._runtime.untrack_goal(self)
        for waiter in waiters:
            waiter.resolve(response, error)
        if self._dispatcher is not None and self._on_done is not None:
            self._dispatcher.push(self)

    def _dispatch(self, item: Any) -> None:
        if item is self:
            _invoke(self._runtime, cast(Callable[[Any], Any], self._on_done), self)
        else:
            _invoke(self._runtime, cast(Callable[[Any], Any], self._on_feedback), item)

    @property
    def done(self) -> bool:
        return self._response is not None

    @property
    def status(self) -> GoalStatus:
        if self._response is not None:
            return GoalStatus(self._response.status)
        return GoalStatus(self._goal.status) if self._goal else GoalStatus.UNKNOWN

    def feedback(self) -> list[Any]:
        """Feedback received since the last call (the newest 100)."""
        items = []
        while self._feedback:
            items.append(self._feedback.popleft())
        return items

    async def result(self, timeout: Optional[float] = None) -> Any:
        """The action's Result message, once the goal is done."""
        with self._lock:
            response = self._response
            if response is None:
                waiter = Waiter()
                self._waiters.add(waiter)
        if response is None:
            what = f"result of the goal on {self.action}"
            response = await self._runtime.wait(
                waiter, timeout, what, self._waiters, *self._groups
            )
        return response.result

    async def cancel(self, timeout: float = 2.0) -> bool:
        """Request cancellation; True if the server accepted it. Raises
        EgressDenied if an egress hook gating cancels refuses."""
        if self.done:
            return False
        client = self._client
        await self._runtime.agate("cancel", client.name, client.msg_type, self.id)
        waiter = Waiter().bind(self._goal.cancel_goal_async())
        what = f"cancel of the goal on {self.action}"
        response = await self._runtime.wait(waiter, timeout, what, *self._groups)
        return response.return_code == CancelGoal.Response.ERROR_NONE


class SyncGoalHandle:
    """Blocking twins of GoalHandle.result and GoalHandle.cancel."""

    def __init__(self, handle: GoalHandle) -> None:
        self._handle = handle

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._handle._runtime.run(self._handle.result(timeout))

    def cancel(self, timeout: float = 2.0) -> bool:
        return self._handle._runtime.run(self._handle.cancel(timeout))


_node_warned: set[tuple[str, int]] = set()  # call sites warned about .node


class ROS2Base:
    """Runtime ownership and lifecycle shared by ROS2Connector and
    ROS2HRIConnector."""

    def _attach(
        self,
        node_name: Optional[str],
        runtime: Optional[ROS2Runtime],
        executor: Optional[ExecutorKind],
        use_sim_time: bool,
        executor_type: Optional[str] = None,
    ) -> None:
        if executor_type is not None:  # R5: remove
            warnings.warn(
                "executor_type is deprecated and ignored: RAI runs a single-threaded"
                " or events executor (executor='single' | 'events')",
                DeprecationWarning,
                stacklevel=3,
            )
        if runtime is None and (node_name or executor or use_sim_time):
            runtime = ROS2Runtime(
                node_name, executor=executor, use_sim_time=use_sim_time
            )
        self._runtime = acquire(runtime)
        self._closed = False
        self._close_lock = threading.Lock()
        self._waiters: set[Waiter] = set()
        self._subscriptions: dict[str, Subscription] = {}
        self._runtime.on_close(self.close)

    # ------------------------------------------------------------ lifecycle

    @property
    def closed(self) -> bool:
        return self._closed or self._runtime.closed

    def _check(self) -> None:
        if self.closed:
            raise ConnectorClosedError(f"{type(self).__name__} is closed")

    def close(self) -> None:
        """Close the connector: its subscriptions stop, pending calls raise
        ConnectorClosedError, and the runtime closes with its last user."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        for subscription in list(self._subscriptions.values()):
            subscription.close()
        fail(self._waiters, ConnectorClosedError(f"{type(self).__name__} was closed"))
        self._runtime.off_close(self.close)
        release(self._runtime)

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()

    @property
    def egress(self) -> EgressHooks:
        """The egress hooks of this connector's runtime (shared by every
        connector on it). rai.communication.ros2.egress.global_egress covers
        every runtime."""
        return self._runtime.egress

    @property
    def _node(self) -> Node:
        """Internal: the runtime's node, for RAI code not yet ported to the
        connector API (listed for A9 in rai3-plan)."""
        return self._runtime.node

    @property
    def node(self) -> Node:
        """Deprecated: bypasses the connector, its lifecycle and egress hooks.
        Warns once per call site."""
        caller = sys._getframe(1)
        site = (caller.f_code.co_filename, caller.f_lineno)
        if site not in _node_warned:
            _node_warned.add(site)
            warnings.warn(
                "connector.node bypasses the connector and its egress hooks"
                " (rai.safety: EStop, ConfirmActuation, RateLimit, AuditLog do not"
                " see what is sent through it) and will be removed in RAI 3.0;"
                " use the connector API",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._runtime.node

    # ------------------------------------------------------------ topics

    def _subscribe(
        self,
        topic: str,
        callback: Optional[Callable[[Any], Any]],
        msg_type: Any,
        qos: QoSProfile,
        maxsize: Optional[int],
        raw: bool = False,
        decode: Callable[[Any], Any] = lambda msg: msg,
    ) -> tuple[str, Subscription]:
        self._check()
        entry, created = self._runtime.topic(topic, msg_type, qos, raw)
        key = uuid.uuid4().hex
        subscription = Subscription(
            self._runtime,
            entry,
            topic,
            callback,
            maxsize,
            lambda: self._subscriptions.pop(key, None),
            replay=created,
            decode=decode,
        )
        self._subscriptions[key] = subscription
        return key, subscription

    async def _receive(
        self,
        topic: str,
        msg_type: Any,
        timeout: float,
        max_age: Optional[float],
        qos: QoSProfile,
    ) -> tuple[Any, float, float]:
        self._check()
        entry = self._runtime.topic(topic, msg_type, qos)[0]
        with entry.lock:
            sample = entry.last
            if sample is not None and (
                max_age is None or time.monotonic() - sample[1] <= max_age
            ):
                return sample
            waiter = Waiter()
            entry.waiters.append(waiter)
        try:
            what = f"receive from {topic}"
            return await self._runtime.wait(waiter, timeout, what, self._waiters)
        finally:
            with entry.lock:
                if waiter in entry.waiters:
                    entry.waiters.remove(waiter)


class ROS2Connector(ROS2Base, BaseConnector[ROS2Message]):
    """ROS 2 topics, services, actions and TF on a shared ROS2Runtime.

    The async methods work from any event loop; ``connector.sync`` has blocking
    twins. Pass node_name, executor or use_sim_time to get a private runtime
    (its own node); otherwise every connector in the process shares one.
    """

    def __init__(
        self,
        node_name: Optional[str] = None,
        *,
        runtime: Optional[ROS2Runtime] = None,
        executor: Optional[ExecutorKind] = None,
        use_sim_time: bool = False,
        executor_type: Optional[str] = None,  # R5: deprecated, ignored
        destroy_subscribers: bool = False,  # R5: deprecated, ignored
    ) -> None:
        BaseConnector.__init__(self)
        self._attach(node_name, runtime, executor, use_sim_time, executor_type)
        self._goals: dict[str, GoalHandle] = {}  # R5: handles of start_action
        self.sync = SyncROS2Connector(self)

    def close(self) -> None:
        super().close()
        self.callback_executor.shutdown(wait=False)

    def shutdown(self) -> None:  # R5: old name of close()
        self.close()

    # ------------------------------------------------------------ topics

    def subscribe(
        self,
        topic: str,
        callback: Optional[Callable[[Any], Any]] = None,
        *,
        msg_type: Any = None,
        qos: QoSProfile = QoS.AUTO,
        maxsize: Optional[int] = 10,
        raw: bool = False,
    ) -> Subscription:
        """Subscribe to a topic. msg_type (class or "pkg/msg/Name") defaults to
        the type in the ROS graph. raw=True delivers serialized bytes (relays
        and recorders only)."""
        return self._subscribe(topic, callback, msg_type, qos, maxsize, raw)[1]

    async def receive(
        self,
        topic: str,
        *,
        msg_type: Any = None,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
        qos: QoSProfile = QoS.AUTO,
    ) -> Any:
        """The latest message on topic. Without max_age a cached sample of any
        age is returned (latched topics); with it, only one at most max_age
        seconds old, waiting up to timeout for it."""
        return (await self._receive(topic, msg_type, timeout, max_age, qos))[0]

    async def publish(
        self, topic: str, msg: Any, *, qos: QoSProfile = QoS.COMMAND
    ) -> None:
        """Publish a ROS message. COMMAND QoS never replays old commands to
        late subscribers; use QoS.LATCHED for state such as a map. Raises
        EgressDenied if an egress hook refuses."""
        self._check()
        await self._runtime.apublish(topic, msg, qos)

    # -------------------------------------------------- services and actions

    async def call(
        self,
        service: str,
        request: Any,
        *,
        srv_type: Any = None,
        timeout: float = 5.0,
    ) -> Any:
        """Call a service and return its response. Raises ROS2TimeoutError if
        the service is not available or does not answer within timeout, and
        EgressDenied if an egress hook refuses (hook time is not counted)."""
        self._check()
        client = self._runtime.client(
            service, srv_type or _companion(request, "_Request"), action=False
        )
        await self._runtime.agate("service", client.name, client.msg_type, request)
        deadline = await self._runtime.ready(client, timeout, f"Service {service}")
        future = client.entity.call_async(request)
        waiter = Waiter().bind(future)
        try:
            return await self._runtime.wait(
                waiter,
                max(0.0, deadline - time.monotonic()),
                f"Service call {service}",
                client.pending,
                self._waiters,
            )
        except TimeoutError:
            client.entity.remove_pending_request(future)
            raise

    def release_client(self, service: str) -> bool:
        """Destroy the cached client of service; calls in flight on it raise."""
        return False if self.closed else self._runtime.release(service)

    async def send_goal(
        self,
        action: str,
        goal: Any,
        *,
        action_type: Any = None,
        accept_timeout: float = 5.0,
        on_feedback: Optional[Callable[[Any], Any]] = None,
        on_done: Optional[Callable[[GoalHandle], Any]] = None,
    ) -> GoalHandle:
        """Send a goal and return its handle once accepted. A goal accepted
        after accept_timeout is cancelled. Raises ROS2GoalRejectedError, and
        EgressDenied if an egress hook refuses (hook time is not counted)."""
        self._check()
        client = self._runtime.client(
            action, action_type or _companion(goal, "_Goal"), action=True
        )
        await self._runtime.agate("goal", client.name, client.msg_type, goal)
        deadline = await self._runtime.ready(client, accept_timeout, f"Action {action}")
        handle = GoalHandle(
            self._runtime, client, action, (self._waiters,), on_feedback, on_done
        )
        future = client.entity.send_goal_async(
            goal, feedback_callback=handle._feedback_callback
        )
        try:
            goal_handle = await self._runtime.wait(
                Waiter().bind(future),
                max(0.0, deadline - time.monotonic()),
                f"Goal on {action}",
                client.pending,
                self._waiters,
            )
        except TimeoutError:
            future.add_done_callback(_cancel_late_goal)
            raise
        if not goal_handle.accepted:
            raise ROS2GoalRejectedError(f"Goal on {action} was rejected")
        handle._accepted(goal_handle)
        return handle

    def create_service(
        self,
        service_name: str,
        on_request: Callable[[Any, Any], Any],
        on_done: Optional[Callable[[Any, Any], Any]] = None,
        *,
        service_type: Any,
        **kwargs: Any,
    ) -> str:
        """Serve requests with on_request(request, response) -> response. It
        runs off the executor thread, one request at a time."""
        self._check()
        return self._runtime.serve(service_name, service_type, on_request)

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable[[Any], Any],
        *,
        action_type: Any,
        **kwargs: Any,
    ) -> str:
        """Serve goals with execute(goal_handle) -> Result, run off the executor
        thread. kwargs go to rclpy's ActionServer."""
        self._check()
        return self._runtime.serve_action(
            action_name, action_type, generate_feedback_callback, **kwargs
        )

    # ------------------------------------------------------ TF, time, graph

    async def transform(
        self, target_frame: str, source_frame: str, *, timeout: float = 1.0
    ) -> TransformStamped:
        """The latest transform; raises tf2_ros.LookupException after timeout."""
        self._check()
        buffer = self._runtime.tf_buffer()
        deadline = time.monotonic() + timeout
        while not buffer.can_transform(target_frame, source_frame, Time()):
            if time.monotonic() >= deadline:
                raise LookupException(
                    f"Could not find transform from {source_frame} to"
                    f" {target_frame} in {timeout} seconds"
                )
            await asyncio.sleep(POLL_S)
            self._check()
        return buffer.lookup_transform(target_frame, source_frame, Time())

    def now(self) -> Time:
        self._check()
        return self._runtime.node.get_clock().now()

    def resolve(self, name: str, kind: str = "topic") -> str:
        """The fully qualified name ("cmd_vel" -> "/cmd_vel", remaps applied);
        kind is "topic", "service" or "action"."""
        self._check()
        return self._runtime.resolve(name, kind)

    def get_topics_names_and_types(self) -> list[tuple[str, list[str]]]:
        self._check()
        return self._runtime.node.get_topic_names_and_types()

    def get_services_names_and_types(self) -> list[tuple[str, list[str]]]:
        self._check()
        return self._runtime.node.get_service_names_and_types()

    def get_actions_names_and_types(self) -> list[tuple[str, list[str]]]:
        self._check()
        return rclpy.action.get_action_names_and_types(self._runtime.node)

    @property
    def node_name(self) -> str:
        """The fully qualified name of this connector's node."""
        return self._runtime.node.get_fully_qualified_name()

    def get_node_names(self) -> list[str]:
        """Fully qualified names of the nodes in the graph (duplicates kept)."""
        self._check()
        return sorted(self._runtime.node.get_fully_qualified_node_names())

    def get_node_interfaces(self, node: str) -> dict[str, list[tuple[str, list[str]]]]:
        """Names and types a node ("/ns/name") offers and uses, by kind:
        publishers, subscribers, services, clients, action_servers and
        action_clients. Raises ValueError if the node is not in the graph."""
        self._check()
        if node not in self.get_node_names():
            raise ValueError(f"Node {node} is not in the ROS graph")
        namespace, _, name = node.rpartition("/")
        args = (name, namespace or "/")
        ros = self._runtime.node
        return {
            "publishers": ros.get_publisher_names_and_types_by_node(*args),
            "subscribers": ros.get_subscriber_names_and_types_by_node(*args),
            "services": ros.get_service_names_and_types_by_node(*args),
            "clients": ros.get_client_names_and_types_by_node(*args),
            "action_servers": rclpy.action.get_action_server_names_and_types_by_node(
                ros, *args
            ),
            "action_clients": rclpy.action.get_action_client_names_and_types_by_node(
                ros, *args
            ),
        }

    def get_topic_endpoints(self, topic: str) -> tuple[list[Any], list[Any]]:
        """Publishers and subscriptions of a topic as rclpy TopicEndpointInfo
        (node_name, node_namespace, topic_type, qos_profile)."""
        self._check()
        name, ros = self.resolve(topic), self._runtime.node
        return (
            ros.get_publishers_info_by_topic(name),
            ros.get_subscriptions_info_by_topic(name),
        )

    def release_topic(self, topic: str, *, raw: bool = False) -> bool:
        """Destroy the shared subscription to topic if no Subscription uses it,
        so its publishers stop sending here. Returns whether it was destroyed."""
        return False if self.closed else self._runtime.release_topic(topic, raw)

    def tf_frames(self) -> dict[str, dict[str, Any]]:
        """The frames in the TF buffer (listening starts on first use): child
        frame -> parent, broadcaster, rate (Hz), most_recent_transform and
        oldest_transform (seconds; 0 for static transforms)."""
        self._check()
        return _parse_frames(self._runtime.tf_buffer().all_frames_as_yaml())

    # ------------------------------------- R5: compatibility with the 2.x API

    def unregister_callback(self, callback_id: str) -> None:
        """Stop a callback registered with register_callback (R5: use
        Subscription.close)."""
        if self._closed:
            return
        subscription = self._subscriptions.pop(callback_id, None)
        if subscription is None:
            raise ConnectorException(f"Callback with id {callback_id} not found.")
        subscription.close()

    def send_message(
        self,
        message: ROS2Message,
        target: str,
        *,
        msg_type: str,
        auto_qos_matching: bool = True,  # ignored: publishers use QoS presets
        qos_profile: Optional[QoSProfile] = None,
        **kwargs: Any,
    ) -> None:
        self._check()
        msg = _build(msg_type, message.payload)
        self._runtime.publish(target, msg, qos_profile or QoS.COMMAND)

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,  # ignored: QoS.AUTO is the default
        max_age: Optional[float] = None,
        **kwargs: Any,
    ) -> ROS2Message:
        msg, _, stamp = self._runtime.run(
            self._receive(
                source, msg_type, timeout_sec, max_age, qos_profile or QoS.AUTO
            )
        )
        return ROS2Message(
            payload=msg, metadata={"msg_type": str(type(msg))}, timestamp=stamp
        )

    def register_callback(
        self,
        source: str,
        callback: Callable[[Any], Any],
        raw: bool = False,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        **kwargs: Any,
    ) -> str:
        def wrapped(msg: Any) -> Any:
            meta = {"msg_type": str(type(msg))}
            return callback(ROS2Message(payload=msg, metadata=meta))

        return self._subscribe(
            source, callback if raw else wrapped, msg_type, qos_profile or QoS.AUTO, 10
        )[0]

    def service_call(
        self,
        message: ROS2Message,
        target: str,
        timeout_sec: float = 5.0,
        *,
        msg_type: str,
        reuse_client: bool = True,  # clients are always cached now
        **kwargs: Any,
    ) -> ROS2Message:
        self._check()
        srv_cls = message_class(msg_type)
        request = _build(srv_cls.Request, message.payload)
        response = self.sync.call(
            target, request, srv_type=srv_cls, timeout=timeout_sec
        )
        meta = {"msg_type": str(type(response)), "service": target}
        return ROS2Message(payload=response, metadata=meta)

    def start_action(
        self,
        action_data: Optional[ROS2Message],
        target: str,
        on_feedback: Callable[[Any], None] = lambda _: None,
        on_done: Callable[[Any], None] = lambda _: None,
        timeout_sec: float = 1.0,
        *,
        msg_type: str,
        **kwargs: Any,
    ) -> str:
        self._check()
        if not isinstance(action_data, ROS2Message):
            raise ValueError("Action data must be of type ROS2Message")
        action_cls = message_class(msg_type)
        handle = self.sync.send_goal(
            target,
            _build(action_cls.Goal, action_data.payload),
            action_type=action_cls,
            accept_timeout=timeout_sec,
            on_feedback=on_feedback,
            on_done=lambda h: on_done(h._result_future),
        )
        for goal_id in [k for k, h in self._goals.items() if h.done][:-32]:
            del self._goals[goal_id]
        self._goals[handle.id] = handle
        return handle.id

    def terminate_action(self, action_handle: str, **kwargs: Any) -> bool:
        self._check()
        handle = self._goals.get(action_handle)
        if handle is None:
            raise ValueError(f"Unknown action handle: {action_handle}")
        return handle.sync.cancel()

    def get_transform(
        self, target_frame: str, source_frame: str, timeout_sec: float = 5.0
    ) -> TransformStamped:
        return self.sync.transform(target_frame, source_frame, timeout=timeout_sec)


def _parse_frames(text: str) -> dict[str, dict[str, Any]]:
    """tf2's all_frames_as_yaml (a fixed, flat format) without a YAML parser."""
    frames: dict[str, dict[str, Any]] = {}
    frame: Optional[dict[str, Any]] = None
    for line in text.splitlines():
        if not line.strip() or line.strip() == "[]":
            continue
        if not line.startswith(" "):
            frame = frames.setdefault(line.rstrip().removesuffix(":"), {})
        elif frame is not None:
            key, _, value = line.strip().partition(":")
            value = value.strip().strip("'")
            try:
                frame[key] = float(value)
            except ValueError:
                frame[key] = value
    return frames


def _build(msg_type: Any, payload: Any) -> Any:
    cls = message_class(msg_type)
    if isinstance(payload, cls):
        return payload
    return compat.set_message_fields(cls(), payload or {})


def _cancel_late_goal(future: Any) -> None:  # executor thread
    try:
        goal = future.result()
        if goal is not None and goal.accepted:
            goal.cancel_goal_async()
    except Exception:
        pass


class SyncROS2Connector:
    """Blocking twins of the ROS2Connector async API: the same coroutines, run
    on the connector's private event loop. Not usable from async callbacks."""

    def __init__(self, connector: ROS2Connector) -> None:
        self._c = connector

    def publish(self, topic: str, msg: Any, **kwargs: Any) -> None:
        return self._c._runtime.run(self._c.publish(topic, msg, **kwargs))

    def receive(self, topic: str, **kwargs: Any) -> Any:
        return self._c._runtime.run(self._c.receive(topic, **kwargs))

    def call(self, service: str, request: Any, **kwargs: Any) -> Any:
        return self._c._runtime.run(self._c.call(service, request, **kwargs))

    def send_goal(self, action: str, goal: Any, **kwargs: Any) -> GoalHandle:
        return self._c._runtime.run(self._c.send_goal(action, goal, **kwargs))

    def transform(self, target_frame: str, source_frame: str, **kwargs: Any) -> Any:
        return self._c._runtime.run(
            self._c.transform(target_frame, source_frame, **kwargs)
        )


__all__ = [
    "GoalHandle",
    "GoalStatus",
    "QoS",
    "ROS2Connector",
    "Subscription",
    "SyncROS2Connector",
]
