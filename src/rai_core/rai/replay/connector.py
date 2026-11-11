# Copyright (C) 2026 Robotec.AI
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

"""ReplayConnector: the ROS2Connector API served from a recording, offline."""

import asyncio
import bisect
import contextvars
import inspect
import logging
import threading
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from rclpy.clock import ClockType
from rclpy.serialization import deserialize_message
from rclpy.time import Time

from rai.communication.exceptions import ConnectorClosedError
from rai.communication.ros2.connectors.ros2_connector import (
    GoalStatus,
    QoS,
    ROS2Connector,
    _companion,
    _parse_frames,
)
from rai.communication.ros2.egress import (
    EgressCheck,
    EgressDenied,
    EgressHooks,
    global_egress,
    match,
    type_name,
)
from rai.communication.ros2.exceptions import ROS2GoalRejectedError, ROS2TimeoutError
from rai.communication.ros2.runtime import message_class
from rai.origin import egress_origin
from rai.recorder import Recording

from .source import Bag, open_bag

logger = logging.getLogger(__name__)

_NS = 1_000_000_000
_TF = ("/tf", "/tf_static")


class ReplayEnded(ROS2TimeoutError):
    """A wait with no timeout reached the end of the recording."""


@dataclass(frozen=True)
class EgressRecord:
    """A publish, service call, goal or cancel the replay did not send."""

    kind: str  # publish, service, goal, cancel
    name: str  # resolved topic, service or action name
    type: str  # "pkg/msg/Name", "pkg/srv/Name" or "pkg/action/Name"
    payload: Any  # the message, request, goal, or the goal id of a cancel
    stamp: int  # ns on the replay's clock
    origin: Optional[str] = None  # the tool that sent it, if any


@dataclass
class ScriptedGoal:
    """How a scripted action server treats a goal."""

    result: Any = None  # the Result message (default: an empty one)
    feedback: Iterable[Any] = ()
    status: GoalStatus = GoalStatus.SUCCEEDED
    accepted: bool = True
    duration: float = 0.0  # virtual seconds until the goal is done


Response = Union[Any, Callable[[Any], Any]]
"""A scripted response: a message, an exception to raise, or a function of the
request (or goal) returning either, sync or async."""


@dataclass
class _Subscription:
    connector: "ReplayConnector"
    topic: str
    callback: Optional[Callable[[Any], Any]]
    raw: bool
    buffer: deque[Any] = field(default_factory=deque)
    closed: bool = False


class ReplaySubscription:
    """The replay twin of :class:`~rai.communication.ros2.Subscription`.
    Callbacks run inline, on the thread that advances the clock, in stamp
    order. In pull mode a wait advances the clock to the next message; ``async
    for`` ends at the end of the recording."""

    def __init__(self, state: _Subscription) -> None:
        self._state = state
        self.topic = state.topic

    @property
    def closed(self) -> bool:
        return self._state.closed

    async def get(self, timeout: Optional[float] = None) -> Any:
        state = self._state
        connector = state.connector
        with connector._lock:
            if state.closed:
                raise ConnectorClosedError(f"Subscription to {self.topic} is closed")
            if not state.buffer:
                connector._wait_for(state.topic, timeout, lambda: bool(state.buffer))
            return state.buffer.popleft()

    def __aiter__(self) -> "ReplaySubscription":
        return self

    async def __anext__(self) -> Any:
        try:
            return await self.get()
        except (ConnectorClosedError, ReplayEnded):
            raise StopAsyncIteration from None

    def close(self) -> None:
        state = self._state
        with state.connector._lock:
            state.closed = True
            subs = state.connector._subs.get(state.topic, [])
            if state in subs:
                subs.remove(state)


class ReplayGoalHandle:
    """The replay twin of :class:`~rai.communication.ros2.GoalHandle`."""

    def __init__(
        self,
        connector: "ReplayConnector",
        action: str,
        action_cls: Any,
        script: ScriptedGoal,
        on_feedback: Optional[Callable[[Any], Any]],
        on_done: Optional[Callable[["ReplayGoalHandle"], Any]],
    ) -> None:
        self.id = uuid.uuid4().hex
        self.action = action
        self.type = type_name(action_cls)
        self._connector = connector
        self._cls = action_cls
        self._script = script
        self._status = GoalStatus.ACCEPTED
        self._done_at = connector._clock + int(script.duration * _NS)
        self._feedback: deque[Any] = deque(script.feedback, maxlen=100)
        self._on_done = on_done
        self._notified = False
        self.sync = _SyncGoalHandle(self)
        if on_feedback is not None:
            for item in list(self._feedback):
                connector._invoke(on_feedback, item)

    def _check_done(self) -> None:
        if (
            self._status == GoalStatus.ACCEPTED
            and self._connector._clock >= self._done_at
        ):
            self._status = GoalStatus(self._script.status)
        if self.done and not self._notified:
            self._notified = True
            if self._on_done is not None:
                self._connector._invoke(self._on_done, self)

    @property
    def done(self) -> bool:
        return self._status not in (GoalStatus.ACCEPTED, GoalStatus.EXECUTING)

    @property
    def status(self) -> GoalStatus:
        return self._status

    def feedback(self) -> list[Any]:
        items = list(self._feedback)
        self._feedback.clear()
        return items

    async def result(self, timeout: Optional[float] = None) -> Any:
        connector = self._connector
        with connector._lock:
            if not self.done:
                connector._wait_until(
                    self._done_at, timeout, f"result of the goal on {self.action}"
                )
                self._check_done()
        if self._script.result is not None:
            return self._script.result
        return self._cls.Result()

    async def cancel(self, timeout: float = 2.0) -> bool:
        connector = self._connector
        await connector._gate("cancel", self.action, self._cls, self.id)
        if self.done:
            return False
        with connector._lock:
            self._status = GoalStatus.CANCELED
            self._check_done()
        return True


class _SyncGoalHandle:
    def __init__(self, handle: ReplayGoalHandle) -> None:
        self._handle = handle

    def result(self, timeout: Optional[float] = None) -> Any:
        return _run(self._handle.result(timeout))

    def cancel(self, timeout: float = 2.0) -> bool:
        return _run(self._handle.cancel(timeout))


class ReplayConnector(ROS2Connector):
    """The :class:`~rai.communication.ros2.ROS2Connector` API served from a
    recording, with no ROS graph, no DDS and a virtual clock, for fast and
    deterministic tests of tools, aggregators and agents.

    - Topics: ``receive`` returns the latest recorded message at the virtual
      time; ``subscribe`` delivers messages as the clock advances. A wait for
      a message (a ``receive`` with nothing fresh enough, ``Subscription.get``)
      advances the clock to the next recorded message, or by the timeout.
    - Egress: ``publish``, ``call``, ``send_goal`` and goal cancels run the
      egress hooks (global and :attr:`egress`) and are recorded in
      :attr:`sent` instead of being sent. Calls and goals are answered from
      ``services`` and ``actions`` (scripted responses); unscripted ones time
      out like an absent server.
    - ``now()`` is the virtual time; ``transform`` uses recorded ``/tf`` and
      ``/tf_static``; graph queries list the recorded topics and the scripted
      services and actions.

    The clock starts at ``start`` (default: the session start of a flight
    recording, else the first message) and moves with :meth:`advance`,
    :meth:`advance_to`, :meth:`seek` and waits. :class:`AgentReplay` moves it
    along a recorded agent session.
    """

    def __init__(
        self,
        source: Union[str, Path, Recording, Bag],
        *,
        services: Optional[Mapping[str, Response]] = None,
        actions: Optional[Mapping[str, Response]] = None,
        start: Optional[int] = None,
        latched: Iterable[str] = ("/tf_static",),
        node_name: str = "/rai_replay",
    ) -> None:
        """
        Args:
            source: a flight recording, a rosbag2 file or directory, or a Bag.
            services: service name -> scripted response (see :data:`Response`).
            actions: action name -> scripted goal: a :class:`ScriptedGoal`, a
                Result message, an exception, or a function of the goal.
            start: the initial virtual time (ns on the recording's clock).
            latched: topics whose latest message a new subscription gets at once.
            node_name: what ``node_name`` and graph queries report.
        """
        self.bag = open_bag(source)
        self.services: dict[str, Response] = {
            _absolute(k): v for k, v in (services or {}).items()
        }
        self.actions: dict[str, Response] = {
            _absolute(k): v for k, v in (actions or {}).items()
        }
        self.sent: list[EgressRecord] = []
        self.denied: list[EgressDenied] = []
        self.latched = {_absolute(t) for t in latched}
        self._node_name = node_name
        self._lock = threading.RLock()
        self._closed = False
        self._hooks = EgressHooks()
        self._subs: dict[str, list[_Subscription]] = {}
        self._goals: list[ReplayGoalHandle] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._decoded: dict[int, Any] = {}
        self._tf: Any = None
        self._tf_fed = 0
        self.logger = logger
        # every ROS message, in stamp order; per topic, indexes into it
        self._messages = list(self.bag.messages())
        self._stamps = [m.stamp for m in self._messages]
        self._by_topic: dict[str, list[int]] = {}
        for i, m in enumerate(self._messages):
            self._by_topic.setdefault(m.topic, []).append(i)
        self._topic_stamps = {
            t: [self._stamps[i] for i in idx] for t, idx in self._by_topic.items()
        }
        first = self._stamps[0] if self._stamps else 0
        session = self.bag.session.get("start_ns")
        if start is None:
            start = min(first, int(session)) if session else first
        self._clock = start
        self._pos = bisect.bisect_right(self._stamps, start)
        self.start = start
        self.sync = SyncReplayConnector(self)  # type: ignore[assignment]

    # ------------------------------------------------------------ lifecycle

    @property
    def _runtime(self) -> Any:  # type: ignore[override]
        raise AttributeError(
            "ReplayConnector has no ROS runtime: use the connector API"
            " (receive, subscribe, publish, call, send_goal, transform, now)"
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _check(self) -> None:
        if self._closed:
            raise ConnectorClosedError("ReplayConnector is closed")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for subs in self._subs.values():
                for state in subs:
                    state.closed = True
            self._subs.clear()

    shutdown = close

    @property
    def egress(self) -> EgressHooks:
        """Egress hooks of this connector (global_egress runs first)."""
        return self._hooks

    @property
    def node_name(self) -> str:
        return self._node_name

    # ------------------------------------------------------------ clock

    @property
    def clock_ns(self) -> int:
        return self._clock

    @property
    def end(self) -> int:
        """The stamp of the last recorded message."""
        return self._stamps[-1] if self._stamps else self._clock

    def now(self) -> Time:
        self._check()
        return Time(nanoseconds=self._clock, clock_type=ClockType.ROS_TIME)

    def advance(self, seconds: float) -> None:
        """Move the clock forward, delivering the messages on the way."""
        self.advance_to(self._clock + int(seconds * _NS))

    def advance_to(self, stamp: int) -> None:
        """Move the clock forward to stamp (ns); earlier stamps are ignored."""
        with self._lock:
            while self._pos < len(self._messages) and self._stamps[self._pos] <= stamp:
                i = self._pos
                self._pos += 1
                self._clock = max(self._clock, self._stamps[i])
                self._deliver(i)
            self._clock = max(self._clock, stamp)
            for goal in list(self._goals):
                goal._check_done()
                if goal.done:
                    self._goals.remove(goal)

    def seek(self, stamp: int) -> None:
        """Set the clock to stamp (ns), also backwards. Nothing is delivered;
        subscriptions get the messages after stamp from now on."""
        with self._lock:
            self._clock = stamp
            self._pos = bisect.bisect_right(self._stamps, stamp)
            self._tf, self._tf_fed = None, 0

    def _next(self, topic: str) -> Optional[int]:
        """Stamp of the next message on topic after the clock."""
        stamps = self._topic_stamps.get(topic, [])
        j = bisect.bisect_right(stamps, self._clock)
        return stamps[j] if j < len(stamps) else None

    def _wait_for(
        self, topic: str, timeout: Optional[float], ready: Callable[[], bool]
    ) -> None:
        """Advance message by message on topic until ready(), within timeout
        (virtual seconds). Raises ROS2TimeoutError (ReplayEnded if timeout is
        None and the recording ends)."""
        deadline = None if timeout is None else self._clock + int(timeout * _NS)
        while not ready():
            next_stamp = self._next(topic)
            if next_stamp is None or (deadline is not None and next_stamp > deadline):
                if deadline is None:
                    raise ReplayEnded(f"receive from {topic}: the recording ended")
                self.advance_to(deadline)
                raise ROS2TimeoutError(
                    f"receive from {topic} timed out after {timeout:g} s"
                )
            self.advance_to(next_stamp)

    def _wait_until(self, stamp: int, timeout: Optional[float], what: str) -> None:
        if timeout is not None and stamp > self._clock + int(timeout * _NS):
            self.advance(timeout)
            raise ROS2TimeoutError(f"{what} timed out after {timeout:g} s")
        self.advance_to(stamp)

    # ------------------------------------------------------------ topics

    def _decode(self, i: int) -> Any:
        msg = self._decoded.get(i)
        if msg is None:
            m = self._messages[i]
            msg = deserialize_message(m.data, message_class(m.type))
            self._decoded[i] = msg
        return msg

    def _deliver(self, i: int) -> None:
        topic = self._messages[i].topic
        for state in list(self._subs.get(topic, ())):
            if state.closed:
                continue
            msg = self._messages[i].data if state.raw else self._decode(i)
            if state.callback is None:
                state.buffer.append(msg)
            else:
                self._invoke(state.callback, msg)

    def _invoke(self, callback: Callable[[Any], Any], arg: Any) -> None:
        try:
            result = callback(arg)
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(_await(result))
                else:
                    task = loop.create_task(_await(result))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        except Exception:
            logger.exception("ReplayConnector: callback %r failed", callback)

    def _check_type(self, topic: str, msg_type: Any) -> None:
        recorded = self.bag.topics.get(topic)
        if msg_type is None or recorded is None:
            return
        wanted = msg_type if isinstance(msg_type, str) else type_name(msg_type)
        if wanted != recorded:
            raise TypeError(f"{topic} is recorded as {recorded}, not {wanted}")

    def subscribe(  # type: ignore[override]
        self,
        topic: str,
        callback: Optional[Callable[[Any], Any]] = None,
        *,
        msg_type: Any = None,
        qos: Any = QoS.AUTO,
        maxsize: Optional[int] = 10,
        raw: bool = False,
    ) -> ReplaySubscription:
        self._check()
        name = self.resolve(topic)
        self._check_type(name, msg_type)
        state = _Subscription(self, name, callback, raw, deque(maxlen=maxsize))
        with self._lock:
            self._subs.setdefault(name, []).append(state)
            if name in self.latched:
                stamps = self._topic_stamps.get(name, [])
                j = bisect.bisect_right(stamps, self._clock)
                for k in range(j):  # everything latched so far, in order
                    i = self._by_topic[name][k]
                    msg = self._messages[i].data if raw else self._decode(i)
                    if callback is None:
                        state.buffer.append(msg)
                    else:
                        self._invoke(callback, msg)
        return ReplaySubscription(state)

    async def receive(  # type: ignore[override]
        self,
        topic: str,
        *,
        msg_type: Any = None,
        timeout: float = 1.0,
        max_age: Optional[float] = None,
        qos: Any = QoS.AUTO,
    ) -> Any:
        """The latest recorded message at the virtual time. Without a fresh
        enough one, the clock advances to the next message on topic, or by
        timeout (then ROS2TimeoutError)."""
        self._check()
        name = self.resolve(topic)
        self._check_type(name, msg_type)
        with self._lock:

            def latest() -> Optional[int]:
                stamps = self._topic_stamps.get(name, [])
                j = bisect.bisect_right(stamps, self._clock)
                if j == 0:
                    return None
                if max_age is not None and self._clock - stamps[j - 1] > max_age * _NS:
                    return None
                return self._by_topic[name][j - 1]

            self._wait_for(name, timeout, lambda: latest() is not None)
            index = latest()
            assert index is not None
            return self._decode(index)

    async def publish(  # type: ignore[override]
        self, topic: str, msg: Any, *, qos: Any = QoS.COMMAND
    ) -> None:
        """Run the egress hooks and record the message in :attr:`sent`."""
        self._check()
        name = self.resolve(topic)
        await self._gate("publish", name, type(msg), msg)

    # ------------------------------------------------- services and actions

    async def _gate(self, kind: str, name: str, cls: Any, payload: Any) -> None:
        key = (kind, name, type_name(cls))
        if global_egress._entries or self._hooks._entries:
            matched = match((global_egress, self._hooks), key)
            if matched:
                check = EgressCheck(matched, key, payload)
                try:
                    await check.finish(check.step())
                except EgressDenied as e:
                    self.denied.append(e)
                    raise
        with self._lock:
            self.sent.append(
                EgressRecord(*key, payload, self._clock, egress_origin.get())
            )

    async def _respond(self, scripted: Response, arg: Any) -> Any:
        if callable(scripted) and not isinstance(scripted, BaseException):
            scripted = scripted(arg)
            if inspect.isawaitable(scripted):
                scripted = await scripted
        if isinstance(scripted, BaseException):
            raise scripted
        return scripted

    async def call(  # type: ignore[override]
        self,
        service: str,
        request: Any,
        *,
        srv_type: Any = None,
        timeout: float = 5.0,
    ) -> Any:
        """Run the egress hooks, record the request and return the scripted
        response; ROS2TimeoutError if there is none."""
        self._check()
        name = self.resolve(service, "service")
        cls = message_class(srv_type) if srv_type else _companion(request, "_Request")
        await self._gate("service", name, cls, request)
        if name not in self.services:
            with self._lock:
                self.advance(timeout)
            raise ROS2TimeoutError(
                f"Service call {name} timed out after {timeout:g} s"
                " (replay: no scripted response)"
            )
        return await self._respond(self.services[name], request)

    async def send_goal(  # type: ignore[override]
        self,
        action: str,
        goal: Any,
        *,
        action_type: Any = None,
        accept_timeout: float = 5.0,
        on_feedback: Optional[Callable[[Any], Any]] = None,
        on_done: Optional[Callable[[Any], Any]] = None,
    ) -> ReplayGoalHandle:
        """Run the egress hooks, record the goal and answer it from the
        scripted ``actions``; ROS2TimeoutError if there is none."""
        self._check()
        name = self.resolve(action)
        cls = message_class(action_type) if action_type else _companion(goal, "_Goal")
        await self._gate("goal", name, cls, goal)
        if name not in self.actions:
            with self._lock:
                self.advance(accept_timeout)
            raise ROS2TimeoutError(
                f"Goal on {name} timed out after {accept_timeout:g} s"
                " (replay: no scripted action server)"
            )
        script = await self._respond(self.actions[name], goal)
        if not isinstance(script, ScriptedGoal):
            script = ScriptedGoal(result=script)
        if not script.accepted:
            raise ROS2GoalRejectedError(f"Goal on {name} was rejected")
        with self._lock:
            handle = ReplayGoalHandle(self, name, cls, script, on_feedback, on_done)
            self._goals.append(handle)
            handle._check_done()
        return handle

    def create_service(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ReplayConnector cannot serve; script responses")

    def create_action(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ReplayConnector cannot serve; script responses")

    def release_client(self, service: str) -> bool:
        return False

    # ------------------------------------------------------ TF, time, graph

    def _tf_buffer(self) -> Any:
        from tf2_ros import Buffer

        if self._tf is None:
            self._tf, self._tf_fed = Buffer(), 0
        tf_indexes = sorted(i for t in _TF for i in self._by_topic.get(t, ()))
        end = bisect.bisect_right([self._stamps[i] for i in tf_indexes], self._clock)
        for i in tf_indexes[self._tf_fed : end]:
            static = self._messages[i].topic == "/tf_static"
            for transform in self._decode(i).transforms:
                if static:
                    self._tf.set_transform_static(transform, "replay")
                else:
                    self._tf.set_transform(transform, "replay")
        self._tf_fed = max(self._tf_fed, end)
        return self._tf

    async def transform(  # type: ignore[override]
        self, target_frame: str, source_frame: str, *, timeout: float = 1.0
    ) -> Any:
        """The latest recorded transform at the virtual time; waits advance
        the clock through recorded /tf. Raises tf2_ros.LookupException."""
        from tf2_ros import LookupException

        self._check()
        with self._lock:
            deadline = self._clock + int(timeout * _NS)
            while not self._tf_buffer().can_transform(
                target_frame, source_frame, Time()
            ):
                upcoming = [s for t in _TF if (s := self._next(t)) is not None]
                if not upcoming or min(upcoming) > deadline:
                    self.advance_to(deadline)
                    raise LookupException(
                        f"Could not find transform from {source_frame} to"
                        f" {target_frame} in {timeout} seconds"
                    )
                self.advance_to(min(upcoming))
            return self._tf.lookup_transform(target_frame, source_frame, Time())

    def tf_frames(self) -> dict[str, dict[str, Any]]:
        self._check()
        with self._lock:
            return _parse_frames(self._tf_buffer().all_frames_as_yaml())

    def resolve(self, name: str, kind: str = "topic") -> str:
        self._check()
        return _absolute(name)

    def get_topics_names_and_types(self) -> list[tuple[str, list[str]]]:
        self._check()
        topics = dict(self.bag.topics)
        for record in self.sent:
            if record.kind == "publish":
                topics.setdefault(record.name, record.type)
        return sorted((name, [type_]) for name, type_ in topics.items())

    def get_services_names_and_types(self) -> list[tuple[str, list[str]]]:
        self._check()
        return [(n, _types(v, "_Response")) for n, v in sorted(self.services.items())]

    def get_actions_names_and_types(self) -> list[tuple[str, list[str]]]:
        self._check()
        out = []
        for name, value in sorted(self.actions.items()):
            result = value.result if isinstance(value, ScriptedGoal) else value
            out.append((name, _types(result, "_Result")))
        return out

    def get_node_names(self) -> list[str]:
        self._check()
        return [self._node_name]

    def get_node_interfaces(self, node: str) -> dict[str, list[tuple[str, list[str]]]]:
        self._check()
        if node != self._node_name:
            raise ValueError(f"Node {node} is not in the ROS graph")
        published = {r.name: r.type for r in self.sent if r.kind == "publish"}
        subscribed = {t: self.bag.topics.get(t, "") for t in self._subs}
        return {
            "publishers": sorted((n, [t]) for n, t in published.items()),
            "subscribers": sorted((n, [t] if t else []) for n, t in subscribed.items()),
            "services": [],
            "clients": self.get_services_names_and_types(),
            "action_servers": [],
            "action_clients": self.get_actions_names_and_types(),
        }

    def get_topic_endpoints(self, topic: str) -> tuple[list[Any], list[Any]]:
        self._check()
        return [], []

    def release_topic(self, topic: str, *, raw: bool = False) -> bool:
        return False

    # ------------------------------------------------------------ expectations

    def published(self, topic: Optional[str] = None) -> list[Any]:
        """Messages published (recorded, not sent), of one topic or all."""
        name = None if topic is None else _absolute(topic)
        return [
            r.payload
            for r in self.sent
            if r.kind == "publish" and (name is None or r.name == name)
        ]


def _absolute(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"


def _types(value: Any, suffix: str) -> list[str]:
    cls = type(value)
    if not cls.__name__.endswith(suffix):
        return []
    package, kind = cls.__module__.split(".")[:2]
    return [f"{package}/{kind}/{cls.__name__.removesuffix(suffix)}"]


async def _await(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _run(coro: Any) -> Any:
    """Run a coroutine to completion from sync code (also from a loop thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


class SyncReplayConnector:
    """Blocking twins of the ReplayConnector async API."""

    def __init__(self, connector: ReplayConnector) -> None:
        self._c = connector

    def publish(self, topic: str, msg: Any, **kwargs: Any) -> None:
        return _run(self._c.publish(topic, msg, **kwargs))

    def receive(self, topic: str, **kwargs: Any) -> Any:
        return _run(self._c.receive(topic, **kwargs))

    def call(self, service: str, request: Any, **kwargs: Any) -> Any:
        return _run(self._c.call(service, request, **kwargs))

    def send_goal(self, action: str, goal: Any, **kwargs: Any) -> ReplayGoalHandle:
        return _run(self._c.send_goal(action, goal, **kwargs))

    def transform(self, target_frame: str, source_frame: str, **kwargs: Any) -> Any:
        return _run(self._c.transform(target_frame, source_frame, **kwargs))
