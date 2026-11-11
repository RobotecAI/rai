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

"""ROS2Runtime: the node, executor, spin thread and entity registry behind the
connectors (internal; rai3-plan/02 section 3.1).

Threads: the executor thread only takes messages, completes futures and fills
bounded queues. User code (subscription callbacks, service handlers, action
execute callbacks) runs on the worker pool, one call at a time per callback, in
arrival order. The sync API runs the async one on a private event loop.

Egress: every publish, service call, goal and cancel passes the egress hook
chains (``global_egress`` then ``runtime.egress``) first; with no hooks that is
one attribute check per call (rai.communication.ros2.egress).
"""

import asyncio
import atexit
import concurrent.futures
import contextvars
import logging
import threading
import time
import uuid
import weakref
from collections import deque
from typing import Any, Callable, Literal, Optional

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.task import Future

from rai.communication.base_connector import ConnectorClosedError
from rai.communication.ros2 import compat
from rai.communication.ros2.api.conversion import import_message_from_str
from rai.communication.ros2.egress import (
    EgressCheck,
    EgressHooks,
    global_egress,
    match,
    type_name,
)
from rai.communication.ros2.exceptions import ROS2ServiceError, ROS2TimeoutError

logger = logging.getLogger(__name__)

ExecutorKind = Literal["single", "events"]
# EventsExecutor passed the full oracle on Jazzy and Lyrical (rai3-plan/PLAN.md
# 2.1); SingleThreadedExecutor stays available as executor="single".
DEFAULT_EXECUTOR: ExecutorKind = "events"
POLL_S = 0.01  # readiness and TF polling; rclpy's own waits poll at 0.25 s


def _make_executor(kind: Optional[str], context: Context) -> Any:
    if kind is None:
        kind = DEFAULT_EXECUTOR if compat.EventsExecutor is not None else "single"
    if kind == "single":
        return SingleThreadedExecutor(context=context)
    if kind == "events":
        if compat.EventsExecutor is None:
            raise ValueError("EventsExecutor needs rclpy >= 7.1.11")
        return compat.EventsExecutor(context=context)
    raise ValueError(f"Unknown executor {kind!r}: use 'single' or 'events'")


def message_class(msg_type: Any) -> Any:
    return import_message_from_str(msg_type) if isinstance(msg_type, str) else msg_type


class Waiter:
    """An asyncio future on the caller's loop that any thread may complete."""

    __slots__ = ("loop", "future")

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.future: asyncio.Future[Any] = self.loop.create_future()

    def resolve(self, value: Any = None, error: Optional[BaseException] = None) -> None:
        try:
            self.loop.call_soon_threadsafe(self._resolve, value, error)
        except RuntimeError:  # the caller's loop is closed
            pass

    def _resolve(self, value: Any, error: Optional[BaseException]) -> None:
        if not self.future.done():
            if error is None:
                self.future.set_result(value)
            else:
                self.future.set_exception(error)

    def bind(self, future: Future) -> "Waiter":
        """Complete with the result of an rclpy future."""

        def done(f: Future) -> None:
            try:
                self.resolve(f.result())
            except BaseException as e:
                self.resolve(error=e)

        future.add_done_callback(done)
        return self


def fail(waiters: set[Waiter], error: BaseException) -> None:
    for waiter in list(waiters):
        waiter.resolve(error=error)


class Dispatcher:
    """Runs callback(item) for pushed items on the worker pool, one at a time,
    in push order. maxsize=None never drops; otherwise the oldest item goes."""

    def __init__(
        self,
        runtime: "ROS2Runtime",
        callback: Callable[[Any], Any],
        maxsize: Optional[int],
    ) -> None:
        self._runtime = runtime
        self._callback = callback
        self._queue: deque[Any] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._scheduled = False
        self.closed = False

    def push(self, item: Any) -> None:
        with self._lock:
            if self.closed:
                return
            self._queue.append(item)
            if self._scheduled:
                return
            self._scheduled = True
        try:
            self._runtime.workers.submit(self._drain)
        except RuntimeError:  # the pool is shut down: the runtime is closing
            self.closed = True

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._queue.clear()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if self.closed or not self._queue:
                    self._scheduled = False
                    return
                item = self._queue.popleft()
            try:
                self._callback(item)
            except Exception:
                logger.exception("Error in callback %r", self._callback)


class Topic:
    """One subscription, shared by every consumer of the topic. Keeps the
    latest sample for receive()."""

    def __init__(self, msg_type: Any) -> None:
        self.msg_type = msg_type
        self.subscription: Any = None
        self.last: Optional[tuple[Any, float, float]] = None  # msg, monotonic, wall
        self.consumers: tuple[Callable[[Any], None], ...] = ()
        self.waiters: list[Waiter] = []
        self.lock = threading.Lock()
        self.rematched = False

    def on_message(self, msg: Any) -> None:  # executor thread
        sample = (msg, time.monotonic(), time.time())
        with self.lock:
            self.last = sample
            waiters, self.waiters = self.waiters, []
            consumers = self.consumers
        for waiter in waiters:
            waiter.resolve(sample)
        for consume in consumers:
            try:
                consume(msg)
            except Exception:
                logger.exception("Error dispatching a message")

    def add(self, consume: Callable[[Any], None], replay: bool = False) -> None:
        """Add a consumer; replay hands it a sample that arrived before it
        (a new subscription may get a latched sample at once)."""
        with self.lock:
            self.consumers = (*self.consumers, consume)
            if replay and self.last is not None:
                consume(self.last[0])

    def remove(self, consume: Callable[[Any], None]) -> None:
        with self.lock:
            self.consumers = tuple(c for c in self.consumers if c is not consume)


class Client:
    """A cached service or action client and the waiters of its calls."""

    def __init__(self, entity: Any, msg_type: Any, name: str) -> None:
        self.entity = entity
        self.name = name  # resolved
        self.msg_type = msg_type
        self.pending: set[Waiter] = set()
        self.dropped: Optional[BaseException] = None


class ROS2Runtime:
    """Owns one node, one executor and its spin thread, the entity registry,
    the worker pool and a private event loop for the sync API.

    Lifecycle: RUNNING -> CLOSED. close() fails every pending wait with
    ConnectorClosedError, stops the executor and joins the spin thread, then
    destroys the entities and the node. It is idempotent.
    """

    def __init__(
        self,
        node_name: Optional[str] = None,
        *,
        executor: Optional[ExecutorKind] = None,
        use_sim_time: bool = False,
        cli_args: Optional[list[str]] = None,
    ) -> None:
        if not rclpy.ok():
            # A library must not own the process signals (rai3-plan/01 section 9).
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            _owned_contexts.append(rclpy.get_default_context())
            logger.info("rclpy was not initialized; RAI initialized it")
        self.node = Node(node_name or f"rai_{uuid.uuid4().hex[:12]}", cli_args=cli_args)
        if use_sim_time:
            self.node.set_parameters([Parameter("use_sim_time", value=True)])
        self.context = self.node.context
        self.executor = _make_executor(executor, self.context)
        self.executor.add_node(self.node)
        # A suspended async callback holds its group until it finishes; with a
        # mutually exclusive group a service handler calling another service
        # would starve its own response. Dispatchers serialize user code.
        self.group = ReentrantCallbackGroup()
        self.workers = concurrent.futures.ThreadPoolExecutor(
            max_workers=64, thread_name_prefix="rai_ros2_worker"
        )
        self.users = 0  # connectors sharing this runtime
        self.egress = EgressHooks()  # after global_egress, for this runtime only
        self._lock = threading.RLock()
        self._closed = False
        self._waiters: set[Waiter] = set()
        self._on_close: list[Callable[[], None]] = []
        self._topics: dict[tuple[str, bool], Topic] = {}
        self._publishers: dict[str, Any] = {}
        self._clients: dict[tuple[str, bool], Client] = {}
        self._servers: list[Any] = []
        self._tf_buffer: Any = None
        self._tf_listener: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._spin_thread = threading.Thread(
            target=self._spin, name="rai_ros2_spin", daemon=True
        )
        self._spin_thread.start()
        _track(self)

    # ------------------------------------------------------------ lifecycle

    @property
    def closed(self) -> bool:
        return self._closed or not self.context.ok()

    def check(self) -> None:
        if self.closed:
            raise ConnectorClosedError("The ROS 2 connector is closed")

    def _wake(self) -> None:
        executor = self.executor
        if executor is not None:
            executor.wake()

    def on_close(self, callback: Callable[[], None]) -> None:
        self._on_close.append(callback)

    def off_close(self, callback: Callable[[], None]) -> None:
        if callback in self._on_close:
            self._on_close.remove(callback)

    def _spin(self) -> None:
        while not self._closed and self.context.ok():
            try:
                self.executor.spin()
                return
            except ExternalShutdownException:
                return
            except Exception:  # RAI callbacks are guarded; this is a last resort
                logger.exception("The ROS 2 executor raised; spinning again")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        closed = ConnectorClosedError("The ROS 2 connector was closed")
        for callback in list(self._on_close):
            try:
                callback()
            except Exception:
                logger.exception("Error while closing a connector")
        fail(self._waiters, closed)
        for client in self._clients.values():
            fail(client.pending, closed)
        self.workers.shutdown(wait=False, cancel_futures=True)
        self.executor.shutdown(timeout_sec=0)
        if threading.current_thread() is not self._spin_thread:
            self._spin_thread.join(1.0)
        if not self._spin_thread.is_alive():
            # EventsExecutor holds the rcl context, which then outlives
            # finalization (see _close_at_exit).
            self.executor = None
        self._stop_loop()
        if self._tf_listener is not None:
            self._tf_listener.unregister()
        entities = [c.entity for c in self._clients.values()]  # action clients first
        entities += self._servers + list(self._publishers.values())
        entities += [t.subscription for t in self._topics.values()]
        for entity in entities:
            compat.destroy(self.node, entity)
        self._clients.clear()
        try:
            self.node.destroy_node()
        except Exception:  # already destroyed by rclpy.shutdown() (Lyrical)
            pass

    # ------------------------------------------------------ sync / async glue

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            self.check()
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._loop.run_forever, name="rai_ros2_loop", daemon=True
                )
                self._loop_thread.start()
            return self._loop

    def run(self, coro: Any) -> Any:
        """Run a coroutine of the async API on the private loop and wait for it."""
        try:
            loop = self.loop()
        except ConnectorClosedError:
            coro.close()
            raise
        if threading.current_thread() is self._loop_thread:
            coro.close()
            raise RuntimeError(
                "The sync API cannot be used from async callbacks; await"
            )
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise ConnectorClosedError("The ROS 2 connector was closed") from None

    def _stop_loop(self) -> None:
        loop, thread = self._loop, self._loop_thread
        if loop is None or thread is None:
            return

        async def cancel_tasks() -> None:
            tasks = asyncio.all_tasks() - {asyncio.current_task()}
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=1.0)

        if threading.current_thread() is thread:
            loop.stop()
            return
        try:
            asyncio.run_coroutine_threadsafe(cancel_tasks(), loop).result(2.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread.join(1.0)
        if not thread.is_alive():
            loop.close()

    async def wait(
        self,
        waiter: Waiter,
        timeout: Optional[float],
        what: str,
        *groups: set[Waiter],
    ) -> Any:
        """Await a waiter; ROS2TimeoutError after timeout, ConnectorClosedError
        if the runtime or an owning group is closed meanwhile."""
        for group in (self._waiters, *groups):
            group.add(waiter)
        try:
            self.check()
            return await asyncio.wait_for(waiter.future, timeout)
        except TimeoutError:
            raise ROS2TimeoutError(f"{what} timed out after {timeout:g} s") from None
        finally:
            for group in (self._waiters, *groups):
                group.discard(waiter)

    def call_in_worker(self, fn: Callable[..., Any], *args: Any) -> Future:
        """Run fn on the worker pool; the returned rclpy future is awaitable from
        executor callbacks (async rclpy callbacks never block the executor)."""
        done: Future = Future()

        def job() -> None:
            try:
                done.set_result(fn(*args))
            except BaseException as e:
                done.set_exception(e)

        self.workers.submit(job)
        return done

    # ---------------------------------------------------------- topics

    def resolve(self, name: str, kind: str = "topic") -> str:
        if kind == "service":
            return self.node.resolve_service_name(name)
        return self.node.resolve_topic_name(name)

    def topic(
        self, name: str, msg_type: Any, qos: QoSProfile, raw: bool = False
    ) -> tuple[Topic, bool]:
        """The shared subscription to a topic, and whether it was just created."""
        self.check()
        key = (self.resolve(name), raw)
        cls = message_class(msg_type)
        with self._lock:
            self.check()
            topic = self._topics.get(key)
            if topic is None:
                topic = Topic(cls or self._graph_type(key[0]))
                # BEST_AVAILABLE resolves when the subscription is created: with
                # no publisher yet it becomes best effort, which loses fragments
                # of large messages. Such a subscription is re-created once, at
                # its first match, against the publishers' real offers.
                early = qos.reliability == ReliabilityPolicy.BEST_AVAILABLE
                if early and not self.node.count_publishers(key[0]):
                    topic.subscription = self._subscribe(topic, key, qos, rematch=True)
                else:
                    topic.subscription = self._subscribe(topic, key, qos)
                self._topics[key] = topic
                return topic, True
            if cls is not None and cls is not topic.msg_type:
                raise TypeError(
                    f"{key[0]} is subscribed as {topic.msg_type.__name__}, not {cls.__name__}"
                )
            return topic, False

    def _subscribe(
        self,
        topic: Topic,
        key: tuple[str, bool],
        qos: QoSProfile,
        rematch: bool = False,
    ) -> Any:
        events = SubscriptionEventCallbacks()
        if rematch:
            events.matched = lambda _: self._rematch(topic, key, qos)
        return self.node.create_subscription(
            topic.msg_type,
            key[0],
            topic.on_message,
            qos,
            raw=key[1],
            callback_group=self.group,
            event_callbacks=events,
        )

    def _rematch(self, topic: Topic, key: tuple[str, bool], qos: QoSProfile) -> None:
        try:  # executor thread
            with self._lock:
                if self._closed or not self.node.count_publishers(key[0]):
                    return
                if self._topics.get(key) is not topic or topic.rematched:
                    return
                topic.rematched = True
                old, topic.subscription = (
                    topic.subscription,
                    self._subscribe(topic, key, qos),
                )
            compat.destroy(self.node, old)
        except Exception:
            logger.exception("Could not re-create the subscription to %s", key[0])

    def _graph_type(self, topic: str) -> Any:
        for name, types in self.node.get_topic_names_and_types():
            if name == topic:
                if len(types) != 1:
                    raise ValueError(f"Topic {topic} has several types: {types}")
                return import_message_from_str(types[0])
        raise ValueError(f"Topic {topic} not found; pass msg_type")

    def publish(self, name: str, msg: Any, qos: QoSProfile) -> None:
        """Publish from sync code; raises EgressDenied if a hook refuses."""
        self.check()
        topic = self.resolve(name)
        if global_egress._entries or self.egress._entries:
            self.gate("publish", topic, type(msg), msg)
        self._publish(topic, msg, qos)

    async def apublish(self, name: str, msg: Any, qos: QoSProfile) -> None:
        """Publish from async code; raises EgressDenied if a hook refuses."""
        self.check()
        topic = self.resolve(name)
        if global_egress._entries or self.egress._entries:
            await self.agate("publish", topic, type(msg), msg)
        self._publish(topic, msg, qos)

    def _publish(self, topic: str, msg: Any, qos: QoSProfile) -> None:
        publisher = self._publishers.get(topic)
        if publisher is None:
            with self._lock:
                self.check()
                publisher = self._publishers.get(topic)
                if publisher is None:
                    publisher = self.node.create_publisher(
                        type(msg), topic, qos, callback_group=self.group
                    )
                    self._publishers[topic] = publisher
        if publisher.msg_type is not type(msg):
            raise TypeError(
                f"{topic} publishes {publisher.msg_type.__name__}, not {type(msg).__name__}"
            )
        publisher.publish(msg)

    # ------------------------------------------------------------ egress

    def _egress(
        self, kind: str, name: str, cls: Any, payload: Any
    ) -> Optional[EgressCheck]:
        key = (kind, name, type_name(cls))
        matched = match((global_egress, self.egress), key)
        if not matched:
            return None
        check = EgressCheck(matched, key, payload)
        if threading.current_thread() is self._spin_thread:  # no user code here
            check.refuse("cannot run on the executor thread")
        return check

    def gate(self, kind: str, name: str, cls: Any, payload: Any) -> None:
        """Run the egress hooks for sync callers. Sync hooks run inline; async
        ones on the private loop (on a worker if called from that loop).
        Raises EgressDenied."""
        if not (global_egress._entries or self.egress._entries):
            return
        check = self._egress(kind, name, cls, payload)
        if check is None:
            return
        pending = check.step()
        if pending is None:
            return
        if threading.current_thread() is self._loop_thread:
            ctx = contextvars.copy_context()
            job = self.workers.submit(ctx.run, asyncio.run, check.finish(pending))
            job.result()
        else:
            self.run(check.finish(pending))

    async def agate(self, kind: str, name: str, cls: Any, payload: Any) -> None:
        """Run the egress hooks on the caller's loop. Raises EgressDenied."""
        if not (global_egress._entries or self.egress._entries):
            return
        check = self._egress(kind, name, cls, payload)
        if check is not None:
            await check.finish(check.step())

    # -------------------------------------------------- services and actions

    def client(self, name: str, msg_type: Any, action: bool) -> Client:
        self.check()
        key = (self.resolve(name, "topic" if action else "service"), action)
        cls = message_class(msg_type)
        with self._lock:
            self.check()
            client = self._clients.get(key)
            if client is not None and client.msg_type is not cls:
                self._drop(key, "replaced by a client of another type")
                client = None
            if client is None:
                client = self._clients[key] = Client(
                    self._create(cls, key), cls, key[0]
                )
            return client

    def _create(self, cls: Any, key: tuple[str, bool]) -> Any:
        # rclpy destroys a client the executor is waiting on only after it wakes,
        # so a client of another type may need a few tries (seen on Lyrical).
        deadline = time.monotonic() + 1.0
        while True:
            try:
                if key[1]:
                    return ActionClient(
                        self.node, cls, key[0], callback_group=self.group
                    )
                return self.node.create_client(cls, key[0], callback_group=self.group)
            except ValueError:
                if time.monotonic() > deadline:
                    raise
                self._wake()
                time.sleep(POLL_S)

    def release(self, name: str, action: bool = False) -> bool:
        if self.closed:
            return False
        key = (self.resolve(name, "topic" if action else "service"), action)
        with self._lock:
            return self._drop(key, "released")

    def _drop(self, key: tuple[str, bool], why: str) -> bool:
        client = self._clients.pop(key, None)
        if client is None:
            return False
        client.dropped = ROS2ServiceError(key[0], 0.0, f"client {why}")
        fail(client.pending, client.dropped)
        compat.destroy(self.node, client.entity)
        self._wake()
        return True

    async def ready(self, client: Client, timeout: float, what: str) -> float:
        """Poll readiness at POLL_S; returns the deadline."""
        deadline = time.monotonic() + timeout
        entity = client.entity
        is_ready = (
            entity.server_is_ready
            if isinstance(entity, ActionClient)
            else entity.service_is_ready
        )
        while True:
            self.check()
            if client.dropped is not None:
                raise client.dropped
            if is_ready():
                return deadline
            if time.monotonic() >= deadline:
                raise ROS2TimeoutError(f"{what} not available within {timeout:g} s")
            await asyncio.sleep(POLL_S)

    def serve(
        self, name: str, srv_type: Any, handler: Callable[[Any, Any], Any]
    ) -> str:
        """A service whose handler runs on the worker pool, one request at a time."""
        self.check()
        dispatcher = Dispatcher(self, lambda job: job(), None)

        async def callback(request: Any, response: Any) -> Any:
            done: Future = Future()

            def job() -> None:
                try:
                    done.set_result(handler(request, response))
                except Exception:
                    logger.exception("Service handler of %s failed", name)
                    done.set_result(response)

            dispatcher.push(job)
            return await done

        service = self.node.create_service(
            message_class(srv_type), name, callback, callback_group=self.group
        )
        with self._lock:
            self._servers.append(service)
        return service.srv_name

    def serve_action(
        self,
        name: str,
        action_type: Any,
        execute: Callable[[Any], Any],
        **kwargs: Any,
    ) -> str:
        """An action server whose callbacks never block the executor: execute
        runs on the worker pool, goal and cancel callbacks are guarded."""
        self.check()

        async def run(goal_handle: Any) -> Any:
            return await self.call_in_worker(execute, goal_handle)

        def guard(callback: Callable[[Any], Any], refuse: Any) -> Callable[[Any], Any]:
            def guarded(arg: Any) -> Any:
                try:
                    return callback(arg)
                except Exception:
                    logger.exception("Action callback of %s failed", name)
                    return refuse

            return guarded

        for key, refuse in (
            ("goal_callback", GoalResponse.REJECT),
            ("cancel_callback", CancelResponse.REJECT),
        ):
            if key in kwargs:
                kwargs[key] = guard(kwargs[key], refuse)
        kwargs.setdefault("result_timeout", compat.ACTION_RESULT_TIMEOUT_S)
        kwargs.setdefault("callback_group", self.group)
        server = ActionServer(
            self.node, message_class(action_type), name, run, **kwargs
        )
        with self._lock:
            self._servers.append(server)
        return name

    # ---------------------------------------------------------------- TF

    def tf_buffer(self) -> Any:
        """The TF buffer, created with its listener on first use."""
        with self._lock:
            self.check()
            if self._tf_buffer is None:
                from tf2_ros import Buffer, TransformListener

                self._tf_buffer = Buffer(node=self.node)
                self._tf_listener = TransformListener(self._tf_buffer, self.node)
            return self._tf_buffer


# Runtimes still open at interpreter exit are closed before finalization, which
# would otherwise destroy rclpy objects under a running executor (abort).
_live: "weakref.WeakSet[ROS2Runtime]" = weakref.WeakSet()


# Contexts RAI initialized itself are shut down at exit too: otherwise exit
# unloads the RMW library under live DDS threads (a segfault, seen on Lyrical).
_owned_contexts: list[Context] = []


def _close_at_exit() -> None:
    for runtime in list(_live):
        runtime.close()
    for context in _owned_contexts:
        rclpy.try_shutdown(context=context)
        context.destroy()


def _track(runtime: ROS2Runtime) -> None:
    if not getattr(_track, "registered", False):
        atexit.register(_close_at_exit)
        setattr(_track, "registered", True)
    _live.add(runtime)


_shared: Optional[ROS2Runtime] = None
_shared_lock = threading.Lock()


def acquire(runtime: Optional[ROS2Runtime] = None) -> ROS2Runtime:
    """Take a reference to runtime, or to the process-wide shared runtime."""
    global _shared
    with _shared_lock:
        if runtime is None:
            if _shared is None or _shared.closed:
                _shared = ROS2Runtime()
            runtime = _shared
        runtime.check()
        runtime.users += 1
        return runtime


def release(runtime: ROS2Runtime) -> None:
    """Drop a reference; the last one closes the runtime."""
    with _shared_lock:
        runtime.users -= 1
        last = runtime.users <= 0
    if last:
        runtime.close()
