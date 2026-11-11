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

"""RAI 3.0 acceptance oracle for the ROS 2 connectors.

One test per failure mode (races, hangs, leaks, lost or stale data), asserting
the 3.0 behavior (rai3-plan/PLAN.md 2.1). Most tests drive the 2.x method
names, which stay on the connector as thin shims until R5.

Rules that keep the suite load tolerant: assert counts, ordering and leaks, not
durations; every wait that may hang runs in a daemon thread with a bound, so a
broken connector fails a test instead of the whole run.
"""

import copy
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from example_interfaces.action import Fibonacci
from geometry_msgs.msg import TransformStamped
from PIL import Image as PILImage
from pydub import AudioSegment
from rai.communication.base_connector import ConnectorClosedError
from rai.communication.ros2 import (
    ROS2Connector,
    ROS2HRIConnector,
    ROS2HRIMessage,
    ROS2Message,
)
from rai.communication.ros2.context import NodeDiscovery
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.communication.ros2.runtime import ROS2Runtime
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from tf2_ros import LookupException, StaticTransformBroadcaster

from rai_interfaces.msg import HRIMessage

from .helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)

STRING, INT32 = "std_msgs/msg/String", "std_msgs/msg/Int32"
TRIGGER, FIBONACCI = "std_srvs/srv/Trigger", "example_interfaces/action/Fibonacci"
LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


# -------------------------------------------------------------------- helpers


def unique(prefix: str) -> str:
    return f"/fault_{prefix}_{uuid.uuid4().hex[:8]}"


def wait_until(predicate: Callable[[], Any], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def run_bounded(fn: Callable[[], Any], timeout: float) -> tuple[bool, Any]:
    """Run fn in a daemon thread. Returns (finished, result or raised exception)."""
    out: list[Any] = [None]

    def target() -> None:
        try:
            out[0] = fn()
        except BaseException as e:
            out[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive(), out[0]


def close_quietly(connector: Any) -> None:
    run_bounded(connector.shutdown, 5.0)
    pool = getattr(connector, "callback_executor", None)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def connectors(ros_setup: None):
    """Factory for connectors that are always shut down after the test."""
    made: list[Any] = []

    def make(cls: Any = ROS2Connector, **kwargs: Any) -> Any:
        made.append(cls(**kwargs))
        return made[-1]

    yield make
    for connector in made:
        close_quietly(connector)


@pytest.fixture
def helper_node(ros_setup: None):
    """Factory for plain rclpy nodes spun in their own thread."""
    spun: list[tuple] = []

    def make(executor_cls: Any = MultiThreadedExecutor) -> Node:
        node = Node(f"fault_helper_{uuid.uuid4().hex[:8]}")
        executor = executor_cls()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        spun.append((node, executor, thread))
        return node

    yield make
    for node, executor, thread in spun:
        executor.shutdown(timeout_sec=1.0)
        thread.join(timeout=2.0)
        node.destroy_node()


def wait_for_subscriber(publisher: Any) -> None:
    assert wait_until(lambda: publisher.get_subscription_count() > 0)


def publish_until(publisher: Any, msg: Any, done: Callable[[], Any]) -> bool:
    """Publish msg every 20 ms until done() (discovery is asymmetric)."""
    return wait_until(lambda: publisher.publish(msg) or done())


@contextmanager
def publishing(publisher: Any, msg: Any):
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            publisher.publish(msg)
            stop.wait(0.02)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(1.0)


def inversions(seq: list[int]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if b < a)


def send(connector: Any, topic: str, data: Any, **kwargs: Any) -> None:
    msg_type = STRING if isinstance(data, str) else INT32
    payload = ROS2Message(payload={"data": data})
    connector.send_message(payload, topic, msg_type=msg_type, **kwargs)


def make_trigger_service(node: Node, name: str, handler: Any = None) -> Any:
    def default(request: Any, response: Any) -> Any:
        response.success = True
        return response

    group = ReentrantCallbackGroup()
    return node.create_service(Trigger, name, handler or default, callback_group=group)


def call_trigger(client: Any, timeout: float) -> Any:
    future = client.call_async(Trigger.Request())
    wait_until(future.done, timeout)
    return future.result() if future.done() else None


def make_fibonacci_server(node: Node, name: str, **kwargs: Any) -> ActionServer:
    def execute(goal_handle: Any) -> Any:
        goal_handle.succeed()
        return Fibonacci.Result()

    kwargs.setdefault("execute_callback", execute)
    group = ReentrantCallbackGroup()
    return ActionServer(node, Fibonacci, name, callback_group=group, **kwargs)


def start_goal(connector: Any, name: str, **kwargs: Any) -> str:
    kwargs.setdefault("timeout_sec", 3.0)
    goal = ROS2Message(payload={"order": 1})
    return connector.start_action(goal, name, msg_type=FIBONACCI, **kwargs)


def wait_for_action(connector: Any, name: str) -> None:
    names = lambda: [n for n, _ in connector.get_actions_names_and_types()]
    assert wait_until(lambda: name in names(), 10.0)


# ------------------------------------------------------------------- dispatch


def test_spin_survives_malformed_hri_image(connectors, helper_node):
    topic = unique("hri")
    connector = connectors(ROS2HRIConnector)
    got: list[str] = []
    connector.register_callback(topic, lambda m: got.append(m.text))
    pub = helper_node().create_publisher(HRIMessage, topic, 10)
    wait_for_subscriber(pub)

    bad = HRIMessage(text="bad")
    bad.images = [numpy_to_imgmsg(np.zeros((4, 4), np.float32), "32FC1")]
    pub.publish(bad)
    assert publish_until(pub, HRIMessage(text="good"), lambda: "good" in got)


def test_dispatch_survives_register_unregister_churn(connectors, helper_node):
    topic = unique("churn")
    connector = connectors()
    got: list[int] = []
    connector.register_callback(
        topic, lambda m: got.append(m.data), raw=True, msg_type=INT32
    )
    pub = helper_node().create_publisher(Int32, topic, 10)
    assert publish_until(pub, Int32(data=-1), lambda: got)

    stop = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            ids = [connector.register_callback(topic, lambda m: None) for _ in range(5)]
            for callback_id in ids:
                connector.unregister_callback(callback_id)

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    churner = threading.Thread(target=churn, daemon=True)
    churner.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            pub.publish(Int32(data=0))
            time.sleep(0.0002)
    finally:
        stop.set()
        churner.join(timeout=2.0)
        sys.setswitchinterval(old_interval)
    got.clear()
    assert publish_until(pub, Int32(data=1), lambda: 1 in got)


def test_per_callback_ordering(connectors, helper_node):
    topic = unique("order")
    connector = connectors()
    got: list[int] = []

    def callback(msg: Int32) -> None:
        if msg.data % 50 == 0:
            time.sleep(0.05)  # a parallel dispatcher reorders around this one
        got.append(msg.data)

    connector.register_callback(topic, callback, raw=True, msg_type=INT32)
    pub = helper_node().create_publisher(Int32, topic, 500)
    assert publish_until(pub, Int32(data=-1), lambda: got)
    for i in range(200):
        pub.publish(Int32(data=i))
        time.sleep(0.001)
    wait_until(lambda: got and got[-1] == 199, 5.0)
    received = [i for i in got if i >= 0]
    assert len(received) >= 10
    assert inversions(received) == 0


def test_hri_burst_delivered_in_order(connectors, helper_node):
    topic = unique("hriburst")
    connector = connectors(ROS2HRIConnector)
    got: list[int] = []

    def slow(msg: Any) -> None:
        time.sleep(0.01)
        got.append(int(msg.text))

    connector.register_callback(topic, slow)
    pub = helper_node().create_publisher(HRIMessage, topic, 100)
    assert publish_until(pub, HRIMessage(text="-1"), lambda: got)
    for i in range(50):
        pub.publish(HRIMessage(text=str(i)))
    wait_until(lambda: got and got[-1] == 49, 10.0)
    assert [i for i in got if i >= 0] == list(range(50))


def test_receive_message_not_starved_by_slow_callbacks(connectors, helper_node):
    slow_topic, fast_topic = unique("slow"), unique("fast")
    connector = connectors()
    release = threading.Event()
    started: list[int] = []
    for _ in range(8):
        connector.register_callback(
            slow_topic,
            lambda m: (started.append(m.data), release.wait(10.0)),
            raw=True,
            msg_type=INT32,
        )
    node = helper_node()
    slow_pub = node.create_publisher(Int32, slow_topic, 10)
    fast_pub = node.create_publisher(Int32, fast_topic, 10)
    try:
        assert publish_until(slow_pub, Int32(data=1), lambda: len(started) >= 4)
        with publishing(fast_pub, Int32(data=1)):
            msg = connector.receive_message(fast_topic, timeout_sec=3.0, msg_type=INT32)
        assert msg.payload.data == 1
    finally:
        release.set()


# --------------------------------------------------------------------- topics


def test_receive_message_on_topic_with_registered_callback(connectors, helper_node):
    topic = unique("regrecv")
    connector = connectors()
    connector.register_callback(topic, lambda m: None, msg_type=STRING)
    pub = helper_node().create_publisher(String, topic, 10)
    wait_for_subscriber(pub)
    with publishing(pub, String(data="x")):
        msg = connector.receive_message(topic, timeout_sec=3.0, msg_type=STRING)
    assert msg.payload.data == "x"


def test_receive_message_max_age_rejects_stale_sample(connectors, helper_node):
    topic = unique("stale")
    connector = connectors()
    pub = helper_node().create_publisher(String, topic, 10)
    with publishing(pub, String(data="x")):
        connector.receive_message(topic, timeout_sec=3.0, msg_type=STRING)
    time.sleep(0.5)  # the publisher is silent: the cached sample is now stale
    try:
        msg = connector.receive_message(topic, timeout_sec=0.3, max_age=0.3)
    except TimeoutError:
        return
    pytest.fail(f"returned a sample {time.time() - msg.timestamp:.2f} s old")


def test_receive_message_returns_latched_sample_of_any_age(connectors, helper_node):
    """Without max_age a latched sample (like /map) is the current one."""
    topic = unique("latched")
    helper_node().create_publisher(String, topic, LATCHED).publish(String(data="map"))
    connector = connectors()
    assert wait_until(lambda: connector._node.count_publishers(topic) == 1)
    first = connector.receive_message(topic, timeout_sec=3.0)
    time.sleep(0.4)
    second = connector.receive_message(topic, timeout_sec=0.2)
    assert first.payload.data == second.payload.data == "map"


def test_register_callback_gets_latched_sample(connectors, helper_node):
    topic = unique("latched")
    helper_node().create_publisher(String, topic, LATCHED).publish(String(data="map"))
    connector = connectors()
    got: list[Any] = []
    assert wait_until(lambda: connector._node.count_publishers(topic) == 1)
    connector.register_callback(topic, got.append, msg_type=STRING)
    assert wait_until(lambda: got)
    assert got[0].payload.data == "map"


def test_send_message_does_not_mutate_payload(connectors):
    connector = connectors()
    payload = {"poses": [{"position": {"x": 1.0}}, {"position": {"y": 2.0}}]}
    expected = copy.deepcopy(payload)
    connector.send_message(
        ROS2Message(payload=payload),
        unique("mut"),
        msg_type="geometry_msgs/msg/PoseArray",
    )
    assert payload == expected


def test_publish_is_not_latched_by_default(connectors, helper_node):
    """COMMAND QoS: a late subscriber (a restarted controller, a recorder) must
    never get a stale command replayed, whatever durability it asks for."""
    topic = unique("cmd")
    connector = connectors()
    send(connector, topic, "old")
    node = helper_node()
    got: dict[Any, list[str]] = {
        DurabilityPolicy.VOLATILE: [],
        DurabilityPolicy.TRANSIENT_LOCAL: [],
    }
    for durability, sink in got.items():
        qos = QoSProfile(depth=10, durability=durability)
        node.create_subscription(String, topic, lambda m, s=sink: s.append(m.data), qos)
    # The VOLATILE reader gets new commands, so it is matched.
    assert wait_until(
        lambda: send(connector, topic, "new") or got[DurabilityPolicy.VOLATILE]
    )
    assert "old" not in sum(got.values(), [])


def test_latched_publish_reaches_late_subscriber(connectors, helper_node):
    """Latching is opt-in (LATCHED QoS), e.g. for /map or a mission state."""
    topic = unique("latch")
    connector = connectors()
    send(connector, topic, "map", qos_profile=LATCHED)
    got: list[str] = []
    helper_node().create_subscription(
        String, topic, lambda m: got.append(m.data), LATCHED
    )
    assert wait_until(lambda: got == ["map"])


def test_publish_same_topic_different_types(connectors):
    topic = unique("types")
    connector = connectors()
    send(connector, topic, 1)
    with pytest.raises(Exception, match=topic):
        send(connector, topic, "x")


def test_concurrent_first_use_creates_one_entity(connectors):
    connector = connectors()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # widen the check-then-create window
    try:
        for k in range(10):
            sub_topic, pub_topic = unique(f"dupsub{k}"), unique(f"duppub{k}")
            barrier = threading.Barrier(8)

            def receive() -> None:
                barrier.wait()
                try:
                    connector.receive_message(
                        sub_topic, timeout_sec=0.01, msg_type=INT32
                    )
                except TimeoutError:
                    pass

            def register() -> None:
                barrier.wait()
                connector.register_callback(sub_topic, lambda m: None, msg_type=INT32)

            def publish() -> None:
                barrier.wait()
                send(connector, pub_topic, 1)

            threads = [
                threading.Thread(target=fn, daemon=True)
                for fn in [receive, receive, register, register] + [publish] * 4
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5.0)
            node = connector._node
            subs = [s for s in node.subscriptions if s.topic_name == sub_topic]
            pubs = [p for p in node.publishers if p.topic_name == pub_topic]
            assert (len(subs), len(pubs)) == (1, 1), f"round {k}"
    finally:
        sys.setswitchinterval(old_interval)


def test_relative_and_absolute_topic_names_are_one_topic(connectors, helper_node):
    topic = unique("rel")
    connector = connectors()
    got: list[Any] = []
    connector.register_callback(topic.lstrip("/"), got.append, msg_type=STRING)
    pub = helper_node().create_publisher(String, topic, 10)
    assert publish_until(pub, String(data="x"), lambda: got)
    with publishing(pub, String(data="x")):
        assert connector.receive_message(topic, timeout_sec=3.0).payload.data == "x"
    send(connector, topic, "a")
    send(connector, topic.lstrip("/"), "b")
    node = connector._node
    assert len([s for s in node.subscriptions if s.topic_name == topic]) == 1
    assert len([p for p in node.publishers if p.topic_name == topic]) == 1


def test_explicit_subscription_qos_is_used(connectors):
    connector = connectors()
    topic = unique("qos_explicit")
    connector.register_callback(
        topic, lambda m: None, msg_type=STRING, qos_profile=LATCHED
    )
    (sub,) = [s for s in connector._node.subscriptions if s.topic_name == topic]
    assert sub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_early_subscription_matches_best_effort_publisher(connectors, helper_node):
    topic = unique("sensor")
    connector = connectors()
    got: list[Any] = []
    connector.register_callback(topic, got.append, msg_type=STRING)
    pub = helper_node().create_publisher(String, topic, qos_profile_sensor_data)
    assert publish_until(pub, String(data="x"), lambda: got)


def test_hri_subscription_is_reliable_without_publishers(connectors):
    # HRI QoS is RELIABLE: best effort drops large image frames.
    connector = connectors(ROS2HRIConnector)
    topic = unique("hri_qos")
    connector.register_callback(topic, lambda m: None)
    (sub,) = [s for s in connector._node.subscriptions if s.topic_name == topic]
    assert sub.qos_profile.reliability == ReliabilityPolicy.RELIABLE


def test_hri_message_round_trip(connectors, helper_node):
    topic = unique("hri_instance")
    connector = connectors(ROS2HRIConnector)
    received: list[HRIMessage] = []
    helper_node().create_subscription(HRIMessage, topic, received.append, 10)
    pixels = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    audio = AudioSegment(
        data=bytes(range(256)) * 4, frame_rate=16000, sample_width=2, channels=1
    )
    message = ROS2HRIMessage(
        text="hello",
        images=[PILImage.fromarray(pixels)],
        audios=[audio],
        message_author="ai",
        communication_id="conversation",
        seq_no=3,
        seq_end=True,
    )
    assert wait_until(lambda: connector.send_message(message, topic) or received)
    restored = ROS2HRIMessage.from_ros2(received[0], message_author="ai")
    assert restored.text == "hello"
    assert np.array_equal(np.array(restored.images[0]), pixels)
    assert restored.audios[0].raw_data == audio.raw_data
    assert restored.audios[0].frame_rate == 16000
    assert (restored.communication_id, restored.seq_no, restored.seq_end) == (
        "conversation",
        3,
        True,
    )


# ------------------------------------------------------------------- services


def test_service_call_does_not_mutate_payload(connectors):
    connector = connectors()
    payload = {"parameters": [{"name": "a", "value": {"type": 1, "bool_value": True}}]}
    expected = copy.deepcopy(payload)
    run_bounded(
        lambda: connector.service_call(
            ROS2Message(payload=payload),
            unique("mutsrv"),
            timeout_sec=0.1,
            msg_type="rcl_interfaces/srv/SetParameters",
        ),
        3.0,
    )
    assert payload == expected


def test_service_call_from_service_handler(connectors, helper_node):
    inner, outer = unique("inner"), unique("outer")
    connector = connectors()
    node = helper_node()
    make_trigger_service(node, inner)

    def handler(request: Any, response: Any) -> Any:
        try:
            r = connector.service_call(
                ROS2Message(payload={}), inner, timeout_sec=3.0, msg_type=TRIGGER
            )
            response.success = r.payload.success
        except Exception as e:
            response.message = repr(e)
        return response

    connector.create_service(outer, handler, service_type=TRIGGER)
    client = node.create_client(Trigger, outer)
    assert client.wait_for_service(timeout_sec=3.0)
    response = call_trigger(client, 8.0)
    assert response is not None and response.success, response


@pytest.mark.parametrize("kind", ["service", "action"])
def test_unreachable_server_raises_timeout(connectors, kind):
    connector = connectors()
    msg = ROS2Message(payload={})
    with pytest.raises(TimeoutError):
        if kind == "service":
            connector.service_call(
                msg, unique("nosrv"), timeout_sec=0.3, msg_type=TRIGGER
            )
        else:
            connector.start_action(
                msg, unique("noact"), timeout_sec=0.3, msg_type=FIBONACCI
            )


def test_service_call_timeout_raises(connectors, helper_node):
    name = unique("slowsrv")
    connector = connectors()
    release = threading.Event()

    def slow(request: Any, response: Any) -> Any:
        release.wait(5.0)
        return response

    make_trigger_service(helper_node(), name, slow)
    try:
        with pytest.raises(TimeoutError):
            connector.service_call(
                ROS2Message(payload={}), name, timeout_sec=0.5, msg_type=TRIGGER
            )
    finally:
        release.set()


SERVER = """
import rclpy
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci
from std_srvs.srv import SetBool
rclpy.init()
node = rclpy.create_node("fault_server")
node.create_service(SetBool, "{name}", lambda req, res: setattr(res, "success", True) or res)
ActionServer(node, Fibonacci, "{name}", lambda goal: goal.succeed() or Fibonacci.Result())
rclpy.spin(node)
"""


@contextmanager
def server_process(name: str):
    """A SetBool service and a Fibonacci action named name, in another process
    (in-process, rmw rejects a client of another type at creation)."""
    server = subprocess.Popen([sys.executable, "-c", SERVER.format(name=name)])
    try:
        yield
    finally:
        server.kill()
        server.wait()


def test_wrong_service_type_then_correct_type(connectors):
    name = unique("setbool")
    connector = connectors()
    with server_process(name):
        srv = (name, ["std_srvs/srv/SetBool"])
        assert wait_until(lambda: srv in connector.get_services_names_and_types(), 10.0)
        with pytest.raises(Exception):
            connector.service_call(
                ROS2Message(payload={}), name, timeout_sec=0.3, msg_type=TRIGGER
            )
        # Fast DDS may drop the first response to a fresh client (discovery
        # race); a poisoned cache fails every attempt.
        for attempt in range(3):
            try:
                response = connector.service_call(
                    ROS2Message(payload={"data": True}),
                    name,
                    timeout_sec=2.0,
                    msg_type="std_srvs/srv/SetBool",
                )
                break
            except Exception:
                if attempt == 2:
                    raise
        assert response.payload.success


def test_no_client_leak_without_reuse(connectors, helper_node):
    name = unique("leak")
    connector = connectors()
    make_trigger_service(helper_node(), name)
    before = len(list(connector._node.clients))
    for _ in range(10):
        connector.service_call(
            ROS2Message(payload={}),
            name,
            timeout_sec=3.0,
            msg_type=TRIGGER,
            reuse_client=False,
        )
    assert len(list(connector._node.clients)) - before <= 1


def test_release_client_destroys_client(connectors, helper_node):
    name = unique("release")
    connector = connectors()
    make_trigger_service(helper_node(), name)
    before = len(list(connector._node.clients))
    connector.service_call(
        ROS2Message(payload={}), name, timeout_sec=3.0, msg_type=TRIGGER
    )
    assert len(list(connector._node.clients)) == before + 1
    assert connector.release_client(name)
    assert len(list(connector._node.clients)) == before
    assert not connector.release_client(name)


def test_release_client_fails_call_in_flight(connectors, helper_node):
    name = unique("released")
    connector = connectors()
    called, release = threading.Event(), threading.Event()

    def slow(request: Any, response: Any) -> Any:
        called.set()
        release.wait(10.0)
        return response

    make_trigger_service(helper_node(), name, slow)
    call = lambda: connector.service_call(
        ROS2Message(payload={}), name, timeout_sec=10.0, msg_type=TRIGGER
    )
    result: list[Any] = []
    caller = threading.Thread(
        target=lambda: result.append(run_bounded(call, 12.0)), daemon=True
    )
    try:
        caller.start()
        assert called.wait(5.0)
        finished, released = run_bounded(lambda: connector.release_client(name), 3.0)
        assert finished and released is True
        caller.join(3.0)
        assert result, "the call in flight was not failed"
        error = result[0][1]
        assert isinstance(error, Exception) and name in str(error), error
    finally:
        release.set()


def test_service_handler_does_not_run_concurrently_with_itself(connectors, helper_node):
    name = unique("serial")
    connector = connectors()
    lock = threading.Lock()
    running, overlap = [0], [0]

    def handler(request: Any, response: Any) -> Any:
        with lock:
            running[0] += 1
            overlap[0] = max(overlap[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        response.success = True
        return response

    connector.create_service(name, handler, service_type=TRIGGER)
    client = helper_node().create_client(Trigger, name)
    assert client.wait_for_service(timeout_sec=3.0)
    futures = [client.call_async(Trigger.Request()) for _ in range(3)]
    assert wait_until(lambda: all(f.done() for f in futures), 10.0)
    assert all(f.result().success for f in futures)
    assert overlap[0] == 1


# -------------------------------------------------------------------- actions


def test_action_feedback_order(connectors, helper_node):
    name = unique("feedback")
    connector = connectors()

    def execute(goal_handle: Any) -> Any:
        for i in range(100):
            goal_handle.publish_feedback(Fibonacci.Feedback(sequence=[i]))
            time.sleep(0.001)
        goal_handle.succeed()
        return Fibonacci.Result()

    make_fibonacci_server(helper_node(), name, execute_callback=execute)
    got: list[int] = []
    done = threading.Event()

    def on_feedback(feedback: Any) -> None:
        if feedback.sequence[0] % 25 == 0:
            time.sleep(0.02)  # a parallel dispatcher reorders around this one
        got.append(feedback.sequence[0])

    start_goal(connector, name, on_feedback=on_feedback, on_done=lambda _: done.set())
    assert done.wait(10.0)
    wait_until(lambda: got and got[-1] == 99, 2.0)
    # Feedback is a lossy stream (depth 10); under load some is dropped.
    assert len(got) >= 10
    assert inversions(got) == 0


def test_on_done_runs_after_feedback(connectors, helper_node):
    name = unique("donelast")
    connector = connectors()

    def execute(goal_handle: Any) -> Any:
        for i in range(5):
            goal_handle.publish_feedback(Fibonacci.Feedback(sequence=[i]))
        time.sleep(0.1)  # the feedback reaches the client before the result
        goal_handle.succeed()
        return Fibonacci.Result()

    make_fibonacci_server(helper_node(), name, execute_callback=execute)
    events: list[Any] = []

    def on_feedback(feedback: Any) -> None:
        if feedback.sequence[0] == 0:
            time.sleep(0.3)
        events.append(feedback.sequence[0])

    start_goal(
        connector,
        name,
        on_feedback=on_feedback,
        on_done=lambda _: events.append("done"),
    )
    assert wait_until(lambda: "done" in events, 5.0)
    assert events == [0, 1, 2, 3, 4, "done"]


def test_slow_accept_does_not_orphan_goal(connectors, helper_node):
    name = unique("slowaccept")
    connector = connectors()
    outcomes: list[str] = []

    def goal_callback(request: Any) -> Any:
        time.sleep(1.0)
        return GoalResponse.ACCEPT

    def execute(goal_handle: Any) -> Any:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                outcomes.append("canceled")
                return Fibonacci.Result()
            time.sleep(0.02)
        goal_handle.succeed()
        outcomes.append("succeeded")
        return Fibonacci.Result()

    make_fibonacci_server(
        helper_node(),
        name,
        execute_callback=execute,
        goal_callback=goal_callback,
        cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
    )
    wait_for_action(connector, name)
    with pytest.raises(Exception):
        start_goal(connector, name, timeout_sec=0.3)
    assert wait_until(lambda: outcomes, 5.0)
    assert outcomes == ["canceled"]


def test_rejected_goal_does_not_fire_on_done(connectors, helper_node):
    name = unique("reject")
    connector = connectors()
    make_fibonacci_server(
        helper_node(), name, goal_callback=lambda request: GoalResponse.REJECT
    )
    done: list[Any] = []
    with pytest.raises(Exception):
        start_goal(connector, name, on_done=done.append)
    time.sleep(0.2)
    assert done == []


def test_terminate_action_bounded_when_server_gone(connectors, ros_setup):
    name = unique("gone")
    connector = connectors()
    node = Node(f"fault_gone_{uuid.uuid4().hex[:8]}")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    release = threading.Event()

    def execute(goal_handle: Any) -> Any:
        release.wait(5.0)
        goal_handle.abort()
        return Fibonacci.Result()

    server = make_fibonacci_server(node, name, execute_callback=execute)
    handle = start_goal(connector, name)
    release.set()
    server.destroy()
    executor.shutdown(timeout_sec=1.0)
    thread.join(timeout=2.0)
    node.destroy_node()

    finished, _ = run_bounded(lambda: connector.terminate_action(handle), 5.0)
    assert finished


def test_action_entities_not_leaked(connectors, helper_node):
    name = unique("fibleak")
    connector = connectors()
    make_fibonacci_server(helper_node(), name)
    before = len(list(connector._node.waitables))
    for _ in range(40):
        done = threading.Event()
        start_goal(connector, name, on_done=lambda _: done.set())
        assert done.wait(5.0)
    assert len(list(connector._node.waitables)) - before <= 1


def test_start_action_from_service_handler(connectors, helper_node):
    action, srv = unique("innerfib"), unique("outersrv")
    connector = connectors()
    node = helper_node()
    make_fibonacci_server(node, action)
    done = threading.Event()

    def handler(request: Any, response: Any) -> Any:
        try:
            start_goal(connector, action, on_done=lambda _: done.set())
            response.success = done.wait(3.0)
        except Exception as e:
            response.message = repr(e)
        return response

    connector.create_service(srv, handler, service_type=TRIGGER)
    client = node.create_client(Trigger, srv)
    assert client.wait_for_service(timeout_sec=3.0)
    response = call_trigger(client, 8.0)
    assert response is not None and response.success, response


def test_wrong_action_type_does_not_block_action_name(connectors):
    name = unique("wrongact")
    connector = connectors()
    # No server yet: both goals fail, neither may poison the name.
    for msg_type in ("nav2_msgs/action/NavigateToPose", FIBONACCI):
        with pytest.raises(Exception):
            connector.start_action(
                ROS2Message(payload={}), name, timeout_sec=0.2, msg_type=msg_type
            )
    with server_process(name):
        wait_for_action(connector, name)
        # Matching the cached client to the new server is DDS discovery: up
        # to about 3 s on Fast DDS.
        assert start_goal(connector, name, timeout_sec=10.0)


# ------------------------------------------------------------------ lifecycle


def test_no_user_callbacks_after_shutdown(connectors, helper_node):
    topic = unique("shut")
    connector = connectors()
    started: list[float] = []

    gate = threading.Event()

    def callback(msg: Int32) -> None:
        started.append(time.monotonic())
        gate.wait(10.0)

    connector.register_callback(topic, callback, raw=True, msg_type=INT32)
    pub = helper_node().create_publisher(Int32, topic, 100)
    try:
        assert publish_until(pub, Int32(data=0), lambda: started)
        for i in range(20):  # queued behind the blocked callbacks
            pub.publish(Int32(data=i))
            time.sleep(0.01)
        finished, _ = run_bounded(connector.shutdown, 5.0)
        returned = time.monotonic()
        assert finished
    finally:
        gate.set()
    time.sleep(0.5)  # a callback started late would be recorded by now
    assert all(t <= returned for t in started)


@pytest.mark.parametrize("cls", [ROS2Connector, ROS2HRIConnector])
def test_every_call_after_shutdown_raises_closed(connectors, cls):
    connector = connectors(cls)
    connector.shutdown()
    msg = ROS2Message(payload={})
    hri = cls is ROS2HRIConnector
    out = ROS2HRIMessage(text="x", message_author="ai") if hri else msg
    calls = {
        "send_message": lambda: connector.send_message(out, "/t", msg_type=STRING),
        "receive_message": lambda: connector.receive_message("/t", msg_type=STRING),
        "register_callback": lambda: connector.register_callback(
            "/t", print, msg_type=INT32
        ),
    }
    if not hri:  # 3.0 HRI connectors only send and receive
        calls |= {
            "service_call": lambda: connector.service_call(msg, "/s", msg_type=TRIGGER),
            "start_action": lambda: connector.start_action(
                msg, "/a", msg_type=FIBONACCI
            ),
            "terminate_action": lambda: connector.terminate_action("handle"),
            "create_service": lambda: connector.create_service(
                "/s", print, service_type=TRIGGER
            ),
            "create_action": lambda: connector.create_action(
                "/a", print, action_type=FIBONACCI
            ),
            "get_transform": lambda: connector.get_transform("a", "b", timeout_sec=0.1),
            "get_topics": connector.get_topics_names_and_types,
            "get_services": connector.get_services_names_and_types,
            "get_actions": connector.get_actions_names_and_types,
        }
    for name, call in calls.items():
        finished, error = run_bounded(call, 2.0)
        assert finished and isinstance(error, ConnectorClosedError), (name, error)


def test_cleanup_after_shutdown_is_noop(connectors):
    connector = connectors()
    callback_id = connector.register_callback(
        unique("cleanup"), lambda _: None, msg_type=STRING
    )
    connector.shutdown()
    connector.unregister_callback(callback_id)
    connector.unregister_callback(callback_id)
    assert connector.release_client(unique("nosrv")) is False


def test_concurrent_shutdown_is_safe(connectors):
    connector = connectors()
    results: list[Any] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(run_bounded(connector.shutdown, 5.0))
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    results.append(run_bounded(connector.shutdown, 5.0))  # and once more after
    assert all(r == (True, None) for r in results), results


def test_shutdown_wakes_waiting_receive_message(connectors):
    connector = connectors()
    result: list[Any] = []
    receive = lambda: connector.receive_message(
        unique("never"), timeout_sec=20.0, msg_type=STRING
    )
    threading.Thread(
        target=lambda: result.append(run_bounded(receive, 25.0)), daemon=True
    ).start()
    time.sleep(0.2)
    run_bounded(connector.shutdown, 5.0)
    assert wait_until(lambda: result, 5.0)
    assert isinstance(result[0][1], ConnectorClosedError), result


def test_shutdown_wakes_waiting_service_call(connectors, helper_node):
    name = unique("stuck")
    connector = connectors()
    called, release = threading.Event(), threading.Event()

    def stuck(request: Any, response: Any) -> Any:
        called.set()
        release.wait(20.0)
        return response

    make_trigger_service(helper_node(), name, stuck)
    result: list[Any] = []
    call = lambda: connector.service_call(
        ROS2Message(payload={}), name, timeout_sec=20.0, msg_type=TRIGGER
    )
    threading.Thread(
        target=lambda: result.append(run_bounded(call, 25.0)), daemon=True
    ).start()
    try:
        assert called.wait(5.0)
        run_bounded(connector.shutdown, 5.0)
        assert wait_until(lambda: result, 5.0)
        assert isinstance(result[0][1], ConnectorClosedError), result
    finally:
        release.set()


def stuck_subscription_callback(connectors, helper_node, release):
    topic = unique("stuckcb")
    connector = connectors()
    started = threading.Event()
    connector.register_callback(
        topic, lambda m: (started.set(), release.wait(20.0)), raw=True, msg_type=INT32
    )
    pub = helper_node().create_publisher(Int32, topic, 10)
    assert publish_until(pub, Int32(data=1), started.is_set)
    return connector


def stuck_execute_callback(connectors, helper_node, release):
    name = unique("stuckfib")
    server = connectors()
    started = threading.Event()

    def execute(goal_handle: Any) -> Any:
        started.set()
        release.wait(20.0)
        goal_handle.abort()
        return Fibonacci.Result()

    server.create_action(name, execute, action_type=FIBONACCI)
    start_goal(connectors(), name)
    assert started.wait(5.0)
    return server


@pytest.mark.parametrize(
    "make_stuck",
    [
        stuck_subscription_callback,
        stuck_execute_callback,
    ],
)
def test_shutdown_bounded_with_stuck_callback(connectors, helper_node, make_stuck):
    release = threading.Event()
    try:
        connector = make_stuck(connectors, helper_node, release)
        # 3.0 acceptance is under 1 s; the bound is generous for loaded hosts.
        finished, error = run_bounded(connector.shutdown, 5.0)
        assert finished and error is None, error
    finally:
        release.set()


def test_shutdown_from_registered_callback(connectors, helper_node):
    topic = unique("stopmsg")
    connector = connectors()
    result: list[Any] = []
    connector.register_callback(
        topic,
        lambda _: result.append(run_bounded(connector.shutdown, 5.0)),
        msg_type=STRING,
    )
    pub = helper_node().create_publisher(String, topic, 10)
    assert publish_until(pub, String(data="stop"), lambda: result)
    assert result[0] == (True, None)


def test_shutdown_joins_connector_threads(connectors, helper_node):
    node = helper_node(SingleThreadedExecutor)  # no lazily started pool threads
    srv, action, topic = unique("thsrv"), unique("thfib"), unique("thtopic")
    make_trigger_service(node, srv)
    make_fibonacci_server(node, action)
    pub = node.create_publisher(String, topic, 10)
    before = set(threading.enumerate())

    connector = connectors()
    got: list[Any] = []
    connector.register_callback(topic, got.append, msg_type=STRING)
    assert publish_until(pub, String(data="x"), lambda: got)
    connector.service_call(
        ROS2Message(payload={}), srv, timeout_sec=3.0, msg_type=TRIGGER
    )
    done = threading.Event()
    start_goal(connector, action, on_done=lambda _: done.set())
    assert done.wait(5.0)

    finished, error = run_bounded(connector.shutdown, 5.0)
    assert finished and error is None, error
    leftover = lambda: [t.name for t in threading.enumerate() if t not in before]
    assert wait_until(lambda: not leftover()), leftover()


def test_hri_connectors_get_distinct_default_node_names(connectors):
    """Default connectors share the process runtime (one node); private
    runtimes each get a fresh name, not one evaluated at import (8a625cf)."""
    first = connectors(ROS2HRIConnector, runtime=ROS2Runtime())
    second = connectors(ROS2HRIConnector, runtime=ROS2Runtime())
    assert first._node.get_name() != second._node.get_name()
    assert connectors(ROS2HRIConnector)._node is connectors(ROS2Connector)._node


def test_hri_connector_forwards_use_sim_time(connectors):
    connector = connectors(ROS2HRIConnector, use_sim_time=True)
    assert connector._node.get_parameter("use_sim_time").value is True


def test_node_discovery_never_exposes_entries_outside_allowlist(ros_setup):
    """A concurrent dict() reader sees whatever the attributes hold, so every
    dict ever assigned to them must already be filtered."""
    allowed = "/rosout"
    leaked: list[str] = []

    class Spy(NodeDiscovery):
        def __setattr__(self, name: str, value: Any) -> None:
            if name.endswith("_and_types"):
                leaked.extend(k for k in value if k != allowed)
            super().__setattr__(name, value)

    node = Node(f"discovery_{uuid.uuid4().hex[:8]}")
    try:
        Spy(node, allowlist=[allowed]).discovery_callback()
        assert not leaked
    finally:
        node.destroy_node()


EXIT_WITHOUT_SHUTDOWN = """
from rai.communication.ros2 import ROS2Connector, ROS2Context

if {use_context}:
    with ROS2Context():
        connector = ROS2Connector()
        connector.register_callback("/exit_test", print, msg_type="std_msgs/msg/String")
    try:
        connector.get_topics_names_and_types()
    except Exception as e:
        print(type(e).__name__)
else:
    connector = ROS2Connector()
    connector.register_callback("/exit_test", print, msg_type="std_msgs/msg/String")
print("done")
"""


@pytest.mark.parametrize(
    "use_context",
    [
        False,
        True,
    ],
)
def test_process_exits_without_shutdown(use_context):
    script = EXIT_WITHOUT_SHUTDOWN.format(use_context=use_context)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the process did not exit")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "done"
    if use_context:
        assert "ConnectorClosedError" in result.stdout
    assert "Traceback" not in result.stderr, result.stderr


# ---------------------------------------------------------------- performance


def test_steady_stream_is_delivered(connectors, helper_node):
    topic = unique("rate")
    connector = connectors()
    got: list[int] = []
    connector.register_callback(
        topic, lambda m: got.append(m.data), raw=True, msg_type=INT32
    )
    pub = helper_node().create_publisher(Int32, topic, 10)
    assert publish_until(pub, Int32(data=-1), lambda: got)
    start = time.monotonic()
    for i in range(200):  # 100 Hz for 2 s
        time.sleep(max(0.0, start + i / 100 - time.monotonic()))
        pub.publish(Int32(data=i))
    delivered = lambda: len({i for i in got if i >= 0})
    wait_until(lambda: delivered() >= 200, 2.0)
    # RELIABLE at 100 Hz loses nothing; the margin is for a loaded host (the
    # current code delivers 60-150). The 1 kHz gate lives in the benchmark.
    assert delivered() >= 190


def test_tf_listener_created_on_first_use(connectors, helper_node):
    connector = connectors()
    has_tf = lambda: any(
        s.topic_name in ("/tf", "/tf_static") for s in connector._node.subscriptions
    )
    assert not has_tf()
    transform = TransformStamped()
    transform.header.frame_id = "lazy_parent"
    transform.child_frame_id = "lazy_child"
    transform.transform.rotation.w = 1.0
    StaticTransformBroadcaster(helper_node()).sendTransform(transform)
    result = connector.get_transform("lazy_parent", "lazy_child", timeout_sec=3.0)
    assert result.child_frame_id == "lazy_child"
    assert has_tf()


def test_get_transform_honors_fractional_timeout(connectors):
    connector = connectors()
    start = time.monotonic()
    with pytest.raises(LookupException):
        connector.get_transform("nowhere", "nothing", timeout_sec=0.3)
    assert 0.25 < time.monotonic() - start < 3.0
