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
the 3.0 behavior (rai3-plan/PLAN.md 2.1), through the connector API.

Rules that keep the suite load tolerant: assert counts, ordering and leaks, not
durations; every wait that may hang runs in a daemon thread with a bound, so a
broken connector fails a test instead of the whole run.
"""

import signal
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
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rai.communication.exceptions import ConnectorClosedError
from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector
from rai.communication.ros2.context import NodeDiscovery
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.communication.ros2.runtime import ROS2Runtime, message_class
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
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
from std_srvs.srv import SetBool, Trigger
from tf2_ros import LookupException, StaticTransformBroadcaster

from rai_interfaces.msg import HRIMessage

from .helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)

TRIGGER, FIBONACCI = "std_srvs/srv/Trigger", "example_interfaces/action/Fibonacci"
NAVIGATE = message_class("nav2_msgs/action/NavigateToPose")
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
    run_bounded(connector.close, 5.0)


def node_of(connector: Any) -> Node:
    """The runtime's node, to count the entities it holds."""
    return connector._runtime.node


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
    msg = String(data=data) if isinstance(data, str) else Int32(data=data)
    connector.sync.publish(topic, msg, **kwargs)


def call(connector: Any, name: str, timeout: float, request: Any = None) -> Any:
    return connector.sync.call(name, request or Trigger.Request(), timeout=timeout)


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


def start_goal(connector: Any, name: str, **kwargs: Any) -> Any:
    kwargs.setdefault("accept_timeout", 3.0)
    return connector.sync.send_goal(name, Fibonacci.Goal(order=1), **kwargs)


def wait_for_action(connector: Any, name: str) -> None:
    names = lambda: [n for n, _ in connector.get_actions_names_and_types()]
    assert wait_until(lambda: name in names(), 10.0)


# ------------------------------------------------------------------- dispatch


def test_spin_survives_malformed_hri_image(connectors, helper_node):
    topic = unique("hri")
    connector = connectors(ROS2HRIConnector)
    got: list[Any] = []
    connector.subscribe(topic, got.append)
    pub = helper_node().create_publisher(HRIMessage, topic, 10)
    wait_for_subscriber(pub)

    bad = HRIMessage(text="bad")
    truncated = numpy_to_imgmsg(np.zeros((4, 4, 3), np.uint8), "rgb8")
    truncated.height = 40
    bad.images = [truncated, numpy_to_imgmsg(np.zeros((4, 4), np.float32), "32FC1")]
    pub.publish(bad)
    assert publish_until(pub, HRIMessage(text="good"), lambda: got and got[-1].text)
    assert wait_until(lambda: any(m.text == "good" for m in got))
    (decoded,) = [m for m in got if m.text == "bad"]  # the rest of it is kept
    assert [i.encoding for i in decoded.images] == ["32FC1"]


def test_dispatch_survives_subscribe_close_churn(connectors, helper_node):
    topic = unique("churn")
    connector = connectors()
    got: list[int] = []
    connector.subscribe(topic, lambda m: got.append(m.data), msg_type=Int32)
    pub = helper_node().create_publisher(Int32, topic, 10)
    assert publish_until(pub, Int32(data=-1), lambda: got)

    stop = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            subs = [connector.subscribe(topic, lambda m: None) for _ in range(5)]
            for sub in subs:
                sub.close()

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

    connector.subscribe(topic, callback, msg_type=Int32)
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

    connector.subscribe(topic, slow)
    pub = helper_node().create_publisher(HRIMessage, topic, 100)
    assert publish_until(pub, HRIMessage(text="-1"), lambda: got)
    for i in range(50):
        pub.publish(HRIMessage(text=str(i)))
    wait_until(lambda: got and got[-1] == 49, 10.0)
    assert [i for i in got if i >= 0] == list(range(50))


def test_receive_not_starved_by_slow_callbacks(connectors, helper_node):
    slow_topic, fast_topic = unique("slow"), unique("fast")
    connector = connectors()
    release = threading.Event()
    started: list[int] = []
    for _ in range(8):
        connector.subscribe(
            slow_topic,
            lambda m: (started.append(m.data), release.wait(10.0)),
            msg_type=Int32,
        )
    node = helper_node()
    slow_pub = node.create_publisher(Int32, slow_topic, 10)
    fast_pub = node.create_publisher(Int32, fast_topic, 10)
    try:
        assert publish_until(slow_pub, Int32(data=1), lambda: len(started) >= 4)
        with publishing(fast_pub, Int32(data=1)):
            msg = connector.sync.receive(fast_topic, timeout=3.0, msg_type=Int32)
        assert msg.data == 1
    finally:
        release.set()


# --------------------------------------------------------------------- topics


def test_receive_on_topic_with_a_subscription(connectors, helper_node):
    topic = unique("regrecv")
    connector = connectors()
    connector.subscribe(topic, lambda m: None, msg_type=String)
    pub = helper_node().create_publisher(String, topic, 10)
    wait_for_subscriber(pub)
    with publishing(pub, String(data="x")):
        msg = connector.sync.receive(topic, timeout=3.0, msg_type=String)
    assert msg.data == "x"


def test_receive_max_age_rejects_stale_sample(connectors, helper_node):
    topic = unique("stale")
    connector = connectors()
    pub = helper_node().create_publisher(String, topic, 10)
    with publishing(pub, String(data="x")):
        connector.sync.receive(topic, timeout=3.0, msg_type=String)
    time.sleep(0.5)  # the publisher is silent: the cached sample is now stale
    with pytest.raises(TimeoutError):
        connector.sync.receive(topic, timeout=0.3, max_age=0.3)


def test_receive_returns_latched_sample_of_any_age(connectors, helper_node):
    """Without max_age a latched sample (like /map) is the current one."""
    topic = unique("latched")
    helper_node().create_publisher(String, topic, LATCHED).publish(String(data="map"))
    connector = connectors()
    assert wait_until(lambda: node_of(connector).count_publishers(topic) == 1)
    first = connector.sync.receive(topic, timeout=3.0)
    time.sleep(0.4)
    second = connector.sync.receive(topic, timeout=0.2)
    assert first.data == second.data == "map"


def test_subscribe_gets_latched_sample(connectors, helper_node):
    topic = unique("latched")
    helper_node().create_publisher(String, topic, LATCHED).publish(String(data="map"))
    connector = connectors()
    got: list[Any] = []
    assert wait_until(lambda: node_of(connector).count_publishers(topic) == 1)
    connector.subscribe(topic, got.append, msg_type=String)
    assert wait_until(lambda: got)
    assert got[0].data == "map"


def test_publish_does_not_mutate_message(connectors):
    connector = connectors()
    msg = PoseArray(poses=[Pose(), Pose()])
    msg.poses[0].position.x = 1.0
    expected = PoseArray(poses=[Pose(), Pose()])
    expected.poses[0].position.x = 1.0
    connector.sync.publish(unique("mut"), msg)
    assert msg == expected


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
    send(connector, topic, "map", qos=LATCHED)
    got: list[str] = []
    helper_node().create_subscription(
        String, topic, lambda m: got.append(m.data), LATCHED
    )
    assert wait_until(lambda: got == ["map"])


def test_publish_same_topic_different_types(connectors, helper_node):
    """A publisher of a new type replaces the old one (one per name, and the
    RMW refuses two types on one topic in a participant)."""
    topic = unique("types")
    connector = connectors()
    send(connector, topic, 1)
    send(connector, topic, "x")
    publishers = node_of(connector).get_publishers_info_by_topic(topic)
    assert [p.topic_type for p in publishers] == ["std_msgs/msg/String"]
    got: list[str] = []
    helper_node().create_subscription(String, topic, lambda m: got.append(m.data), 10)
    assert wait_until(lambda: send(connector, topic, "y") or got)


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
                    connector.sync.receive(sub_topic, timeout=0.01, msg_type=Int32)
                except TimeoutError:
                    pass

            def subscribe() -> None:
                barrier.wait()
                connector.subscribe(sub_topic, lambda m: None, msg_type=Int32)

            def publish() -> None:
                barrier.wait()
                send(connector, pub_topic, 1)

            threads = [
                threading.Thread(target=fn, daemon=True)
                for fn in [receive, receive, subscribe, subscribe] + [publish] * 4
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5.0)
            node = node_of(connector)
            subs = [s for s in node.subscriptions if s.topic_name == sub_topic]
            pubs = [p for p in node.publishers if p.topic_name == pub_topic]
            assert (len(subs), len(pubs)) == (1, 1), f"round {k}"
    finally:
        sys.setswitchinterval(old_interval)


def test_relative_and_absolute_topic_names_are_one_topic(connectors, helper_node):
    topic = unique("rel")
    connector = connectors()
    got: list[Any] = []
    connector.subscribe(topic.lstrip("/"), got.append, msg_type=String)
    pub = helper_node().create_publisher(String, topic, 10)
    assert publish_until(pub, String(data="x"), lambda: got)
    with publishing(pub, String(data="x")):
        assert connector.sync.receive(topic, timeout=3.0).data == "x"
    send(connector, topic, "a")
    send(connector, topic.lstrip("/"), "b")
    node = node_of(connector)
    assert len([s for s in node.subscriptions if s.topic_name == topic]) == 1
    assert len([p for p in node.publishers if p.topic_name == topic]) == 1


def test_explicit_subscription_qos_is_used(connectors):
    connector = connectors()
    topic = unique("qos_explicit")
    connector.subscribe(topic, lambda m: None, msg_type=String, qos=LATCHED)
    (sub,) = [s for s in node_of(connector).subscriptions if s.topic_name == topic]
    assert sub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_early_subscription_matches_best_effort_publisher(connectors, helper_node):
    topic = unique("sensor")
    connector = connectors()
    got: list[Any] = []
    connector.subscribe(topic, got.append, msg_type=String)
    pub = helper_node().create_publisher(String, topic, qos_profile_sensor_data)
    assert publish_until(pub, String(data="x"), lambda: got)


def test_hri_subscription_is_reliable_without_publishers(connectors):
    # HRI QoS is RELIABLE: best effort drops large image frames.
    connector = connectors(ROS2HRIConnector)
    topic = unique("hri_qos")
    connector.subscribe(topic, lambda m: None)
    (sub,) = [s for s in node_of(connector).subscriptions if s.topic_name == topic]
    assert sub.qos_profile.reliability == ReliabilityPolicy.RELIABLE


# ------------------------------------------------------------------- services


def test_call_does_not_mutate_request(connectors):
    connector = connectors()

    def request() -> Any:
        value = ParameterValue(type=1, bool_value=True)
        return SetParameters.Request(parameters=[Parameter(name="a", value=value)])

    sent, expected = request(), request()
    run_bounded(lambda: call(connector, unique("mutsrv"), 0.1, sent), 3.0)
    assert sent == expected


def test_call_from_service_handler(connectors, helper_node):
    inner, outer = unique("inner"), unique("outer")
    connector = connectors()
    node = helper_node()
    make_trigger_service(node, inner)

    def handler(request: Any, response: Any) -> Any:
        try:
            response.success = call(connector, inner, 3.0).success
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
    with pytest.raises(TimeoutError):
        if kind == "service":
            call(connector, unique("nosrv"), 0.3)
        else:
            start_goal(connector, unique("noact"), accept_timeout=0.3)


def test_call_timeout_raises(connectors, helper_node):
    name = unique("slowsrv")
    connector = connectors()
    release = threading.Event()

    def slow(request: Any, response: Any) -> Any:
        release.wait(5.0)
        return response

    make_trigger_service(helper_node(), name, slow)
    try:
        with pytest.raises(TimeoutError):
            call(connector, name, 0.5)
    finally:
        release.set()


SERVER = """
import signal
from example_interfaces.action import Fibonacci
from std_srvs.srv import SetBool
from rai.communication.ros2 import ROS2Connector
connector = ROS2Connector()
connector.create_service("{name}", lambda req, res: setattr(res, "success", True) or res, service_type=SetBool)
connector.create_action("{name}", lambda goal: goal.succeed() or Fibonacci.Result(), action_type=Fibonacci)
try:
    signal.pause()
except KeyboardInterrupt:
    connector.close()
"""


@contextmanager
def server_process(name: str):
    """A SetBool service and a Fibonacci action named name, in another process
    (in-process, rmw rejects a client of another type at creation). RAI
    servers: a plain rclpy one may drop the response to a client it has not
    discovered yet (test_fresh_connector_clients_of_new_rclpy_servers)."""
    server = subprocess.Popen([sys.executable, "-c", SERVER.format(name=name)])
    try:
        yield
    finally:
        server.send_signal(signal.SIGINT)  # a killed participant lingers
        try:
            server.wait(5.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def test_type_change_replaces_the_entity_of_the_old_type(connectors):
    """rmw_fastrtps refuses an entity whose type differs from one still on the
    same topic in this participant (RCLError on both distros), and rclpy
    destroys an entity the executor waits on only once it wakes. So the old
    client, action client, publisher or idle subscription must be gone before
    the one of the new type is created."""
    connector = connectors()
    srv, act, pub, sub = (unique(p) for p in ("tsrv", "tact", "tpub", "tsub"))
    for i in range(10):
        first = i % 2 == 0
        with pytest.raises(TimeoutError):  # no server: readiness times out
            call(
                connector, srv, 0.05, Trigger.Request() if first else SetBool.Request()
            )
        with pytest.raises(TimeoutError):
            goal = Fibonacci.Goal() if first else NAVIGATE.Goal()
            connector.sync.send_goal(act, goal, accept_timeout=0.05)
        connector.sync.publish(pub, String(data="x") if first else Int32(data=1))
        with pytest.raises(TimeoutError):
            connector.sync.receive(
                sub, msg_type=String if first else Int32, timeout=0.05
            )
    # A subscription with consumers is never replaced under them.
    connector.subscribe(sub, lambda _: None, msg_type=Int32)
    with pytest.raises(TypeError, match=sub):
        connector.sync.receive(sub, msg_type=String, timeout=0.05)


def test_wrong_service_type_then_correct_type(connectors):
    name = unique("setbool")
    connector = connectors()
    with server_process(name):
        srv = (name, ["std_srvs/srv/SetBool"])
        assert wait_until(lambda: srv in connector.get_services_names_and_types(), 10.0)
        with pytest.raises(Exception):
            call(connector, name, 0.3)
        # A poisoned cache would fail this.
        assert call(connector, name, 5.0, SetBool.Request(data=True)).success


FRESH_CLIENTS, SERVER_PROCESSES = 200, 20

MANY_SERVICES = """
import sys
{setup}
print("up", flush=True)
sys.stdin.read()
{teardown}
"""
RCLPY_SERVICES = (
    """
import rclpy
from std_srvs.srv import SetBool
rclpy.init()
node = rclpy.create_node("fault_many")
for i in range({first}, {n}, {step}):
    node.create_service(SetBool, "{prefix}_%d" % i, lambda req, res: setattr(res, "success", True) or res)
import threading
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
""",
    "rclpy.shutdown()",
)
RAI_SERVICES = (
    """
from rai.communication.ros2 import ROS2Connector
connector = ROS2Connector()
for i in range({first}, {n}, {step}):
    connector.create_service("{prefix}_%d" % i, lambda req, res: setattr(res, "success", True) or res, service_type="std_srvs/srv/SetBool")
""",
    "connector.close()",
)


@contextmanager
def many_services(code: tuple[str, str], prefix: str, n: int):
    """n services {prefix}_<i> spread over SERVER_PROCESSES fresh processes:
    each new participant is one more chance for the race. They exit cleanly:
    a killed participant lingers in discovery and slows the next ones."""
    setup, teardown = code
    servers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                MANY_SERVICES.format(
                    setup=setup.format(
                        first=first, n=n, step=SERVER_PROCESSES, prefix=prefix
                    ),
                    teardown=teardown,
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        for first in range(SERVER_PROCESSES)
    ]
    try:
        for server in servers:
            assert server.stdout is not None and server.stdout.readline() == b"up\n"
        yield
    finally:
        for server in servers:
            if server.stdin is not None:
                server.stdin.close()
        for server in servers:
            try:
                server.wait(5.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


RAI_ACTIONS = (
    """
from example_interfaces.action import Fibonacci
from rai.communication.ros2 import ROS2Connector
connector = ROS2Connector()
for i in range({first}, {n}, {step}):
    connector.create_action("{prefix}_%d" % i, lambda goal: goal.succeed() or Fibonacci.Result(), action_type=Fibonacci)
""",
    "connector.close()",
)


def pad_readers(node: Node, prefix: str) -> None:
    """More readers than writers in this participant: a new server learns its
    request writers well before its reply readers (6-36 of 200 lost unfixed).
    The churn also slows discovery of this participant for good, so the
    clients run in a child process (in_child)."""
    for i in range(400):
        node.create_subscription(Int32, f"{prefix}_pad_{i}", lambda _: None, 1)


def in_child(fn: Callable[..., Any], *args: Any) -> Any:
    """fn(*args) in a fresh spawned process: its own DDS participant."""
    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
        return pool.submit(fn, *args).result(timeout=50)


def lost_raw_responses(server: tuple[str, str], action: bool) -> str:
    """Child process: plain rclpy clients call (or send a goal) right after
    service_is_ready() (server_is_ready()). Returns 'lost/total'."""
    import rclpy
    from std_srvs.srv import SetBool

    rclpy.init()
    prefix = unique("fresh")
    node = Node("fault_fresh_clients")
    pad_readers(node, prefix)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    n = 2 * SERVER_PROCESSES if action else FRESH_CLIENTS
    if action:
        clients = [ActionClient(node, Fibonacci, f"{prefix}_{i}") for i in range(n)]
        ready = [c.server_is_ready for c in clients]
        send = [lambda c=c: c.send_goal_async(Fibonacci.Goal(order=1)) for c in clients]
    else:
        clients = [node.create_client(SetBool, f"{prefix}_{i}") for i in range(n)]
        ready = [c.service_is_ready for c in clients]
        send = [lambda c=c: c.call_async(SetBool.Request(data=True)) for c in clients]
    futures: list[Any] = [None] * n

    def send_ready() -> bool:
        for i in range(n):
            if futures[i] is None and ready[i]():
                futures[i] = send[i]()
        return all(f is not None for f in futures)

    with many_services(server, prefix, n):
        assert wait_until(send_ready, 15.0), "servers not discovered"
        wait_until(lambda: all(f.done() for f in futures), 10.0)
    lost = sum(not f.done() for f in futures)
    executor.shutdown(timeout_sec=1.0)
    rclpy.try_shutdown()
    return f"{lost}/{n}"


def test_fresh_clients_get_every_response_from_a_rai_server():
    """Fast DDS drops a response whose reader the server has not matched yet
    (rmw_fastrtps waits 100 ms), and a fresh server process discovers the
    clients late. RAI's servers wait for the requester's reply reader, so
    clients may call right after service_is_ready()."""
    assert in_child(lost_raw_responses, RAI_SERVICES, False) == f"0/{FRESH_CLIENTS}"


def test_fresh_action_clients_get_every_goal_response_from_a_rai_server():
    """A lost goal response leaves a goal running that the client timed out on.
    RAI's action servers answer once the client's reply readers are known."""
    n = 2 * SERVER_PROCESSES
    assert in_child(lost_raw_responses, RAI_ACTIONS, True) == f"0/{n}"


def lost_connector_calls() -> str:
    """Child process: a ROS2Connector calls services of new plain rclpy server
    processes, all at once. Returns 'lost/total' and the first error."""
    import asyncio

    from std_srvs.srv import SetBool

    prefix = unique("fresh")
    with ROS2Connector() as connector:
        pad_readers(node_of(connector), prefix)

        async def call_all() -> list[Any]:
            calls = [
                connector.call(f"{prefix}_{i}", SetBool.Request(data=True), timeout=8.0)
                for i in range(FRESH_CLIENTS)
            ]
            return await asyncio.gather(*calls, return_exceptions=True)

        with many_services(RCLPY_SERVICES, prefix, FRESH_CLIENTS):
            results = asyncio.run(call_all())
    lost = [r for r in results if isinstance(r, BaseException)]
    return f"{len(lost)}/{FRESH_CLIENTS} {lost[0]!r}" if lost else f"0/{FRESH_CLIENTS}"


@pytest.mark.benchmark
def test_fresh_connector_clients_of_new_rclpy_servers():
    """The same race against third-party (plain rclpy) servers, which RAI can
    only mitigate on the client by settling (runtime.SETTLE_NEW_S): a residual
    risk that grows with this participant's size and churn, so report only."""
    print(f"lost connector calls: {in_child(lost_connector_calls)}")


def test_no_client_leak_over_calls(connectors, helper_node):
    name = unique("leak")
    connector = connectors()
    make_trigger_service(helper_node(), name)
    before = len(list(node_of(connector).clients))
    for _ in range(10):
        call(connector, name, 3.0)
    assert len(list(node_of(connector).clients)) - before <= 1


def test_release_client_destroys_client(connectors, helper_node):
    name = unique("release")
    connector = connectors()
    make_trigger_service(helper_node(), name)
    before = len(list(node_of(connector).clients))
    call(connector, name, 3.0)
    assert len(list(node_of(connector).clients)) == before + 1
    assert connector.release_client(name)
    assert len(list(node_of(connector).clients)) == before
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
    result: list[Any] = []
    caller = threading.Thread(
        target=lambda: result.append(
            run_bounded(lambda: call(connector, name, 10.0), 12.0)
        ),
        daemon=True,
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
        start_goal(connector, name, accept_timeout=0.3)
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


def test_cancel_bounded_when_server_gone(connectors, ros_setup):
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

    finished, _ = run_bounded(handle.sync.cancel, 5.0)
    assert finished


def test_action_entities_not_leaked(connectors, helper_node):
    name = unique("fibleak")
    connector = connectors()
    make_fibonacci_server(helper_node(), name)
    before = len(list(node_of(connector).waitables))
    for _ in range(40):
        done = threading.Event()
        start_goal(connector, name, on_done=lambda _: done.set())
        assert done.wait(5.0)
    assert len(list(node_of(connector).waitables)) - before <= 1


def test_send_goal_from_service_handler(connectors, helper_node):
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
    for goal in (NAVIGATE.Goal(), Fibonacci.Goal()):
        with pytest.raises(Exception):
            connector.sync.send_goal(name, goal, accept_timeout=0.2)
    with server_process(name):
        wait_for_action(connector, name)
        # Readiness covers DDS discovery (up to about 3 s on Fast DDS) and
        # the settling for a fresh server process (runtime.SETTLE_NEW_S).
        assert start_goal(connector, name, accept_timeout=6.0)


# ------------------------------------------------------------------ lifecycle


def test_no_user_callbacks_after_shutdown(connectors, helper_node):
    topic = unique("shut")
    connector = connectors()
    started: list[float] = []

    gate = threading.Event()

    def callback(msg: Int32) -> None:
        started.append(time.monotonic())
        gate.wait(10.0)

    connector.subscribe(topic, callback, msg_type=Int32)
    pub = helper_node().create_publisher(Int32, topic, 100)
    try:
        assert publish_until(pub, Int32(data=0), lambda: started)
        for i in range(20):  # queued behind the blocked callbacks
            pub.publish(Int32(data=i))
            time.sleep(0.01)
        finished, _ = run_bounded(connector.close, 5.0)
        returned = time.monotonic()
        assert finished
    finally:
        gate.set()
    time.sleep(0.5)  # a callback started late would be recorded by now
    assert all(t <= returned for t in started)


@pytest.mark.parametrize("cls", [ROS2Connector, ROS2HRIConnector])
def test_every_call_after_shutdown_raises_closed(connectors, cls):
    connector = connectors(cls)
    connector.close()
    hri = cls is ROS2HRIConnector
    calls = {
        "send": lambda: connector.sync.send("/t", "x"),
        "receive": lambda: connector.sync.receive("/t"),
        "subscribe": lambda: connector.subscribe("/t", print),
    }
    if not hri:  # 3.0 HRI connectors only send and receive
        calls = {
            "publish": lambda: connector.sync.publish("/t", String()),
            "receive": lambda: connector.sync.receive("/t", msg_type=String),
            "subscribe": lambda: connector.subscribe("/t", print, msg_type=Int32),
            "call": lambda: call(connector, "/s", 5.0),
            "send_goal": lambda: start_goal(connector, "/a"),
            "create_service": lambda: connector.create_service(
                "/s", print, service_type=TRIGGER
            ),
            "create_action": lambda: connector.create_action(
                "/a", print, action_type=FIBONACCI
            ),
            "transform": lambda: connector.sync.transform("a", "b", timeout=0.1),
            "get_topics": connector.get_topics_names_and_types,
            "get_services": connector.get_services_names_and_types,
            "get_actions": connector.get_actions_names_and_types,
        }
    for name, fn in calls.items():
        finished, error = run_bounded(fn, 2.0)
        assert finished and isinstance(error, ConnectorClosedError), (name, error)


def test_cleanup_after_shutdown_is_noop(connectors):
    connector = connectors()
    subscription = connector.subscribe(
        unique("cleanup"), lambda _: None, msg_type=String
    )
    connector.close()
    subscription.close()
    subscription.close()
    assert connector.release_client(unique("nosrv")) is False


def test_concurrent_shutdown_is_safe(connectors):
    connector = connectors()
    results: list[Any] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(run_bounded(connector.close, 5.0))
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    results.append(run_bounded(connector.close, 5.0))  # and once more after
    assert all(r == (True, None) for r in results), results


def test_shutdown_wakes_waiting_receive(connectors):
    connector = connectors()
    result: list[Any] = []
    receive = lambda: connector.sync.receive(
        unique("never"), timeout=20.0, msg_type=String
    )
    threading.Thread(
        target=lambda: result.append(run_bounded(receive, 25.0)), daemon=True
    ).start()
    time.sleep(0.2)
    run_bounded(connector.close, 5.0)
    assert wait_until(lambda: result, 5.0)
    assert isinstance(result[0][1], ConnectorClosedError), result


def test_shutdown_wakes_waiting_call(connectors, helper_node):
    name = unique("stuck")
    connector = connectors()
    called, release = threading.Event(), threading.Event()

    def stuck(request: Any, response: Any) -> Any:
        called.set()
        release.wait(20.0)
        return response

    make_trigger_service(helper_node(), name, stuck)
    result: list[Any] = []
    threading.Thread(
        target=lambda: result.append(
            run_bounded(lambda: call(connector, name, 20.0), 25.0)
        ),
        daemon=True,
    ).start()
    try:
        assert called.wait(5.0)
        run_bounded(connector.close, 5.0)
        assert wait_until(lambda: result, 5.0)
        assert isinstance(result[0][1], ConnectorClosedError), result
    finally:
        release.set()


def stuck_subscription_callback(connectors, helper_node, release):
    topic = unique("stuckcb")
    connector = connectors()
    started = threading.Event()
    connector.subscribe(
        topic, lambda m: (started.set(), release.wait(20.0)), msg_type=Int32
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
        finished, error = run_bounded(connector.close, 5.0)
        assert finished and error is None, error
    finally:
        release.set()


def test_shutdown_from_subscription_callback(connectors, helper_node):
    topic = unique("stopmsg")
    connector = connectors()
    result: list[Any] = []
    connector.subscribe(
        topic,
        lambda _: result.append(run_bounded(connector.close, 5.0)),
        msg_type=String,
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
    connector.subscribe(topic, got.append, msg_type=String)
    assert publish_until(pub, String(data="x"), lambda: got)
    call(connector, srv, 3.0)
    done = threading.Event()
    start_goal(connector, action, on_done=lambda _: done.set())
    assert done.wait(5.0)

    finished, error = run_bounded(connector.close, 5.0)
    assert finished and error is None, error
    leftover = lambda: [t.name for t in threading.enumerate() if t not in before]
    assert wait_until(lambda: not leftover()), leftover()


def test_hri_connectors_get_distinct_default_node_names(connectors):
    """Default connectors share the process runtime (one node); private
    runtimes each get a fresh name, not one evaluated at import (8a625cf)."""
    first = connectors(ROS2HRIConnector, runtime=ROS2Runtime())
    second = connectors(ROS2HRIConnector, runtime=ROS2Runtime())
    assert node_of(first).get_name() != node_of(second).get_name()
    assert node_of(connectors(ROS2HRIConnector)) is node_of(connectors(ROS2Connector))


def test_hri_connector_forwards_use_sim_time(connectors):
    connector = connectors(ROS2HRIConnector, use_sim_time=True)
    assert node_of(connector).get_parameter("use_sim_time").value is True


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
        connector.subscribe("/exit_test", print, msg_type="std_msgs/msg/String")
    try:
        connector.get_topics_names_and_types()
    except Exception as e:
        print(type(e).__name__)
else:
    connector = ROS2Connector()
    connector.subscribe("/exit_test", print, msg_type="std_msgs/msg/String")
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
    connector.subscribe(topic, lambda m: got.append(m.data), msg_type=Int32)
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
        s.topic_name in ("/tf", "/tf_static") for s in node_of(connector).subscriptions
    )
    assert not has_tf()
    transform = TransformStamped()
    transform.header.frame_id = "lazy_parent"
    transform.child_frame_id = "lazy_child"
    transform.transform.rotation.w = 1.0
    StaticTransformBroadcaster(helper_node()).sendTransform(transform)
    result = connector.sync.transform("lazy_parent", "lazy_child", timeout=3.0)
    assert result.child_frame_id == "lazy_child"
    assert has_tf()


def test_transform_honors_fractional_timeout(connectors):
    connector = connectors()
    start = time.monotonic()
    with pytest.raises(LookupException):
        connector.sync.transform("nowhere", "nothing", timeout=0.3)
    assert 0.25 < time.monotonic() - start < 3.0
