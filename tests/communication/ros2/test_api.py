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

"""The 3.0 connector API: async from any loop, connector.sync, Subscription,
GoalHandle, QoS presets, views and encode_image."""

import asyncio
import base64
import threading
import time
import uuid
from typing import Any

import cv2
import numpy as np
import pytest
from example_interfaces.action import Fibonacci
from rai.communication.base_connector import ConnectorClosedError
from rai.communication.ros2 import (
    GoalStatus,
    QoS,
    ROS2Connector,
    ROS2GoalRejectedError,
    ROS2HRIConnector,
    ROS2Message,
    ROS2TimeoutError,
    encode_image,
    views,
)
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.communication.ros2.runtime import ROS2Runtime
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Int32, String
from std_srvs.srv import SetBool, Trigger

from .helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)


def unique(prefix: str) -> str:
    return f"/api_{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def connector(ros_setup: None):
    with ROS2Connector() as connector:
        yield connector


@pytest.fixture
def server(ros_setup: None):
    node = Node(f"api_server_{uuid.uuid4().hex[:8]}")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown(timeout_sec=1.0)
    thread.join(2.0)
    node.destroy_node()


def fibonacci_server(node: Node, name: str, **kwargs: Any) -> ActionServer:
    def execute(goal_handle: Any) -> Any:
        for i in range(goal_handle.request.order):
            goal_handle.publish_feedback(Fibonacci.Feedback(sequence=[i]))
            time.sleep(0.01)
        goal_handle.succeed()
        return Fibonacci.Result(sequence=[1, 1, 2])

    kwargs.setdefault("execute_callback", execute)
    return ActionServer(
        node, Fibonacci, name, callback_group=ReentrantCallbackGroup(), **kwargs
    )


def publish_until(publisher: Any, msg: Any, done: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        publisher.publish(msg)
        time.sleep(0.02)
    assert done()


def test_async_api_from_a_foreign_event_loop(connector, server):
    topic, srv = unique("topic"), unique("srv")
    server.create_service(
        Trigger, srv, lambda req, res: setattr(res, "success", True) or res
    )
    publisher = server.create_publisher(String, topic, 10)

    async def main() -> tuple[Any, Any]:
        stop = asyncio.Event()

        async def keep_publishing() -> None:
            while not stop.is_set():
                publisher.publish(String(data="hi"))
                await asyncio.sleep(0.02)

        task = asyncio.create_task(keep_publishing())
        msg = await connector.receive(topic, msg_type=String, timeout=5.0)
        stop.set()
        await task
        response = await connector.call(srv, Trigger.Request(), timeout=5.0)
        return msg, response

    msg, response = asyncio.run(main())
    assert isinstance(msg, String) and msg.data == "hi"
    assert response.success


def test_publish_native_message_and_type_conflict(connector, server):
    topic = unique("pub")
    got: list[str] = []
    server.create_subscription(String, topic, lambda m: got.append(m.data), 10)
    deadline = time.monotonic() + 5.0
    while not got and time.monotonic() < deadline:
        connector.sync.publish(topic, String(data="x"))
        time.sleep(0.02)
    assert got[0] == "x"
    with pytest.raises(TypeError, match=topic):
        connector.sync.publish(topic, Int32(data=1))


def test_receive_max_age_and_timeout(connector, server):
    topic = unique("age")
    publisher = server.create_publisher(String, topic, 10)
    sub = connector.subscribe(topic, msg_type=String)
    publish_until(publisher, String(data="x"), lambda: sub._topic.last is not None)
    assert connector.sync.receive(topic).data == "x"  # cached, any age
    time.sleep(0.3)
    with pytest.raises(ROS2TimeoutError):
        connector.sync.receive(topic, timeout=0.2, max_age=0.1)


def test_subscription_async_iteration_and_close(connector, server):
    topic = unique("iter")
    publisher = server.create_publisher(Int32, topic, 10)
    sub = connector.subscribe(topic, msg_type=Int32, maxsize=None)

    async def first_three() -> list[int]:
        got = []
        async for msg in sub:
            got.append(msg.data)
            if len(got) == 3:
                sub.close()
        return got

    threading.Thread(
        target=lambda: publish_until(publisher, Int32(data=7), lambda: sub.closed),
        daemon=True,
    ).start()
    assert asyncio.run(first_three()) == [7, 7, 7]


def test_async_callback_is_awaited_in_order(connector, server):
    topic = unique("acb")
    publisher = server.create_publisher(Int32, topic, 100)
    got: list[int] = []

    async def callback(msg: Int32) -> None:
        await asyncio.sleep(0.001)
        got.append(msg.data)

    connector.subscribe(topic, callback, msg_type=Int32, maxsize=None)
    publish_until(publisher, Int32(data=-1), lambda: got)
    for i in range(50):
        publisher.publish(Int32(data=i))
    deadline = time.monotonic() + 5.0
    while (not got or got[-1] != 49) and time.monotonic() < deadline:
        time.sleep(0.01)
    received = [i for i in got if i >= 0]
    assert received == sorted(received) and received[-1] == 49


def test_service_call_errors(connector, server):
    srv = unique("setbool")
    server.create_service(
        SetBool, srv, lambda req, res: setattr(res, "success", req.data) or res
    )
    assert connector.sync.call(srv, SetBool.Request(data=True), timeout=5.0).success
    with pytest.raises(ROS2TimeoutError):
        connector.sync.call(unique("missing"), SetBool.Request(), timeout=0.2)
    with pytest.raises(AttributeError):
        connector.service_call(
            ROS2Message(payload={}),
            srv,
            msg_type="std_srvs/srv/NotExistingService",
        )


def test_goal_handle_feedback_result_status(connector, server):
    name = unique("fib")
    fibonacci_server(server, name)
    feedback: list[int] = []
    done = threading.Event()
    handle = connector.sync.send_goal(
        name,
        Fibonacci.Goal(order=5),
        on_feedback=lambda f: feedback.append(f.sequence[0]),
        on_done=lambda h: done.set(),
    )
    assert handle.id
    assert list(handle.sync.result(timeout=5.0).sequence) == [1, 1, 2]
    assert handle.done and handle.status == GoalStatus.SUCCEEDED
    assert done.wait(5.0)
    assert feedback == sorted(feedback) and feedback[-1] == 4
    assert not handle.sync.cancel()  # already done


def test_goal_cancel(connector, server):
    name = unique("cancel")

    def execute(goal_handle: Any) -> Any:
        deadline = time.monotonic() + 5.0
        while not goal_handle.is_cancel_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        goal_handle.canceled()
        return Fibonacci.Result()

    fibonacci_server(
        server,
        name,
        execute_callback=execute,
        cancel_callback=lambda _: CancelResponse.ACCEPT,
    )
    handle = connector.sync.send_goal(name, Fibonacci.Goal(order=1))
    assert handle.sync.cancel()
    handle.sync.result(timeout=5.0)
    assert handle.status == GoalStatus.CANCELED


def test_goal_rejected_and_slow_accept(connector, server):
    rejected, slow = unique("reject"), unique("slow")
    fibonacci_server(server, rejected, goal_callback=lambda _: GoalResponse.REJECT)

    def accept_late(_: Any) -> Any:
        time.sleep(0.5)
        return GoalResponse.ACCEPT

    fibonacci_server(server, slow, goal_callback=accept_late)
    with pytest.raises(ROS2GoalRejectedError):
        connector.sync.send_goal(rejected, Fibonacci.Goal(order=1))
    # The server is ready but accepts after the timeout: deterministic, and
    # one exception type on both distros.
    with pytest.raises(ROS2TimeoutError):
        connector.sync.send_goal(slow, Fibonacci.Goal(order=1), accept_timeout=0.2)


def test_failing_feedback_callback_does_not_break_the_goal(connector, server):
    name = unique("fbfail")
    fibonacci_server(server, name)

    def failing(_: Any) -> None:
        raise ValueError("boom")

    handle = connector.sync.send_goal(
        name, Fibonacci.Goal(order=3), on_feedback=failing
    )
    handle.sync.result(timeout=5.0)
    assert handle.status == GoalStatus.SUCCEEDED
    assert len(handle.feedback()) == 3


def test_create_service_and_action(connector, server):
    srv, action = unique("own_srv"), unique("own_fib")

    def execute(goal_handle: Any) -> Any:
        goal_handle.succeed()
        return Fibonacci.Result(sequence=[42])

    connector.create_service(
        srv,
        lambda req, res: setattr(res, "success", True) or res,
        service_type="std_srvs/srv/Trigger",
    )
    connector.create_action(action, execute, action_type=Fibonacci)
    with ROS2Connector(runtime=ROS2Runtime()) as other:
        assert other.sync.call(srv, Trigger.Request(), timeout=5.0).success
        handle = other.sync.send_goal(action, Fibonacci.Goal(order=1))
        assert list(handle.sync.result(timeout=5.0).sequence) == [42]


def test_shared_runtime_is_reference_counted(ros_setup):
    first, second = ROS2Connector(), ROS2HRIConnector()
    runtime = first._runtime
    assert second._runtime is runtime
    first.close()
    first.close()  # idempotent
    assert not runtime.closed
    with pytest.raises(ConnectorClosedError):
        first.sync.receive("/x", msg_type=String)
    second.close()
    assert runtime.closed
    with ROS2Connector(node_name=f"private_{uuid.uuid4().hex[:6]}") as private:
        assert private._runtime is not runtime


def test_sync_api_from_an_async_callback_raises(connector, server):
    topic = unique("syncinasync")
    publisher = server.create_publisher(Int32, topic, 10)
    errors: list[BaseException] = []

    async def callback(_: Int32) -> None:
        try:
            connector.sync.receive(topic)
        except RuntimeError as e:
            errors.append(e)

    connector.subscribe(topic, callback, msg_type=Int32)
    publish_until(publisher, Int32(data=1), lambda: errors)


def test_resolve_and_now(connector):
    assert connector.resolve("cmd_vel") == "/cmd_vel"
    assert connector.resolve("srv", "service") == "/srv"
    assert connector.now().nanoseconds > 0


def test_qos_presets():
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

    assert QoS.COMMAND.durability == DurabilityPolicy.VOLATILE
    assert QoS.LATCHED.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert QoS.HRI.reliability == ReliabilityPolicy.RELIABLE
    assert QoS.SENSOR.reliability == ReliabilityPolicy.BEST_EFFORT
    assert QoS.AUTO.reliability == ReliabilityPolicy.BEST_AVAILABLE


def test_image_view_is_zero_copy():
    pixels = np.random.default_rng(0).integers(0, 255, (4, 6, 3), dtype=np.uint8)
    msg = numpy_to_imgmsg(pixels, "rgb8")
    view = views.image(msg)
    assert np.array_equal(view, pixels)
    assert np.shares_memory(view, np.frombuffer(msg.data, np.uint8))
    padded = numpy_to_imgmsg(np.zeros((2, 3), np.uint16), "mono16")
    padded.step, padded.data = 8, bytes(range(16))
    assert views.image(padded).shape == (2, 3)


def test_points_view():
    fields = [
        PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate("xyz")
    ]
    data = np.arange(12, dtype=np.float32)
    msg = PointCloud2(height=1, width=3, point_step=16, row_step=48, fields=fields)
    msg.data = np.pad(data.reshape(3, 4)[:, :3], ((0, 0), (0, 1))).tobytes()
    cloud = views.points(msg)
    assert cloud.shape == (3,)
    assert list(cloud["y"]) == [1.0, 5.0, 9.0]


def test_encode_image_downscales_and_encodes_jpeg():
    image = np.zeros((2000, 1000, 3), np.uint8)
    image[..., 0] = 255  # red in RGB
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encode_image(image)), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == (1024, 512, 3)
    assert decoded[..., 2].mean() > 200  # cv2 decodes BGR: red stays red
