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

"""Connector-side egress hooks (R6): every write path is gated, matching by type
and resolved name, leases, fail-closed decisions, and example EStop and
RateLimit hooks that show the API fits A8's built-ins."""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from example_interfaces.action import Fibonacci
from geometry_msgs.msg import Twist, TwistStamped
from rai.communication.ros2 import (
    ROS2Connector,
    ROS2HRIConnector,
    ROS2Message,
)
from rai.communication.ros2 import runtime as runtime_module
from rai.communication.ros2.egress import (
    ALL_KINDS,
    MOTION,
    Allow,
    Deny,
    Egress,
    EgressDecision,
    EgressDenied,
    egress_origin,
    global_egress,
    type_name,
)
from rai.communication.ros2.runtime import ROS2Runtime
from rclpy.action import CancelResponse
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)

TRIGGER, FIBONACCI = "std_srvs/srv/Trigger", "example_interfaces/action/Fibonacci"
TWIST = "geometry_msgs/msg/Twist"


def unique(prefix: str) -> str:
    return f"/egress_{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def client(ros_setup: None) -> Iterator[ROS2Connector]:
    connector = ROS2Connector(f"egress_client_{uuid.uuid4().hex[:8]}")
    yield connector
    connector.close()


@pytest.fixture
def server(ros_setup: None) -> Iterator[ROS2Connector]:
    connector = ROS2Connector(f"egress_server_{uuid.uuid4().hex[:8]}")
    yield connector
    connector.close()


def serve_trigger(server: ROS2Connector, calls: list[Any]) -> str:
    name = unique("srv")

    def handler(request: Any, response: Any) -> Any:
        calls.append(request)
        response.success = True
        return response

    server.create_service(name, handler, service_type=TRIGGER)
    return name


def serve_fibonacci(server: ROS2Connector, goals: list[Any], hold: float = 0.0) -> str:
    name = unique("action")

    def execute(goal_handle: Any) -> Any:
        goals.append(goal_handle.request)
        deadline = time.monotonic() + hold
        while time.monotonic() < deadline and not goal_handle.is_cancel_requested:
            time.sleep(0.01)
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return Fibonacci.Result()

    server.create_action(
        name,
        execute,
        action_type=FIBONACCI,
        cancel_callback=lambda _: CancelResponse.ACCEPT,
    )
    return name


class Recorder:
    """A hook that records what it saw and answers with a fixed decision."""

    def __init__(self, decision: Any = Allow()) -> None:
        self.decision = decision
        self.seen: list[Egress] = []
        self.threads: list[str] = []
        self.__name__ = "recorder"

    def __call__(self, egress: Egress) -> Any:
        self.seen.append(egress)
        self.threads.append(threading.current_thread().name)
        return self.decision


class AsyncRecorder(Recorder):
    async def __call__(self, egress: Egress) -> Any:  # type: ignore[override]
        await asyncio.sleep(0)
        return super().__call__(egress)


# ------------------------------------------------------------ every path


def test_type_name() -> None:
    assert type_name(Twist) == TWIST
    assert type_name(Trigger) == TRIGGER
    assert type_name(Fibonacci) == FIBONACCI
    assert TWIST in MOTION and type_name(TwistStamped) in MOTION


def test_publish_paths_are_gated(client: ROS2Connector) -> None:
    hook = client.egress.add(Recorder(Deny("no")), types={"std_msgs/msg/String"})
    topic = unique("pub")
    with pytest.raises(EgressDenied, match="denied by recorder: no"):
        client.sync.publish(topic, String(data="a"))
    with pytest.raises(EgressDenied):
        asyncio.run(client.publish(topic, String(data="b")))
    with pytest.raises(EgressDenied):
        client.send_message(
            ROS2Message(payload={"data": "c"}), topic, msg_type="std_msgs/msg/String"
        )
    seen = hook.seen
    assert [e.payload.data for e in seen] == ["a", "b", "c"]
    assert {(e.kind, e.name, e.type) for e in seen} == {
        ("publish", topic, "std_msgs/msg/String")
    }
    # A denied publish creates no publisher.
    assert not client._runtime._publishers


def test_hri_publish_is_gated(ros_setup: None) -> None:
    runtime = ROS2Runtime(f"egress_hri_{uuid.uuid4().hex[:8]}")
    hri = ROS2HRIConnector(runtime=runtime)
    try:
        hook = runtime.egress.add(Recorder(Deny("quiet")))
        with pytest.raises(EgressDenied, match="quiet"):
            hri.sync.send("/to_human", "hi")
        assert hook.seen[0].type == "rai_interfaces/msg/HRIMessage"
    finally:
        hri.close()


def test_service_call_is_gated(client: ROS2Connector, server: ROS2Connector) -> None:
    calls: list[Any] = []
    name = serve_trigger(server, calls)
    hook = client.egress.add(Recorder(Deny("no calls")), types={TRIGGER})
    with pytest.raises(EgressDenied, match="no calls"):
        client.sync.call(name, Trigger.Request(), timeout=10.0)
    with pytest.raises(EgressDenied):
        client.service_call(ROS2Message(payload={}), name, 10.0, msg_type=TRIGGER)
    assert [(e.kind, e.name, e.type) for e in hook.seen] == [
        ("service", name, TRIGGER)
    ] * 2
    hook.decision = Allow()
    assert client.sync.call(name, Trigger.Request(), timeout=10.0).success
    assert len(calls) == 1


def test_send_goal_is_gated(client: ROS2Connector, server: ROS2Connector) -> None:
    goals: list[Any] = []
    name = serve_fibonacci(server, goals)
    hook = client.egress.add(Recorder(Deny("parked")), types={FIBONACCI})

    async def send() -> Any:
        return await client.send_goal(
            name, Fibonacci.Goal(order=3), accept_timeout=10.0
        )

    with pytest.raises(EgressDenied, match="parked"):
        asyncio.run(send())
    with pytest.raises(EgressDenied):
        client.sync.send_goal(name, Fibonacci.Goal(order=4), accept_timeout=10.0)
    assert [(e.kind, e.name, e.payload.order) for e in hook.seen] == [
        ("goal", name, 3),
        ("goal", name, 4),
    ]
    hook.decision = Allow()
    handle = client.sync.send_goal(name, Fibonacci.Goal(order=5), accept_timeout=10.0)
    handle.sync.result(timeout=10.0)
    assert [g.order for g in goals] == [5]


def test_cancel_is_gated_only_when_asked(
    client: ROS2Connector, server: ROS2Connector
) -> None:
    name = serve_fibonacci(server, [], hold=10.0)
    writes = client.egress.add(Recorder(), types={FIBONACCI})  # default kinds
    handle = client.sync.send_goal(name, Fibonacci.Goal(order=1), accept_timeout=10.0)
    cancels = client.egress.add(
        Recorder(Deny("keep going")), types={FIBONACCI}, kinds={"cancel"}
    )
    with pytest.raises(EgressDenied, match="keep going"):
        handle.sync.cancel()
    assert [e.kind for e in writes.seen] == [
        "goal"
    ]  # the write hook never saw the cancel
    assert [(e.kind, e.payload) for e in cancels.seen] == [("cancel", handle.id)]
    client.egress.remove(cancels)
    assert handle.sync.cancel(timeout=5.0)
    handle.sync.result(timeout=10.0)


# --------------------------------------------------------------- matching


def test_unmatched_egress_skips_the_hook(client: ROS2Connector) -> None:
    hook = client.egress.add(Recorder(Deny("motion")), types=MOTION)
    client.sync.publish(unique("chatter"), String(data="ok"))
    assert hook.seen == []
    with pytest.raises(EgressDenied):
        client.sync.publish(unique("anything"), Twist())


def test_type_and_name_matching_with_remap(ros_setup: None) -> None:
    runtime = ROS2Runtime(
        f"egress_remap_{uuid.uuid4().hex[:8]}",
        cli_args=["--ros-args", "-r", "cmd_vel:=/robot/cmd_vel"],
    )
    connector = ROS2Connector(runtime=runtime)
    try:
        by_type = connector.egress.add(Recorder(), types={TWIST})
        by_old_name = connector.egress.add(Recorder(), names={"/cmd_vel"})
        by_pattern = connector.egress.add(Recorder(), names={"/robot/*"})
        connector.sync.publish("cmd_vel", Twist())
        connector.sync.publish("/robot/arm", String())
        assert [e.name for e in by_type.seen] == ["/robot/cmd_vel"]
        assert by_old_name.seen == []
        assert [e.name for e in by_pattern.seen] == ["/robot/cmd_vel", "/robot/arm"]
    finally:
        connector.close()


def test_no_hooks_skips_matching(
    client: ROS2Connector, server: ROS2Connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any) -> Any:
        raise AssertionError("matched egress with no hooks registered")

    monkeypatch.setattr(runtime_module, "match", boom)
    name = serve_trigger(server, [])
    client.sync.publish(unique("free"), String())
    assert client.sync.call(name, Trigger.Request(), timeout=10.0).success


def test_global_hooks_cover_every_runtime(client: ROS2Connector) -> None:
    hook = global_egress.add(Recorder(Deny("global")), names={"/egress_global_*"})
    try:
        with pytest.raises(EgressDenied, match="global"):
            client.sync.publish(unique("global"), String())
        other = ROS2Connector(f"egress_other_{uuid.uuid4().hex[:8]}")
        try:
            with pytest.raises(EgressDenied):
                other.sync.publish(unique("global"), String())
        finally:
            other.close()
    finally:
        global_egress.remove(hook)
    client.sync.publish(unique("global"), String())


# ----------------------------------------------------------------- leases


def test_lease_reuse_expiry_and_revoke(client: ROS2Connector) -> None:
    leased = client.egress.add(Recorder(Allow(lease_s=0.3)), types={TWIST})
    every_time = client.egress.add(Recorder(), types={TWIST})
    topic, other = unique("cmd_vel"), unique("cmd_vel")
    for _ in range(10):
        client.sync.publish(topic, Twist())
    assert len(leased.seen) == 1
    assert len(every_time.seen) == 10  # a lease covers only its own hook
    client.sync.publish(other, Twist())
    assert len(leased.seen) == 2  # leases are per name
    time.sleep(0.4)
    client.sync.publish(topic, Twist())
    assert len(leased.seen) == 3  # expired
    client.egress.revoke(leased)
    client.sync.publish(topic, Twist())
    assert len(leased.seen) == 4
    leased.decision = Deny("stop")  # a lease does not outlive a revoke
    client.egress.revoke()
    with pytest.raises(EgressDenied):
        client.sync.publish(topic, Twist())


# ------------------------------------------------------------ fail closed


def raises(egress: Egress) -> Any:
    raise RuntimeError("bug")


async def slow(egress: Egress) -> Any:
    await asyncio.sleep(5.0)
    return Allow()


async def returns_none(egress: Egress) -> Any:
    return None


@pytest.mark.parametrize(
    "hook, message",
    [
        (raises, "by raises: failed \\(RuntimeError: bug\\)"),
        (slow, "by slow: timed out after 0.1 s"),
        (returns_none, "by returns_none: returned None"),
        (lambda e: True, "returned True"),
    ],
)
def test_fail_closed(client: ROS2Connector, hook: Any, message: str) -> None:
    client.egress.add(hook, timeout_s=0.1)
    with pytest.raises(EgressDenied, match=message):
        client.sync.publish(unique("fail"), String())
    with pytest.raises(EgressDenied, match=message):
        asyncio.run(client.publish(unique("fail"), String()))


# ---------------------------------------------------------------- threads


def test_hooks_never_run_on_the_executor_thread(client: ROS2Connector) -> None:
    sync_hook = client.egress.add(Recorder(), types={"std_msgs/msg/String"})
    async_hook = client.egress.add(AsyncRecorder(), types={"std_msgs/msg/String"})
    client.sync.publish(unique("t"), String())  # sync API: the private loop
    asyncio.run(client.publish(unique("t"), String()))  # async API: the caller's loop
    client.send_message(
        ROS2Message(payload={}), unique("t"), msg_type="std_msgs/msg/String"
    )
    threads = sync_hook.threads + async_hook.threads
    assert "rai_ros2_spin" not in threads
    assert async_hook.threads == ["rai_ros2_loop", "MainThread", "rai_ros2_loop"]


def test_async_hook_from_an_async_callback(
    client: ROS2Connector, server: ROS2Connector
) -> None:
    """Sync publish from a callback running on the private loop still runs
    async hooks (on a worker), instead of deadlocking that loop."""
    hook = client.egress.add(AsyncRecorder(), names={"/egress_relay_out_*"})
    source, sink = unique("relay_in"), unique("relay_out")
    done = threading.Event()
    errors: list[BaseException] = []

    async def relay(msg: Any) -> None:
        try:
            client.send_message(
                ROS2Message(payload={"data": msg.data}),
                sink,
                msg_type="std_msgs/msg/String",
            )
        except BaseException as e:
            errors.append(e)
        done.set()

    client.subscribe(source, relay, msg_type=String)
    deadline = time.monotonic() + 10.0
    while not done.is_set() and time.monotonic() < deadline:
        server.sync.publish(source, String(data="x"))
        done.wait(0.1)
    assert done.is_set() and not errors
    assert hook.seen[0].payload.data == "x"


# ----------------------------------------------------- observing decisions


def test_decisions_are_observable(
    client: ROS2Connector, caplog: pytest.LogCaptureFixture
) -> None:
    decisions: list[EgressDecision] = []
    client.egress.observe(decisions.append)
    client.egress.add(Recorder(Allow(lease_s=10.0)), types={TWIST})
    topic = unique("watched")
    token = egress_origin.set("test_tool")
    try:
        with caplog.at_level(logging.DEBUG, logger="rai.communication.ros2.egress"):
            client.sync.publish(topic, Twist())
            asyncio.run(client.publish(topic, Twist()))
    finally:
        egress_origin.reset(token)
    assert [(d.allowed, d.leased) for d in decisions] == [(True, False), (True, True)]
    assert decisions[1].egress.origin == "test_tool"
    assert "allowed by recorder (lease)" in caplog.text


# ---------------------------------------------- examples of A8 built-ins


class EStop:
    """A latch: while triggered, every MOTION write is denied. Cancels pass
    (default kinds), so stopping a goal is never blocked."""

    def __init__(self) -> None:
        self.triggered = False
        self.__name__ = "estop"

    def __call__(self, egress: Egress) -> Any:
        return Deny("e-stop is active") if self.triggered else Allow()


class RateLimit:
    """A token bucket per (kind, name)."""

    def __init__(self, max_calls: int, per_s: float) -> None:
        self.max_calls, self.per_s = max_calls, per_s
        self.buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self.__name__ = "rate_limit"

    def __call__(self, egress: Egress) -> Any:
        now = time.monotonic()
        key = (egress.kind, egress.name)
        tokens, last = self.buckets.get(key, (float(self.max_calls), now))
        tokens = min(
            self.max_calls, tokens + (now - last) * self.max_calls / self.per_s
        )
        if tokens < 1:
            self.buckets[key] = (tokens, now)
            return Deny(f"rate limited to {self.max_calls} per {self.per_s:g} s")
        self.buckets[key] = (tokens - 1, now)
        return Allow()


def test_example_estop(client: ROS2Connector, server: ROS2Connector) -> None:
    name = serve_fibonacci(server, [], hold=10.0)
    estop = client.egress.add(EStop(), types=MOTION | {FIBONACCI})
    handle = client.sync.send_goal(name, Fibonacci.Goal(order=1), accept_timeout=10.0)
    client.sync.publish(unique("cmd_vel"), Twist())
    estop.triggered = True
    with pytest.raises(EgressDenied, match="e-stop is active"):
        client.sync.publish(unique("cmd_vel"), Twist())
    with pytest.raises(EgressDenied):
        client.sync.send_goal(name, Fibonacci.Goal(order=2), accept_timeout=10.0)
    client.sync.publish(unique("chatter"), String())  # not MOTION
    assert handle.sync.cancel(timeout=5.0)  # the stop path stays open
    estop.triggered = False
    client.sync.publish(unique("cmd_vel"), Twist())


def test_example_rate_limit(client: ROS2Connector) -> None:
    client.egress.add(RateLimit(5, 10.0), types={TWIST})
    topic = unique("cmd_vel")
    for _ in range(5):
        client.sync.publish(topic, Twist())
    with pytest.raises(EgressDenied, match="rate limited"):
        client.sync.publish(topic, Twist())
    client.sync.publish(unique("cmd_vel"), Twist())  # per name


def test_kinds_are_validated(client: ROS2Connector) -> None:
    with pytest.raises(ValueError):
        client.egress.add(Recorder(), kinds={"subscribe"})
    client.egress.add(Recorder(), kinds=ALL_KINDS)
    assert len(client.egress) == 1
