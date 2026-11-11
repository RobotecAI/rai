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

"""A8: the built-in safety hooks (rai.safety) on both hook points."""

import asyncio
import json
import time
import typing
import uuid
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from example_interfaces.action import Fibonacci
from geometry_msgs.msg import Twist
from langchain_core.messages import AIMessage, ToolMessage
from rai import safety
from rai.communication.ros2 import ROS2Connector, ROS2HRIConnector
from rai.communication.ros2 import runtime as rt
from rai.communication.ros2.connectors.ros2_connector import GoalStatus
from rai.communication.ros2.egress import (
    MOTION,
    EgressDecision,
    EgressDenied,
    global_egress,
)
from rai.safety import (
    AuditLog,
    AuditRecord,
    ConfirmActuation,
    EStop,
    HRIAsker,
    RateLimit,
)
from rai.tools import Effect, RaiTool, ToolDecision, global_tool_hooks, rai_tool
from rai.tools.ros2.cli import ROS2CLIToolkit
from rclpy.action import CancelResponse
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from rai_interfaces.msg import HRIMessage
from tests.communication.ros2.helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)

FIBONACCI = "example_interfaces/action/Fibonacci"
TWIST = "geometry_msgs/msg/Twist"
MOVING = MOTION | {FIBONACCI}  # the test action stands in for a Nav2 goal


def unique(prefix: str) -> str:
    return f"/safety_{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _clean_hooks() -> Iterator[None]:
    yield
    global_egress.clear()
    global_egress._observers = ()
    global_tool_hooks.clear()
    global_tool_hooks._observers = ()
    rt._goal_watchers = ()


@pytest.fixture
def client(ros_setup: None) -> Iterator[ROS2Connector]:
    connector = ROS2Connector(f"safety_client_{uuid.uuid4().hex[:8]}")
    yield connector
    connector.close()


@pytest.fixture
def server(ros_setup: None) -> Iterator[ROS2Connector]:
    connector = ROS2Connector(f"safety_server_{uuid.uuid4().hex[:8]}")
    yield connector
    connector.close()


def serve_fibonacci(server: ROS2Connector, hold: float = 20.0) -> str:
    name = unique("action")

    def execute(goal_handle: Any) -> Any:
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


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


def publish_until(
    client: ROS2Connector, topic: str, msg: Any, condition: Callable[[], bool]
) -> None:
    """Publish until condition holds: the first messages may precede discovery."""
    deadline = time.monotonic() + 10.0
    while not condition():
        assert time.monotonic() < deadline, f"no effect of {topic}"
        client.sync.publish(topic, msg)
        time.sleep(0.1)


def invoke(tool: RaiTool, **args: Any) -> ToolMessage:
    call = {"type": "tool_call", "name": tool.name, "args": args, "id": "c1"}
    return asyncio.run(tool.ainvoke(call))


def publisher_tool(client: ROS2Connector, topic: str) -> RaiTool:
    @rai_tool(Effect.ACTUATE, name="drive")
    async def drive(seconds: float = 0.0) -> str:
        """Publish cmd_vel at 30 Hz for seconds."""
        await client.publish(topic, Twist())
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(1 / 30)
            await client.publish(topic, Twist())
        return "driven"

    return drive


@rai_tool(Effect.READ)
async def look() -> str:
    """Read something."""
    return "seen"


class FakeAsker:
    """Answers questions with a fixed answer after delay_s; records them."""

    def __init__(self, answer: Any = True, delay_s: float = 0.0) -> None:
        self.answer = answer
        self.delay_s = delay_s
        self.questions: list[str] = []
        self.before_answer: Callable[[], None] = lambda: None

    async def __call__(self, question: str) -> bool:
        self.questions.append(question)
        await asyncio.sleep(self.delay_s)
        self.before_answer()
        return self.answer


# ------------------------------------------------------------ EStop


def test_estop_denies_motion_and_actuate_tools_until_released(
    client: ROS2Connector,
) -> None:
    estop = EStop().install()
    topic = unique("cmd_vel")
    drive = publisher_tool(client, topic)
    client.sync.publish(topic, Twist())
    assert invoke(drive).content == "driven"

    estop.engage("test")
    assert estop.engaged
    with pytest.raises(EgressDenied, match=r"e-stop engaged \(test\)"):
        client.sync.publish(topic, Twist())
    client.sync.publish(unique("chatter"), String())  # not MOTION
    message = invoke(drive)
    assert message.status == "error"
    assert message.content.startswith("Denied by EStop: e-stop engaged (test)")
    assert invoke(look).content == "seen"  # READ tools still run

    estop.release()
    client.sync.publish(topic, Twist())
    assert invoke(drive).content == "driven"
    estop.uninstall()
    assert len(global_egress) == 0 and len(global_tool_hooks) == 0


def test_estop_cancels_goals_in_flight_and_cancels_pass(
    client: ROS2Connector, server: ROS2Connector
) -> None:
    name = serve_fibonacci(server)
    estop = EStop(types=MOVING).install(client)
    decisions: list[EgressDecision] = []
    client.egress.observe(decisions.append)
    first = client.sync.send_goal(name, Fibonacci.Goal(order=1), accept_timeout=10.0)
    second = client.sync.send_goal(name, Fibonacci.Goal(order=2), accept_timeout=10.0)
    assert {h.id for h in client._runtime.goals(MOVING)} == {first.id, second.id}

    futures = estop.engage("operator")
    assert len(futures) == 2
    assert all(f.result(timeout=10.0) for f in futures)  # the stop path is open
    for handle in (first, second):
        handle.sync.result(timeout=10.0)
        assert handle.status == GoalStatus.CANCELED
    assert client._runtime.goals() == []
    assert not [d for d in decisions if d.egress.kind == "cancel"]  # never gated
    with pytest.raises(EgressDenied):
        client.sync.send_goal(name, Fibonacci.Goal(order=3), accept_timeout=10.0)

    estop.release()
    handle = client.sync.send_goal(name, Fibonacci.Goal(order=4), accept_timeout=10.0)
    assert handle.sync.cancel(timeout=10.0)


def test_estop_stops_a_tool_that_is_running(
    client: ROS2Connector, server: ROS2Connector
) -> None:
    name = serve_fibonacci(server)
    estop = EStop(types=MOVING).install()
    drive = publisher_tool(client, unique("cmd_vel"))

    @rai_tool(Effect.ACTUATE)
    async def navigate() -> str:
        """Send a goal and wait for it."""
        handle = await client.send_goal(name, Fibonacci.Goal(order=1))
        await handle.result(timeout=20.0)
        return handle.status.name

    async def main() -> tuple[Any, Any]:
        calls = [
            asyncio.create_task(
                tool.ainvoke({"type": "tool_call", "name": n, "args": a, "id": n})
            )
            for tool, n, a in (
                (drive, "drive", {"seconds": 20.0}),
                (navigate, "navigate", {}),
            )
        ]
        await asyncio.sleep(0.5)
        estop.engage("mid-call")
        return tuple(await asyncio.wait_for(asyncio.gather(*calls), 10.0))

    driving, navigating = asyncio.run(main())
    assert driving.status == "error"
    assert driving.content.startswith("Denied by EStop (publish on /safety_cmd_vel_")
    assert navigating.content == "CANCELED"


def test_estop_cancels_a_goal_that_raced_the_engage(
    client: ROS2Connector, server: ROS2Connector
) -> None:
    name = serve_fibonacci(server)
    estop = EStop(types=MOVING).install(tools=None)
    estop.uninstall()  # no gate: only the goal watcher, as if the gate passed
    rt.watch_goals(estop._on_goal)
    estop._engaged = True
    handle = client.sync.send_goal(name, Fibonacci.Goal(order=1), accept_timeout=10.0)
    handle.sync.result(timeout=10.0)
    assert handle.status == GoalStatus.CANCELED


def test_estop_from_ros_topic_and_service(
    client: ROS2Connector, server: ROS2Connector
) -> None:
    estop = EStop().install(client).listen(server)  # the default names
    topic = unique("cmd_vel")

    publish_until(client, "/rai/estop", Bool(data=True), lambda: estop.engaged)
    assert "/rai/estop" in estop.reason
    with pytest.raises(EgressDenied):
        client.sync.publish(topic, Twist())
    publish_until(client, "/rai/estop", Bool(data=False), lambda: not estop.engaged)
    client.sync.publish(topic, Twist())

    response = client.sync.call(
        "/rai/estop/set", SetBool.Request(data=True), timeout=10.0
    )
    assert response.success and response.message == "engaged"
    assert estop.engaged
    with pytest.raises(EgressDenied):
        client.sync.publish(topic, Twist())
    response = client.sync.call(
        "/rai/estop/set", SetBool.Request(data=False), timeout=10.0
    )
    assert response.message == "released" and not estop.engaged
    client.sync.publish(topic, Twist())


def test_estop_on_engage_callbacks(ros_setup: None) -> None:
    estop = EStop()
    seen: list[str] = []
    estop.on_engage(lambda reason: 1 / 0)  # a failing callback does not stop others
    estop.on_engage(seen.append)
    estop.engage("why")
    assert seen == ["why"]


# ------------------------------------------------------------ ConfirmActuation


def test_confirm_tool_call_approved_once_for_its_egress(client: ROS2Connector) -> None:
    asker = FakeAsker(True)
    ConfirmActuation(asker).install()
    drive = publisher_tool(client, unique("cmd_vel"))
    assert invoke(drive, seconds=0.3).content == "driven"
    assert len(asker.questions) == 1  # the tool call; its publishes are covered
    assert "drive(seconds=0.3)" in asker.questions[0]
    assert invoke(look).content == "seen" and len(asker.questions) == 1


@pytest.mark.parametrize(
    ("asker", "reason"),
    [
        (FakeAsker(False), "the operator denied it"),
        (FakeAsker("yes"), "the operator denied it"),  # only True approves
        (FakeAsker(True, delay_s=5.0), "no answer from the operator within 0.2 s"),
    ],
)
def test_confirm_denials_reach_the_llm(
    client: ROS2Connector, asker: FakeAsker, reason: str
) -> None:
    ConfirmActuation(asker, timeout_s=0.2).install()
    message = invoke(publisher_tool(client, unique("cmd_vel")))
    assert message.status == "error"
    assert message.content == f"Denied by ConfirmActuation: {reason}"


def test_confirm_egress_lease_covers_a_30_hz_stream(client: ROS2Connector) -> None:
    asker = FakeAsker(True)
    confirm = ConfirmActuation(asker, lease_s=10.0).install(client, tools=None)
    topic = unique("cmd_vel")
    for _ in range(30):  # one second at 30 Hz
        client.sync.publish(topic, Twist())
        time.sleep(1 / 30)
    assert len(asker.questions) == 1
    assert f"publish {TWIST} on {topic}" in asker.questions[0]
    client.sync.publish(unique("cmd_vel"), Twist())  # another topic asks again
    assert len(asker.questions) == 2
    client.egress.revoke(confirm.egress_hook)
    client.sync.publish(topic, Twist())
    assert len(asker.questions) == 3

    asker.answer = False
    with pytest.raises(EgressDenied, match="the operator denied it"):
        client.sync.publish(unique("cmd_vel"), Twist())


def test_hri_asker(ros_setup: None) -> None:
    hri = ROS2HRIConnector(f"safety_hri_{uuid.uuid4().hex[:8]}")
    human = ROS2Connector(f"safety_human_{uuid.uuid4().hex[:8]}")
    to_human, from_human = unique("to_human"), unique("from_human")
    asker = HRIAsker(hri, to_human, from_human)
    reply = {"text": "yes"}

    def answer(msg: Any) -> None:
        human.sync.publish(from_human, HRIMessage(text=reply["text"]))

    human.subscribe(to_human, answer, msg_type="rai_interfaces/msg/HRIMessage")
    received: list[Any] = []
    hri.register_callback(from_human, received.append)
    try:
        # discovery both ways: warm-up questions nobody waits for are ignored
        deadline = time.monotonic() + 10.0
        while not received:
            assert time.monotonic() < deadline
            hri.send_message(hri.build_message(AIMessage("warm-up")), to_human)
            time.sleep(0.2)
        assert asyncio.run(asyncio.wait_for(asker("Move?"), 10.0)) is True
        reply["text"] = "no"
        assert asyncio.run(asyncio.wait_for(asker("Move?"), 10.0)) is False
    finally:
        asker.close()
        human.close()
        hri.close()


# ------------------------------------------------------------ RateLimit


def test_rate_limit_tools_and_egress(client: ROS2Connector) -> None:
    RateLimit(2, per_s=60.0, types={TWIST}).install()
    drive = publisher_tool(client, unique("cmd_vel"))
    assert invoke(drive).content == "driven"  # tool 1/2, its publish 1/2
    assert invoke(drive).content == "driven"
    message = invoke(drive)
    assert message.status == "error"
    assert message.content.startswith("Denied by RateLimit: rate limited to 2 per 60 s")
    assert invoke(look).content == "seen"  # READ tools are not limited

    topic = unique("cmd_vel")
    client.sync.publish(topic, Twist())
    client.sync.publish(topic, Twist())
    with pytest.raises(EgressDenied, match="rate limited"):
        client.sync.publish(topic, Twist())
    client.sync.publish(unique("cmd_vel"), Twist())  # per topic


def test_rate_limit_refills() -> None:
    limit = RateLimit(10, per_s=1.0, burst=1)
    assert limit.take("k") is None
    wait_s = limit.take("k")
    assert wait_s is not None and 0 < wait_s <= 0.1
    time.sleep(0.12)
    assert limit.take("k") is None


# ------------------------------------------------------------ AuditLog


def test_audit_records_both_hook_points(client: ROS2Connector, tmp_path: Path) -> None:
    estop = EStop().install()
    asker = FakeAsker(True)
    ConfirmActuation(asker, lease_s=10.0, tool_effects=()).install()
    audit = AuditLog(tmp_path / "audit.jsonl").install()
    seen: list[AuditRecord] = []
    audit.subscribe(seen.append)
    drive = publisher_tool(client, unique("cmd_vel"))
    topic = unique("cmd_vel")

    assert invoke(drive).content == "driven"  # tool allow; egress allow + lease
    client.sync.publish(topic, Twist())  # allow + lease
    client.sync.publish(topic, Twist())  # leased
    estop.engage("audit")
    invoke(drive)  # tool deny
    with pytest.raises(EgressDenied):
        client.sync.publish(topic, Twist())  # egress deny
    audit.close()

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == len(seen) and records == [r.to_dict() for r in seen]
    assert set(records[0]) >= {
        "time", "side", "origin", "kind", "name", "type", "decision", "hook",
        "reason", "lease_s", "leased",
    }  # fmt: skip
    tool = [r for r in records if r["side"] == "tool"]
    assert [(r["hook"], r["decision"]) for r in tool] == [
        ("EStop", "allow"),
        ("ConfirmActuation", "allow"),
        ("EStop", "deny"),
    ]
    assert tool[0]["name"] == "drive" and tool[0]["type"] == "actuate"
    assert tool[2]["reason"] == "e-stop engaged (audit)"
    egress = [r for r in records if r["side"] == "egress" and r["name"] == topic]
    assert [(r["hook"], r["decision"], r["lease_s"], r["leased"]) for r in egress] == [
        ("EStop", "allow", 0.0, False),
        ("ConfirmActuation", "allow", 10.0, False),
        ("EStop", "allow", 0.0, False),
        ("ConfirmActuation", "allow", 0.0, True),
        ("EStop", "deny", 0.0, False),
    ]
    assert egress[0]["origin"] is None and egress[0]["type"] == TWIST
    from_tool = [r for r in records if r["side"] == "egress" and r["origin"] == "drive"]
    assert from_tool and all(r["kind"] == "publish" for r in from_tool)


def test_audit_never_denies_or_blocks(client: ROS2Connector, tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl", maxsize=1).install(client)
    audit.subscribe(lambda record: time.sleep(0.2))  # a slow subscriber
    audit.subscribe(lambda record: 1 / 0)  # a failing one
    RateLimit(1000).install(client, tools=None)  # something to observe
    topic = unique("cmd_vel")
    start = time.monotonic()
    for _ in range(10):
        client.sync.publish(topic, Twist())
    assert time.monotonic() - start < 1.0  # the queue overflowed, nobody waited
    assert audit.dropped > 0
    audit.on_egress("not a decision")  # type: ignore[arg-type]
    audit.on_tool(ToolDecision(None, "h", True))  # type: ignore[arg-type]
    audit.close()


def test_audit_alone_never_gates(client: ROS2Connector) -> None:
    audit = AuditLog().install(client)  # observers only: nothing to decide
    assert len(client.egress) == 0
    client.sync.publish(unique("cmd_vel"), Twist())
    audit.close()


# ------------------------------------------------------------ install


def test_install_standard_set(client: ROS2Connector, tmp_path: Path) -> None:
    asker = FakeAsker(True)
    guard = safety.install(
        confirm=asker,
        rate_limits=[RateLimit(100, types={TWIST})],
        audit_path=tmp_path / "audit.jsonl",
        listen=False,
    )
    assert [e.name for e in global_egress._entries] == [
        "EStop",
        "RateLimit",
        "ConfirmActuation",
        "EStop",
    ]
    assert len(global_tool_hooks) == 4
    assert guard.estop is not None and guard.confirm is not None

    # an e-stop engaged while the human is being asked wins
    asker.before_answer = lambda: guard.estop.engage("during the question")  # type: ignore[union-attr]
    with pytest.raises(EgressDenied, match="EStop"):
        client.sync.publish(unique("cmd_vel"), Twist())
    guard.uninstall()
    assert len(global_egress) == 0 and len(global_tool_hooks) == 0
    assert not global_egress._observers and not global_tool_hooks._observers
    assert (tmp_path / "audit.jsonl").read_text()


def test_install_listens_on_ros(client: ROS2Connector) -> None:
    guard = safety.install(client)
    assert guard.confirm is None and guard.audit is None and not guard.rate_limits
    try:
        estop = guard.estop
        assert estop is not None
        publish_until(client, "/rai/estop", Bool(data=True), lambda: estop.engaged)
        with pytest.raises(EgressDenied):
            client.sync.publish(unique("cmd_vel"), Twist())
    finally:
        guard.uninstall()


# ------------------------------------------------------------ bypasses


def test_node_warns_once_per_call_site(client: ROS2Connector) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            client.node  # noqa: B018
        client.node  # noqa: B018
    messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert len(messages) == 2
    assert "egress hooks" in messages[0] and "EStop" in messages[0]


def test_cli_toolkit_has_no_write_commands() -> None:
    commands = set()
    for tool in ROS2CLIToolkit().get_tools():
        field = tool.args_schema.model_fields["command"]  # type: ignore[union-attr]
        commands |= set(typing.get_args(field.annotation))
    assert commands  # the toolkit is still there
    assert not commands & {"pub", "call", "send_goal", "set", "delete", "load"}
