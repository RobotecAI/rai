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

"""The manifest of a small live graph: publishers, a service, an action server
and one MOTION topic, served by a second node."""

import pytest

pytest.importorskip("rclpy")

import asyncio
import json
import time
import uuid
from typing import Any

from geometry_msgs.msg import Twist
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.manifest import (
    GetRobotManifestTool,
    ManifestPrompt,
    ManifestSource,
    RobotManifest,
    RobotSpec,
)
from rai.tools.ros2 import (
    GetROS2ImageConfiguredTool,
    ROS2ActionToolkit,
    ROS2ServicesToolkit,
    ROS2TopicsToolkit,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

TOKEN = f"mf{uuid.uuid4().hex[:8]}"
NS = f"/{TOKEN}"
STATUS, CAMERA, CMD_VEL = f"{NS}/status", f"{NS}/camera", f"{NS}/cmd_vel"
RESET, FIBONACCI = f"{NS}/reset", f"{NS}/fibonacci"
ROBOT_NODE = f"/{TOKEN}_robot"


def fibonacci(goal_handle: Any) -> Any:
    from example_interfaces.action import Fibonacci

    goal_handle.succeed()
    return Fibonacci.Result()


@pytest.fixture(scope="module")
def robot():
    """The robot side: three publishers, a service and an action server."""
    with ROS2Connector(f"{TOKEN}_robot") as robot:
        robot.sync.publish(STATUS, String(data="ok"))
        robot.sync.publish(CAMERA, Image())
        robot.sync.publish(CMD_VEL, Twist())
        robot.create_service(
            RESET, lambda req, res: res, service_type="std_srvs/srv/Trigger"
        )
        robot.create_action(
            FIBONACCI, fibonacci, action_type="example_interfaces/action/Fibonacci"
        )
        yield robot


@pytest.fixture(scope="module")
def connector(robot):
    with ROS2Connector(f"{TOKEN}_rai") as connector:
        yield connector


def spec() -> RobotSpec:
    return RobotSpec.model_validate(
        {
            "name": "testbot",
            "limits": {"max_linear_speed_mps": 0.5},
            "safety": ["Stop when asked."],
            "capabilities": [
                {"name": "drive", "topics": [CMD_VEL]},
                {"name": "count", "actions": [FIBONACCI]},
                {"name": "dock", "actions": [f"{NS}/dock"]},
            ],
            "forbidden": [RESET],
        }
    )


def tools(connector: ROS2Connector) -> list[Any]:
    access = dict(connector=connector, forbidden=[CAMERA])
    return [
        *ROS2TopicsToolkit(**access).get_tools(),
        *ROS2ServicesToolkit(**access).get_tools(),
        *ROS2ActionToolkit(**access).get_tools(),
        GetROS2ImageConfiguredTool(connector=connector, topic=CAMERA),
    ]


def wait_for(source: ManifestSource, check, timeout: float = 15.0) -> RobotManifest:
    """Fresh nodes take up to a few seconds to discover each other (Fast DDS)."""
    deadline = time.monotonic() + timeout
    while True:
        manifest = source.current()
        if check(manifest) or time.monotonic() > deadline:
            return manifest
        time.sleep(0.1)


def names(manifest: RobotManifest, kind: str) -> set[str]:
    return {e.name for e in manifest.entries(kind)}  # type: ignore[arg-type]


def complete(manifest: RobotManifest) -> bool:
    return (
        {STATUS, CAMERA, CMD_VEL} <= names(manifest, "topic")
        and RESET in names(manifest, "service")
        and FIBONACCI in names(manifest, "action")
        and ROBOT_NODE in manifest.nodes
    )


@pytest.fixture(scope="module")
def source(connector):
    source = ManifestSource(
        connector, spec=spec(), tools=tools(connector), include=[f"*{TOKEN}*"]
    )
    assert complete(wait_for(source, complete)), source.current().render()
    return source


def test_live_graph_is_annotated(source):
    manifest = source.current()
    assert manifest.nodes == [ROBOT_NODE]  # the manifest's own node is left out
    assert names(manifest, "topic") == {STATUS, CAMERA, CMD_VEL}
    assert names(manifest, "service") == {RESET}  # no parameter services
    assert names(manifest, "action") == {FIBONACCI}  # no _action internals
    topics = {e.name: e for e in manifest.topics}
    assert topics[CMD_VEL].motion and topics[CMD_VEL].capabilities == ["drive"]
    assert topics[CMD_VEL].access == "rw" and topics[STATUS].access == "rw"
    assert not topics[STATUS].motion
    assert topics[CAMERA].access == "" and topics[CAMERA].tools == [
        "get_ros2_camera_image"
    ]
    assert topics[STATUS].types == ["std_msgs/msg/String"]
    service, action = manifest.services[0], manifest.actions[0]
    assert service.forbidden and service.access == "w"
    assert action.access == "w" and action.capabilities == ["count"]
    assert manifest.generic_tools["topic"] == [
        "publish_ros2_message",
        "receive_ros2_message",
        "get_ros2_image",
    ]
    assert manifest.generic_tools["service"] == ["call_ros2_service"]
    assert manifest.generic_tools["action"] == ["start_ros2_action"]
    dock = {c.name: c for c in manifest.capabilities}["dock"]
    assert dock.missing == [f"{NS}/dock"]


def test_render_of_the_live_graph(source):
    text = source.render()
    assert text == source.current().render()
    assert f"{CMD_VEL} geometry_msgs/msg/Twist [MOTION, rw, cap:drive]" in text
    assert f"{RESET} std_srvs/srv/Trigger [FORBIDDEN, w]" in text
    assert f"{CAMERA} sensor_msgs/msg/Image [tool:get_ros2_camera_image]" in text
    assert "Generic tools: actions: start_ros2_action;" in text


def test_tool_returns_text_and_json(source):
    tool = GetRobotManifestTool(source=source)
    assert tool.effect.value == "read"
    text = asyncio.run(tool.ainvoke({}))
    assert "# Robot manifest: testbot" in text
    data = json.loads(asyncio.run(tool.ainvoke({"format": "json"})))
    assert data["name"] == "testbot" and data["fingerprint"]


class RecordingModel(BaseChatModel):
    """Records the system message of every call and answers 'ok'."""

    systems: list[str] = Field(default_factory=list)
    lengths: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = [m for m in messages if isinstance(m, SystemMessage)]
        self.systems.append(system[0].text if system else "")
        self.lengths.append(len(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


def test_refresh_on_graph_change_reaches_the_prompt(robot, connector):
    source = ManifestSource(connector, spec=spec(), include=[f"*{TOKEN}*"])
    wait_for(source, complete)
    model = RecordingModel()
    agent = ReActAgent(
        model,
        [],
        system_prompt="You drive testbot.",
        middleware=[ManifestPrompt(source, min_interval_s=0.0)],
    )
    asyncio.run(agent.ainvoke("hello"))
    first = model.systems[-1]
    assert first.startswith("You drive testbot.\n\n# Robot manifest: testbot")
    assert f"{NS}/added" not in first

    assert not source.changed() and source.refresh() is None
    robot.sync.publish(f"{NS}/added", String(data="new"))
    deadline = time.monotonic() + 15
    while not source.changed() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert source.changed()

    asyncio.run(agent.ainvoke("again"))
    second = model.systems[-1]
    assert f"{NS}/added std_msgs/msg/String" in second
    assert second.count("# Robot manifest") == 1
    # the manifest never enters the history: one system message per call,
    # and the history holds only the turns
    assert all(not isinstance(m, SystemMessage) for m in agent.history)
    assert not any("Robot manifest" in m.text for m in agent.history)
    assert model.lengths == [2, 4]
